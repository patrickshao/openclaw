import { type ChildProcessWithoutNullStreams, spawn } from "node:child_process";
import fs from "node:fs/promises";
import { createInterface, type Interface } from "node:readline";
import type { ResolvedQmdDaemonConfig } from "./backend-config.js";
import { createSubsystemLogger } from "../logging/subsystem.js";

const log = createSubsystemLogger("memory");

const SIGKILL_GRACE_MS = 5_000;
const MAX_BACKOFF_MS = 30_000;
const STABILITY_RESET_MS = 60_000;

type DaemonState = "stopped" | "starting" | "ready" | "error";

type PendingRequest = {
  resolve: (value: unknown) => void;
  reject: (error: Error) => void;
  timer?: NodeJS.Timeout;
};

type JsonRpcResponse = {
  jsonrpc?: string;
  id?: string | number | null;
  result?: unknown;
  error?: { code?: number; message?: string; data?: unknown };
};

export type QmdDaemonParams = {
  command: string;
  env: NodeJS.ProcessEnv;
  cwd: string;
  pidFilePath: string;
  daemonConfig: ResolvedQmdDaemonConfig;
};

export interface QmdDaemonQueryResult {
  docid?: string;
  file?: string;
  title?: string;
  score?: number;
  snippet?: string;
  context?: string | null;
}

export class QmdDaemon {
  private child: ChildProcessWithoutNullStreams | null = null;
  private reader: Interface | null = null;
  private readonly pending = new Map<string, PendingRequest>();
  private nextId = 1;
  private state: DaemonState = "stopped";
  private lastQueryAt = 0;
  private idleTimer: NodeJS.Timeout | null = null;
  private backoffMs = 1_000;
  private lastStartAt = 0;
  private restartAttempt = 0;
  private lastCrashReason = "startup";
  private startPromise: Promise<void> | null = null;
  private initialized = false;
  private queryMutex: Promise<void> = Promise.resolve();

  private readonly command: string;
  private readonly env: NodeJS.ProcessEnv;
  private readonly cwd: string;
  private readonly pidFilePath: string;
  private readonly config: ResolvedQmdDaemonConfig;

  constructor(params: QmdDaemonParams) {
    this.command = params.command;
    this.env = params.env;
    this.cwd = params.cwd;
    this.pidFilePath = params.pidFilePath;
    this.config = params.daemonConfig;
  }

  isReady(): boolean {
    return this.state === "ready";
  }

  /**
   * Verify daemon responsiveness using a lightweight MCP request.
   * If the daemon is wedged, tear it down so the caller can restart cleanly.
   */
  async ensureHealthy(timeoutMs = 2_000): Promise<boolean> {
    if (!this.isReady()) {
      return false;
    }
    try {
      await this.rpcRequest("tools/list", undefined, { timeoutMs });
      return true;
    } catch (err) {
      log.warn(`qmd daemon health check failed: ${String(err)}`);
      this.handleCrash("health_check_failed");
      return false;
    }
  }

  /** Clean up orphaned daemon from a previous run via PID file. */
  async cleanupOrphan(): Promise<void> {
    try {
      const pidStr = await fs.readFile(this.pidFilePath, "utf-8").catch(() => null);
      if (!pidStr) {
        return;
      }
      const pid = parseInt(pidStr.trim(), 10);
      if (!Number.isFinite(pid) || pid <= 0) {
        await fs.rm(this.pidFilePath, { force: true });
        return;
      }
      try {
        // Check if process exists
        process.kill(pid, 0);
        // Process exists — kill it
        log.warn(`killing orphaned qmd daemon (pid ${pid})`);
        process.kill(pid, "SIGTERM");
        // Give it a moment then force-kill
        await new Promise<void>((resolve) => setTimeout(resolve, 2_000));
        try {
          process.kill(pid, 0);
          process.kill(pid, "SIGKILL");
        } catch {
          // Already dead
        }
      } catch {
        // Process doesn't exist — stale PID file
      }
      await fs.rm(this.pidFilePath, { force: true });
    } catch (err) {
      log.debug(`orphan cleanup failed: ${String(err)}`);
    }
  }

  async start(): Promise<void> {
    if (this.state === "ready") {
      return;
    }
    if (this.startPromise) {
      return this.startPromise;
    }
    this.startPromise = this.doStart().finally(() => {
      this.startPromise = null;
    });
    return this.startPromise;
  }

  private async doStart(): Promise<void> {
    this.state = "starting";
    try {
      this.lastStartAt = Date.now();

      const child = spawn(this.command, ["mcp"], {
        stdio: ["pipe", "pipe", "pipe"],
        env: this.env,
        cwd: this.cwd,
      });
      this.child = child;
      this.reader = createInterface({ input: child.stdout });

      this.reader.on("line", (line) => {
        const trimmed = line.trim();
        if (!trimmed) {
          return;
        }
        this.handleLine(trimmed);
      });

      child.stderr?.on("data", (chunk: Buffer) => {
        const lines = chunk.toString().split(/\r?\n/);
        for (const line of lines) {
          if (!line.trim()) {
            continue;
          }
          log.debug(`qmd daemon stderr: ${line.trim()}`);
        }
      });

      child.on("error", (err) => {
        this.failAll(err instanceof Error ? err : new Error(String(err)));
        if (this.state === "ready" || this.state === "starting") {
          this.handleCrash("child_error");
        }
      });

      child.on("close", (code, signal) => {
        if (code !== 0 && code !== null) {
          const reason = signal ? `signal ${signal}` : `code ${code}`;
          this.failAll(new Error(`qmd daemon exited (${reason})`));
        } else {
          this.failAll(new Error("qmd daemon closed"));
        }
        if (this.state === "ready" || this.state === "starting") {
          log.debug("qmd daemon process closed (will restart on next query)");
          const closeReason =
            code !== null && code !== undefined
              ? `child_close_code_${code}`
              : signal
                ? `child_close_signal_${signal.toLowerCase()}`
                : "child_close";
          this.handleCrash(closeReason);
        }
      });

      // Write PID file
      if (child.pid) {
        await fs.writeFile(this.pidFilePath, String(child.pid), "utf-8");
      }

      // MCP initialize handshake
      await this.rpcRequest(
        "initialize",
        {
          protocolVersion: "2024-11-05",
          capabilities: {},
          clientInfo: { name: "openclaw-qmd", version: "1.0.0" },
        },
        { timeoutMs: this.config.coldStartTimeoutMs },
      );

      // Send initialized notification (no id = notification, no response expected)
      this.rpcNotify("notifications/initialized");

      this.initialized = true;
      this.state = "ready";
      this.resetBackoff();
      this.resetIdleTimer();
      this.restartAttempt = 0;
      this.lastCrashReason = "none";
      log.info("qmd daemon started (ready)");
    } catch (err) {
      this.state = "error";
      log.debug(`qmd daemon start failed: ${String(err)}`);
      await this.cleanup();
      throw err;
    }
  }

  async stop(): Promise<void> {
    this.clearIdleTimer();
    if (this.state === "stopped") {
      return;
    }
    this.state = "stopped";
    await this.cleanup();
    log.info("qmd daemon stopped");
  }

  async query(
    text: string,
    opts: { tool: string; limit: number; collection?: string; timeoutMs: number },
  ): Promise<QmdDaemonQueryResult[]> {
    return await this.withQueryLock(async () => {
      if (!this.child || this.state !== "ready") {
        throw new Error("qmd daemon not ready");
      }

      this.lastQueryAt = Date.now();
      this.resetIdleTimer();

      const result = await this.rpcRequest<{
        content?: Array<{ type: string; text?: string }>;
        structuredContent?: { results?: QmdDaemonQueryResult[] };
      }>(
        "tools/call",
        {
          name: opts.tool,
          arguments: {
            query: text,
            limit: opts.limit,
            ...(opts.collection ? { collection: opts.collection } : {}),
          },
        },
        { timeoutMs: opts.timeoutMs },
      );

      // Check for MCP-level errors (isError flag)
      if ((result as Record<string, unknown>)?.isError) {
        const errText =
          Array.isArray(result?.content) && result.content[0]?.type === "text"
            ? (result.content[0] as { text: string }).text
            : "unknown daemon error";
        throw new Error(`qmd daemon tool error: ${errText}`);
      }

      // Parse result — QMD returns structured data in structuredContent.results
      if (result?.structuredContent?.results && Array.isArray(result.structuredContent.results)) {
        return result.structuredContent.results;
      }

      // Fallback: try parsing text content as JSON
      if (Array.isArray(result?.content) && result.content.length > 0) {
        const first = result.content[0];
        if (first?.type === "text" && typeof first.text === "string") {
          try {
            const parsed: unknown = JSON.parse(first.text);
            if (Array.isArray(parsed)) {
              return parsed as QmdDaemonQueryResult[];
            }
          } catch {
            // Not JSON — structuredContent is preferred anyway
          }
        }
      }
      return [];
    });
  }

  private async withQueryLock<T>(fn: () => Promise<T>): Promise<T> {
    const previous = this.queryMutex;
    let release: () => void = () => undefined;
    this.queryMutex = new Promise<void>((resolve) => {
      release = resolve;
    });
    await previous.catch(() => undefined);
    try {
      return await fn();
    } finally {
      release();
    }
  }

  // --- JSON-RPC over stdio (same pattern as iMessage/Signal clients) ---

  private async rpcRequest<T = unknown>(
    method: string,
    params?: Record<string, unknown>,
    opts?: { timeoutMs?: number },
  ): Promise<T> {
    if (!this.child?.stdin) {
      throw new Error("qmd daemon not running");
    }
    const id = this.nextId++;
    const payload = { jsonrpc: "2.0", id, method, params: params ?? {} };
    const line = `${JSON.stringify(payload)}\n`;
    const timeoutMs = opts?.timeoutMs ?? 10_000;

    const response = new Promise<T>((resolve, reject) => {
      const key = String(id);
      const timer =
        timeoutMs > 0
          ? setTimeout(() => {
              this.pending.delete(key);
              if (
                method === "tools/call" &&
                (this.state === "ready" || this.state === "starting")
              ) {
                log.warn(
                  `qmd daemon request timed out (${method}, ${timeoutMs}ms); resetting daemon`,
                );
                this.handleCrash(`rpc_timeout_${method.replace("/", "_")}`);
              }
              reject(new Error(`qmd daemon rpc timeout (${method})`));
            }, timeoutMs)
          : undefined;
      this.pending.set(key, {
        resolve: (value) => resolve(value as T),
        reject,
        timer,
      });
    });

    this.child.stdin.write(line);
    return await response;
  }

  private rpcNotify(method: string, params?: Record<string, unknown>): void {
    if (!this.child?.stdin) {
      return;
    }
    // Notifications have no id — server won't respond
    const payload = { jsonrpc: "2.0", method, params: params ?? {} };
    this.child.stdin.write(`${JSON.stringify(payload)}\n`);
  }

  private handleLine(line: string): void {
    let parsed: JsonRpcResponse;
    try {
      parsed = JSON.parse(line) as JsonRpcResponse;
    } catch {
      log.debug(`qmd daemon: failed to parse: ${line.slice(0, 200)}`);
      return;
    }

    if (parsed.id !== undefined && parsed.id !== null) {
      const key = String(parsed.id);
      const pending = this.pending.get(key);
      if (!pending) {
        return;
      }
      if (pending.timer) {
        clearTimeout(pending.timer);
      }
      this.pending.delete(key);

      if (parsed.error) {
        const msg = parsed.error.message ?? "qmd daemon rpc error";
        pending.reject(new Error(msg));
      } else {
        pending.resolve(parsed.result);
      }
    }
    // Notifications from server (no id) are ignored for now
  }

  private failAll(err: Error): void {
    for (const [_key, pending] of this.pending) {
      if (pending.timer) {
        clearTimeout(pending.timer);
      }
      pending.reject(err);
    }
    this.pending.clear();
  }

  // --- Lifecycle ---

  private handleCrash(reason = "unknown"): void {
    if (this.state === "stopped") {
      return;
    }
    this.state = "error";
    this.initialized = false;
    this.lastCrashReason = reason;
    this.cleanup().catch(() => undefined);

    const timeSinceStart = Date.now() - this.lastStartAt;
    if (timeSinceStart > STABILITY_RESET_MS) {
      this.resetBackoff();
      this.restartAttempt = 1;
    } else {
      this.backoffMs = Math.min(this.backoffMs * 2, MAX_BACKOFF_MS);
      this.restartAttempt += 1;
    }
    const uptimeMs = Math.max(0, timeSinceStart);
    log.info(
      `qmd daemon restart scheduled reason=${reason} attempt=${this.restartAttempt} uptimeMs=${uptimeMs} backoffMs=${this.backoffMs}`,
    );
  }

  private resetBackoff(): void {
    this.backoffMs = 1_000;
  }

  /** Wait for backoff period if needed before restarting. */
  async waitForBackoff(): Promise<void> {
    if (this.backoffMs > 1_000) {
      log.info(
        `qmd daemon restart backoff: ${this.backoffMs}ms reason=${this.lastCrashReason} attempt=${this.restartAttempt}`,
      );
      await new Promise<void>((resolve) => setTimeout(resolve, this.backoffMs));
    }
  }

  private resetIdleTimer(): void {
    this.clearIdleTimer();
    const timeout = this.config.idleTimeoutMs;
    if (timeout <= 0) {
      return;
    }
    this.idleTimer = setTimeout(() => {
      this.onIdle();
    }, timeout);
  }

  private clearIdleTimer(): void {
    if (this.idleTimer) {
      clearTimeout(this.idleTimer);
      this.idleTimer = null;
    }
  }

  private onIdle(): void {
    log.info("qmd daemon idle timeout — shutting down");
    void this.stop();
  }

  private async cleanup(): Promise<void> {
    this.clearIdleTimer();
    this.initialized = false;

    this.reader?.close();
    this.reader = null;

    if (this.child) {
      const child = this.child;
      this.child = null;

      child.stdin?.end();

      try {
        child.kill("SIGTERM");
        await new Promise<void>((resolve) => {
          const timer = setTimeout(() => {
            try {
              child.kill("SIGKILL");
            } catch {
              // ignore
            }
            resolve();
          }, SIGKILL_GRACE_MS);
          child.on("exit", () => {
            clearTimeout(timer);
            resolve();
          });
        });
      } catch {
        // ignore
      }
    }

    // Reject any remaining pending requests
    this.failAll(new Error("qmd daemon stopped"));

    // Remove PID file
    await fs.rm(this.pidFilePath, { force: true }).catch(() => undefined);
  }
}
