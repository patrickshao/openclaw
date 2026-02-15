import { spawn } from "node:child_process";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import type { ResolvedQmdDaemonConfig } from "./backend-config.js";
import { createSubsystemLogger } from "../logging/subsystem.js";

const log = createSubsystemLogger("memory");

const QMD_HTTP_PORT = 8181;
const QMD_HTTP_URL = `http://localhost:${QMD_HTTP_PORT}/mcp`;
const QMD_PID_FILE = path.join(os.homedir(), ".cache", "qmd", "mcp.pid");
const MAX_BACKOFF_MS = 30_000;
const STABILITY_RESET_MS = 60_000;
const HEALTH_POLL_INTERVAL_MS = 200;
const HEALTH_POLL_MAX_MS = 10_000;

type DaemonState = "stopped" | "starting" | "ready" | "error";

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
  private nextId = 1;
  private state: DaemonState = "stopped";
  private lastQueryAt = 0;
  private idleTimer: NodeJS.Timeout | null = null;
  private backoffMs = 1_000;
  private lastStartAt = 0;
  private startPromise: Promise<void> | null = null;

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

  /** Clean up orphaned daemon from a previous run. Uses `qmd mcp stop` which reads the PID file. */
  async cleanupOrphan(): Promise<void> {
    try {
      const pidStr = await fs.readFile(QMD_PID_FILE, "utf-8").catch(() => null);
      if (!pidStr) {
        return;
      }
      const pid = parseInt(pidStr.trim(), 10);
      if (!Number.isFinite(pid) || pid <= 0) {
        await fs.rm(QMD_PID_FILE, { force: true });
        return;
      }
      try {
        process.kill(pid, 0); // Check if alive
        log.warn(`killing orphaned qmd HTTP daemon (pid ${pid})`);
        await this.runCommand(["mcp", "stop"], 5_000);
      } catch {
        // Process doesn't exist — stale PID file
        await fs.rm(QMD_PID_FILE, { force: true });
      }
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

      // Launch the HTTP daemon (detaches itself)
      await this.runCommand(["mcp", "--http", "--daemon"], this.config.coldStartTimeoutMs);

      // Poll until the daemon is accepting HTTP requests
      await this.pollUntilReady();

      this.state = "ready";
      this.resetBackoff();
      this.resetIdleTimer();
      log.info("qmd HTTP daemon started (ready)");
    } catch (err) {
      this.state = "error";
      log.debug(`qmd HTTP daemon start failed: ${String(err)}`);
      // Try to stop any partially started daemon
      await this.runCommand(["mcp", "stop"], 5_000).catch(() => undefined);
      throw err;
    }
  }

  private async pollUntilReady(): Promise<void> {
    const deadline = Date.now() + HEALTH_POLL_MAX_MS;
    while (Date.now() < deadline) {
      try {
        const resp = await fetch(QMD_HTTP_URL, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            jsonrpc: "2.0",
            id: this.nextId++,
            method: "initialize",
            params: {
              protocolVersion: "2024-11-05",
              capabilities: {},
              clientInfo: { name: "openclaw-qmd", version: "1.0.0" },
            },
          }),
          signal: AbortSignal.timeout(2_000),
        });
        if (resp.ok) {
          return;
        }
      } catch {
        // Not ready yet
      }
      await new Promise<void>((resolve) => setTimeout(resolve, HEALTH_POLL_INTERVAL_MS));
    }
    throw new Error("qmd HTTP daemon failed to become ready within timeout");
  }

  async stop(): Promise<void> {
    this.clearIdleTimer();
    if (this.state === "stopped") {
      return;
    }
    this.state = "stopped";
    try {
      await this.runCommand(["mcp", "stop"], 5_000);
    } catch (err) {
      log.debug(`qmd mcp stop failed: ${String(err)}`);
    }
    log.info("qmd HTTP daemon stopped");
  }

  async query(
    text: string,
    opts: { tool: string; limit: number; collection?: string; timeoutMs: number },
  ): Promise<QmdDaemonQueryResult[]> {
    if (this.state !== "ready") {
      throw new Error("qmd daemon not ready");
    }

    this.lastQueryAt = Date.now();
    this.resetIdleTimer();

    const id = this.nextId++;
    const payload = {
      jsonrpc: "2.0",
      id,
      method: "tools/call",
      params: {
        name: opts.tool,
        arguments: {
          query: text,
          limit: opts.limit,
          ...(opts.collection ? { collection: opts.collection } : {}),
        },
      },
    };

    let resp: Response;
    try {
      resp = await fetch(QMD_HTTP_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
        signal: AbortSignal.timeout(opts.timeoutMs),
      });
    } catch (err) {
      // Fetch failed — daemon may have crashed. Mark as error for restart.
      this.handleCrash();
      throw new Error(`qmd HTTP daemon request failed: ${String(err)}`, { cause: err });
    }

    if (!resp.ok) {
      const body = await resp.text().catch(() => "");
      throw new Error(`qmd HTTP daemon returned ${resp.status}: ${body.slice(0, 500)}`);
    }

    const json = (await resp.json()) as JsonRpcResponse;

    if (json.error) {
      const msg = json.error.message ?? "qmd daemon rpc error";
      throw new Error(msg);
    }

    const result = json.result as
      | {
          isError?: boolean;
          content?: Array<{ type: string; text?: string }>;
          structuredContent?: { results?: QmdDaemonQueryResult[] };
        }
      | undefined;

    // Check for MCP-level errors (isError flag)
    if (result?.isError) {
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
          // Not JSON
        }
      }
    }
    return [];
  }

  // --- Lifecycle ---

  private handleCrash(): void {
    if (this.state === "stopped") {
      return;
    }
    this.state = "error";
    this.clearIdleTimer();

    const timeSinceStart = Date.now() - this.lastStartAt;
    if (timeSinceStart > STABILITY_RESET_MS) {
      this.resetBackoff();
    } else {
      this.backoffMs = Math.min(this.backoffMs * 2, MAX_BACKOFF_MS);
    }
  }

  private resetBackoff(): void {
    this.backoffMs = 1_000;
  }

  /** Wait for backoff period if needed before restarting. */
  async waitForBackoff(): Promise<void> {
    if (this.backoffMs > 1_000) {
      log.info(`qmd daemon restart backoff: ${this.backoffMs}ms`);
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

  /** Run a qmd CLI command and wait for it to complete. */
  private runCommand(args: string[], timeoutMs: number): Promise<void> {
    return new Promise((resolve, reject) => {
      const child = spawn(this.command, args, {
        stdio: ["ignore", "pipe", "pipe"],
        env: this.env,
        cwd: this.cwd,
      });
      let stderr = "";
      child.stderr?.on("data", (chunk: Buffer) => {
        stderr += chunk.toString();
      });
      const timer = setTimeout(() => {
        child.kill("SIGKILL");
        reject(new Error(`qmd ${args.join(" ")} timed out after ${timeoutMs}ms`));
      }, timeoutMs);
      child.on("error", (err) => {
        clearTimeout(timer);
        reject(err);
      });
      child.on("close", (code) => {
        clearTimeout(timer);
        if (code === 0) {
          resolve();
        } else {
          reject(new Error(`qmd ${args.join(" ")} failed (code ${code}): ${stderr.slice(0, 500)}`));
        }
      });
    });
  }
}
