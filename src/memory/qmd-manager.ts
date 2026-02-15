import { spawn } from "node:child_process";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import readline from "node:readline";
import type { OpenClawConfig } from "../config/config.js";
import type {
  MemoryEmbeddingProbeResult,
  MemoryProviderStatus,
  MemorySearchManager,
  MemorySearchResult,
  MemorySource,
  MemorySyncProgressUpdate,
} from "./types.js";
import { resolveAgentWorkspaceDir } from "../agents/agent-scope.js";
import { resolveStateDir } from "../config/paths.js";
import { createSubsystemLogger } from "../logging/subsystem.js";
import { deriveQmdScopeChannel, deriveQmdScopeChatType, isQmdScopeAllowed } from "./qmd-scope.js";
import {
  listSessionFilesForAgent,
  buildSessionEntry,
  type SessionFileEntry,
} from "./session-files.js";
import { requireNodeSqlite } from "./sqlite.js";

type SqliteDatabase = import("node:sqlite").DatabaseSync;
import type {
  ResolvedMemoryBackendConfig,
  ResolvedQmdConfig,
  ResolvedQmdDaemonConfig,
} from "./backend-config.js";
import { parseQmdQueryJson, type QmdQueryResult } from "./qmd-query-parser.js";

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

export interface QmdDaemonQueryResult {
  docid?: string;
  file?: string;
  title?: string;
  score?: number;
  snippet?: string;
  context?: string | null;
}

const log = createSubsystemLogger("memory");

const SNIPPET_HEADER_RE = /@@\s*-([0-9]+),([0-9]+)/;
const SEARCH_PENDING_UPDATE_WAIT_MS = 500;
const MAX_QMD_OUTPUT_CHARS = 200_000;
const NUL_MARKER_RE = /(?:\^@|\\0|\\x00|\\u0000|null\s*byte|nul\s*byte)/i;

type CollectionRoot = {
  path: string;
  kind: MemorySource;
};

type SessionExporterConfig = {
  dir: string;
  retentionMs?: number;
  collectionName: string;
};

type QmdManagerMode = "full" | "status";

export class QmdMemoryManager implements MemorySearchManager {
  static async create(params: {
    cfg: OpenClawConfig;
    agentId: string;
    resolved: ResolvedMemoryBackendConfig;
    mode?: QmdManagerMode;
  }): Promise<QmdMemoryManager | null> {
    const resolved = params.resolved.qmd;
    if (!resolved) {
      return null;
    }
    const manager = new QmdMemoryManager({ cfg: params.cfg, agentId: params.agentId, resolved });
    await manager.initialize(params.mode ?? "full");
    return manager;
  }

  private readonly cfg: OpenClawConfig;
  private readonly agentId: string;
  private readonly qmd: ResolvedQmdConfig;
  private readonly workspaceDir: string;
  private readonly stateDir: string;
  private readonly agentStateDir: string;
  private readonly qmdDir: string;
  private readonly xdgConfigHome: string;
  private readonly xdgCacheHome: string;
  private readonly indexPath: string;
  private readonly env: NodeJS.ProcessEnv;
  private readonly collectionRoots = new Map<string, CollectionRoot>();
  private readonly sources = new Set<MemorySource>();
  private readonly docPathCache = new Map<
    string,
    { rel: string; abs: string; source: MemorySource }
  >();
  private readonly exportedSessionState = new Map<
    string,
    {
      hash: string;
      mtimeMs: number;
      target: string;
    }
  >();
  private readonly maxQmdOutputChars = MAX_QMD_OUTPUT_CHARS;
  private readonly sessionExporter: SessionExporterConfig | null;
  private daemonEnabled: boolean = false;
  private daemonConfig: ResolvedQmdDaemonConfig | null = null;
  private daemonState: DaemonState = "stopped";
  private daemonNextId = 1;
  private daemonLastQueryAt = 0;
  private daemonIdleTimer: NodeJS.Timeout | null = null;
  private daemonBackoffMs = 1_000;
  private daemonLastStartAt = 0;
  private daemonStartPromise: Promise<void> | null = null;
  private updateTimer: NodeJS.Timeout | null = null;
  private pendingUpdate: Promise<void> | null = null;
  private queuedForcedUpdate: Promise<void> | null = null;
  private queuedForcedRuns = 0;
  private closed = false;
  private db: SqliteDatabase | null = null;
  private lastUpdateAt: number | null = null;
  private lastEmbedAt: number | null = null;
  private attemptedNullByteCollectionRepair = false;

  private constructor(params: {
    cfg: OpenClawConfig;
    agentId: string;
    resolved: ResolvedQmdConfig;
  }) {
    this.cfg = params.cfg;
    this.agentId = params.agentId;
    this.qmd = params.resolved;
    this.workspaceDir = resolveAgentWorkspaceDir(params.cfg, params.agentId);
    this.stateDir = resolveStateDir(process.env, os.homedir);
    this.agentStateDir = path.join(this.stateDir, "agents", this.agentId);
    this.qmdDir = path.join(this.agentStateDir, "qmd");
    // QMD uses XDG base dirs for its internal state.
    // Collections are managed via `qmd collection add` and stored inside the index DB.
    // - config:  $XDG_CONFIG_HOME (contexts, etc.)
    // - cache:   $XDG_CACHE_HOME/qmd/index.sqlite
    this.xdgConfigHome = path.join(this.qmdDir, "xdg-config");
    this.xdgCacheHome = path.join(this.qmdDir, "xdg-cache");
    this.indexPath = path.join(this.xdgCacheHome, "qmd", "index.sqlite");

    this.env = {
      ...process.env,
      XDG_CONFIG_HOME: this.xdgConfigHome,
      XDG_CACHE_HOME: this.xdgCacheHome,
      NO_COLOR: "1",
    };
    this.sessionExporter = this.qmd.sessions.enabled
      ? {
          dir: this.qmd.sessions.exportDir ?? path.join(this.qmdDir, "sessions"),
          retentionMs: this.qmd.sessions.retentionDays
            ? this.qmd.sessions.retentionDays * 24 * 60 * 60 * 1000
            : undefined,
          collectionName: this.pickSessionCollectionName(),
        }
      : null;
    if (this.qmd.daemon.enabled) {
      this.daemonEnabled = true;
      this.daemonConfig = this.qmd.daemon;
    }
    if (this.sessionExporter) {
      this.qmd.collections = [
        ...this.qmd.collections,
        {
          name: this.sessionExporter.collectionName,
          path: this.sessionExporter.dir,
          pattern: "**/*.md",
          kind: "sessions",
        },
      ];
    }
  }

  private async initialize(mode: QmdManagerMode): Promise<void> {
    this.bootstrapCollections();
    if (mode === "status") {
      return;
    }

    if (this.daemonEnabled) {
      await this.daemonCleanupOrphan();
    }
    await fs.mkdir(this.xdgConfigHome, { recursive: true });
    await fs.mkdir(this.xdgCacheHome, { recursive: true });
    await fs.mkdir(path.dirname(this.indexPath), { recursive: true });

    // QMD stores its ML models under $XDG_CACHE_HOME/qmd/models/.  Because we
    // override XDG_CACHE_HOME to isolate the index per-agent, qmd would not
    // find models installed at the default location (~/.cache/qmd/models/) and
    // would attempt to re-download them on every invocation.  Symlink the
    // default models directory into our custom cache so the index stays
    // isolated while models are shared.
    await this.symlinkSharedModels();

    await this.ensureCollections();

    if (this.qmd.update.onBoot) {
      const bootRun = this.runUpdate("boot", true);
      if (this.qmd.update.waitForBootSync) {
        await bootRun.catch((err) => {
          log.warn(`qmd boot update failed: ${String(err)}`);
        });
      } else {
        void bootRun.catch((err) => {
          log.warn(`qmd boot update failed: ${String(err)}`);
        });
      }
    }
    if (this.qmd.update.intervalMs > 0) {
      this.updateTimer = setInterval(() => {
        void this.runUpdate("interval").catch((err) => {
          log.warn(`qmd update failed (${String(err)})`);
        });
      }, this.qmd.update.intervalMs);
    }
  }

  private bootstrapCollections(): void {
    this.collectionRoots.clear();
    this.sources.clear();
    for (const collection of this.qmd.collections) {
      const kind: MemorySource = collection.kind === "sessions" ? "sessions" : "memory";
      this.collectionRoots.set(collection.name, { path: collection.path, kind });
      this.sources.add(kind);
    }
  }

  private async ensureCollections(): Promise<void> {
    // QMD collections are persisted inside the index database and must be created
    // via the CLI. Prefer listing existing collections when supported, otherwise
    // fall back to best-effort idempotent `qmd collection add`.
    const existing = new Set<string>();
    try {
      const result = await this.runQmd(["collection", "list", "--json"], {
        timeoutMs: this.qmd.update.commandTimeoutMs,
      });
      const parsed = JSON.parse(result.stdout) as unknown;
      if (Array.isArray(parsed)) {
        for (const entry of parsed) {
          if (typeof entry === "string") {
            existing.add(entry);
          } else if (entry && typeof entry === "object") {
            const name = (entry as { name?: unknown }).name;
            if (typeof name === "string") {
              existing.add(name);
            }
          }
        }
      }
    } catch {
      // ignore; older qmd versions might not support list --json.
    }

    for (const collection of this.qmd.collections) {
      if (existing.has(collection.name)) {
        continue;
      }
      try {
        await this.addCollection(collection.path, collection.name, collection.pattern);
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        if (this.isCollectionAlreadyExistsError(message)) {
          continue;
        }
        log.warn(`qmd collection add failed for ${collection.name}: ${message}`);
      }
    }
  }

  private isCollectionAlreadyExistsError(message: string): boolean {
    const lower = message.toLowerCase();
    return lower.includes("already exists") || lower.includes("exists");
  }

  private isCollectionMissingError(message: string): boolean {
    const lower = message.toLowerCase();
    return (
      lower.includes("not found") || lower.includes("does not exist") || lower.includes("missing")
    );
  }

  private async addCollection(pathArg: string, name: string, pattern: string): Promise<void> {
    await this.runQmd(["collection", "add", pathArg, "--name", name, "--mask", pattern], {
      timeoutMs: this.qmd.update.commandTimeoutMs,
    });
  }

  private async removeCollection(name: string): Promise<void> {
    await this.runQmd(["collection", "remove", name], {
      timeoutMs: this.qmd.update.commandTimeoutMs,
    });
  }

  private shouldRepairNullByteCollectionError(err: unknown): boolean {
    const message = err instanceof Error ? err.message : String(err);
    const lower = message.toLowerCase();
    return (
      (lower.includes("enotdir") || lower.includes("not a directory")) &&
      NUL_MARKER_RE.test(message)
    );
  }

  private async tryRepairNullByteCollections(err: unknown, reason: string): Promise<boolean> {
    if (this.attemptedNullByteCollectionRepair) {
      return false;
    }
    if (!this.shouldRepairNullByteCollectionError(err)) {
      return false;
    }
    this.attemptedNullByteCollectionRepair = true;
    log.warn(
      `qmd update failed with suspected null-byte collection metadata (${reason}); rebuilding managed collections and retrying once`,
    );
    for (const collection of this.qmd.collections) {
      try {
        await this.removeCollection(collection.name);
      } catch (removeErr) {
        const removeMessage = removeErr instanceof Error ? removeErr.message : String(removeErr);
        if (!this.isCollectionMissingError(removeMessage)) {
          log.warn(`qmd collection remove failed for ${collection.name}: ${removeMessage}`);
        }
      }
      try {
        await this.addCollection(collection.path, collection.name, collection.pattern);
      } catch (addErr) {
        const addMessage = addErr instanceof Error ? addErr.message : String(addErr);
        if (!this.isCollectionAlreadyExistsError(addMessage)) {
          log.warn(`qmd collection add failed for ${collection.name}: ${addMessage}`);
        }
      }
    }
    return true;
  }

  async search(
    query: string,
    opts?: { maxResults?: number; minScore?: number; sessionKey?: string },
  ): Promise<MemorySearchResult[]> {
    if (!this.isScopeAllowed(opts?.sessionKey)) {
      this.logScopeDenied(opts?.sessionKey);
      return [];
    }
    const trimmed = query.trim();
    if (!trimmed) {
      return [];
    }
    await this.waitForPendingUpdateBeforeSearch();
    const limit = Math.min(
      this.qmd.limits.maxResults,
      opts?.maxResults ?? this.qmd.limits.maxResults,
    );
    const collectionNames = this.listManagedCollectionNames();

    // Try daemon path first
    const daemonResults = await this.tryDaemonSearch(trimmed, limit, collectionNames);
    if (daemonResults !== null) {
      return daemonResults;
    }

    if (collectionNames.length === 0) {
      log.warn("qmd query skipped: no managed collections configured");
      return [];
    }
    const qmdSearchCommand = this.qmd.searchMode;
    let parsed: QmdQueryResult[];
    try {
      if (qmdSearchCommand === "query" && collectionNames.length > 1) {
        parsed = await this.runQueryAcrossCollections(trimmed, limit, collectionNames);
      } else {
        const args = this.buildSearchArgs(qmdSearchCommand, trimmed, limit);
        args.push(...this.buildCollectionFilterArgs(collectionNames));
        // Always scope to managed collections (default + custom). Even for `search`/`vsearch`,
        // pass collection filters; if a given QMD build rejects these flags, we fall back to `query`.
        const result = await this.runQmd(args, { timeoutMs: this.qmd.limits.timeoutMs });
        parsed = parseQmdQueryJson(result.stdout, result.stderr);
      }
    } catch (err) {
      if (qmdSearchCommand !== "query" && this.isUnsupportedQmdOptionError(err)) {
        log.warn(
          `qmd ${qmdSearchCommand} does not support configured flags; retrying search with qmd query`,
        );
        try {
          if (collectionNames.length > 1) {
            parsed = await this.runQueryAcrossCollections(trimmed, limit, collectionNames);
          } else {
            const fallbackArgs = this.buildSearchArgs("query", trimmed, limit);
            fallbackArgs.push(...this.buildCollectionFilterArgs(collectionNames));
            const fallback = await this.runQmd(fallbackArgs, {
              timeoutMs: this.qmd.limits.timeoutMs,
            });
            parsed = parseQmdQueryJson(fallback.stdout, fallback.stderr);
          }
        } catch (fallbackErr) {
          log.warn(`qmd query fallback failed: ${String(fallbackErr)}`);
          throw fallbackErr instanceof Error ? fallbackErr : new Error(String(fallbackErr));
        }
      } else {
        log.warn(`qmd ${qmdSearchCommand} failed: ${String(err)}`);
        throw err instanceof Error ? err : new Error(String(err));
      }
    }
    const results: MemorySearchResult[] = [];
    for (const entry of parsed) {
      const doc = await this.resolveDocLocation(entry.docid);
      if (!doc) {
        continue;
      }
      const snippet = entry.snippet?.slice(0, this.qmd.limits.maxSnippetChars) ?? "";
      const lines = this.extractSnippetLines(snippet);
      const score = typeof entry.score === "number" ? entry.score : 0;
      const minScore = opts?.minScore ?? 0;
      if (score < minScore) {
        continue;
      }
      results.push({
        path: doc.rel,
        startLine: lines.startLine,
        endLine: lines.endLine,
        score,
        snippet,
        source: doc.source,
      });
    }
    return this.clampResultsByInjectedChars(results.slice(0, limit));
  }

  async sync(params?: {
    reason?: string;
    force?: boolean;
    progress?: (update: MemorySyncProgressUpdate) => void;
  }): Promise<void> {
    if (params?.progress) {
      params.progress({ completed: 0, total: 1, label: "Updating QMD index…" });
    }
    await this.runUpdate(params?.reason ?? "manual", params?.force);
    if (params?.progress) {
      params.progress({ completed: 1, total: 1, label: "QMD index updated" });
    }
  }

  async readFile(params: {
    relPath: string;
    from?: number;
    lines?: number;
  }): Promise<{ text: string; path: string }> {
    const relPath = params.relPath?.trim();
    if (!relPath) {
      throw new Error("path required");
    }
    const absPath = this.resolveReadPath(relPath);
    if (!absPath.endsWith(".md")) {
      throw new Error("path required");
    }
    const stat = await fs.lstat(absPath);
    if (stat.isSymbolicLink() || !stat.isFile()) {
      throw new Error("path required");
    }
    if (params.from !== undefined || params.lines !== undefined) {
      const text = await this.readPartialText(absPath, params.from, params.lines);
      return { text, path: relPath };
    }
    const content = await fs.readFile(absPath, "utf-8");
    if (!params.from && !params.lines) {
      return { text: content, path: relPath };
    }
    const lines = content.split("\n");
    const start = Math.max(1, params.from ?? 1);
    const count = Math.max(1, params.lines ?? lines.length);
    const slice = lines.slice(start - 1, start - 1 + count);
    return { text: slice.join("\n"), path: relPath };
  }

  status(): MemoryProviderStatus {
    const counts = this.readCounts();
    return {
      backend: "qmd",
      provider: "qmd",
      model: "qmd",
      requestedProvider: "qmd",
      files: counts.totalDocuments,
      chunks: counts.totalDocuments,
      dirty: false,
      workspaceDir: this.workspaceDir,
      dbPath: this.indexPath,
      sources: Array.from(this.sources),
      sourceCounts: counts.sourceCounts,
      vector: { enabled: true, available: true },
      batch: {
        enabled: false,
        failures: 0,
        limit: 0,
        wait: false,
        concurrency: 0,
        pollIntervalMs: 0,
        timeoutMs: 0,
      },
      custom: {
        qmd: {
          collections: this.qmd.collections.length,
          lastUpdateAt: this.lastUpdateAt,
        },
      },
    };
  }

  async probeEmbeddingAvailability(): Promise<MemoryEmbeddingProbeResult> {
    return { ok: true };
  }

  async probeVectorAvailability(): Promise<boolean> {
    return true;
  }

  async close(): Promise<void> {
    if (this.closed) {
      return;
    }
    this.closed = true;
    if (this.updateTimer) {
      clearInterval(this.updateTimer);
      this.updateTimer = null;
    }
    this.queuedForcedRuns = 0;
    await this.pendingUpdate?.catch(() => undefined);
    await this.queuedForcedUpdate?.catch(() => undefined);
    if (this.daemonEnabled) {
      await this.daemonStop();
    }
    if (this.db) {
      this.db.close();
      this.db = null;
    }
  }

  private async runUpdate(
    reason: string,
    force?: boolean,
    opts?: { fromForcedQueue?: boolean },
  ): Promise<void> {
    if (this.closed) {
      return;
    }
    if (this.pendingUpdate) {
      if (force) {
        return this.enqueueForcedUpdate(reason);
      }
      return this.pendingUpdate;
    }
    if (this.queuedForcedUpdate && !opts?.fromForcedQueue) {
      if (force) {
        return this.enqueueForcedUpdate(reason);
      }
      return this.queuedForcedUpdate;
    }
    if (this.shouldSkipUpdate(force)) {
      return;
    }
    const run = async () => {
      if (this.sessionExporter) {
        await this.exportSessions();
      }
      try {
        await this.runQmd(["update"], { timeoutMs: this.qmd.update.updateTimeoutMs });
      } catch (err) {
        if (!(await this.tryRepairNullByteCollections(err, reason))) {
          throw err;
        }
        await this.runQmd(["update"], { timeoutMs: this.qmd.update.updateTimeoutMs });
      }
      const embedIntervalMs = this.qmd.update.embedIntervalMs;
      const shouldEmbed =
        Boolean(force) ||
        this.lastEmbedAt === null ||
        (embedIntervalMs > 0 && Date.now() - this.lastEmbedAt > embedIntervalMs);
      if (shouldEmbed) {
        try {
          await this.runQmd(["embed"], { timeoutMs: this.qmd.update.embedTimeoutMs });
          this.lastEmbedAt = Date.now();
        } catch (err) {
          log.warn(`qmd embed failed (${reason}): ${String(err)}`);
        }
      }
      this.lastUpdateAt = Date.now();
      this.docPathCache.clear();
    };
    this.pendingUpdate = run().finally(() => {
      this.pendingUpdate = null;
    });
    await this.pendingUpdate;
  }

  private enqueueForcedUpdate(reason: string): Promise<void> {
    this.queuedForcedRuns += 1;
    if (!this.queuedForcedUpdate) {
      this.queuedForcedUpdate = this.drainForcedUpdates(reason).finally(() => {
        this.queuedForcedUpdate = null;
      });
    }
    return this.queuedForcedUpdate;
  }

  private async drainForcedUpdates(reason: string): Promise<void> {
    await this.pendingUpdate?.catch(() => undefined);
    while (!this.closed && this.queuedForcedRuns > 0) {
      this.queuedForcedRuns -= 1;
      await this.runUpdate(`${reason}:queued`, true, { fromForcedQueue: true });
    }
  }

  /**
   * Symlink the default QMD models directory into our custom XDG_CACHE_HOME so
   * that the pre-installed ML models (~/.cache/qmd/models/) are reused rather
   * than re-downloaded for every agent.  If the default models directory does
   * not exist, or a models directory/symlink already exists in the target, this
   * is a no-op.
   */
  private async symlinkSharedModels(): Promise<void> {
    // process.env is never modified — only this.env (passed to child_process
    // spawn) overrides XDG_CACHE_HOME.  So reading it here gives us the
    // user's original value, which is where `qmd` downloaded its models.
    //
    // On Windows, well-behaved apps (including Rust `dirs` / Go os.UserCacheDir)
    // store caches under %LOCALAPPDATA% rather than ~/.cache.  Fall back to
    // LOCALAPPDATA when XDG_CACHE_HOME is not set on Windows.
    const defaultCacheHome =
      process.env.XDG_CACHE_HOME ||
      (process.platform === "win32" ? process.env.LOCALAPPDATA : undefined) ||
      path.join(os.homedir(), ".cache");
    const defaultModelsDir = path.join(defaultCacheHome, "qmd", "models");
    const targetModelsDir = path.join(this.xdgCacheHome, "qmd", "models");
    try {
      // Check if the default models directory exists.
      // Missing path is normal on first run and should be silent.
      const stat = await fs.stat(defaultModelsDir).catch((err: unknown) => {
        if ((err as NodeJS.ErrnoException).code === "ENOENT") {
          return null;
        }
        throw err;
      });
      if (!stat?.isDirectory()) {
        return;
      }
      // Check if something already exists at the target path
      try {
        await fs.lstat(targetModelsDir);
        // Already exists (directory, symlink, or file) – leave it alone
        return;
      } catch {
        // Does not exist – proceed to create symlink
      }
      // On Windows, creating directory symlinks requires either Administrator
      // privileges or Developer Mode.  Fall back to a directory junction which
      // works without elevated privileges (junctions are always absolute-path,
      // which is fine here since both paths are already absolute).
      try {
        await fs.symlink(defaultModelsDir, targetModelsDir, "dir");
      } catch (symlinkErr: unknown) {
        const code = (symlinkErr as NodeJS.ErrnoException).code;
        if (process.platform === "win32" && (code === "EPERM" || code === "ENOTSUP")) {
          await fs.symlink(defaultModelsDir, targetModelsDir, "junction");
        } else {
          throw symlinkErr;
        }
      }
      log.debug(`symlinked qmd models: ${defaultModelsDir} → ${targetModelsDir}`);
    } catch (err) {
      // Non-fatal: if we can't symlink, qmd will fall back to downloading
      log.warn(`failed to symlink qmd models directory: ${String(err)}`);
    }
  }

  private async runQmd(
    args: string[],
    opts?: { timeoutMs?: number },
  ): Promise<{ stdout: string; stderr: string }> {
    return await new Promise((resolve, reject) => {
      const child = spawn(this.qmd.command, args, {
        env: this.env,
        cwd: this.workspaceDir,
      });
      let stdout = "";
      let stderr = "";
      let stdoutTruncated = false;
      let stderrTruncated = false;
      const timer = opts?.timeoutMs
        ? setTimeout(() => {
            child.kill("SIGKILL");
            reject(new Error(`qmd ${args.join(" ")} timed out after ${opts.timeoutMs}ms`));
          }, opts.timeoutMs)
        : null;
      child.stdout.on("data", (data) => {
        const next = appendOutputWithCap(stdout, data.toString("utf8"), this.maxQmdOutputChars);
        stdout = next.text;
        stdoutTruncated = stdoutTruncated || next.truncated;
      });
      child.stderr.on("data", (data) => {
        const next = appendOutputWithCap(stderr, data.toString("utf8"), this.maxQmdOutputChars);
        stderr = next.text;
        stderrTruncated = stderrTruncated || next.truncated;
      });
      child.on("error", (err) => {
        if (timer) {
          clearTimeout(timer);
        }
        reject(err);
      });
      child.on("close", (code) => {
        if (timer) {
          clearTimeout(timer);
        }
        if (stdoutTruncated || stderrTruncated) {
          reject(
            new Error(
              `qmd ${args.join(" ")} produced too much output (limit ${this.maxQmdOutputChars} chars)`,
            ),
          );
          return;
        }
        if (code === 0) {
          resolve({ stdout, stderr });
        } else {
          reject(new Error(`qmd ${args.join(" ")} failed (code ${code}): ${stderr || stdout}`));
        }
      });
    });
  }

  private async readPartialText(absPath: string, from?: number, lines?: number): Promise<string> {
    const start = Math.max(1, from ?? 1);
    const count = Math.max(1, lines ?? Number.POSITIVE_INFINITY);
    const handle = await fs.open(absPath);
    const stream = handle.createReadStream({ encoding: "utf-8" });
    const rl = readline.createInterface({
      input: stream,
      crlfDelay: Infinity,
    });
    const selected: string[] = [];
    let index = 0;
    try {
      for await (const line of rl) {
        index += 1;
        if (index < start) {
          continue;
        }
        if (selected.length >= count) {
          break;
        }
        selected.push(line);
      }
    } finally {
      rl.close();
      await handle.close();
    }
    return selected.slice(0, count).join("\n");
  }

  private ensureDb(): SqliteDatabase {
    if (this.db) {
      return this.db;
    }
    const { DatabaseSync } = requireNodeSqlite();
    this.db = new DatabaseSync(this.indexPath, { readOnly: true });
    // Keep QMD recall responsive when the updater holds a write lock.
    this.db.exec("PRAGMA busy_timeout = 1");
    return this.db;
  }

  private async exportSessions(): Promise<void> {
    if (!this.sessionExporter) {
      return;
    }
    const exportDir = this.sessionExporter.dir;
    await fs.mkdir(exportDir, { recursive: true });
    const files = await listSessionFilesForAgent(this.agentId);
    const keep = new Set<string>();
    const tracked = new Set<string>();
    const cutoff = this.sessionExporter.retentionMs
      ? Date.now() - this.sessionExporter.retentionMs
      : null;
    for (const sessionFile of files) {
      const entry = await buildSessionEntry(sessionFile);
      if (!entry) {
        continue;
      }
      if (cutoff && entry.mtimeMs < cutoff) {
        continue;
      }
      const target = path.join(exportDir, `${path.basename(sessionFile, ".jsonl")}.md`);
      tracked.add(sessionFile);
      const state = this.exportedSessionState.get(sessionFile);
      if (!state || state.hash !== entry.hash || state.mtimeMs !== entry.mtimeMs) {
        await fs.writeFile(target, this.renderSessionMarkdown(entry), "utf-8");
      }
      this.exportedSessionState.set(sessionFile, {
        hash: entry.hash,
        mtimeMs: entry.mtimeMs,
        target,
      });
      keep.add(target);
    }
    const exported = await fs.readdir(exportDir).catch(() => []);
    for (const name of exported) {
      if (!name.endsWith(".md")) {
        continue;
      }
      const full = path.join(exportDir, name);
      if (!keep.has(full)) {
        await fs.rm(full, { force: true });
      }
    }
    for (const [sessionFile, state] of this.exportedSessionState) {
      if (!tracked.has(sessionFile) || !state.target.startsWith(exportDir + path.sep)) {
        this.exportedSessionState.delete(sessionFile);
      }
    }
  }

  private renderSessionMarkdown(entry: SessionFileEntry): string {
    const header = `# Session ${path.basename(entry.absPath, path.extname(entry.absPath))}`;
    const body = entry.content?.trim().length ? entry.content.trim() : "(empty)";
    return `${header}\n\n${body}\n`;
  }

  private pickSessionCollectionName(): string {
    const existing = new Set(this.qmd.collections.map((collection) => collection.name));
    if (!existing.has("sessions")) {
      return "sessions";
    }
    let counter = 2;
    let candidate = `sessions-${counter}`;
    while (existing.has(candidate)) {
      counter += 1;
      candidate = `sessions-${counter}`;
    }
    return candidate;
  }

  private async resolveDocLocation(
    docid?: string,
  ): Promise<{ rel: string; abs: string; source: MemorySource } | null> {
    if (!docid) {
      return null;
    }
    const normalized = docid.startsWith("#") ? docid.slice(1) : docid;
    if (!normalized) {
      return null;
    }
    const cached = this.docPathCache.get(normalized);
    if (cached) {
      return cached;
    }
    const db = this.ensureDb();
    let row: { collection: string; path: string } | undefined;
    try {
      const exact = db
        .prepare("SELECT collection, path FROM documents WHERE hash = ? AND active = 1 LIMIT 1")
        .get(normalized) as { collection: string; path: string } | undefined;
      row = exact;
      if (!row) {
        row = db
          .prepare(
            "SELECT collection, path FROM documents WHERE hash LIKE ? AND active = 1 LIMIT 1",
          )
          .get(`${normalized}%`) as { collection: string; path: string } | undefined;
      }
    } catch (err) {
      if (this.isSqliteBusyError(err)) {
        log.debug(`qmd index is busy while resolving doc path: ${String(err)}`);
        throw this.createQmdBusyError(err);
      }
      throw err;
    }
    if (!row) {
      return null;
    }
    const location = this.toDocLocation(row.collection, row.path);
    if (!location) {
      return null;
    }
    this.docPathCache.set(normalized, location);
    return location;
  }

  private extractSnippetLines(snippet: string): { startLine: number; endLine: number } {
    const match = SNIPPET_HEADER_RE.exec(snippet);
    if (match) {
      const start = Number(match[1]);
      const count = Number(match[2]);
      if (Number.isFinite(start) && Number.isFinite(count)) {
        return { startLine: start, endLine: start + count - 1 };
      }
    }
    const lines = snippet.split("\n").length;
    return { startLine: 1, endLine: lines };
  }

  private readCounts(): {
    totalDocuments: number;
    sourceCounts: Array<{ source: MemorySource; files: number; chunks: number }>;
  } {
    try {
      const db = this.ensureDb();
      const rows = db
        .prepare(
          "SELECT collection, COUNT(*) as c FROM documents WHERE active = 1 GROUP BY collection",
        )
        .all() as Array<{ collection: string; c: number }>;
      const bySource = new Map<MemorySource, { files: number; chunks: number }>();
      for (const source of this.sources) {
        bySource.set(source, { files: 0, chunks: 0 });
      }
      let total = 0;
      for (const row of rows) {
        const root = this.collectionRoots.get(row.collection);
        const source = root?.kind ?? "memory";
        const entry = bySource.get(source) ?? { files: 0, chunks: 0 };
        entry.files += row.c ?? 0;
        entry.chunks += row.c ?? 0;
        bySource.set(source, entry);
        total += row.c ?? 0;
      }
      return {
        totalDocuments: total,
        sourceCounts: Array.from(bySource.entries()).map(([source, value]) => ({
          source,
          files: value.files,
          chunks: value.chunks,
        })),
      };
    } catch (err) {
      log.warn(`failed to read qmd index stats: ${String(err)}`);
      return {
        totalDocuments: 0,
        sourceCounts: Array.from(this.sources).map((source) => ({ source, files: 0, chunks: 0 })),
      };
    }
  }

  private logScopeDenied(sessionKey?: string): void {
    const channel = deriveQmdScopeChannel(sessionKey) ?? "unknown";
    const chatType = deriveQmdScopeChatType(sessionKey) ?? "unknown";
    const key = sessionKey?.trim() || "<none>";
    log.warn(
      `qmd search denied by scope (channel=${channel}, chatType=${chatType}, session=${key})`,
    );
  }

  private isScopeAllowed(sessionKey?: string): boolean {
    return isQmdScopeAllowed(this.qmd.scope, sessionKey);
  }

  private toDocLocation(
    collection: string,
    collectionRelativePath: string,
  ): { rel: string; abs: string; source: MemorySource } | null {
    const root = this.collectionRoots.get(collection);
    if (!root) {
      return null;
    }
    const normalizedRelative = collectionRelativePath.replace(/\\/g, "/");
    const absPath = path.normalize(path.resolve(root.path, collectionRelativePath));
    const relativeToWorkspace = path.relative(this.workspaceDir, absPath);
    const relPath = this.buildSearchPath(
      collection,
      normalizedRelative,
      relativeToWorkspace,
      absPath,
    );
    return { rel: relPath, abs: absPath, source: root.kind };
  }

  private buildSearchPath(
    collection: string,
    collectionRelativePath: string,
    relativeToWorkspace: string,
    absPath: string,
  ): string {
    const insideWorkspace = this.isInsideWorkspace(relativeToWorkspace);
    if (insideWorkspace) {
      const normalized = relativeToWorkspace.replace(/\\/g, "/");
      if (!normalized) {
        return path.basename(absPath);
      }
      return normalized;
    }
    const sanitized = collectionRelativePath.replace(/^\/+/, "");
    return `qmd/${collection}/${sanitized}`;
  }

  private isInsideWorkspace(relativePath: string): boolean {
    if (!relativePath) {
      return true;
    }
    if (relativePath.startsWith("..")) {
      return false;
    }
    if (relativePath.startsWith(`..${path.sep}`)) {
      return false;
    }
    return !path.isAbsolute(relativePath);
  }

  private resolveReadPath(relPath: string): string {
    if (relPath.startsWith("qmd/")) {
      const [, collection, ...rest] = relPath.split("/");
      if (!collection || rest.length === 0) {
        throw new Error("invalid qmd path");
      }
      const root = this.collectionRoots.get(collection);
      if (!root) {
        throw new Error(`unknown qmd collection: ${collection}`);
      }
      const joined = rest.join("/");
      const resolved = path.resolve(root.path, joined);
      if (!this.isWithinRoot(root.path, resolved)) {
        throw new Error("qmd path escapes collection");
      }
      return resolved;
    }
    const absPath = path.resolve(this.workspaceDir, relPath);
    if (!this.isWithinWorkspace(absPath)) {
      throw new Error("path escapes workspace");
    }
    return absPath;
  }

  private isWithinWorkspace(absPath: string): boolean {
    const normalizedWorkspace = this.workspaceDir.endsWith(path.sep)
      ? this.workspaceDir
      : `${this.workspaceDir}${path.sep}`;
    if (absPath === this.workspaceDir) {
      return true;
    }
    const candidate = absPath.endsWith(path.sep) ? absPath : `${absPath}${path.sep}`;
    return candidate.startsWith(normalizedWorkspace);
  }

  private isWithinRoot(root: string, candidate: string): boolean {
    const normalizedRoot = root.endsWith(path.sep) ? root : `${root}${path.sep}`;
    if (candidate === root) {
      return true;
    }
    const next = candidate.endsWith(path.sep) ? candidate : `${candidate}${path.sep}`;
    return next.startsWith(normalizedRoot);
  }

  private clampResultsByInjectedChars(results: MemorySearchResult[]): MemorySearchResult[] {
    const budget = this.qmd.limits.maxInjectedChars;
    if (!budget || budget <= 0) {
      return results;
    }
    let remaining = budget;
    const clamped: MemorySearchResult[] = [];
    for (const entry of results) {
      if (remaining <= 0) {
        break;
      }
      const snippet = entry.snippet ?? "";
      if (snippet.length <= remaining) {
        clamped.push(entry);
        remaining -= snippet.length;
      } else {
        const trimmed = snippet.slice(0, Math.max(0, remaining));
        clamped.push({ ...entry, snippet: trimmed });
        break;
      }
    }
    return clamped;
  }

  private shouldSkipUpdate(force?: boolean): boolean {
    if (force) {
      return false;
    }
    const debounceMs = this.qmd.update.debounceMs;
    if (debounceMs <= 0) {
      return false;
    }
    if (!this.lastUpdateAt) {
      return false;
    }
    return Date.now() - this.lastUpdateAt < debounceMs;
  }

  private isSqliteBusyError(err: unknown): boolean {
    const message = err instanceof Error ? err.message : String(err);
    const normalized = message.toLowerCase();
    return normalized.includes("sqlite_busy") || normalized.includes("database is locked");
  }

  private isUnsupportedQmdOptionError(err: unknown): boolean {
    const message = err instanceof Error ? err.message : String(err);
    const normalized = message.toLowerCase();
    return (
      normalized.includes("unknown flag") ||
      normalized.includes("unknown option") ||
      normalized.includes("unrecognized option") ||
      normalized.includes("flag provided but not defined") ||
      normalized.includes("unexpected argument")
    );
  }

  private createQmdBusyError(err: unknown): Error {
    const message = err instanceof Error ? err.message : String(err);
    return new Error(`qmd index busy while reading results: ${message}`);
  }

  private async waitForPendingUpdateBeforeSearch(): Promise<void> {
    const pending = this.pendingUpdate;
    if (!pending) {
      return;
    }
    await Promise.race([
      pending.catch(() => undefined),
      new Promise<void>((resolve) => setTimeout(resolve, SEARCH_PENDING_UPDATE_WAIT_MS)),
    ]);
  }

  private async runQueryAcrossCollections(
    query: string,
    limit: number,
    collectionNames: string[],
  ): Promise<QmdQueryResult[]> {
    log.debug(
      `qmd query multi-collection workaround active (${collectionNames.length} collections)`,
    );
    const bestByDocId = new Map<string, QmdQueryResult>();
    for (const collectionName of collectionNames) {
      const args = this.buildSearchArgs("query", query, limit);
      args.push("-c", collectionName);
      const result = await this.runQmd(args, { timeoutMs: this.qmd.limits.timeoutMs });
      const parsed = parseQmdQueryJson(result.stdout, result.stderr);
      for (const entry of parsed) {
        if (typeof entry.docid !== "string" || !entry.docid.trim()) {
          continue;
        }
        const prev = bestByDocId.get(entry.docid);
        const prevScore = typeof prev?.score === "number" ? prev.score : Number.NEGATIVE_INFINITY;
        const nextScore = typeof entry.score === "number" ? entry.score : Number.NEGATIVE_INFINITY;
        if (!prev || nextScore > prevScore) {
          bestByDocId.set(entry.docid, entry);
        }
      }
    }
    return [...bestByDocId.values()].toSorted((a, b) => (b.score ?? 0) - (a.score ?? 0));
  }

  private listManagedCollectionNames(): string[] {
    const seen = new Set<string>();
    const names: string[] = [];
    for (const collection of this.qmd.collections) {
      const name = collection.name?.trim();
      if (!name || seen.has(name)) {
        continue;
      }
      seen.add(name);
      names.push(name);
    }
    return names;
  }

  private resolveMcpTool(): string {
    // HTTP daemon uses qmd_-prefixed tool names
    switch (this.qmd.searchMode) {
      case "search":
        return "qmd_search";
      case "vsearch":
        return "qmd_vector_search";
      case "query":
        return "qmd_deep_search";
      default:
        return "qmd_deep_search";
    }
  }

  /**
   * Attempt to search via the daemon. Returns null if daemon is not enabled
   * or fails (caller should fall back to spawn-per-query).
   */
  private async tryDaemonSearch(
    query: string,
    limit: number,
    collections: string[] = [],
  ): Promise<MemorySearchResult[] | null> {
    if (!this.daemonEnabled || !this.daemonConfig) {
      return null;
    }

    const tool = this.resolveMcpTool();
    // MCP tool accepts a single collection string; pass only when exactly one collection
    const collection = collections.length === 1 ? collections[0] : undefined;
    try {
      if (this.daemonState === "ready") {
        // Warm path
        const raw = await this.daemonQuery(query, {
          tool,
          limit,
          collection,
          timeoutMs: this.daemonConfig.warmTimeoutMs,
        });
        return this.formatDaemonResults(raw, limit);
      }

      // Cold start path — daemon enabled but not running
      await this.daemonWaitForBackoff();
      await this.daemonStart();
      const raw = await this.daemonQuery(query, {
        tool,
        limit,
        collection,
        timeoutMs: this.daemonConfig.coldStartTimeoutMs,
      });
      return this.formatDaemonResults(raw, limit);
    } catch (err) {
      // If query mode fails due to context size, retry with vsearch (no reranker)
      const errMsg = String(err);
      if (tool === "query" && errMsg.includes("context size")) {
        log.warn("qmd daemon query hit context size limit, retrying with vsearch");
        try {
          const timeoutMs =
            this.daemonState === "ready"
              ? this.daemonConfig.warmTimeoutMs
              : this.daemonConfig.coldStartTimeoutMs;
          const raw = await this.daemonQuery(query, {
            tool: "qmd_vector_search",
            limit,
            collection,
            timeoutMs,
          });
          return this.formatDaemonResults(raw, limit);
        } catch (retryErr) {
          log.warn(`qmd daemon vsearch fallback also failed: ${String(retryErr)}`);
        }
      }
      log.warn(`qmd daemon search failed, falling back to spawn-per-query: ${errMsg}`);
      return null;
    }
  }

  private async formatDaemonResults(
    raw: Array<{ docid?: string; file?: string; score?: number; snippet?: string }>,
    limit: number,
  ): Promise<MemorySearchResult[]> {
    const results: MemorySearchResult[] = [];
    for (const entry of raw) {
      // Prefer file path from structured response; fall back to docid hash lookup
      let doc: { rel: string; abs: string; source: MemorySource } | null = null;
      if (entry.file) {
        doc = this.resolveDocFromFile(entry.file);
      }
      if (!doc) {
        doc = await this.resolveDocLocation(entry.docid);
      }
      if (!doc) {
        continue;
      }
      const snippet = entry.snippet?.slice(0, this.qmd.limits.maxSnippetChars) ?? "";
      const lines = this.extractSnippetLines(snippet);
      const score = typeof entry.score === "number" ? entry.score : 0;
      results.push({
        path: doc.rel,
        startLine: lines.startLine,
        endLine: lines.endLine,
        score,
        snippet,
        source: doc.source,
      });
    }
    return this.clampResultsByInjectedChars(results.slice(0, limit));
  }

  /** Resolve a relative file path from QMD structured response to a doc location. */
  private resolveDocFromFile(
    file: string,
  ): { rel: string; abs: string; source: MemorySource } | null {
    if (!file) {
      return null;
    }
    // Determine source based on collection path prefix
    const source: MemorySource = file.startsWith("sessions/") ? "sessions" : "memory";
    const abs = path.resolve(this.qmdDir, file);
    return { rel: file, abs, source };
  }

  private buildCollectionFilterArgs(collectionNames: string[]): string[] {
    if (collectionNames.length === 0) {
      return [];
    }
    const names = collectionNames.filter(Boolean);
    return names.flatMap((name) => ["-c", name]);
  }

  private buildSearchArgs(
    command: "query" | "search" | "vsearch",
    query: string,
    limit: number,
  ): string[] {
    if (command === "query") {
      return ["query", query, "--json", "-n", String(limit)];
    }
    return [command, query, "--json", "-n", String(limit)];
  }

  // --- Inline daemon methods ---

  private async daemonCleanupOrphan(): Promise<void> {
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
        await this.daemonRunCommand(["mcp", "stop"], 5_000);
      } catch {
        // Process doesn't exist — stale PID file
        await fs.rm(QMD_PID_FILE, { force: true });
      }
    } catch (err) {
      log.debug(`orphan cleanup failed: ${String(err)}`);
    }
  }

  private async daemonStart(): Promise<void> {
    if (this.daemonState === "ready") {
      return;
    }
    if (this.daemonStartPromise) {
      return this.daemonStartPromise;
    }
    this.daemonStartPromise = this.daemonDoStart().finally(() => {
      this.daemonStartPromise = null;
    });
    return this.daemonStartPromise;
  }

  private async daemonDoStart(): Promise<void> {
    if (!this.daemonConfig) {
      return;
    }
    this.daemonState = "starting";
    try {
      this.daemonLastStartAt = Date.now();

      // Launch the HTTP daemon (detaches itself)
      await this.daemonRunCommand(
        ["mcp", "--http", "--daemon"],
        this.daemonConfig.coldStartTimeoutMs,
      );

      // Poll until the daemon is accepting HTTP requests
      await this.daemonPollUntilReady();

      this.daemonState = "ready";
      this.daemonResetBackoff();
      this.daemonResetIdleTimer();
      log.info("qmd HTTP daemon started (ready)");
    } catch (err) {
      this.daemonState = "error";
      log.debug(`qmd HTTP daemon start failed: ${String(err)}`);
      // Try to stop any partially started daemon
      await this.daemonRunCommand(["mcp", "stop"], 5_000).catch(() => undefined);
      throw err;
    }
  }

  private async daemonPollUntilReady(): Promise<void> {
    const deadline = Date.now() + HEALTH_POLL_MAX_MS;
    while (Date.now() < deadline) {
      try {
        const resp = await fetch(QMD_HTTP_URL, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            jsonrpc: "2.0",
            id: this.daemonNextId++,
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

  private async daemonStop(): Promise<void> {
    this.daemonClearIdleTimer();
    if (this.daemonState === "stopped") {
      return;
    }
    this.daemonState = "stopped";
    try {
      await this.daemonRunCommand(["mcp", "stop"], 5_000);
    } catch (err) {
      log.debug(`qmd mcp stop failed: ${String(err)}`);
    }
    log.info("qmd HTTP daemon stopped");
  }

  private async daemonQuery(
    text: string,
    opts: { tool: string; limit: number; collection?: string; timeoutMs: number },
  ): Promise<QmdDaemonQueryResult[]> {
    if (this.daemonState !== "ready") {
      throw new Error("qmd daemon not ready");
    }

    this.daemonLastQueryAt = Date.now();
    this.daemonResetIdleTimer();

    const id = this.daemonNextId++;
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
      this.daemonHandleCrash();
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

  private daemonHandleCrash(): void {
    if (this.daemonState === "stopped") {
      return;
    }
    this.daemonState = "error";
    this.daemonClearIdleTimer();

    const timeSinceStart = Date.now() - this.daemonLastStartAt;
    if (timeSinceStart > STABILITY_RESET_MS) {
      this.daemonResetBackoff();
    } else {
      this.daemonBackoffMs = Math.min(this.daemonBackoffMs * 2, MAX_BACKOFF_MS);
    }
  }

  private daemonResetBackoff(): void {
    this.daemonBackoffMs = 1_000;
  }

  private async daemonWaitForBackoff(): Promise<void> {
    if (this.daemonBackoffMs > 1_000) {
      log.info(`qmd daemon restart backoff: ${this.daemonBackoffMs}ms`);
      await new Promise<void>((resolve) => setTimeout(resolve, this.daemonBackoffMs));
    }
  }

  private daemonResetIdleTimer(): void {
    this.daemonClearIdleTimer();
    const timeout = this.daemonConfig?.idleTimeoutMs ?? 0;
    if (timeout <= 0) {
      return;
    }
    this.daemonIdleTimer = setTimeout(() => {
      this.daemonOnIdle();
    }, timeout);
  }

  private daemonClearIdleTimer(): void {
    if (this.daemonIdleTimer) {
      clearTimeout(this.daemonIdleTimer);
      this.daemonIdleTimer = null;
    }
  }

  private daemonOnIdle(): void {
    log.info("qmd daemon idle timeout — shutting down");
    void this.daemonStop();
  }

  private daemonRunCommand(args: string[], timeoutMs: number): Promise<void> {
    return new Promise((resolve, reject) => {
      const child = spawn(this.qmd.command, args, {
        stdio: ["ignore", "pipe", "pipe"],
        env: this.env,
        cwd: this.workspaceDir,
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

function appendOutputWithCap(
  current: string,
  chunk: string,
  maxChars: number,
): { text: string; truncated: boolean } {
  const appended = current + chunk;
  if (appended.length <= maxChars) {
    return { text: appended, truncated: false };
  }
  return { text: appended.slice(-maxChars), truncated: true };
}
