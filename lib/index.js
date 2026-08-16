/**
 * `audit-gate` — a DeepSeek Harness completion gate.
 *
 * When the model is about to finish a task, this plugin audits the agent's
 * workspace with a **cheap-model subagent**: it detects the project language,
 * lets the subagent generate and run the language-appropriate checks (lint /
 * typecheck / tests / build), review code style, hunt for real logic bugs,
 * verify the requested functionality works end-to-end, and check documentation
 * coverage, then returns a structured verdict. The task may only finish when
 * the audit passes — and crucially, it may NOT finish on an inconclusive
 * audit (by default): an audit that could not be completed is reported as
 * such, never as a pass.
 *
 * Every audit resolves to one of three statuses:
 *   `passed`        — the subagent completed and reported concrete checks,
 *                     all of which passed (evidence required: an empty check
 *                     list is inconclusive, not a pass).
 *   `failed`        — the subagent completed and at least one concrete check
 *                     failed.
 *   `inconclusive`  — the audit could not produce a trustworthy verdict:
 *                     subagent error/refusal/max-tokens, malformed or empty
 *                     structured output, a rejected result promise, an
 *                     abort, the total-audit deadline, or the audit itself
 *                     modifying the workspace.
 *
 * The host derives the overall status from per-check results only; the model
 * is never asked for, and never trusted with, a top-level "passed" verdict.
 *
 * Command execution is HOST-controlled: the audit subagent is given read-only
 * tools plus a single `audit_exec` tool, and never a general bash. `audit_exec`
 * runs allowlisted commands (npm scripts, allowlisted executables, or
 * workspace-relative binaries) as argv (no shell), in a scrubbed environment,
 * under a per-command timeout and process-group cleanup, and records
 * host-owned evidence (argv, exit code, duration, output digest/tail). The
 * final verdict merges that evidence authoritatively: an executed check's
 * pass/fail is decided by the host, not by the model. A workspace snapshot
 * taken before and after the audit catches the audit itself modifying the
 * repo (treated as inconclusive).
 *
 * A pass issues a short-lived **audit certificate** bound to the workspace
 * state, task context, goal, and configuration. A monotonic `ctx.tools.guard()`
 * (which runs after every `tools/pre-execute` layer and can never be re-allowed
 * by a later listener) denies `update_goal complete` unless a fresh certificate
 * for the current state exists. This makes the completion invariant
 * non-reorderable.
 *
 * Three surfaces, sharing one audit runner, one snapshot-keyed result cache,
 * and the certificate:
 *
 *   (A) Goal-completion gate  — a `tools/pre-execute` waterfall listener that
 *       runs the audit and DENIES `update_goal complete` while the audit is
 *       not `passed`; the pass certificate it issues is then enforced by the
 *       monotonic guard.
 *
 *   (B) Turn-end gate         — an `agent/turn-stopping` (serial) listener for
 *       non-goal tasks. When the model stops, it audits and, unless the audit
 *       is `passed` (or `inconclusive` with `onInconclusive: warn|allow`),
 *       steers a fix prompt into the same turn so the model keeps working.
 *       No-progress is judged by comparing workspace snapshots, not wording.
 *
 *   (C) `run_audit` tool      — a model-facing tool so the model can run the
 *       same audit proactively and see a structured report before it attempts
 *       completion, avoiding a denial round-trip.
 *
 * Recursion safety: the audit itself is a subagent, and a subagent inherits the
 * parent's preset (which may mount this very plugin). The runner therefore
 * only audits top-level agents (`delegationDepth` 0); any gate or tool
 * trigger on a subagent resolves to a non-blocking "not applicable" pass, so
 * an audit child can never audit itself.
 *
 * @module dsh-audit-gate
 */
import { readdir, readFile, stat } from "node:fs/promises";
import { isAbsolute, join, relative, resolve } from "node:path";
import { createHash } from "node:crypto";
import z from "@deepseek-ai/schemastery";
import { createUserMessage, boundContextSummary } from "@deepseek-ai/dsh-llm";
import { defineTool } from "@deepseek-ai/dsh-tools";
import { installSettingsSection, settingsNamespace } from "@deepseek-ai/dsh-settings";
import { deadline, timeoutOf } from "@deepseek-ai/dsh-timeout";

export const name = "audit-gate";

/** Settings namespace: the runtime switches surfaced to the Web settings UI. */
const NS = settingsNamespace("audit-gate");

/** Hard dependencies: prompt, agents, tools, commands, subagents, controlled subprocess. */
export const inject = ["systemPrompt", "agents", "tools", "commands", "subagents", "subprocess"];

/** Plugin configuration schema (the loader normalizes defaults). */
export const Config = z.object({
  /** Master switch. */
  enabled: z.boolean().default(true),
  /** Gate (A): deny `update_goal` complete until the audit passes. */
  guardCompletion: z.boolean().default(true),
  /** Gate (B): audit at turn end — `off` | `modified` (only when files changed) | `always`. */
  guardTurnEnd: z.union(["off", "modified", "always"]).default("modified"),
  /** Max fix iterations the turn-end gate injects before giving up. */
  maxAttempts: z.number().step(1).min(1).max(64).default(5),
  /** What an inconclusive audit (could not be verified) means for completion: `deny` | `warn` | `allow`. */
  onInconclusive: z.union(["deny", "warn", "allow"]).default("deny"),
  /** Optional workspace override; defaults to the agent's session cwd. */
  workdir: z.string(),
  /** Tool names that mark the turn as "modified" for the `modified` trigger and invalidate the audit cache/certificate. */
  mutatingTools: z.array(z.string()).default(["write", "edit", "bash", "pwsh"]),
  /** Register the model-facing `run_audit` tool (C). */
  registerTool: z.boolean().default(true),
  /** Name of the model-facing audit tool. */
  toolName: z.string().default("run_audit"),
  /** Name of the host-controlled check-runner tool the audit subagent uses instead of bash. */
  auditExecName: z.string().default("audit_exec"),
  /** Provider serving the cheap audit subagent. */
  auditProvider: z.string().default("deepseek-official"),
  /** Cheap model that performs the audit. */
  auditModel: z.string().default("deepseek-v4-flash"),
  /** Subagent provider to spawn the audit with. */
  subagentProvider: z.string().default("spawn"),
  /** Total wall-clock budget for one audit run, before it is declared inconclusive. */
  maxAuditTimeoutMs: z.number().step(1).min(1000).max(3600000).default(300000),
  /** Default per-command budget for audit_exec checks. */
  commandTimeoutMs: z.number().step(1).min(1000).max(600000).default(120000),
  /** How many chars of evidence output are retained. */
  outputMaxChars: z.number().step(1).min(0).max(1_000_000).default(4000),
  /** Have the audit subagent review code style. */
  checkStyle: z.boolean().default(true),
  /** Have the audit subagent hunt for real logic bugs in the source (fault-finding review). */
  checkLogic: z.boolean().default(true),
  /** Have the audit subagent verify the requested functionality actually works end-to-end. */
  checkFunction: z.boolean().default(true),
  /** Have the audit subagent check documentation coverage. */
  checkDocs: z.boolean().default(true),
  /** Optional style-guide file (absolute or workspace-relative) handed to the audit subagent. */
  styleGuide: z.string(),
});

/**
 * The subset of `Config` that is a live, user-owned runtime switch. It is
 * registered as the `audit-gate` settings namespace, so `enabled`,
 * `guardCompletion`, `guardTurnEnd`, and `maxAttempts` can be flipped at
 * runtime (from the Web settings UI or `~/.dsh/settings.yaml`) without
 * editing the composition. Everything else stays composition config and takes
 * effect on (re)load.
 */
const SettingsSchema = z.object({
  enabled: z.boolean().default(true),
  guardCompletion: z.boolean().default(true),
  guardTurnEnd: z.union(["off", "modified", "always"]).default("modified"),
  maxAttempts: z.number().step(1).min(1).max(64).default(5),
});

const TURN_END_MODES = new Set(["off", "modified", "always"]);
const INCONCLUSIVE_MODES = new Set(["deny", "warn", "allow"]);
const DEFAULT_MUTATING = ["write", "edit", "bash", "pwsh"];
const MAX_CHECKS = 8;
const AUDIT_MAX_ATTEMPTS = 2;
const AUDIT_CMD_TIMEOUT_CODE = "AUDIT_CMD_TIMEOUT";
const STDOUT_MAX_BYTES = 64 * 1024;
const STDOUT_SPILL_BYTES = 4 * 1024 * 1024;
const GRACE_MS = 3000;

/** Validate and default the (possibly raw) config. */
export function resolveConfig(config) {
  const guardTurnEnd = config.guardTurnEnd ?? "modified";
  if (!TURN_END_MODES.has(guardTurnEnd)) {
    throw new TypeError("audit-gate: guardTurnEnd must be one of off|modified|always");
  }
  const onInconclusive = config.onInconclusive ?? "deny";
  if (!INCONCLUSIVE_MODES.has(onInconclusive)) {
    throw new TypeError("audit-gate: onInconclusive must be one of deny|warn|allow");
  }
  const maxAttempts = config.maxAttempts ?? 5;
  if (!Number.isSafeInteger(maxAttempts) || maxAttempts < 1) {
    throw new TypeError("audit-gate: maxAttempts must be a positive safe integer");
  }
  const maxAuditTimeoutMs = config.maxAuditTimeoutMs ?? 300000;
  if (!Number.isSafeInteger(maxAuditTimeoutMs) || maxAuditTimeoutMs < 1000) {
    throw new TypeError("audit-gate: maxAuditTimeoutMs must be a positive safe integer >= 1000");
  }
  return {
    enabled: config.enabled ?? true,
    guardCompletion: config.guardCompletion ?? true,
    guardTurnEnd,
    onInconclusive,
    maxAttempts,
    workdir: config.workdir,
    mutatingTools: new Set(config.mutatingTools ?? DEFAULT_MUTATING),
    registerTool: config.registerTool ?? true,
    toolName: config.toolName ?? "run_audit",
    auditExecName: config.auditExecName ?? "audit_exec",
    auditProvider: config.auditProvider ?? "deepseek-official",
    auditModel: config.auditModel ?? "deepseek-v4-flash",
    subagentProvider: config.subagentProvider ?? "spawn",
    maxAuditTimeoutMs,
    commandTimeoutMs: config.commandTimeoutMs ?? 120000,
    outputMaxChars: config.outputMaxChars ?? 4000,
    checkStyle: config.checkStyle ?? true,
    checkLogic: config.checkLogic ?? true,
    checkFunction: config.checkFunction ?? true,
    checkDocs: config.checkDocs ?? true,
    styleGuide: config.styleGuide,
  };
}

/** Validate and default the live runtime-switch subset (defensive; the settings schema already validates). */
function resolveSwitches(switches) {
  const guardTurnEnd = switches.guardTurnEnd ?? "modified";
  if (!TURN_END_MODES.has(guardTurnEnd)) {
    throw new TypeError("audit-gate: guardTurnEnd must be one of off|modified|always");
  }
  const maxAttempts = switches.maxAttempts ?? 5;
  if (!Number.isSafeInteger(maxAttempts) || maxAttempts < 1) {
    throw new TypeError("audit-gate: maxAttempts must be a positive safe integer");
  }
  return {
    enabled: switches.enabled ?? true,
    guardCompletion: switches.guardCompletion ?? true,
    guardTurnEnd,
    maxAttempts,
  };
}

export function sha256(text) {
  return createHash("sha256").update(String(text ?? "")).digest("hex");
}

/** Keep the tail of a (possibly huge) text. */
export function trimOutput(text, max) {
  if (max <= 0) return "";
  const t = String(text ?? "");
  if (t.length <= max) return t;
  return `${t.slice(-max)}\n…[truncated]`;
}

/** Keep the head and the tail of a (possibly huge) text — for requirements where the head carries intent. */
export function trimHeadTail(text, max) {
  if (max <= 0) return "";
  const t = String(text ?? "");
  if (t.length <= max) return t;
  const head = Math.ceil(max * 0.65);
  const tail = max - head;
  return `${t.slice(0, head)}\n…[truncated]…\n${t.slice(-tail)}`;
}

function isPlainObject(value) {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

// ── language detection ──────────────────────────────────────────────────────

/** Manifest signals per language family, with the idiomatic checks hint. */
const LANGUAGE_SIGNALS = [
  { files: ["package.json"], label: "JavaScript/TypeScript (Node.js)", hint: "Use the package's npm scripts (lint/typecheck/test/build); run npx tsc --noEmit when a tsconfig exists." },
  { files: ["tsconfig.json"], label: "TypeScript", hint: "Run npx tsc --noEmit and the package's npm lint/test scripts." },
  { files: ["pyproject.toml", "requirements.txt", "setup.py"], label: "Python", hint: "Use ruff check (or pylint/flake8) and pytest; python -m compileall -q . for syntax." },
  { files: ["go.mod"], label: "Go", hint: "Run gofmt -l ., go vet ./..., and go test ./..." },
  { files: ["Cargo.toml"], label: "Rust", hint: "Run cargo fmt --check, cargo clippy -- -D warnings, and cargo test." },
  { files: ["pom.xml", "build.gradle", "build.gradle.kts"], label: "Java (Maven/Gradle)", hint: "Run the build's check tasks: mvn verify or ./gradlew check." },
  { files: ["composer.json"], label: "PHP", hint: "Run composer validate and php -l on sources." },
  { files: ["Gemfile"], label: "Ruby", hint: "Run rubocop and bundle exec rspec (or rake test)." },
  { files: ["CMakeLists.txt"], label: "C/C++ (CMake)", hint: "Configure, build, and run ctest; honor configured warnings." },
  { match: (name) => name.endsWith(".csproj"), label: "C#", hint: "Run dotnet build and dotnet test." },
];

/** Directories skipped by the bounded manifest scan. */
const SCAN_IGNORE_DIRS = new Set([".git", "node_modules", "target", "dist", "build", ".venv", "vendor", "out", "coverage", "__pycache__", ".pytest_cache", ".nyc_output"]);
/** Bounded walk: depth and total file-name budget so a huge repo cannot stall the gate. */
const SCAN_MAX_DEPTH = 2;
const SCAN_MAX_NAMES = 400;

/** Collect basenames under `workdir` up to SCAN_MAX_DEPTH, skipping ignored dirs. */
async function collectManifestNames(workdir) {
  const names = new Set();
  const seen = new Set();
  const walk = async (dir, depth) => {
    if (depth > SCAN_MAX_DEPTH || names.size >= SCAN_MAX_NAMES || seen.has(dir)) return;
    seen.add(dir);
    let entries;
    try {
      entries = await readdir(dir, { withFileTypes: true });
    } catch {
      return;
    }
    for (const entry of entries) {
      if (names.size >= SCAN_MAX_NAMES) return;
      if (entry.isDirectory()) {
        if (SCAN_IGNORE_DIRS.has(entry.name)) continue;
        await walk(join(dir, entry.name), depth + 1);
      } else {
        names.add(entry.name);
      }
    }
  };
  await walk(workdir, 0);
  return names;
}

/**
 * Deterministically detect the project language(s) from manifest file basenames
 * anywhere within a shallow scan of the workspace (monorepo-friendly). Falls
 * back to a generic directive when nothing is known.
 */
export async function detectLanguages(workdir) {
  const names = await collectManifestNames(workdir);
  const found = [];
  const seen = new Set();
  for (const signal of LANGUAGE_SIGNALS) {
    const hit =
      signal.files !== undefined
        ? [...names].some((n) => signal.files.includes(n))
        : [...names].some((n) => signal.match(n));
    if (hit && !seen.has(signal.label)) {
      seen.add(signal.label);
      found.push(signal);
    }
  }
  if (found.length === 0) {
    return [{ label: "unknown", hint: "Detect the language from the repository contents, then run its idiomatic checks." }];
  }
  return found;
}

/** Read an optional style-guide file, resolved against the workspace when relative. */
async function readStyleGuide(workdir, styleGuide) {
  if (!styleGuide) return undefined;
  const target = isAbsolute(styleGuide) ? styleGuide : join(workdir, styleGuide);
  try {
    return trimHeadTail(await readFile(target, "utf8"), 8000);
  } catch {
    return undefined;
  }
}

/** Pull the plain text out of one message's content blocks. */
function messageText(message) {
  return (message?.content ?? [])
    .filter((block) => block.type === "text")
    .map((block) => block.text ?? "")
    .filter(Boolean)
    .join("\n")
    .trim();
}

/**
 * Capture what this task was asked to deliver, for the functional audit: the
 * goal objective (goal-based tasks) plus the most recent direct human requests
 * from the session. Plugin-injected runtime snapshots are skipped. Requests are
 * truncated head-and-tail so the opening intent survives.
 */
function taskContextFor(agent, goals) {
  const lines = [];
  if (goals !== undefined) {
    try {
      const goal = goals.get(agent);
      if (typeof goal?.objective === "string" && goal.objective.length > 0) {
        lines.push(`Goal: ${trimHeadTail(goal.objective, 1200)}`);
      }
    } catch {}
  }
  try {
    const messages = agent.session.deriveMessages();
    const requests = [];
    for (let i = messages.length - 1; i >= 0 && requests.length < 3; i--) {
      const message = messages[i];
      if (message?.role !== "user") continue;
      if (message?.source?.kind !== "user") continue; // skip plugin/system snapshots
      const text = messageText(message);
      if (text.length === 0) continue;
      requests.unshift(trimHeadTail(text, 600));
    }
    if (requests.length > 0) {
      lines.push("Recent user request(s):", ...requests);
    }
  } catch {}
  const joined = lines.join("\n").trim();
  return joined.length === 0 ? undefined : trimHeadTail(joined, 2400);
}

// ── host-controlled command execution ───────────────────────────────────────

/** Environment variables stripped from audit_exec commands (credentials and cloud/CI secrets). */
const SECRET_ENV_PATTERN = /(?:API_?KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|DEEPSEEK_API_KEY|AWS_|AZURE_|GOOGLE_|GCP_|GITHUB_|GITLAB_|NPM_TOKEN|SSH_)/i;

/** Build a scrubbed environment for a controlled check command. */
function scrubbedEnv() {
  const out = {};
  for (const [key, value] of Object.entries(process.env)) {
    if (key === "SSH_AUTH_SOCK") continue;
    if (SECRET_ENV_PATTERN.test(key)) continue;
    out[key] = value;
  }
  return out;
}

/** Executables an `audit_exec` `executable` runner may invoke directly. */
export const ALLOWED_EXECUTABLES = new Set([
  "ruff", "flake8", "pylint", "pyright", "black", "isort", "pytest", "python", "python3",
  "tsc", "eslint", "prettier",
  "go", "gofmt", "goimports",
  "cargo", "clippy", "rustfmt",
  "dotnet",
  "mvn", "gradle",
  "ruby", "rubocop", "rspec",
  "php", "composer",
  "cmake", "ctest", "make",
  "gcc", "g++", "clang", "clang-tidy",
  "shellcheck", "shfmt",
  "git",
]);

/** Build an argv array for a validated runner/target. Throws on anything not allowlisted. */
export function resolveExecutableArgv(workdir, runner, target, args) {
  const argv = (args ?? []).map(String);
  if (runner === "npm-script") {
    if (typeof target !== "string" || !/^[a-zA-Z0-9:_.-]+$/.test(target)) {
      throw new Error(`audit_exec: invalid npm script name ${JSON.stringify(target)}`);
    }
    return ["npm", "run", target, ...argv];
  }
  if (runner === "executable") {
    if (typeof target !== "string" || !ALLOWED_EXECUTABLES.has(target)) {
      throw new Error(`audit_exec: executable ${JSON.stringify(target)} is not in the allowlist`);
    }
    return [target, ...argv];
  }
  if (runner === "file") {
    return [resolveWorkspaceFileArgv(workdir, target), ...argv];
  }
  throw new Error(`audit_exec: unknown runner ${JSON.stringify(runner)}`);
}

/** Resolve and confine a workspace-relative executable path for the `file` runner. */
function resolveWorkspaceFileArgv(workdir, target) {
  if (typeof target !== "string" || target.length === 0) throw new Error("audit_exec: file runner needs a non-empty relative path");
  if (isAbsolute(target)) throw new Error("audit_exec: file runner target must be relative to the workspace");
  const resolved = resolve(workdir, target);
  const rel = relative(workdir, resolved);
  if (rel.startsWith("..") || isAbsolute(rel)) throw new Error("audit_exec: file runner target escapes the workspace");
  return resolved;
}

/**
 * Run one controlled host command as argv (never a shell), with a deadline,
 * a detached process tree, scrubbed environment, and bounded output.
 */
export async function runHostCommand(ctx, argv, cwd, { signal, timeoutMs, env, stdoutMaxBytes = STDOUT_MAX_BYTES, stderrMaxBytes = STDOUT_MAX_BYTES }) {
  const t0 = Date.now();
  const budget = Math.max(1, Math.min(timeoutMs ?? 120000, 600000));
  const d = deadline(signal, budget, AUDIT_CMD_TIMEOUT_CODE);
  const collect = (maxBytes) => ({ maxBytes, spill: { maxBytes: STDOUT_SPILL_BYTES } });
  try {
    const handle = ctx.subprocess.spawn({
      argv,
      cwd,
      stdio: { stdin: "ignore", stdout: collect(stdoutMaxBytes), stderr: collect(stderrMaxBytes) },
      graceMs: GRACE_MS,
      signal: d.signal,
      env,
    });
    const outcome = await handle.done;
    const stdout = handle.collected?.stdout?.readFrom(0) ?? { text: "" };
    const stderr = handle.collected?.stderr?.readFrom(0) ?? { text: "" };
    const timedOut = timeoutOf(d.signal, AUDIT_CMD_TIMEOUT_CODE) !== undefined;
    const aborted = d.signal.aborted && !timedOut;
    return {
      exitCode: outcome.exitCode ?? null,
      timedOut,
      aborted,
      durationMs: Date.now() - t0,
      stdout: stdout.text ?? "",
      stderr: stderr.text ?? "",
    };
  } catch (error) {
    return { exitCode: null, timedOut: false, aborted: false, durationMs: Date.now() - t0, error: error?.message ?? String(error), stdout: "", stderr: "" };
  } finally {
    d[Symbol.dispose]?.();
  }
}

/**
 * Snapshot the workspace state for cache keying and self-modification
 * detection. Git repos use HEAD + status (state) and HEAD + tracked diffs
 * (modification) so test artifacts under untracked files do not trip the
 * self-modification check; non-git repos fall back to a bounded file walk.
 */
export async function workspaceState(ctx, workdir, signal) {
  const env = scrubbedEnv();
  const head = await runHostCommand(ctx, ["git", "-C", workdir, "rev-parse", "HEAD"], workdir, { signal, timeoutMs: 10000, env });
  if (head.exitCode === 0) {
    const [status, diff, cached] = await Promise.all([
      runHostCommand(ctx, ["git", "-C", workdir, "status", "--porcelain"], workdir, { signal, timeoutMs: 10000, env }),
      runHostCommand(ctx, ["git", "-C", workdir, "diff"], workdir, { signal, timeoutMs: 10000, env }),
      runHostCommand(ctx, ["git", "-C", workdir, "diff", "--cached"], workdir, { signal, timeoutMs: 10000, env }),
    ]);
    const h = head.stdout.trim();
    return {
      isGit: true,
      stateDigest: sha256(`${h}\n${status.stdout}`),
      modDigest: sha256(`${h}\n${diff.stdout}\n${cached.stdout}`),
      status: status.stdout,
    };
  }
  const entries = [];
  const seen = new Set();
  const walk = async (dir, depth) => {
    if (depth > SCAN_MAX_DEPTH || entries.length >= SCAN_MAX_NAMES || seen.has(dir)) return;
    seen.add(dir);
    let names;
    try {
      names = await readdir(dir, { withFileTypes: true });
    } catch {
      return;
    }
    for (const entry of names) {
      if (entries.length >= SCAN_MAX_NAMES) return;
      const full = join(dir, entry.name);
      if (entry.isDirectory()) {
        if (SCAN_IGNORE_DIRS.has(entry.name)) continue;
        await walk(full, depth + 1);
      } else {
        try {
          const s = await stat(full);
          entries.push(`${relative(workdir, full)}\t${s.mtimeMs}\t${s.size}`);
        } catch {}
      }
    }
  };
  await walk(workdir, 0);
  const text = entries.sort().join("\n");
  return { isGit: false, stateDigest: sha256(text), modDigest: sha256(text), status: "" };
}

/** A short "changed files" summary (git porcelain) for the audit prompt. */
function changedFilesSummary(state) {
  const lines = (state.status ?? "").split("\n").filter(Boolean).slice(0, 30);
  return lines.length === 0 ? "none detected" : lines.join("\n");
}

// ── audit subagent contract ─────────────────────────────────────────────────

/**
 * Structured verdict the audit subagent must report. There is deliberately NO
 * top-level `passed` field: the host derives the overall status from the
 * per-check results only, so the model cannot contradict its own checks.
 */
export const VERDICT_SCHEMA = {
  type: "object",
  properties: {
    summary: { type: "string" },
    checks: {
      type: "array",
      items: {
        type: "object",
        properties: {
          name: { type: "string" },
          passed: { type: "boolean" },
          detail: { type: "string" },
        },
        required: ["name", "passed"],
      },
    },
  },
  required: ["summary", "checks"],
};

/** Prompt for the cheap-model audit subagent. */
function buildAuditPrompt(workdir, languages, cfg, styleGuide, taskContext, changed) {
  const languageLines = languages.map((l) => `- ${l.label}: ${l.hint}`).join("\n");
  const parts = [
    "You are the audit stage of an automatic completion gate. Audit the repository at:",
    `  ${workdir}`,
    "",
    "Detected language(s):",
    languageLines,
    "",
    "Changed files (from git status):",
    changed,
    "",
    "1. Generate the concrete check plan for this repository — the idiomatic checks for the detected language(s):",
    "   lint, typecheck, tests, build, or equivalent. Honor existing package/build files and scripts.",
    `   Run each check by calling the ${cfg.auditExecName} tool (the ONLY way to run commands here). Give each call a`,
    "   stable `name`, choose `runner` (npm-script | executable | file) and `target`, and pass argv in `args`.",
    "   The tool returns host-recorded evidence (argv, exit code, duration, output) — your check's `passed`",
    "   MUST match that evidence. Never attempt to run commands any other way.",
  ];
  if (cfg.checkStyle) {
    parts.push(
      "",
      "2. Review code style: naming, formatting, consistency, and obvious anti-patterns.",
    );
    if (styleGuide) {
      parts.push("   Follow the repository style guide:", "", styleGuide, "");
    } else {
      parts.push("   Judge against the language's conventional style. Flag only clear, defensible deviations.");
    }
  } else {
    parts.push("", "2. (style review disabled)");
  }
  if (cfg.checkLogic) {
    parts.push(
      "",
      "3. FAULT-FINDING logic review — READ the actual source code and hunt for REAL bugs, not style:",
      "   - index/off-by-one errors, wrong comparisons, inverted conditions;",
      "   - null/undefined/type assumptions that crash at runtime;",
      "   - race conditions, unhandled async rejections, resource leaks (unclosed handles/subscriptions);",
      "   - incorrect algorithm or wrong result vs the stated intent / API contract;",
      "   - error handling gaps: swallowed errors, wrong error paths, missing cleanup;",
      "   - security-relevant flaws: injection, unsafe eval/dynamic code, secrets committed in code.",
      "   Focus on the changed files listed above, then the rest of the repository.",
      "   Only flag concrete, defensible defects with file:line evidence. Do NOT pad the list — a quiet",
      "   pass is better than inventing problems. Each real finding is its own check entry.",
    );
  } else {
    parts.push("", "3. (logic review disabled)");
  }
  if (cfg.checkFunction) {
    parts.push(
      "",
      "4. FUNCTIONAL audit — verify the requested functionality actually works end-to-end:",
      "   - Task context (what this task was asked to deliver):",
      taskContext
        ? `     ${taskContext}`
        : "     (no explicit task context captured — infer the intended functionality from git diff and the repository)",
      "   - Identify the functionality this task was meant to deliver.",
      "   - Verify it actually works: run the relevant program/entrypoint/tests via the audit_exec tool and",
      "     EXERCISE the feature where possible. Confirm the behavior matches the stated requirement.",
      "   - Flag concrete failures of the REQUESTED behavior: unimplemented or half-implemented features, behavior",
      "     that contradicts the request, broken flows or implied edge cases, crashes when the feature is used.",
      "   - Each functional gap is its own check entry, with evidence.",
    );
  } else {
    parts.push("", "4. (functional audit disabled)");
  }
  parts.push(
    cfg.checkDocs
      ? "5. Check documentation coverage: README present and accurate, public API and signatures documented,"
        + " key modules or functions with obvious missing docs. Flag concrete gaps, not pedantry."
      : "5. (documentation coverage check disabled)",
    "",
    "Constraints:",
    `- NEVER modify source files, never commit, never install packages, never spawn agents, never call ${cfg.toolName}.`,
    `- Commands run ONLY through ${cfg.auditExecName}; you have no bash. Prefer read-only checks where possible.`,
    `- Keep the audit bounded: at most ${MAX_CHECKS} checks.`,
    "- A check fails only when there is a concrete, defensible problem.",
    "",
    "When finished, call structured_output exactly once with:",
    "- summary: one or two sentences.",
    "- checks: at least one object per check — name (short label), passed (bool), detail (actionable; quote the",
    "  relevant output or file:line when possible). A check has passed: false when there is a concrete,",
    "  defensible problem. For every command you ran via audit_exec, use the SAME name as that call's `name`",
    "  and set passed to match the returned evidence. The host derives the overall verdict from EVERY check,",
    "  so never report an empty check list unless you genuinely could not run any check — an empty list is",
    "  an inconclusive audit.",
  );
  return parts.join("\n");
}

// ── audit result model ──────────────────────────────────────────────────────

/**
 * Map one raw check from the subagent to a host-owned verdict.
 * @returns `{ name, run, status: "passed"|"failed", output }`.
 */
export function verdictOf(check) {
  return {
    name: String(check?.name ?? "check"),
    run: "llm-audit",
    status: check?.passed === true ? "passed" : "failed",
    output: trimOutput(check?.detail, 4000),
  };
}

/** A verdict that was never reached: the audit could not be trusted. */
function inconclusive(cause, summary, checks = []) {
  return { status: "inconclusive", cause, summary, checks };
}

/** Host-recorded evidence from audit_exec calls during one audit run, keyed by child agent id. */
const evidenceSinks = new Map(); // agentId -> Evidence[]

/**
 * Overlay host-recorded execution evidence onto the subagent's verdict so an
 * executed check's pass/fail is decided by the host (exit code), not by the
 * model's restatement. Semantic findings without matching evidence stand.
 */
export function mergeEvidence(result, evidence) {
  if (result.status === "inconclusive") return result;
  const byName = new Map(evidence.map((entry) => [entry.name, entry]));
  const checks = result.checks.map((c) => {
    const ev = byName.get(c.name);
    if (ev === undefined) return c;
    return {
      ...c,
      run: `audit:${ev.runner}`,
      status: ev.passed ? "passed" : "failed",
      output: ev.outputTail || c.output,
      evidence: {
        command: ev.argv,
        exitCode: ev.exitCode,
        timedOut: ev.timedOut,
        durationMs: ev.durationMs,
        outputDigest: ev.outputDigest,
      },
    };
  });
  return {
    ...result,
    checks,
    status: checks.some((c) => c.status === "failed") ? "failed" : "passed",
  };
}

/**
 * One full subagent audit attempt. The child gets read-only tools plus
 * audit_exec (never bash) and a total deadline.
 * @returns `{ result: OverallResult, retryable: boolean }`.
 */
async function runAuditOnce(ctx, agent, cfg, signal, prompt) {
  const auditTools = ["read", "glob", "grep", cfg.auditExecName];
  let run;
  let result;
  try {
    run = await ctx.subagents.start(cfg.subagentProvider, {
      label: "audit-gate",
      parent: agent,
      prompt,
      signal,
      outputSchema: VERDICT_SCHEMA,
      toolFilter: { allow: auditTools },
      agentOptions: { provider: cfg.auditProvider, model: cfg.auditModel },
    });
  } catch (error) {
    return {
      result: inconclusive("start-error", `audit could not start: ${error?.message ?? String(error)}`),
      retryable: false,
    };
  }
  const childId = run.localAgent?.id;
  if (childId !== undefined) evidenceSinks.set(childId, []);

  let deadlineFired = false;
  let timer;
  const deadlinePromise = new Promise((_, reject) => {
    timer = setTimeout(() => {
      deadlineFired = true;
      reject(new Error("audit deadline"));
    }, cfg.maxAuditTimeoutMs + GRACE_MS);
  });
  try {
    result = await Promise.race([run.result, deadlinePromise]);
  } catch (error) {
    return {
      result: deadlineFired
        ? inconclusive("deadline", `audit exceeded the ${cfg.maxAuditTimeoutMs}ms budget`)
        : inconclusive("result-rejected", `audit result promise rejected: ${error?.message ?? String(error)}`),
      retryable: !deadlineFired,
    };
  } finally {
    clearTimeout(timer);
    await run.dispose().catch(() => {});
  }

  const evidence = childId !== undefined ? (evidenceSinks.get(childId) ?? []) : [];
  if (childId !== undefined) evidenceSinks.delete(childId);

  const structured = result.structured;
  if (result.stopReason !== "completed" || !isPlainObject(structured) || !Array.isArray(structured.checks)) {
    const reason =
      typeof structured?.summary === "string" && structured.summary.length > 0
        ? structured.summary
        : `audit did not complete (${result.stopReason})`;
    const retryable = result.stopReason === "error" || result.stopReason === "max-tokens" || result.stopReason === "completed";
    return { result: inconclusive(result.stopReason === "completed" ? "malformed-output" : String(result.stopReason), reason), retryable };
  }
  const checks = structured.checks.slice(0, MAX_CHECKS).map(verdictOf);
  if (checks.length === 0) {
    return {
      result: inconclusive("empty-checks", "audit reported no checks — an empty check list is not evidence of a pass"),
      retryable: false,
    };
  }
  const base = { status: checks.some((c) => c.status === "failed") ? "failed" : "passed", summary: String(structured.summary ?? ""), checks };
  return { result: mergeEvidence(base, evidence), retryable: false };
}

function delay(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Run the whole audit for one agent by delegating to a cheap-model subagent,
 * retrying transient infrastructure failures once, with a total budget and a
 * workspace self-modification check. Returns an {@link OverallResult}.
 * @param baseline - `{ stateDigest, modDigest }` captured by the cache layer.
 * @returns `{ status, cause?, summary, checks }`.
 */
export async function runAudit(ctx, agent, cfg, signal, baseline) {
  if (signal?.aborted) {
    return inconclusive("aborted", "audit aborted by caller");
  }
  const workdir = cfg.workdir ?? agent.session.header.cwd ?? process.cwd();
  const goals = ctx.get("goals");
  const [languages, styleGuide] = await Promise.all([
    detectLanguages(workdir),
    readStyleGuide(workdir, cfg.styleGuide),
  ]);
  const taskContext = taskContextFor(agent, goals);
  const changed = changedFilesSummary(baseline);
  const prompt = buildAuditPrompt(workdir, languages, cfg, styleGuide, taskContext, changed);

  for (let attempt = 1; ; attempt++) {
    const outcome = await runAuditOnce(ctx, agent, cfg, signal, prompt);
    if (signal?.aborted) return outcome.result;
    if (outcome.result.status === "inconclusive") {
      // The audit itself must not have modified the workspace.
      const after = await workspaceState(ctx, workdir, signal);
      if (after.modDigest !== baseline.modDigest) {
        return inconclusive("self-modified", "the audit run modified the workspace; its verdict is not trusted");
      }
    }
    if (!outcome.retryable || attempt >= AUDIT_MAX_ATTEMPTS) {
      return outcome.result;
    }
    ctx.logger.warn(`audit-gate: audit subagent did not complete (attempt ${attempt}/${AUDIT_MAX_ATTEMPTS}); retrying`);
    await delay(1200);
  }
}

/** Render a compact PASS/FAIL/INCONCLUSIVE summary plus failing output. */
export function renderVerdictLines(checks) {
  const lines = [];
  for (const c of checks) {
    const label = c.status === "passed" ? "PASS" : c.status === "failed" ? "FAIL" : "INCONCLUSIVE";
    lines.push(`${label}: ${c.name}  (${c.run})`);
    if (c.status !== "passed" && c.output) lines.push(c.output);
  }
  return lines;
}

/** Turn-end gate: the fix prompt injected into the same turn. */
export function renderFeedback(result, attempts, maxAttempts) {
  const failedCount = result.checks.filter((c) => c.status === "failed").length;
  const lines = ["<audit_gate>"];
  if (result.status === "failed") {
    lines.push(
      `Automatic verification failed (fix attempt ${attempts}/${maxAttempts}).`,
      `${failedCount} check(s) failing. Fix the issues below and then stop; ` +
        "the gate re-runs automatically, and this task only ends when the audit passes.",
      "",
    );
  } else {
    lines.push(
      `Automatic verification could not be completed (fix attempt ${attempts}/${maxAttempts}).`,
      `The audit is inconclusive: ${result.summary}. This task may not finish on an unverified audit; ` +
        "resolve the audit problem (or run the audit again), then stop.",
      "",
    );
  }
  lines.push(...renderVerdictLines(result.checks), "</audit_gate>");
  return [{ type: "text", text: lines.join("\n") }];
}

/** Completion gate: the denial reason surfaced as the `update_goal` tool error. */
export function renderDenialReason(result) {
  const failedCount = result.checks.filter((c) => c.status === "failed").length;
  const lines = [];
  if (result.status === "failed") {
    lines.push(
      `Automatic verification failed before completion (${failedCount} check(s) failing). ` +
        "Fix the findings below, then mark the goal complete again once they pass.",
    );
  } else {
    lines.push(
      "Automatic verification could not be completed, so completion is blocked. " +
        `The audit is inconclusive: ${result.summary}. Resolve the audit problem, re-run the audit until it passes, then retry completion.`,
    );
  }
  lines.push("", ...renderVerdictLines(result.checks));
  return lines.join("\n");
}

/** Structured, lossless-JSON report returned by the `run_audit` tool. */
export function reportValue(result) {
  return {
    status: result.status,
    passed: result.status === "passed",
    summary: result.summary,
    ...(result.cause !== undefined ? { cause: result.cause } : {}),
    results: result.checks.map((c) => ({
      name: c.name,
      run: c.run,
      status: c.status,
      output: c.output ?? "",
      ...(c.evidence !== undefined ? { evidence: c.evidence } : {}),
    })),
  };
}

/** Human-readable render of the structured report. */
export function renderReport(report) {
  const label = report.status === "passed" ? "PASS" : report.status === "failed" ? "FAIL" : "INCONCLUSIVE";
  const lines = [label, report.summary, ""];
  for (const r of report.results) {
    const rLabel = r.status === "passed" ? "PASS" : r.status === "failed" ? "FAIL" : "INCONCLUSIVE";
    lines.push(`${rLabel}: ${r.name}  (${r.run})`);
    if (r.status !== "passed" && r.output) lines.push(r.output);
  }
  return [{ type: "text", text: lines.join("\n") }];
}

/** Human-readable switch state for the `/audit` status command. */
function renderStatus(lv, cfg) {
  return [
    "Audit gate:",
    `  enabled: ${lv.enabled}`,
    `  guardCompletion: ${lv.guardCompletion}`,
    `  guardTurnEnd: ${lv.guardTurnEnd}`,
    `  maxAttempts: ${lv.maxAttempts}`,
    `  onInconclusive: ${cfg.onInconclusive}`,
    `  audit: ${cfg.auditModel} (${cfg.auditProvider}) via ${cfg.subagentProvider}`,
    `  budget: ${cfg.maxAuditTimeoutMs}ms · cmd: ${cfg.commandTimeoutMs}ms`,
    `  style: ${cfg.checkStyle ? "on" : "off"} · logic: ${cfg.checkLogic ? "on" : "off"} · function: ${cfg.checkFunction ? "on" : "off"} · docs: ${cfg.checkDocs ? "on" : "off"}`,
    ...(cfg.styleGuide ? [`  styleGuide: ${cfg.styleGuide}`] : []),
    "",
    "Commands: /audit (status) · /audit-toggle [on|off] (flip with no argument)",
  ].join("\n");
}

export function apply(ctx, config) {
  const cfg = resolveConfig(config);

  // Runtime switches: composition entry is the base; the user's settings
  // section (`~/.dsh/settings.yaml` or the Web settings UI) layers on top and
  // is hot-reloaded. `live()` is read at every gate trigger.
  const switchesBase = {
    enabled: cfg.enabled,
    guardCompletion: cfg.guardCompletion,
    guardTurnEnd: cfg.guardTurnEnd,
    maxAttempts: cfg.maxAttempts,
  };
  let source = () => switchesBase;
  installSettingsSection(ctx, NS, SettingsSchema, switchesBase, {
    setSource: (next) => {
      source = next;
    },
    onChange: () => {},
  });

  /** Static composition config overlaid with the live runtime switches. */
  const live = () => ({ ...cfg, ...resolveSwitches(source()) });

  // Model guidance: make the gate legible and cooperative, and only promise
  // surfaces the composition actually enabled.
  const guidance = [
    "An automatic verification gate audits this workspace before a task may finish.",
    `The audit is performed by a ${cfg.auditModel} subagent that generates and runs language-appropriate checks, reviews code style, hunts for real logic bugs, verifies the requested functionality works end-to-end, and checks documentation coverage.`,
  ];
  if (cfg.guardCompletion) {
    guidance.push(
      'For goal-based work, update_goal action "complete" is denied while the audit fails or is inconclusive — fix the findings and retry completion.',
    );
  }
  if (cfg.registerTool) {
    guidance.push(`You may run the ${cfg.toolName} tool any time to see the current audit result.`);
  }
  guidance.push(
    `Commands may be run through the ${cfg.auditExecName} tool, a controlled runner (allowlisted executables, npm scripts, workspace binaries; no shell; credentials scrubbed; host-recorded evidence).`,
  );
  if (cfg.guardTurnEnd !== "off") {
    guidance.push("For non-goal work, after you stop, the gate re-runs the audit and hands any failures back to you to fix.");
  }
  guidance.push("Treat audit findings as remaining work, not as permission to end. Do not attempt to bypass or disable the gate.");
  ctx.systemPrompt.section({
    name: "tool:audit-gate",
    order: 120,
    text: guidance.join(" "),
  });

  // Per-agent turn-local state: bounded fix loop, workspace-digest no-progress,
  // mutation-versioned result cache (snapshot key + single-flight), and the
  // audit certificate the monotonic guard enforces.
  const states = new Map(); // agent -> state

  function stateFor(agent) {
    let st = states.get(agent);
    if (st === undefined) {
      st = {
        attempts: 0,
        lastDigest: undefined,
        dirty: false,
        dirtyVersion: 0,
        completed: undefined,
        inFlight: undefined,
        certificate: undefined,
      };
      states.set(agent, st);
    }
    return st;
  }

  /** Subagents are never audited — the audit itself is a subagent, so gating one would recurse. */
  function isSubagent(agent) {
    return (agent?.session?.header?.delegationDepth ?? 0) > 0;
  }

  /**
   * Whether the completion gate owns this agent's current task: only an ACTIVE
   * goal is driven to completion by update_goal. A completed/paused/blocked
   * goal no longer shields the turn-end gate from ordinary turns.
   */
  function goalOwnsCompletion(agent) {
    const goals = ctx.get("goals");
    if (goals === undefined) return false;
    try {
      return goals.get(agent)?.phase === "active";
    } catch {
      return false;
    }
  }

  function workdirFor(agent) {
    return cfg.workdir ?? agent.session.header.cwd ?? process.cwd();
  }

  /** Digest the audit-relevant identity: workspace state, task context, goal, config. */
  async function auditKey(agent) {
    const workdir = workdirFor(agent);
    const signal = undefined;
    const state = await workspaceState(ctx, workdir, signal);
    const task = taskContextFor(agent, ctx.get("goals")) ?? "";
    let goalRef = "";
    const goals = ctx.get("goals");
    if (goals !== undefined) {
      try {
        const goal = goals.get(agent);
        if (goal?.id !== undefined) goalRef = `${goal.id}:${goal.revision}:${goal.phase}`;
      } catch {}
    }
    const configText = JSON.stringify({
      model: cfg.auditModel, provider: cfg.auditProvider, runner: cfg.auditExecName,
      style: cfg.checkStyle, logic: cfg.checkLogic, function: cfg.checkFunction, docs: cfg.checkDocs,
      styleGuide: cfg.styleGuide, onInconclusive: cfg.onInconclusive,
    });
    return { key: sha256([state.stateDigest, sha256(task), goalRef, configText].join("|")), state };
  }

  /**
   * Audit with a per-agent cache keyed by workspace/task/config snapshot
   * (single-flight), invalidated by any mutating tool result, and a pass
   * certificate issued to the monotonic guard.
   * @returns `{ key, digest, state, result }`.
   */
  async function runAuditCached(agent, signal) {
    if (isSubagent(agent)) {
      return {
        key: "n/a",
        digest: undefined,
        state: { stateDigest: "n/a", modDigest: "n/a", isGit: false, status: "" },
        result: {
          status: "passed",
          summary: "audit applies to top-level agents only",
          checks: [{ name: "audit", run: "llm-audit", status: "passed", output: "not applicable" }],
        },
      };
    }
    const st = stateFor(agent);
    const { key, state } = await auditKey(agent);
    const digest = state.stateDigest;
    if (st.completed !== undefined && st.completed.key === key) {
      // Same snapshot → the cached verdict still holds; re-issue the certificate
      // at the current dirtyVersion so the guard stays satisfied.
      st.certificate = { dirtyVersion: st.dirtyVersion, key, status: st.completed.result.status, auditedAt: Date.now() };
      return st.completed;
    }
    if (st.inFlight !== undefined && st.inFlight.key === key) return st.inFlight.promise;

    const promise = runAudit(ctx, agent, cfg, signal, state)
      .then((result) => {
        if (!signal?.aborted) {
          const entry = { key, digest, state, result };
          st.completed = entry;
          st.certificate = { dirtyVersion: st.dirtyVersion, key, status: result.status, auditedAt: Date.now() };
        }
        return { key, digest, state, result };
      })
      .finally(() => {
        if (st.inFlight?.promise === promise) st.inFlight = undefined;
      });
    st.inFlight = { key, promise };
    return promise;
  }

  // Reset turn-local state at the canonical turn boundary.
  ctx.on("session/event", (session, event) => {
    if (event.type !== "turn/start") return;
    const agent = ctx.agents.get(session.id);
    if (agent === undefined || agent.session !== session) return;
    const st = stateFor(agent);
    st.dirty = false;
    st.attempts = 0;
    st.lastDigest = undefined;
    st.completed = undefined;
    st.certificate = undefined; // a new turn may carry new user input / external changes
  });

  // Mark the turn "modified" and invalidate the cache + certificate when a
  // mutating tool runs — regardless of success, because a failing command may
  // still have modified files before it errored.
  ctx.on("tools/result", (exec, result) => {
    if (exec.agent === undefined) return;
    if (!cfg.mutatingTools.has(exec.name)) return;
    const st = stateFor(exec.agent);
    st.dirty = true;
    st.dirtyVersion += 1;
    st.certificate = undefined;
  });

  // (A) Goal-completion gate — run the audit in tools/pre-execute; the pass
  // certificate it issues is enforced by the monotonic guard below.
  ctx.on("tools/pre-execute", (exec, next) => {
    if (exec.name !== "update_goal" || exec.agent === undefined) return next();
    const lv = live();
    if (!lv.enabled || !lv.guardCompletion) return next();
    if (isSubagent(exec.agent)) return next();
    const args = exec.arguments;
    if (typeof args !== "object" || args === null || args.action !== "complete") return next();
    return runAuditCached(exec.agent, exec.signal).then((entry) => {
      const result = entry.result;
      if (result.status === "passed") return next();
      const blocking = result.status === "failed" || lv.onInconclusive === "deny";
      if (!blocking) return next(); // warn/allow: let it through, the report/logs carry the warning
      return { kind: "deny", reason: renderDenialReason(result) };
    });
  });

  // Monotonic completion guard: runs after every tools/pre-execute layer and
  // can never be re-allowed by a later listener. `update_goal complete` is
  // denied unless a fresh audit certificate for the current workspace state
  // exists (and, for an inconclusive certificate, onInconclusive allows it).
  ctx.tools.guard((exec) => {
    if (exec.name !== "update_goal" || exec.agent === undefined) return;
    const lv = live();
    if (!lv.enabled || !lv.guardCompletion) return;
    if (isSubagent(exec.agent)) return;
    const args = exec.arguments;
    if (typeof args !== "object" || args === null || args.action !== "complete") return;
    const st = stateFor(exec.agent);
    const cert = st.certificate;
    if (cert === undefined || cert.dirtyVersion !== st.dirtyVersion) {
      return "Completion requires a fresh audit certificate for the current workspace state. Run the audit so it issues a certificate on pass.";
    }
    if (cert.status !== "passed" && lv.onInconclusive === "deny") {
      return "The last audit was inconclusive; completion is blocked until the audit passes.";
    }
  });

  // (B) Turn-end gate — for tasks that do not use the goal system.
  ctx.on("agent/turn-stopping", async ({ agent, signal }) => {
    const lv = live();
    if (!lv.enabled || lv.guardTurnEnd === "off") return;
    if (agent === undefined || isSubagent(agent)) return;
    if (goalOwnsCompletion(agent)) return; // the completion gate owns active goal tasks
    const st = stateFor(agent);

    const firstForTurn = st.attempts === 0;
    const shouldAudit =
      lv.guardTurnEnd === "always" ||
      (lv.guardTurnEnd === "modified" && (st.dirty || !firstForTurn));
    if (!shouldAudit) return;
    if (signal.aborted) return;

    const entry = await runAuditCached(agent, signal);
    if (signal.aborted) return;
    const result = entry.result;

    if (result.status === "passed") {
      st.attempts = 0;
      st.lastDigest = undefined;
      return; // passes → let the turn close
    }
    const blocking = result.status === "failed" || lv.onInconclusive === "deny";
    if (!blocking) {
      st.attempts = 0;
      st.lastDigest = undefined;
      return; // inconclusive with warn/allow → close the turn
    }

    // No-progress: compare the workspace digest the audit saw with the one the
    // previous steering saw. If the model changed nothing since the last steer,
    // stop injecting rather than looping on the same failing state.
    const noProgress = entry.digest !== undefined && st.lastDigest !== undefined && entry.digest === st.lastDigest;
    st.lastDigest = entry.digest;

    if (noProgress || st.attempts >= lv.maxAttempts) {
      ctx.logger.warn(
        `audit-gate: verification still failing for agent "${agent.id}" after ` +
          `${st.attempts} attempt(s); closing the turn with unresolved failures.`,
      );
      return; // bounded give-up → let the turn close
    }

    st.attempts += 1;
    agent.steer(
      createUserMessage({
        content: renderFeedback(result, st.attempts, lv.maxAttempts),
        source: {
          kind: "plugin",
          plugin: "audit-gate",
          form: "notice",
          summary: boundContextSummary(`verification ${result.status === "failed" ? "failed" : "inconclusive"}; fix and retry`),
        },
      }),
    );
  });

  // Host-controlled check runner used by the audit subagent instead of bash.
  ctx.tools.register(
    defineTool({
      name: cfg.auditExecName,
      description:
        "Run one controlled, allowlisted command as argv (no shell) and return host-recorded evidence: " +
        "exit code, duration, and output. Runners: `npm-script` (target = package.json script), `executable` " +
        "(target = allowlisted binary like ruff, pytest, tsc, go, cargo, dotnet, make), or `file` (target = " +
        "workspace-relative executable path). Credentials are scrubbed from the environment. Use this instead " +
        "of a general shell to run checks.",
      parameters: {
        name: { type: "string", description: "Stable label for this check; the host correlates evidence by it." },
        runner: { type: "string", enum: ["npm-script", "executable", "file"] },
        target: { type: "string", description: "npm script name, allowlisted executable, or workspace-relative file path." },
        args: { type: "array", items: { type: "string" }, description: "Optional argv arguments (never shell-expanded)." },
        timeoutMs: { type: "number", description: "Optional per-command timeout in milliseconds." },
        description: { type: "string", description: "Short description of the check." },
      },
      required: ["name", "runner", "target"],
      output: {
        schema: { type: "json" },
        render: (_args, value) => [
          { type: "text", text: value?.passed ? `PASS ${value.name} (${value.command?.join(" ")})` : `FAIL ${value.name} (${value.command?.join(" ")})` },
        ],
      },
      async execute(args, exec) {
        const agent = exec.agent;
        if (agent === undefined) return { ok: false, error: "audit_exec requires a calling agent" };
        let argv;
        try {
          argv = resolveExecutableArgv(workdirFor(agent), args.runner, args.target, args.args);
        } catch (error) {
          return { ok: false, error: error?.message ?? String(error) };
        }
        const workdir = workdirFor(agent);
        const timeoutMs = Number.isFinite(args.timeoutMs) && args.timeoutMs > 0 ? Math.min(args.timeoutMs, 600000) : cfg.commandTimeoutMs;
        const run = await runHostCommand(ctx, argv, workdir, { signal: exec.signal, timeoutMs, env: scrubbedEnv() });
        const evidence = {
          name: String(args.name ?? args.target ?? "check"),
          runner: String(args.runner),
          target: String(args.target ?? ""),
          argv,
          exitCode: run.exitCode,
          timedOut: run.timedOut,
          aborted: run.aborted,
          durationMs: run.durationMs,
          passed: run.exitCode === 0 && !run.timedOut && !run.aborted,
          outputTail: trimOutput(`${run.stdout}\n${run.stderr}`, cfg.outputMaxChars),
          outputDigest: sha256(`${run.stdout}\n${run.stderr}`),
        };
        const sink = evidenceSinks.get(agent.id);
        if (sink !== undefined) sink.push(evidence);
        return {
          ok: true,
          name: evidence.name,
          runner: evidence.runner,
          command: argv,
          passed: evidence.passed,
          exitCode: evidence.exitCode,
          timedOut: evidence.timedOut,
          aborted: evidence.aborted,
          durationMs: evidence.durationMs,
          outputTail: evidence.outputTail,
          outputDigest: evidence.outputDigest,
        };
      },
    }),
  );

  // (C) Model-facing tool: run the same audit proactively and see the report.
  if (cfg.registerTool) {
    ctx.tools.register(
      defineTool({
        name: cfg.toolName,
        description:
          "Run this workspace's automatic audit and return a structured pass/fail report. The audit is " +
          "performed by a cheap-model subagent that generates and runs language-appropriate checks (lint, " +
          "typecheck, tests, build) through a host-controlled runner, reviews code style, hunts for real logic " +
          "bugs, verifies the requested functionality works end-to-end, and checks documentation coverage. Use " +
          "it to confirm your work passes before marking a task complete; the automatic completion gate enforces " +
          "the same audit.",
        parameters: {},
        output: {
          schema: { type: "json" },
          render: (_args, value) => renderReport(value),
        },
        async execute(_args, exec) {
          const agent = exec.agent;
          if (agent === undefined) {
            return { status: "inconclusive", passed: false, summary: "run_audit requires a calling agent", results: [] };
          }
          const entry = await runAuditCached(agent, exec.signal);
          return reportValue(entry.result);
        },
      }),
    );
  }

  // (D) Human slash commands: inspect and toggle the runtime switch.
  ctx.commands.register({
    name: "audit",
    description: "show the audit gate's current switch state",
    handler: () => ({ kind: "success", text: renderStatus(live(), cfg) }),
  });

  ctx.commands.register({
    name: "audit-toggle",
    description: "turn the audit gate on or off (no argument flips)",
    input: { hint: "[on|off]" },
    handler: async (invocation) => {
      const arg = invocation.rawInput.trim().toLowerCase();
      let target;
      if (arg === "") target = !live().enabled;
      else if (arg === "on") target = true;
      else if (arg === "off") target = false;
      else {
        return {
          kind: "error",
          text: `Usage: /audit-toggle [on|off]\nUnknown argument: ${invocation.rawInput.trim()}`,
        };
      }

      const settings = ctx.get("settings");
      if (settings === undefined) {
        return {
          kind: "error",
          text:
            "Settings are not configured, so the switch cannot be changed at runtime. " +
            "Edit the composition (agent.cordis.yml) or ~/.dsh/settings.yaml instead.",
        };
      }

      try {
        await settings.update(NS, { enabled: target });
      } catch (error) {
        return {
          kind: "error",
          text: `Could not update the audit gate switch: ${error?.message ?? String(error)}`,
        };
      }
      return {
        kind: "success",
        text: `Audit gate ${target ? "enabled" : "disabled"}.\n\n${renderStatus(live(), cfg)}`,
      };
    },
  });

  // Release per-agent state when an agent goes away.
  ctx.on("agent/disposed", ({ agent }) => {
    states.delete(agent);
    evidenceSinks.delete(agent.id);
  });
}
