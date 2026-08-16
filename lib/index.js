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
 *                     structured output, a rejected result promise, or an
 *                     abort. Whether an inconclusive audit blocks completion
 *                     is governed by `onInconclusive` (default `deny`).
 *
 * The host derives the overall status from per-check results only; the model
 * is never asked for, and never trusted with, a top-level "passed" verdict.
 *
 * Three surfaces, sharing one audit runner and one per-agent result cache:
 *
 *   (A) Goal-completion gate  — a `tools/pre-execute` waterfall listener that
 *       DENIES `update_goal` with action `complete` while the audit is not
 *       `passed`. This is authoritative for goal-based tasks: the goal is
 *       never marked complete until the audit passes.
 *
 *   (B) Turn-end gate         — an `agent/turn-stopping` (serial) listener for
 *       non-goal tasks. When the model stops, it audits and, unless the audit
 *       is `passed` (or `inconclusive` with `onInconclusive: warn|allow`),
 *       steers a fix prompt into the same turn so the model keeps working.
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
import { readdir, readFile } from "node:fs/promises";
import { isAbsolute, join } from "node:path";
import z from "@deepseek-ai/schemastery";
import { createUserMessage, boundContextSummary } from "@deepseek-ai/dsh-llm";
import { defineTool } from "@deepseek-ai/dsh-tools";
import { installSettingsSection, settingsNamespace } from "@deepseek-ai/dsh-settings";

export const name = "audit-gate";

/** Settings namespace: the runtime switches surfaced to the Web settings UI. */
const NS = settingsNamespace("audit-gate");

/** Hard dependencies: prompt registry, agent registry, tool registry, command registry, subagent service. */
export const inject = ["systemPrompt", "agents", "tools", "commands", "subagents"];

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
  /** Tool names that mark the turn as "modified" for the `modified` trigger and the audit cache. */
  mutatingTools: z.array(z.string()).default(["write", "edit", "bash", "pwsh"]),
  /** Register the model-facing `run_audit` tool (C). */
  registerTool: z.boolean().default(true),
  /** Name of the model-facing audit tool. */
  toolName: z.string().default("run_audit"),
  /** Provider serving the cheap audit subagent. */
  auditProvider: z.string().default("deepseek-official"),
  /** Cheap model that performs the audit. */
  auditModel: z.string().default("deepseek-v4-flash"),
  /** Subagent provider to spawn the audit with. */
  subagentProvider: z.string().default("spawn"),
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
 * editing the composition. Everything else (the audit model, subagent
 * provider, `onInconclusive`, style/docs toggles, style guide) stays
 * composition config and takes effect on (re)load.
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
    auditProvider: config.auditProvider ?? "deepseek-official",
    auditModel: config.auditModel ?? "deepseek-v4-flash",
    subagentProvider: config.subagentProvider ?? "spawn",
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
const SCAN_IGNORE_DIRS = new Set([".git", "node_modules", "target", "dist", "build", ".venv", "vendor", "out"]);
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
function buildAuditPrompt(workdir, languages, cfg, styleGuide, taskContext) {
  const languageLines = languages.map((l) => `- ${l.label}: ${l.hint}`).join("\n");
  const parts = [
    "You are the audit stage of an automatic completion gate. Audit the repository at:",
    `  ${workdir}`,
    "",
    "Detected language(s):",
    languageLines,
    "",
    "1. Generate the concrete check plan for this repository — the idiomatic checks for the detected language(s):",
    "   lint, typecheck, tests, build, or equivalent. Honor existing package/build files and scripts.",
    "   Run each check with the bash tool (the workdir above is the repository root) and record pass/fail",
    "   plus the key failing output.",
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
      "   Focus on the work this task changed first: if this is a git repository, run `git status --short`,",
      "   `git diff`, and `git diff --cached` to see what changed, and inspect those files closely.",
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
      "   - Verify it actually works: if the change is runnable, run the relevant program/entrypoint/tests and",
      "     EXERCISE the feature (start servers/CLIs with a bounded lifetime; kill them when done). Confirm the",
      "     behavior matches the stated requirement. Where it is not safely runnable, verify by careful reading",
      "     that the flow is complete and wired up: entry points reach the implementation, no missing integration.",
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
    "- NEVER modify source files, never commit, never install packages, never spawn agents, never call run_audit",
    "  or any audit-gate mechanism. Running the program to verify behavior (checks, tests, exercising the",
    "  feature) is allowed; write only to temp/scratch areas if needed.",
    `- Keep the audit bounded: at most ${MAX_CHECKS} checks.`,
    "- A check fails only when there is a concrete, defensible problem.",
    "",
    "When finished, call structured_output exactly once with:",
    "- summary: one or two sentences.",
    "- checks: at least one object per check — name (short label), passed (bool), detail (actionable; quote the",
    "  relevant output or file:line when possible). A check has passed: false when there is a concrete,",
    "  defensible problem. The host derives the overall verdict from EVERY check, so never report an empty",
    "  check list unless you genuinely could not run any check — an empty list is an inconclusive audit.",
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

/**
 * One full subagent audit attempt.
 * @returns `{ result: OverallResult, retryable: boolean }`.
 */
async function runAuditOnce(ctx, agent, cfg, signal, prompt) {
  let run;
  try {
    run = await ctx.subagents.start(cfg.subagentProvider, {
      label: "audit-gate",
      parent: agent,
      prompt,
      signal,
      outputSchema: VERDICT_SCHEMA,
      agentOptions: { provider: cfg.auditProvider, model: cfg.auditModel },
    });
  } catch (error) {
    return {
      result: inconclusive("start-error", `audit could not start: ${error?.message ?? String(error)}`),
      retryable: false,
    };
  }

  let result;
  try {
    result = await run.result;
  } catch (error) {
    return {
      result: inconclusive("result-rejected", `audit result promise rejected: ${error?.message ?? String(error)}`),
      retryable: true,
    };
  } finally {
    await run.dispose().catch(() => {});
  }

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
  const status = checks.some((c) => c.status === "failed") ? "failed" : "passed";
  return { result: { status, summary: String(structured.summary ?? ""), checks }, retryable: false };
}

/** How many times a failed/transient audit run is re-spawned before giving up. */
const AUDIT_MAX_ATTEMPTS = 2;

function delay(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Run the whole audit for one agent by delegating to a cheap-model subagent,
 * retrying transient infrastructure failures once so a blip does not skip the
 * audit entirely. Returns an {@link OverallResult} — never a bare pass.
 * @returns `{ status, cause?, summary, checks }`.
 */
export async function runAudit(ctx, agent, cfg, signal) {
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
  const prompt = buildAuditPrompt(workdir, languages, cfg, styleGuide, taskContext);

  for (let attempt = 1; ; attempt++) {
    const outcome = await runAuditOnce(ctx, agent, cfg, signal, prompt);
    if (signal?.aborted) return outcome.result;
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

/** Stable fingerprint of current failures, for no-progress detection. */
export function fingerprint(result) {
  return JSON.stringify(
    result.checks.filter((c) => c.status !== "passed").map((c) => ({ name: c.name, output: c.output })),
  );
}

/** Structured, lossless-JSON report returned by the `run_audit` tool. */
export function reportValue(result) {
  return {
    status: result.status,
    passed: result.status === "passed",
    summary: result.summary,
    ...(result.cause !== undefined ? { cause: result.cause } : {}),
    results: result.checks.map((c) => ({ name: c.name, run: c.run, status: c.status, output: c.output ?? "" })),
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
  // is hot-reloaded. `live()` is read at every gate trigger, so flipping the
  // switch takes effect on the very next completion/turn-end.
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
    onChange: () => {}, // gates read live() each trigger; no eager work needed
  });

  /** Static composition config overlaid with the live runtime switches. */
  const live = () => ({ ...cfg, ...resolveSwitches(source()) });

  // Model guidance: make the gate legible and cooperative, and only promise
  // surfaces the composition actually enabled (so the prompt never advertises
  // a disabled gate or a tool that is not registered).
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
  if (cfg.guardTurnEnd !== "off") {
    guidance.push("For non-goal work, after you stop, the gate re-runs the audit and hands any failures back to you to fix.");
  }
  guidance.push("Treat audit findings as remaining work, not as permission to end. Do not attempt to bypass or disable the gate.");
  ctx.systemPrompt.section({
    name: "tool:audit-gate",
    order: 120,
    text: guidance.join(" "),
  });

  // Per-agent turn-local state: bounded fix loop, no-progress detection, and a
  // mutation-versioned result cache shared by the gates and the tool.
  const states = new Map(); // agent -> { attempts, lastFingerprint, dirty, dirtyVersion, completed, inFlight }

  function stateFor(agent) {
    let st = states.get(agent);
    if (st === undefined) {
      st = {
        attempts: 0,
        lastFingerprint: undefined,
        dirty: false,
        dirtyVersion: 0,
        completed: undefined,
        inFlight: undefined,
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

  /**
   * Audit with a per-agent cache (completed result + single-flight promise),
   * invalidated by any mutating tool result regardless of success/failure.
   */
  async function runAuditCached(agent, signal) {
    if (isSubagent(agent)) {
      return {
        status: "passed",
        summary: "audit applies to top-level agents only",
        checks: [{ name: "audit", run: "llm-audit", status: "passed", output: "not applicable" }],
      };
    }
    const st = stateFor(agent);
    if (st.completed !== undefined && st.completed.dirtyVersion === st.dirtyVersion) {
      return st.completed.result;
    }
    if (st.inFlight !== undefined && st.inFlight.dirtyVersion === st.dirtyVersion) {
      return st.inFlight.promise;
    }
    const promise = runAudit(ctx, agent, cfg, signal)
      .then((result) => {
        if (!signal?.aborted) st.completed = { dirtyVersion: st.dirtyVersion, result };
        return result;
      })
      .finally(() => {
        if (st.inFlight?.promise === promise) st.inFlight = undefined;
      });
    st.inFlight = { dirtyVersion: st.dirtyVersion, promise };
    return promise;
  }

  // Reset turn-local state at the canonical turn boundary (fires before the
  // turn's first step, so dirtiness from one turn never leaks into the next).
  ctx.on("session/event", (session, event) => {
    if (event.type !== "turn/start") return;
    const agent = ctx.agents.get(session.id);
    if (agent === undefined || agent.session !== session) return;
    const st = stateFor(agent);
    st.dirty = false;
    st.attempts = 0;
    st.lastFingerprint = undefined;
    st.completed = undefined; // a new turn may carry new user input / external changes
  });

  // Mark an agent's current turn "modified" and invalidate the audit cache when
  // a mutating tool runs — regardless of success, because a failing command may
  // still have modified files before it errored.
  ctx.on("tools/result", (exec, result) => {
    if (exec.agent === undefined) return;
    if (!cfg.mutatingTools.has(exec.name)) return;
    const st = stateFor(exec.agent);
    st.dirty = true;
    st.dirtyVersion += 1;
  });

  // (A) Goal-completion gate — deny `update_goal` complete while the audit is
  // not passed (failed, or inconclusive when onInconclusive is deny).
  ctx.on("tools/pre-execute", (exec, next) => {
    if (exec.name !== "update_goal" || exec.agent === undefined) return next();
    const lv = live();
    if (!lv.enabled || !lv.guardCompletion) return next();
    if (isSubagent(exec.agent)) return next();
    const args = exec.arguments;
    if (typeof args !== "object" || args === null || args.action !== "complete") return next();
    return runAuditCached(exec.agent, exec.signal).then((result) => {
      if (result.status === "passed") return next();
      const blocking = result.status === "failed" || lv.onInconclusive === "deny";
      if (!blocking) return next(); // warn/allow: let it through, the report/logs carry the warning
      return { kind: "deny", reason: renderDenialReason(result) };
    });
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

    const result = await runAuditCached(agent, signal);
    if (signal.aborted) return;

    if (result.status === "passed") {
      st.attempts = 0;
      st.lastFingerprint = undefined;
      return; // passes → let the turn close
    }
    const blocking = result.status === "failed" || lv.onInconclusive === "deny";
    if (!blocking) {
      st.attempts = 0;
      st.lastFingerprint = undefined;
      return; // inconclusive with warn/allow → close the turn
    }

    const fp = fingerprint(result);
    const noProgress = fp === st.lastFingerprint;
    st.lastFingerprint = fp;

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

  // (C) Model-facing tool: run the same audit proactively and see the report.
  if (cfg.registerTool) {
    ctx.tools.register(
      defineTool({
        name: cfg.toolName,
        description:
          "Run this workspace's automatic audit and return a structured pass/fail report. The audit is " +
          "performed by a cheap-model subagent that generates and runs language-appropriate checks (lint, " +
          "typecheck, tests, build), reviews code style, hunts for real logic bugs, verifies the requested " +
          "functionality works end-to-end, and checks documentation coverage. Use it to confirm your work " +
          "passes before marking a task complete; the automatic completion gate enforces the same audit.",
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
          const result = await runAuditCached(agent, exec.signal);
          return reportValue(result);
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
  });
}
