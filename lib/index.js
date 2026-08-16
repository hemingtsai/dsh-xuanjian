/**
 * `audit-gate` — a DeepSeek Harness completion gate.
 *
 * When the model is about to finish a task, this plugin audits the agent's
 * workspace with a **cheap-model subagent**: it detects the project language,
 * lets the subagent generate and run the language-appropriate checks (lint /
 * typecheck / tests / build), review code style, hunt for real logic bugs,
 * verify the requested functionality works end-to-end, and check documentation
 * coverage, then returns a structured verdict. If a required check fails, the
 * failure is handed back to the model so it fixes the code, and the task is
 * not allowed to end until the audit passes (or a configured safety bound is
 * reached).
 *
 * Three surfaces, sharing one audit runner and one per-agent result cache:
 *
 *   (A) Goal-completion gate  — a `tools/pre-execute` waterfall listener that
 *       DENIES `update_goal` with action `complete` while the audit fails.
 *       This is authoritative for goal-based tasks: the goal is never marked
 *       complete until the audit passes, so goal semantics are preserved.
 *
 *   (B) Turn-end gate         — an `agent/turn-stopping` (serial) listener for
 *       non-goal tasks. When the model stops, it audits and, on failure, steers
 *       a fix prompt into the same turn so the model keeps working until the
 *       audit passes.
 *
 *   (C) `run_audit` tool      — a model-facing tool so the model can run the
 *       same audit proactively and see a structured report before it attempts
 *       completion, avoiding a denial round-trip.
 *
 * Recursion safety: the audit itself is a subagent, and a subagent inherits the
 * parent's preset (which may mount this very plugin). The runner therefore
 * only audits top-level agents (`delegationDepth` 0); any gate or tool
 * trigger on a subagent resolves to a non-blocking "not applicable" verdict,
 * so an audit child can never audit itself.
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
  /** Optional workspace override; defaults to the agent's session cwd. */
  workdir: z.string(),
  /** Tool names that mark the turn as "modified" for the `modified` trigger. */
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
 * provider, style/docs toggles, style guide) stays composition config and
 * takes effect on (re)load.
 */
const SettingsSchema = z.object({
  enabled: z.boolean().default(true),
  guardCompletion: z.boolean().default(true),
  guardTurnEnd: z.union(["off", "modified", "always"]).default("modified"),
  maxAttempts: z.number().step(1).min(1).max(64).default(5),
});

const TURN_END_MODES = new Set(["off", "modified", "always"]);
const DEFAULT_MUTATING = ["write", "edit", "bash", "pwsh"];

/** Validate and default the (possibly raw) config. */
function resolveConfig(config) {
  const guardTurnEnd = config.guardTurnEnd ?? "modified";
  if (!TURN_END_MODES.has(guardTurnEnd)) {
    throw new TypeError("audit-gate: guardTurnEnd must be one of off|modified|always");
  }
  const maxAttempts = config.maxAttempts ?? 5;
  if (!Number.isSafeInteger(maxAttempts) || maxAttempts < 1) {
    throw new TypeError("audit-gate: maxAttempts must be a positive safe integer");
  }
  return {
    enabled: config.enabled ?? true,
    guardCompletion: config.guardCompletion ?? true,
    guardTurnEnd,
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

/** Keep the tail of a (possibly huge) output stream. */
function trimOutput(text, max) {
  if (max <= 0) return "";
  const t = String(text ?? "");
  if (t.length <= max) return t;
  return `${t.slice(-max)}\n…[truncated]`;
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
  { files: [".csproj"], label: "C#", hint: "Run dotnet build and dotnet test." },
];

/**
 * Deterministically detect the project language(s) from manifest files in the
 * workspace root. Falls back to a generic directive when nothing is known.
 */
async function detectLanguages(workdir) {
  let names;
  try {
    names = await readdir(workdir);
  } catch {
    names = [];
  }
  const found = [];
  const seen = new Set();
  for (const signal of LANGUAGE_SIGNALS) {
    if (signal.files.some((file) => names.includes(file))) {
      if (!seen.has(signal.label)) {
        seen.add(signal.label);
        found.push(signal);
      }
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
    return trimOutput(await readFile(target, "utf8"), 8000);
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
 * from the session. Plugin-injected runtime snapshots are skipped.
 */
function taskContextFor(agent, goals) {
  const lines = [];
  if (goals !== undefined) {
    try {
      const goal = goals.get(agent);
      if (typeof goal?.objective === "string" && goal.objective.length > 0) {
        lines.push(`Goal: ${goal.objective}`);
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
      requests.unshift(text);
    }
    if (requests.length > 0) {
      lines.push("Recent user request(s):", ...requests.map((r) => trimOutput(r, 600)));
    }
  } catch {}
  const joined = lines.join("\n").trim();
  return joined.length === 0 ? undefined : trimOutput(joined, 2400);
}

// ── audit subagent contract ─────────────────────────────────────────────────

/** Structured verdict the audit subagent must report. */
const VERDICT_SCHEMA = {
  type: "object",
  properties: {
    passed: { type: "boolean" },
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
  required: ["passed", "summary", "checks"],
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
    "- Keep the audit bounded: at most 8 checks.",
    "- A check fails only when there is a concrete, defensible problem.",
    "",
    "When finished, call structured_output exactly once with:",
    "- passed: true only if EVERY check passed.",
    "- summary: one or two sentences.",
    "- checks: one object per check — name (short label), passed (bool), detail (actionable; quote the",
    "  relevant output or file:line when possible).",
  );
  return parts.join("\n");
}

/** One full subagent audit attempt. */
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
      verdicts: [{ name: "audit", run: "llm-audit", required: false, passed: false, output: `audit could not start: ${error?.message ?? String(error)}` }],
      retryable: false,
    };
  }

  let result;
  try {
    result = await run.result;
  } finally {
    await run.dispose().catch(() => {});
  }

  const structured = result.structured;
  if (result.stopReason !== "completed" || !isPlainObject(structured) || !Array.isArray(structured.checks)) {
    const reason =
      typeof structured?.summary === "string" && structured.summary.length > 0
        ? structured.summary
        : `audit did not complete (${result.stopReason})`;
    return {
      verdicts: [{ name: "audit", run: "llm-audit", required: false, passed: false, output: `${reason} — not blocking this run.` }],
      // Transient infrastructure failures (transport/error, max-tokens) are worth retrying once;
      // a refusal is a deliberate decline and won't change on a fresh run.
      retryable: result.stopReason === "error" || result.stopReason === "max-tokens",
    };
  }

  const verdicts = structured.checks.slice(0, 16).map((check) => ({
    name: String(check?.name ?? "check"),
    run: "llm-audit",
    required: true,
    passed: check?.passed === true,
    output: trimOutput(check?.detail, 4000),
  }));
  if (verdicts.length === 0) {
    return [{
      verdicts: [{ name: "audit", run: "llm-audit", required: true, passed: structured.passed === true, output: trimOutput(structured.summary, 4000) }],
      retryable: false,
    }][0];
  }
  return { verdicts, retryable: false };
}

/** How many times a failed/transient audit run is re-spawned before giving up. */
const AUDIT_MAX_ATTEMPTS = 2;

function delay(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Run the whole audit for one agent by delegating to a cheap-model subagent,
 * retrying transient infrastructure failures once so a blip does not skip the
 * audit entirely.
 * @returns one verdict per check: `{ name, run, required, passed, output }`.
 */
async function runAudit(ctx, agent, cfg, signal) {
  if (signal?.aborted) {
    return [{ name: "audit", run: "llm-audit", required: false, passed: false, output: "audit aborted" }];
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
    if (signal?.aborted) return outcome.verdicts;
    if (!outcome.retryable || attempt >= AUDIT_MAX_ATTEMPTS) {
      return outcome.verdicts;
    }
    ctx.logger.warn(`audit-gate: audit subagent did not complete (attempt ${attempt}/${AUDIT_MAX_ATTEMPTS}); retrying`);
    await delay(1200);
  }
}

/** Required checks that failed. */
function failedRequired(verdicts) {
  return verdicts.filter((v) => !v.passed && v.required);
}

/** Render a compact PASS/FAIL summary plus failing output. */
function renderVerdictLines(verdicts) {
  const lines = [];
  for (const v of verdicts) {
    lines.push(`${v.passed ? "PASS" : "FAIL"}: ${v.name}  (${v.run})`);
    if (!v.passed && v.output) lines.push(v.output);
  }
  return lines;
}

/** Turn-end gate: the fix prompt injected into the same turn. */
function renderFeedback(verdicts, attempts, maxAttempts) {
  const failed = failedRequired(verdicts);
  const lines = [
    "<audit_gate>",
    `Automatic verification failed (fix attempt ${attempts}/${maxAttempts}).`,
    `${failed.length} required check(s) are failing. Fix the issues below and then stop; ` +
      "the gate re-runs automatically, and this task only ends when the audit passes.",
    "",
    ...renderVerdictLines(verdicts),
    "</audit_gate>",
  ];
  return [{ type: "text", text: lines.join("\n") }];
}

/** Completion gate: the denial reason surfaced as the `update_goal` tool error. */
function renderDenialReason(verdicts) {
  const failed = failedRequired(verdicts);
  return [
    `Automatic verification failed before completion (${failed.length} required check(s) failing). ` +
      "Fix the findings below, then mark the goal complete again once they pass.",
    "",
    ...renderVerdictLines(verdicts),
  ].join("\n");
}

/** Stable fingerprint of current failures, for no-progress detection. */
function fingerprint(verdicts) {
  return JSON.stringify(failedRequired(verdicts).map((v) => ({ name: v.name, output: v.output })));
}

/** Structured, lossless-JSON report returned by the `run_audit` tool. */
function reportValue(verdicts) {
  const failed = failedRequired(verdicts);
  return {
    passed: failed.length === 0,
    summary:
      failed.length === 0
        ? `all ${verdicts.length} check(s) passed`
        : `${failed.length} required check(s) failed`,
    results: verdicts.map((v) => ({
      name: v.name,
      run: v.run,
      required: v.required,
      passed: v.passed,
      output: v.output ?? "",
    })),
  };
}

/** Human-readable render of the structured report. */
function renderReport(report) {
  const lines = [report.passed ? "PASS" : "FAIL", report.summary, ""];
  for (const r of report.results) {
    lines.push(`${r.passed ? "PASS" : "FAIL"}: ${r.name}  (${r.run})`);
    if (!r.passed && r.output) lines.push(r.output);
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

  // Model guidance: makes the gate legible and cooperative rather than a surprise.
  ctx.systemPrompt.section({
    name: "tool:audit-gate",
    order: 120,
    text:
      "An automatic verification gate (when enabled) audits this workspace before a task may finish. " +
      `The audit is performed by a ${cfg.auditModel} subagent that generates and runs language-appropriate ` +
      "checks, reviews code style, hunts for real logic bugs, verifies the requested functionality works " +
      "end-to-end, and checks documentation coverage. For goal-based work, update_goal " +
      'action "complete" is denied while the audit fails — fix the findings and retry completion. ' +
      `You may run the ${cfg.toolName} tool any time to see the current audit result. For non-goal work, after ` +
      "you stop, the gate re-runs the audit and hands any failures back to you to fix. Treat audit " +
      "findings as remaining work, not as permission to end. Do not attempt to bypass or disable the gate.",
  });

  // Per-agent turn-local state: bounded fix loop, no-progress detection, and a
  // mutation-versioned result cache shared by the gates and the tool.
  const states = new Map(); // agent -> { attempts, lastFingerprint, dirty, dirtyVersion, cache }

  function stateFor(agent) {
    let st = states.get(agent);
    if (st === undefined) {
      st = {
        attempts: 0,
        lastFingerprint: undefined,
        dirty: false,
        dirtyVersion: 0,
        cache: undefined,
      };
      states.set(agent, st);
    }
    return st;
  }

  /** Subagents are never audited — the audit itself is a subagent, so gating one would recurse. */
  function isSubagent(agent) {
    return (agent?.session?.header?.delegationDepth ?? 0) > 0;
  }

  function goalExists(agent) {
    const goals = ctx.get("goals");
    if (goals === undefined) return false;
    try {
      return goals.get(agent) !== undefined;
    } catch {
      return false;
    }
  }

  function workdirFor(agent) {
    return cfg.workdir ?? agent.session.header.cwd ?? process.cwd();
  }

  /** Audit with a per-agent cache invalidated by any model code mutation. */
  async function runAuditCached(agent, signal) {
    if (isSubagent(agent)) {
      return [{ name: "audit", run: "llm-audit", required: false, passed: true, output: "audit applies to top-level agents only" }];
    }
    const st = stateFor(agent);
    if (st.cache !== undefined && st.cache.dirtyVersion === st.dirtyVersion) return st.cache.verdicts;
    const verdicts = await runAudit(ctx, agent, cfg, signal);
    if (!signal?.aborted) st.cache = { dirtyVersion: st.dirtyVersion, verdicts };
    return verdicts;
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
    st.cache = undefined; // a new turn may carry new user input / external changes
  });

  // Mark an agent's current turn "modified" and invalidate the audit cache when
  // a mutating tool succeeds.
  ctx.on("tools/result", (exec, result) => {
    if (exec.agent === undefined) return;
    if (!cfg.mutatingTools.has(exec.name)) return;
    if (result?.isError === true) return;
    const st = stateFor(exec.agent);
    st.dirty = true;
    st.dirtyVersion += 1;
  });

  // (A) Goal-completion gate — deny `update_goal` complete until the audit passes.
  ctx.on("tools/pre-execute", (exec, next) => {
    if (exec.name !== "update_goal" || exec.agent === undefined) return next();
    const lv = live();
    if (!lv.enabled || !lv.guardCompletion) return next();
    if (isSubagent(exec.agent)) return next();
    const args = exec.arguments;
    if (typeof args !== "object" || args === null || args.action !== "complete") return next();
    return runAuditCached(exec.agent, exec.signal).then((verdicts) => {
      if (failedRequired(verdicts).length === 0) return next();
      return { kind: "deny", reason: renderDenialReason(verdicts) };
    });
  });

  // (B) Turn-end gate — for tasks that do not use the goal system.
  ctx.on("agent/turn-stopping", async ({ agent, signal }) => {
    const lv = live();
    if (!lv.enabled || lv.guardTurnEnd === "off") return;
    if (agent === undefined || isSubagent(agent)) return;
    if (goalExists(agent)) return; // the completion gate owns goal tasks
    const st = stateFor(agent);

    const firstForTurn = st.attempts === 0;
    const shouldAudit =
      lv.guardTurnEnd === "always" ||
      (lv.guardTurnEnd === "modified" && (st.dirty || !firstForTurn));
    if (!shouldAudit) return;
    if (signal.aborted) return;

    const verdicts = await runAuditCached(agent, signal);
    if (signal.aborted) return;

    const failed = failedRequired(verdicts);
    if (failed.length === 0) {
      st.attempts = 0;
      st.lastFingerprint = undefined;
      return; // passes → let the turn close
    }

    st.attempts += 1;
    const fp = fingerprint(verdicts);
    const noProgress = fp === st.lastFingerprint;
    st.lastFingerprint = fp;

    if (st.attempts >= lv.maxAttempts || noProgress) {
      ctx.logger.warn(
        `audit-gate: verification still failing for agent "${agent.id}" after ` +
          `${st.attempts} attempt(s); closing the turn with unresolved failures.`,
      );
      return; // bounded give-up → let the turn close
    }

    agent.steer(
      createUserMessage({
        content: renderFeedback(verdicts, st.attempts, lv.maxAttempts),
        source: {
          kind: "plugin",
          plugin: "audit-gate",
          form: "notice",
          summary: boundContextSummary(`verification failed (${failed.length} check(s)); fix and retry`),
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
          "typecheck, tests, build), reviews code style, and checks documentation coverage. Use it to confirm " +
          "your work passes before marking a task complete; the automatic completion gate enforces the same audit.",
        parameters: {},
        output: {
          schema: { type: "json" },
          render: (_args, value) => renderReport(value),
        },
        async execute(_args, exec) {
          const agent = exec.agent;
          if (agent === undefined) {
            return { passed: false, summary: "run_audit requires a calling agent", results: [] };
          }
          const verdicts = await runAuditCached(agent, exec.signal);
          return reportValue(verdicts);
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
