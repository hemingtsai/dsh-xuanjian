/**
 * `audit-gate` — a DeepSeek Harness completion gate.
 *
 * When the model is about to finish a task, this plugin runs a configurable
 * full audit suite (lint / typecheck / tests / build / anything shell-based)
 * against the agent's workspace. If a required check fails, the failure is
 * handed back to the model so it fixes the code, and the task is not allowed
 * to end until verification passes (or a configured safety bound is reached).
 *
 * Two complementary gates, sharing one audit runner:
 *
 *   (A) Goal-completion gate  — a `tools/pre-execute` waterfall listener that
 *       DENIES `update_goal` with action `complete` while the suite fails.
 *       This is authoritative for goal-based tasks: the goal is never marked
 *       complete until verification passes, so goal semantics are preserved.
 *
 *   (B) Turn-end gate         — an `agent/turn-stopping` (serial) listener for
 *       non-goal tasks. When the model stops, it audits and, on failure, steers
 *       a fix prompt into the same turn so the model keeps working until the
 *       suite passes.
 *
 * @module dsh-audit-gate
 */
import z from "@deepseek-ai/schemastery";
import { createUserMessage, boundContextSummary } from "@deepseek-ai/dsh-llm";

export const name = "audit-gate";

/** Hard dependencies: the shell executor, the prompt registry, the agent registry. */
export const inject = ["bash", "systemPrompt", "agents"];

/** One audit command. */
const CommandSchema = z.object({
  /** Short label shown in feedback. */
  name: z.string().min(1),
  /** Shell command (run as `bash -c <run>`). */
  run: z.string().min(1),
  /** Optional per-command timeout (ms); capped by the shell executor. */
  timeoutMs: z.number().step(1).min(1).max(Number.MAX_SAFE_INTEGER),
  /** Whether a failing exit code blocks completion (default true). */
  required: z.boolean().default(true),
  /** How many chars of failing output are fed back to the model (default 4000). */
  maxOutputChars: z.number().step(1).min(0).max(1_000_000).default(4000),
});

/** Plugin configuration schema (the loader normalizes defaults). */
export const Config = z.object({
  /** Master switch. */
  enabled: z.boolean().default(true),
  /** Gate (A): deny `update_goal` complete until the suite passes. */
  guardCompletion: z.boolean().default(true),
  /** Gate (B): audit at turn end — `off` | `modified` (only when files changed) | `always`. */
  guardTurnEnd: z.union(["off", "modified", "always"]).default("modified"),
  /** Which agents are gated: `root` (top-level only) | `all`. */
  scope: z.union(["root", "all"]).default("root"),
  /** Max fix iterations the turn-end gate injects before giving up. */
  maxAttempts: z.number().step(1).min(1).max(64).default(5),
  /** Optional workspace override; defaults to the agent's session cwd. */
  workdir: z.string(),
  /** Tool names that mark the turn as "modified" for the `modified` trigger. */
  mutatingTools: z.array(z.string()).default(["write", "edit", "bash", "pwsh"]),
  /** The audit suite. */
  commands: z.array(CommandSchema).default([]),
});

const TURN_END_MODES = new Set(["off", "modified", "always"]);
const SCOPES = new Set(["root", "all"]);
const DEFAULT_MUTATING = ["write", "edit", "bash", "pwsh"];

/** Validate and default the (possibly raw) config. */
function resolveConfig(config) {
  const guardTurnEnd = config.guardTurnEnd ?? "modified";
  if (!TURN_END_MODES.has(guardTurnEnd)) {
    throw new TypeError("audit-gate: guardTurnEnd must be one of off|modified|always");
  }
  const scope = config.scope ?? "root";
  if (!SCOPES.has(scope)) throw new TypeError("audit-gate: scope must be root|all");
  const maxAttempts = config.maxAttempts ?? 5;
  if (!Number.isSafeInteger(maxAttempts) || maxAttempts < 1) {
    throw new TypeError("audit-gate: maxAttempts must be a positive safe integer");
  }
  return {
    enabled: config.enabled ?? true,
    guardCompletion: config.guardCompletion ?? true,
    guardTurnEnd,
    scope,
    maxAttempts,
    workdir: config.workdir,
    mutatingTools: new Set(config.mutatingTools ?? DEFAULT_MUTATING),
    commands: (config.commands ?? []).map((c) => ({
      name: c.name,
      run: c.run,
      timeoutMs: c.timeoutMs,
      required: c.required ?? true,
      maxOutputChars: c.maxOutputChars ?? 4000,
    })),
  };
}

/** Keep the tail of a (possibly huge) output stream. */
function trimOutput(text, max) {
  if (max <= 0) return "";
  const t = String(text ?? "");
  if (t.length <= max) return t;
  return `${t.slice(-max)}\n…[truncated]`;
}

/**
 * Run the whole audit suite in order.
 * @returns one verdict per command:
 *   { name, run, required, passed, exitCode?, timedOut?, output, error?, aborted? }
 */
async function runAudit(ctx, workdir, commands, signal) {
  const verdicts = [];
  for (const cmd of commands) {
    if (signal?.aborted) {
      verdicts.push({
        name: cmd.name,
        run: cmd.run,
        required: cmd.required,
        passed: false,
        aborted: true,
        output: "",
      });
      continue;
    }
    let result;
    try {
      const spec = ctx.bash.resolve({
        command: cmd.run,
        workdir,
        ...(cmd.timeoutMs === undefined ? {} : { timeoutMs: cmd.timeoutMs }),
        signal,
      });
      result = await ctx.bash.run(spec);
    } catch (error) {
      verdicts.push({
        name: cmd.name,
        run: cmd.run,
        required: cmd.required,
        passed: false,
        error: error?.message ?? String(error),
        output: "",
      });
      continue;
    }
    const passed = result.exitCode === 0;
    const stdout = trimOutput(result.stdout?.text, cmd.maxOutputChars);
    const stderr = trimOutput(result.stderr?.text, cmd.maxOutputChars);
    const output = result.timedOut
      ? `(timed out after ${result.timeoutMs}ms)\n${stdout || stderr}`
      : stdout || stderr;
    verdicts.push({
      name: cmd.name,
      run: cmd.run,
      required: cmd.required,
      passed,
      exitCode: result.exitCode,
      timedOut: result.timedOut === true,
      output,
    });
  }
  return verdicts;
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
    if (v.aborted) lines.push("(aborted before it could finish)");
    if (v.error) lines.push(`error: ${v.error}`);
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
      "the gate re-runs automatically, and this task only ends when verification passes.",
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

export function apply(ctx, config) {
  const cfg = resolveConfig(config);
  if (!cfg.enabled || cfg.commands.length === 0) return;

  // Model guidance: makes the gate legible and cooperative rather than a surprise.
  ctx.systemPrompt.section({
    name: "tool:audit-gate",
    order: 120,
    text:
      "An automatic verification gate audits this workspace before a task may finish. " +
      'For goal-based work, update_goal action "complete" is denied while the verification ' +
      "suite fails — fix the findings and retry completion. For non-goal work, after you stop " +
      "the gate re-runs the suite and hands any failures back to you to fix. Treat verification " +
      "failures as remaining work, not as permission to end. Do not attempt to bypass or disable the gate.",
  });

  // Per-agent turn-local state (bounded fix loop + no-progress detection).
  const states = new Map(); // agent -> { attempts, lastFingerprint, dirty }

  function stateFor(agent) {
    let st = states.get(agent);
    if (st === undefined) {
      st = { attempts: 0, lastFingerprint: undefined, dirty: false };
      states.set(agent, st);
    }
    return st;
  }

  function inScope(agent) {
    if (cfg.scope === "all") return true;
    return ctx.agents.roots().includes(agent);
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

  // Reset turn-local state at the canonical turn boundary (fires before the
  // turn's first step, so dirtiness accumulated during one turn never leaks
  // into the next). Session events are scope-routed to the session's scope,
  // which nests under this preset's standing scope.
  ctx.on("session/event", (session, event) => {
    if (event.type !== "turn/start") return;
    const agent = ctx.agents.get(session.id);
    if (agent === undefined || agent.session !== session) return;
    const st = stateFor(agent);
    st.dirty = false;
    st.attempts = 0;
    st.lastFingerprint = undefined;
  });

  // Mark an agent's current turn "modified" when a mutating tool succeeds.
  ctx.on("tools/result", (exec, result) => {
    if (exec.agent === undefined) return;
    if (!cfg.mutatingTools.has(exec.name)) return;
    if (result?.isError === true) return;
    stateFor(exec.agent).dirty = true;
  });

  // (A) Goal-completion gate — deny `update_goal` complete until verified.
  if (cfg.guardCompletion) {
    ctx.on("tools/pre-execute", (exec, next) => {
      if (exec.name !== "update_goal" || exec.agent === undefined) return next();
      if (!inScope(exec.agent)) return next();
      const args = exec.arguments;
      if (typeof args !== "object" || args === null || args.action !== "complete") return next();
      return runAudit(ctx, workdirFor(exec.agent), cfg.commands, exec.signal).then((verdicts) => {
        if (failedRequired(verdicts).length === 0) return next();
        return { kind: "deny", reason: renderDenialReason(verdicts) };
      });
    });
  }

  // (B) Turn-end gate — for tasks that do not use the goal system.
  if (cfg.guardTurnEnd !== "off") {
    ctx.on("agent/turn-stopping", async ({ agent, signal }) => {
      if (!inScope(agent)) return;
      if (goalExists(agent)) return; // the completion gate owns goal tasks
      const st = stateFor(agent);

      const firstForTurn = st.attempts === 0;
      const shouldAudit =
        cfg.guardTurnEnd === "always" ||
        (cfg.guardTurnEnd === "modified" && (st.dirty || !firstForTurn));
      if (!shouldAudit) return;
      if (signal.aborted) return;

      const verdicts = await runAudit(ctx, workdirFor(agent), cfg.commands, signal);
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

      if (st.attempts >= cfg.maxAttempts || noProgress) {
        ctx.logger.warn(
          `audit-gate: verification still failing for agent "${agent.id}" after ` +
            `${st.attempts} attempt(s); closing the turn with unresolved failures.`,
        );
        return; // bounded give-up → let the turn close
      }

      agent.steer(
        createUserMessage({
          content: renderFeedback(verdicts, st.attempts, cfg.maxAttempts),
          source: {
            kind: "plugin",
            plugin: "audit-gate",
            form: "notice",
            summary: boundContextSummary(`verification failed (${failed.length} check(s)); fix and retry`),
          },
        }),
      );
    });
  }

  // Release per-agent state when an agent goes away.
  ctx.on("agent/disposed", ({ agent }) => {
    states.delete(agent);
  });
}

export default apply;
