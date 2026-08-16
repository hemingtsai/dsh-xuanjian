/**
 * `audit-gate` — a DeepSeek Harness completion gate.
 *
 * When the model is about to finish a task, this plugin runs a configurable
 * full audit suite (lint / typecheck / tests / build / anything shell-based)
 * against the agent's workspace. If a required check fails, the failure is
 * handed back to the model so it fixes the code, and the task is not allowed
 * to end until verification passes (or a configured safety bound is reached).
 *
 * Three surfaces, sharing one audit runner and one per-agent result cache:
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
 *   (C) `run_audit` tool      — a model-facing tool so the model can run the
 *       same suite proactively and see a structured report before it attempts
 *       completion, avoiding a denial round-trip.
 *
 * @module dsh-audit-gate
 */
import z from "@deepseek-ai/schemastery";
import { createUserMessage, boundContextSummary } from "@deepseek-ai/dsh-llm";
import { defineTool } from "@deepseek-ai/dsh-tools";
import { installSettingsSection, settingsNamespace } from "@deepseek-ai/dsh-settings";

export const name = "audit-gate";

/** Settings namespace: the runtime switches surfaced to the Web settings UI. */
const NS = settingsNamespace("audit-gate");

/** Hard dependencies: shell executor, prompt registry, agent registry, tool registry, command registry. */
export const inject = ["shell", "systemPrompt", "agents", "tools", "commands"];

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
  /** Register the model-facing `run_audit` tool (C). */
  registerTool: z.boolean().default(true),
  /** Name of the model-facing audit tool. */
  toolName: z.string().default("run_audit"),
  /** The audit suite. */
  commands: z.array(CommandSchema).default([]),
});

/**
 * The subset of `Config` that is a live, user-owned runtime switch. It is
 * registered as the `audit-gate` settings namespace, so `enabled`,
 * `guardCompletion`, `guardTurnEnd`, and `maxAttempts` can be flipped at
 * runtime (from the Web settings UI or `~/.dsh/settings.yaml`) without
 * editing the composition. Everything else (the audit `commands`, `scope`,
 * `workdir`, `mutatingTools`, `registerTool`, `toolName`) stays composition
 * config and takes effect on (re)load.
 */
const SettingsSchema = z.object({
  enabled: z.boolean().default(true),
  guardCompletion: z.boolean().default(true),
  guardTurnEnd: z.union(["off", "modified", "always"]).default("modified"),
  maxAttempts: z.number().step(1).min(1).max(64).default(5),
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
    registerTool: config.registerTool ?? true,
    toolName: config.toolName ?? "run_audit",
    commands: (config.commands ?? []).map((c) => ({
      name: c.name,
      run: c.run,
      timeoutMs: c.timeoutMs,
      required: c.required ?? true,
      maxOutputChars: c.maxOutputChars ?? 4000,
    })),
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
      const spec = ctx.shell.resolve({
        command: cmd.run,
        workdir,
        ...(cmd.timeoutMs === undefined ? {} : { timeoutMs: cmd.timeoutMs }),
        signal,
      });
      result = await ctx.shell.run(spec);
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
      exitCode: v.exitCode ?? null,
      timedOut: v.timedOut === true,
      error: v.error ?? null,
      output: v.output ?? "",
    })),
  };
}

/** Human-readable render of the structured report. */
function renderReport(report) {
  const lines = [report.passed ? "PASS" : "FAIL", report.summary, ""];
  for (const r of report.results) {
    lines.push(`${r.passed ? "PASS" : "FAIL"}: ${r.name}  (${r.run})`);
    if (r.timedOut) lines.push("(timed out)");
    if (r.error) lines.push(`error: ${r.error}`);
    if (!r.passed && r.output) lines.push(r.output);
  }
  return [{ type: "text", text: lines.join("\n") }];
}

/** Human-readable switch state for the `/audit` status command. */
function renderStatus(lv) {
  return [
    "Audit gate:",
    `  enabled: ${lv.enabled}`,
    `  guardCompletion: ${lv.guardCompletion}`,
    `  guardTurnEnd: ${lv.guardTurnEnd}`,
    `  maxAttempts: ${lv.maxAttempts}`,
    `  suite: ${lv.commands.length} check(s)`,
    "",
    "Commands: /audit (status) · /audit-toggle [on|off] (flip with no argument)",
  ].join("\n");
}

export function apply(ctx, config) {
  const cfg = resolveConfig(config);
  if (cfg.commands.length === 0) return;

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
      'For goal-based work, update_goal action "complete" is denied while the verification ' +
      `suite fails — fix the findings and retry completion. You may run the ${cfg.toolName} tool ` +
      "any time to check the suite yourself before attempting completion. For non-goal work, after " +
      "you stop the gate re-runs the suite and hands any failures back to you to fix. Treat " +
      "verification failures as remaining work, not as permission to end. Do not attempt to bypass " +
      "or disable the gate.",
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

  /** Audit with a per-agent cache invalidated by any model code mutation. */
  async function runAuditCached(agent, commands, signal) {
    const st = stateFor(agent);
    if (st.cache !== undefined && st.cache.dirtyVersion === st.dirtyVersion) return st.cache.verdicts;
    const verdicts = await runAudit(ctx, workdirFor(agent), commands, signal);
    st.cache = { dirtyVersion: st.dirtyVersion, verdicts };
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

  // (A) Goal-completion gate — deny `update_goal` complete until verified.
  ctx.on("tools/pre-execute", (exec, next) => {
    if (exec.name !== "update_goal" || exec.agent === undefined) return next();
    const lv = live();
    if (!lv.enabled || !lv.guardCompletion) return next();
    if (!inScope(exec.agent)) return next();
    const args = exec.arguments;
    if (typeof args !== "object" || args === null || args.action !== "complete") return next();
    return runAuditCached(exec.agent, cfg.commands, exec.signal).then((verdicts) => {
      if (failedRequired(verdicts).length === 0) return next();
      return { kind: "deny", reason: renderDenialReason(verdicts) };
    });
  });

  // (B) Turn-end gate — for tasks that do not use the goal system.
  ctx.on("agent/turn-stopping", async ({ agent, signal }) => {
    const lv = live();
    if (!lv.enabled || lv.guardTurnEnd === "off") return;
    if (!inScope(agent)) return;
    if (goalExists(agent)) return; // the completion gate owns goal tasks
    const st = stateFor(agent);

    const firstForTurn = st.attempts === 0;
    const shouldAudit =
      lv.guardTurnEnd === "always" ||
      (lv.guardTurnEnd === "modified" && (st.dirty || !firstForTurn));
    if (!shouldAudit) return;
    if (signal.aborted) return;

    const verdicts = await runAuditCached(agent, cfg.commands, signal);
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

  // (C) Model-facing tool: run the same suite proactively and see the report.
  if (cfg.registerTool) {
    ctx.tools.register(
      defineTool({
        name: cfg.toolName,
        description:
          "Run this workspace's configured verification/audit suite (lint, typecheck, tests, build, or " +
          "any shell checks) and return a structured pass/fail report. Use it to confirm your work passes " +
          "before marking a task complete; the automatic completion gate enforces the same suite.",
        parameters: {
          names: {
            type: "array",
            items: { type: "string" },
            description: "Optional subset of command names to run; omit to run the full suite.",
          },
        },
        output: {
          schema: { type: "json" },
          render: (_args, value) => renderReport(value),
        },
        async execute(args, exec) {
          const agent = exec.agent;
          if (agent === undefined) {
            return { passed: false, summary: "run_audit requires a calling agent", results: [] };
          }
          let commands = cfg.commands;
          if (Array.isArray(args.names) && args.names.length > 0) {
            const wanted = new Set(args.names);
            commands = cfg.commands.filter((c) => wanted.has(c.name));
            if (commands.length === 0) {
              return {
                passed: false,
                summary: `no configured check matches the given names: ${args.names.join(", ")}`,
                results: [],
              };
            }
          }
          const verdicts = await runAuditCached(agent, commands, exec.signal);
          return reportValue(verdicts);
        },
      }),
    );
  }

  // (D) Human slash commands: inspect and toggle the runtime switch.
  ctx.commands.register({
    name: "audit",
    description: "show the audit gate's current switch state",
    handler: () => ({ kind: "success", text: renderStatus(live()) }),
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
        text: `Audit gate ${target ? "enabled" : "disabled"}.\n\n${renderStatus(live())}`,
      };
    },
  });

  // Release per-agent state when an agent goes away.
  ctx.on("agent/disposed", ({ agent }) => {
    states.delete(agent);
  });
}

