import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, writeFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { Context, Service } from "@deepseek-ai/cordis";

// Load the plugin the way the DSH loader does: the module namespace (with
// name/inject/Config/apply), not the bare apply function.
const mod = await import("../lib/index.js");
const plugin = mod.default ?? mod;

const completed = (checks) => ({ output: [], structured: { summary: "s", checks }, stopReason: "completed" });
const stopped = (stopReason, summary) => ({ output: [], structured: { summary: summary ?? `stop: ${stopReason}` }, stopReason });

async function makeHarness(options = {}) {
  const root = new Context();
  let spawns = 0;
  let steers = 0;
  const prompts = [];
  const guards = [];
  // options.result: static result, or options.results: array consumed per spawn, or options.resultProvider
  const queue = [...(options.results ?? [])];
  const resultProvider = options.resultProvider ?? (() => queue.length > 0 ? queue.shift() : options.result);

  class MockSystemPrompt extends Service { constructor(c) { super(c, "systemPrompt"); } section() {} tools() {} }
  class MockAgents extends Service { constructor(c) { super(c, "agents"); } roots() { return []; } get() { return undefined; } }
  class MockTools extends Service { constructor(c) { super(c, "tools"); this.guards = guards; } register() {} guard(fn) { guards.push(fn); return () => {}; } }
  class MockCommands extends Service { constructor(c) { super(c, "commands"); } register() { return () => {}; } }
  class MockSettings extends Service { constructor(c) { super(c, "settings"); } register(ns, schema, opts) { return { get() { return opts.base; }, watch() {} }; } }
  class MockGoals extends Service {
    constructor(c) { super(c, "goals"); }
    get() { return options.goal; }
  }
  class MockSubagents extends Service {
    constructor(c) { super(c, "subagents"); }
    start(name, req) {
      spawns++;
      prompts.push(req.prompt);
      const r = resultProvider(req);
      if (r && r.reject) {
        return { result: new Promise((_, reject) => setTimeout(() => reject(new Error("seam fault")), 2)), dispose: async () => {} };
      }
      return { result: Promise.resolve(r), dispose: async () => {} };
    }
  }
  // workspaceState probes git and falls back to a bounded file walk; a mock that
  // "fails git" keeps the harness deterministic over an empty / temp workspace.
  class MockSubprocess extends Service {
    constructor(c) { super(c, "subprocess"); }
    spawn(spec) {
      return {
        done: Promise.resolve({ exitCode: 1, signal: undefined, killed: false }),
        collected: { stdout: { readFrom: () => ({ text: "" }) }, stderr: { readFrom: () => ({ text: "" }) } },
      };
    }
  }
  for (const S of [MockSystemPrompt, MockAgents, MockTools, MockCommands, MockSettings, MockGoals, MockSubagents, MockSubprocess]) root.plugin(S);

  const pluginConfig = {
    guardTurnEnd: options.guardTurnEnd ?? "modified",
    maxAttempts: options.maxAttempts ?? 5,
    onInconclusive: options.onInconclusive ?? "deny",
    workdir: options.workdir ?? "/tmp/audit-test",
  };
  await root.plugin(plugin, pluginConfig);

  const agent = { id: "root-1", session: { header: { cwd: options.workdir ?? "/tmp/audit-test", delegationDepth: 0 } } };
  agent.steer = () => { steers++; };

  return {
    root, agent, guards,
    get spawns() { return spawns; },
    get steers() { return steers; },
    get prompts() { return prompts; },
    complete: (args = {}) =>
      root.waterfall(root, "tools/pre-execute", { name: "update_goal", agent, arguments: { action: "complete", ...args }, signal: new AbortController().signal }, () => Promise.resolve({ kind: "allow" })),
    mutate: (isError = false) =>
      root.emit(agent, "tools/result", { name: "edit", agent, arguments: {} }, { isError }),
    turnStop: (turn = 1) =>
      root.serial(agent, "agent/turn-stopping", { turn, signal: new AbortController().signal, agent }),
  };
}

test("gate allows when audit passes", async () => {
  const h = await makeHarness({ result: completed([{ name: "lint", passed: true, detail: "ok" }]) });
  const g = await h.complete();
  assert.equal(g.kind, "allow");
  assert.equal(h.spawns, 1);
});

test("gate denies when audit fails", async () => {
  const h = await makeHarness({ result: completed([{ name: "lint", passed: false, detail: "boom" }]) });
  const g = await h.complete();
  assert.equal(g.kind, "deny");
  assert.match(g.reason, /verification failed before completion \(1 check/);
});

test("gate denies on inconclusive by default", async () => {
  const h = await makeHarness({ result: stopped("error") });
  const g = await h.complete();
  assert.equal(g.kind, "deny", "inconclusive must not be allowed by default");
  assert.match(g.reason, /could not be completed/);
  assert.equal(h.spawns, 2, "transient error is retried once before denying");
});

test("gate honors onInconclusive: allow", async () => {
  const h = await makeHarness({ onInconclusive: "allow", result: stopped("refusal") });
  const g = await h.complete();
  assert.equal(g.kind, "allow");
});

test("gate honors onInconclusive: warn", async () => {
  const h = await makeHarness({ onInconclusive: "warn", result: stopped("max-tokens") });
  const g = await h.complete();
  assert.equal(g.kind, "allow");
});

test("empty checks is inconclusive and blocks by default", async () => {
  const h = await makeHarness({ result: completed([]) });
  const g = await h.complete();
  assert.equal(g.kind, "deny");
  assert.match(g.reason, /reported no checks/);
});

test("run.result rejection becomes a controlled inconclusive deny, not a thrown fault", async () => {
  const h = await makeHarness({ result: { reject: true } });
  const g = await h.complete();
  assert.equal(g.kind, "deny");
  assert.match(g.reason, /could not be completed/);
  assert.match(g.reason, /rejected/);
});

test("an inconsistent model verdict cannot be reported as pass (passed is derived)", async () => {
  // Model says passed:true (extra field, ignored) but every check failed.
  const h = await makeHarness({
    result: { output: [], structured: { summary: "s", passed: true, checks: [{ name: "t", passed: false, detail: "x" }] }, stopReason: "completed" },
  });
  const g = await h.complete();
  assert.equal(g.kind, "deny");
});

test("turn-end gate: pass closes, fail steers", async () => {
  const h = await makeHarness({ result: completed([{ name: "t", passed: false, detail: "nope" }]) });
  h.mutate();
  await h.turnStop();
  assert.equal(h.steers, 1);
});

test("maxAttempts steers exactly N times (1, 2, 5)", async (t) => {
  for (const n of [1, 2, 5]) {
    const dir = await mkdtemp(join(tmpdir(), "audit-attempts-"));
    t.after(() => rm(dir, { recursive: true, force: true }));
    let seed = 0;
    const h = await makeHarness({
      maxAttempts: n,
      workdir: dir,
      resultProvider: () => completed([{ name: "t", passed: false, detail: `fail-${++seed}` }]),
    });
    for (let k = 0; k <= n; k++) {
      await writeFile(join(dir, "state.txt"), `iter-${k}`); // real change → new digest → progress
      h.mutate();
      await h.turnStop(k + 1);
    }
    assert.equal(h.steers, n, `maxAttempts=${n} should steer exactly ${n} times`);
  }
});

test("goal phase: only an ACTIVE goal shields the turn-end gate", async () => {
  // active goal → no audit, no steer
  let h = await makeHarness({ goal: { phase: "active", objective: "x" }, result: completed([{ name: "t", passed: false, detail: "x" }]) });
  h.mutate();
  await h.turnStop();
  assert.equal(h.spawns, 0, "active goal defers to the completion gate");
  assert.equal(h.steers, 0);

  // completed goal → the turn-end gate audits again
  h = await makeHarness({ goal: { phase: "complete", objective: "x" }, result: completed([{ name: "t", passed: false, detail: "x" }]) });
  h.mutate();
  await h.turnStop();
  assert.ok(h.spawns > 0, "completed goal does not shield the turn-end gate");
});

test("cache: digest-keyed, invalidated when the workspace actually changes", async (t) => {
  const dir = await mkdtemp(join(tmpdir(), "audit-cache-"));
  t.after(() => rm(dir, { recursive: true, force: true }));
  const h = await makeHarness({ workdir: dir, result: completed([{ name: "t", passed: true, detail: "ok" }]) });
  assert.equal((await h.complete()).kind, "allow");
  assert.equal(h.spawns, 1);
  assert.equal((await h.complete()).kind, "allow");
  assert.equal(h.spawns, 1, "unchanged workspace → cache hit");
  await writeFile(join(dir, "code.txt"), "changed");
  assert.equal((await h.complete()).kind, "allow");
  assert.equal(h.spawns, 2, "workspace change invalidates the cache");
});

test("monotonic guard denies completion after the certificate is invalidated", async (t) => {
  const dir = await mkdtemp(join(tmpdir(), "audit-guard-"));
  t.after(() => rm(dir, { recursive: true, force: true }));
  const h = await makeHarness({ workdir: dir, result: completed([{ name: "t", passed: true, detail: "ok" }]) });
  const completeExec = () => ({ name: "update_goal", agent: h.agent, arguments: { action: "complete" }, signal: new AbortController().signal });
  assert.equal((await h.complete()).kind, "allow", "fresh audit passes");
  assert.equal(h.guards[0](completeExec()), undefined, "guard allows a fresh pass certificate");

  h.mutate(); // a mutating tool invalidates the certificate (dirtyVersion bump)
  assert.match(h.guards[0](completeExec()), /fresh audit certificate/, "guard denies an invalidated certificate");

  await h.complete(); // re-audit re-issues the certificate on the same call
  assert.equal(h.guards[0](completeExec()), undefined, "guard allows after re-audit");
});

test("single-flight: concurrent audits spawn one subagent", async () => {
  const h = await makeHarness({ result: completed([{ name: "t", passed: true, detail: "ok" }]) });
  const [a, b] = await Promise.all([h.complete(), h.complete()]);
  assert.equal(a.kind, "allow");
  assert.equal(b.kind, "allow");
  assert.equal(h.spawns, 1, "two concurrent triggers must not spawn two subagents");
});

test("subagent triggers never audit (recursion guard)", async () => {
  const h = await makeHarness({ result: completed([{ name: "t", passed: false, detail: "x" }]) });
  const child = { id: "child", session: { header: { cwd: "/tmp/audit-test", delegationDepth: 1 } } };
  const g = await h.root.waterfall(h.root, "tools/pre-execute", { name: "update_goal", agent: child, arguments: { action: "complete" }, signal: new AbortController().signal }, () => Promise.resolve({ kind: "allow" }));
  assert.equal(g.kind, "allow");
  assert.equal(h.spawns, 0);
});
