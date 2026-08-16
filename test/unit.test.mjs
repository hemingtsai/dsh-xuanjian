import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, mkdir, writeFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  resolveConfig,
  trimOutput,
  trimHeadTail,
  detectLanguages,
  verdictOf,
  reportValue,
  renderDenialReason,
  renderFeedback,
  VERDICT_SCHEMA,
  ALLOWED_EXECUTABLES,
  resolveExecutableArgv,
  mergeEvidence,
  workspaceState,
} from "../lib/index.js";

test("resolveConfig: defaults and validation", () => {
  const cfg = resolveConfig({});
  assert.equal(cfg.enabled, true);
  assert.equal(cfg.guardCompletion, true);
  assert.equal(cfg.guardTurnEnd, "modified");
  assert.equal(cfg.onInconclusive, "deny");
  assert.equal(cfg.maxAttempts, 5);
  assert.equal(cfg.auditModel, "deepseek-v4-flash");

  assert.throws(() => resolveConfig({ guardTurnEnd: "sometimes" }), /guardTurnEnd/);
  assert.throws(() => resolveConfig({ onInconclusive: "maybe" }), /onInconclusive/);
  assert.throws(() => resolveConfig({ maxAttempts: 0 }), /maxAttempts/);
});

test("trimOutput keeps the tail; trimHeadTail keeps head and tail", () => {
  const long = "A".repeat(100) + "TAIL";
  const tail = trimOutput(long, 10);
  assert.ok(tail.includes("TAIL"));
  assert.ok(tail.includes("truncated"));
  assert.ok(tail.length <= 10 + 1 + "…[truncated]".length, "tail stays bounded");
  const ht = trimHeadTail(long, 20);
  assert.ok(ht.startsWith("AAA"), "head survives");
  assert.ok(ht.includes("TAIL"), "tail survives");
  assert.ok(ht.includes("truncated"));
});

test("detectLanguages: manifest at root, nested (monorepo), .csproj matcher, unknown fallback", async (t) => {
  const dir = await mkdtemp(join(tmpdir(), "audit-detect-"));
  t.after(() => rm(dir, { recursive: true, force: true }));

  await writeFile(join(dir, "package.json"), "{}");
  let langs = await detectLanguages(dir);
  assert.ok(langs.some((l) => l.label.startsWith("JavaScript/TypeScript")));

  await rm(join(dir, "package.json"));
  await mkdir(join(dir, "services", "api"), { recursive: true });
  await writeFile(join(dir, "services", "api", "go.mod"), "module x");
  langs = await detectLanguages(dir);
  assert.ok(langs.some((l) => l.label === "Go"), "nested go.mod detected");

  await writeFile(join(dir, "Demo.csproj"), "<Project/>");
  langs = await detectLanguages(dir);
  assert.ok(langs.some((l) => l.label === "C#"), ".csproj matcher works");

  await rm(join(dir, "services", "api", "go.mod"));
  await rm(join(dir, "Demo.csproj"));
  langs = await detectLanguages(dir);
  assert.equal(langs[0].label, "unknown", "fallback is unknown");
});

test("verdictOf maps model check pass/fail", () => {
  assert.equal(verdictOf({ name: "t", passed: true }).status, "passed");
  assert.equal(verdictOf({ name: "t", passed: false }).status, "failed");
  assert.equal(verdictOf({}).status, "failed", "missing passed is a failure, never a pass");
});

test("reportValue: passed/failed/inconclusive are reported truthfully", () => {
  const passed = reportValue({ status: "passed", summary: "all good", checks: [{ name: "a", run: "llm-audit", status: "passed", output: "" }] });
  assert.equal(passed.passed, true);
  assert.equal(passed.status, "passed");
  assert.equal(passed.summary, "all good");

  const failed = reportValue({ status: "failed", summary: "nope", checks: [{ name: "a", run: "llm-audit", status: "failed", output: "x" }] });
  assert.equal(failed.passed, false);
  assert.equal(failed.status, "failed");

  const inconclusive = reportValue({ status: "inconclusive", cause: "empty-checks", summary: "no checks", checks: [] });
  assert.equal(inconclusive.passed, false, "inconclusive is never reported as passed");
  assert.equal(inconclusive.status, "inconclusive");
  assert.equal(inconclusive.cause, "empty-checks");
});

test("renderDenialReason distinguishes failed vs inconclusive", () => {
  const failed = renderDenialReason({ status: "failed", summary: "s", checks: [{ name: "t", run: "llm-audit", status: "failed", output: "boom" }] });
  assert.match(failed, /verification failed before completion \(1 check/);
  const inconclusive = renderDenialReason({ status: "inconclusive", summary: "subagent error", checks: [] });
  assert.match(inconclusive, /could not be completed/);
  assert.match(inconclusive, /subagent error/);
  assert.doesNotMatch(inconclusive, /verification failed before completion/);
});

test("renderFeedback shows the correct phase and attempt count", () => {
  const failed = renderFeedback({ status: "failed", summary: "s", checks: [{ name: "t", run: "llm-audit", status: "failed", output: "boom" }] }, 2, 5);
  assert.match(failed[0].text, /fix attempt 2\/5/);
  const inconclusive = renderFeedback({ status: "inconclusive", summary: "no checks", checks: [] }, 1, 5);
  assert.match(inconclusive[0].text, /inconclusive/);
});

test("resolveExecutableArgv: allowlists runners and confines file paths", () => {
  const wd = "/tmp/repo";
  assert.deepEqual(resolveExecutableArgv(wd, "npm-script", "test", []), ["npm", "run", "test"]);
  assert.deepEqual(resolveExecutableArgv(wd, "executable", "ruff", ["check", "."]), ["ruff", "check", "."]);
  assert.throws(() => resolveExecutableArgv(wd, "executable", "evil-binary", []), /allowlist/);
  assert.throws(() => resolveExecutableArgv(wd, "npm-script", "test && rm -rf /", []), /invalid npm script/);
  assert.throws(() => resolveExecutableArgv(wd, "shell", "bash", []), /unknown runner/);
  assert.throws(() => resolveExecutableArgv(wd, "file", "../outside", []), /escapes/);
  assert.throws(() => resolveExecutableArgv(wd, "file", "/abs/path", []), /relative/);
  const fileArgv = resolveExecutableArgv(wd, "file", "bin/tool", []);
  assert.equal(fileArgv[0], "/tmp/repo/bin/tool");
  assert.ok(ALLOWED_EXECUTABLES.has("pytest"));
});

test("mergeEvidence: host exit code overrides the model's restatement", () => {
  const model = {
    status: "passed",
    summary: "s",
    checks: [
      { name: "lint", run: "llm-audit", status: "passed", output: "" },
      { name: "style", run: "llm-audit", status: "failed", output: "indent" },
    ],
  };
  const evidence = [{ name: "lint", runner: "executable", argv: ["eslint", "."], exitCode: 1, passed: false, timedOut: false, outputTail: "boom", outputDigest: "x" }];
  const merged = mergeEvidence(model, evidence);
  assert.equal(merged.checks[0].status, "failed", "host evidence flips the model's pass to fail");
  assert.equal(merged.checks[0].run, "audit:executable");
  assert.equal(merged.status, "failed");
});

test("workspaceState (non-git walk) reflects file changes (self-modification detection)", async (t) => {
  const dir = await mkdtemp(join(tmpdir(), "audit-state-"));
  t.after(() => rm(dir, { recursive: true, force: true }));
  await writeFile(join(dir, "a.txt"), "one");
  const ctx = {
    subprocess: { spawn: () => ({ done: Promise.resolve({ exitCode: 1 }), collected: {} }) },
  };
  const before = await workspaceState(ctx, dir, undefined);
  assert.equal(before.isGit, false);
  await writeFile(join(dir, "a.txt"), "two");
  const after = await workspaceState(ctx, dir, undefined);
  assert.notEqual(after.stateDigest, before.stateDigest, "content change must change the digest");
  assert.notEqual(after.modDigest, before.modDigest);
});

test("VERDICT_SCHEMA has no top-level passed field (host derives it)", () => {
  assert.equal(VERDICT_SCHEMA.properties.passed, undefined);
  assert.deepEqual(VERDICT_SCHEMA.required.sort(), ["checks", "summary"]);
});
