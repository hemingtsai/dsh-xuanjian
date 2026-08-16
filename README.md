# 让任务在完成前先通过全量审计 —— `dsh-audit-gate`

一个 DeepSeek Harness（DSH）插件：**当模型声称「任务完成」时，先用廉价模型 subagent 对代码做全量审计；审计不通过就把失败信息反馈给模型让它修复，修复后再审计，直到通过才真正结束这个 task。**

**可信性承诺：审计只有三种结果——`passed` / `failed` / `inconclusive`，其中「审计没跑成」永远不会被报告成通过。默认（`onInconclusive: deny`）下，审计不明确通过，任务就不能完成。** 整体裁决由宿主从每条检查证据推导，模型无权自报顶层 `passed`。

- **审计 = 自动生成**：按项目语言，由审计 subagent 自己生成并运行合适的检查（lint / typecheck / test / build / 等价命令）。
- **找茬式逻辑审计**：审计 subagent 会真正**读源码**，按 git 变更优先揪真实 bug（越界/空指针/竞态/未处理异步/资源泄漏/错误处理缺口/安全问题），每条带 `file:line` 证据；宁缺毋滥。
- **功能审计**：审计 subagent 会拿任务的**真实需求**（最近的用户请求 / goal objective）来验证「要实现的功能是否真的按需求跑通了」——能跑就跑起来实际用一遍，不能跑就核对链路完整性，抓「没实现/半实现/行为与需求不符/一用就崩」。
- **廉价模型**：审计 subagent 默认用 `deepseek-v4-flash`（远便宜于任务的 `deepseek-v4-pro`），一次 LLM 往返完成「生成检查 + 跑检查 + 审风格 + 找逻辑茬 + 功能验证 + 查文档覆盖」。
- **反馈修复** = 失败输出自动回到模型当前 turn，模型继续改，改完 gate 自动重跑。
- **结束门槛** = 只有审计 `passed` 任务才真正结束；`failed` 必拒；`inconclusive` 默认也拒（`onInconclusive` 可放宽为 `warn`/`allow`）。

---

## 1. 工作原理

DSH 里「任务完成」有两个落地信号，本插件对应两个互补的闸门（gate）加一个主动验证工具（tool），
共用同一个审计执行器（按 agent 缓存，代码没变时不会重复开 subagent）：

```
                 ┌────────────────────────────────────────────┐
                 │              audit-gate (this plugin)       │
                 │                                            │
  task signals ──┤  (A) tools/pre-execute   ──> update_goal    │
                 │      complete 被 DENY，直到审计通过          │
                 │                                            │
                 │  (B) agent/turn-stopping ──> 无 goal 任务    │
                 │      注入修复提示，turn 不结束               │
                 │                                            │
  model ────────>│  (C) run_audit tool ──> 主动自查 + 结构化报告 │
                 └───────────────┬────────────────────────────┘
                                 │  runAuditCached(agent, signal)
                                 ▼
                  ctx.subagents.start("spawn", {
                    agentOptions: { model: auditModel },   // 默认 deepseek-v4-flash
                    outputSchema: verdict,                  // 结构化裁决
                  })   ──>  审计 subagent：生成检查并执行
                             + 审代码风格 + 找逻辑茬 + 功能验证 + 查文档覆盖
```

### (A) 目标类任务的「完成闸门」（精确、保语义）

DSH 的目标系统里，模型用 `update_goal`（action `complete`）宣布完成。本插件监听
`tools/pre-execute`（waterfall，可异步），当工具是 `update_goal` 且 action 为 `complete` 时：

1. 先跑完整审计（`await`，阻塞在工具执行前）。
2. 审计 `passed` → `next()`，放行，goal 正常进入 `complete`。
3. 审计 `failed` → 返回 `{ kind: "deny", reason: <失败摘要> }`，工具调用被拒绝，goal **不会被标记 complete**。
4. 审计 `inconclusive` → 按 `onInconclusive`：默认 `deny`（拒绝并说明「审计未完成，不允许在未验证的情况下完成」）；`warn`/`allow` 则放行但报告/日志保留告警。

模型看到 `update_goal` 返回错误（含失败明细），继续修复，再重试 `complete`。

### (B) 非目标类任务的「turn 结束闸门」（通用兜底）

没有 goal 的单轮/多轮编码任务，模型直接停手结束 turn。本插件监听
`agent/turn-stopping`（serial，`await` 后才提交 turn 边界）：

1. 模型停手、turn 即将关闭时触发。
2. 按 `guardTurnEnd` 触发条件决定是否审计（默认 `modified`：仅当本 turn 改过文件）。
3. 审计通过 → 什么都不做，turn 正常关闭。
4. 审计失败 → 用 `agent.steer(...)` 把失败摘要注入当前 turn 的下一个 step，机器重读 inbox、再跑一个 step，
   模型据此修复；修完停手又触发，自动重跑。
5. 循环直到通过；用 `maxAttempts` 和「无进展指纹」双重兜底，防止死循环。

> 有 active/completed goal 时，(B) 自动让位给 (A)，避免重复审计。

### (C) 主动验证工具 `run_audit`（可选，默认开启）

插件注册一个 `run_audit` 工具：模型在标记完成**之前**主动调用它跑同一套审计，拿到结构化报告
（`passed` / `summary` / `results[]`），减少「试 `complete` 被拒」的往返。结果写入缓存，
同一次 turn 内模型再 `complete`，闸门 (A) 直接复用，不再重复开 subagent。
缓存以「代码是否被改动」为失效条件：模型一旦 `write`/`edit`/`bash`/`pwsh` 改了东西，缓存自动作废、下次重跑。

### 递归安全

审计本身就是 subagent，而 subagent 会继承父级 preset（可能再次装上本插件）。因此审计**只对顶层 agent
（`delegationDepth` 0）执行**：任何 gate 或 `run_audit` 工具若由 subagent 触发，都会得到一个
「仅适用于顶层 agent」的非阻断裁决。这样审计子 agent 永远不可能审计它自己。

---

## 2. 安装与组合

### 作为 npm 包

```bash
# 在部署 DSH 的 node_modules 里安装（或 npm link 本地包）
npm install <this-package>
```

> **peer 依赖约定**：`@deepseek-ai/dsh-llm` / `dsh-tools` / `dsh-settings` / `cordis` 声明为
> **peerDependencies**，由宿主 harness 提供（与 `tool-bash`、`tool-goal` 等官方插件一致）。
> 切勿把它们列进 `dependencies`：那会让 pnpm 往 profile 的 `node_modules` 里塞进**第二份**
> `dsh-tools`，它会在 loader 以 profile 为 baseUrl 解析时遮蔽扁平回退副本，造成两份模块实例、
> `TOOL_RUNTIME_SCHEDULER` symbol 不一致，最终所有工具调度都以
> `Cannot read properties of undefined (reading 'prepare')` 崩溃。只把第三方库（如 `schemastery`）放 dependencies。

在预设的 `agent.cordis.yml`（或 host 组合）里加一行。它**不发布任何 service**，因此可以像
`tool-bash` 一样松散地放在 preset 里：

```yaml
- id: audit-gate
  name: dsh-audit-gate
  config:
    guardTurnEnd: modified      # off | modified | always
    maxAttempts: 5
    onInconclusive: deny        # deny | warn | allow —— 审计未完成时的默认策略
    auditProvider: deepseek-official
    auditModel: deepseek-v4-flash
    subagentProvider: spawn
    checkStyle: true
    checkLogic: true
    checkFunction: true
    checkDocs: true
    # styleGuide: .editorconfig   # 可选：绝对路径或相对工作区的风格指南文件
```

> 推荐放在**预设（agent 平面）**：`agent/turn-stopping`、`tools/pre-execute` 都是按 scope 向上路由的
> 事件，站在 preset 的 standing scope 上能收到其下所有 session 的事件。放在 host 平面（unscoped）同样能收到全部 agent，二选一即可。

---

## 3. 配置参考

| 键 | 类型 | 默认 | 说明 |
| --- | --- | --- | --- |
| `enabled` | bool | `true` | 总开关（**运行时热切换**） |
| `guardCompletion` | bool | `true` | 闸门 (A)：拒绝未通过审计的 `update_goal complete`（**运行时热切换**） |
| `guardTurnEnd` | `off\|modified\|always` | `modified` | 闸门 (B)：turn 结束审计的触发条件（**运行时热切换**） |
| `maxAttempts` | int | `5` | (B) 每个 turn 最多注入几次修复提示（**运行时热切换**；1/2/5 行为有精确测试） |
| `onInconclusive` | `deny\|warn\|allow` | `deny` | 审计无法完成（subagent 报错/拒绝/超 token/结构化缺失/空检查等）时的策略：`deny` 阻断完成（默认，推荐）、`warn` 放行但保留告警、`allow` 放行 |
| `workdir` | string | agent cwd | 审计工作目录覆盖 |
| `mutatingTools` | string[] | `write,edit,bash,pwsh` | `modified` 触发判定 + 缓存失效判定用到的“会改文件”工具名 |
| `registerTool` | bool | `true` | 是否注册 (C) 主动验证工具 |
| `toolName` | string | `run_audit` | 主动验证工具的模型可见名称 |
| `auditProvider` | string | `deepseek-official` | 审计 subagent 的 provider |
| `auditModel` | string | `deepseek-v4-flash` | 审计 subagent 的廉价模型 |
| `subagentProvider` | string | `spawn` | 审计 subagent 的 provider 名 |
| `checkStyle` | bool | `true` | 审计是否审代码风格 |
| `checkLogic` | bool | `true` | 审计是否做找茬式逻辑审计（读源码揪真实 bug） |
| `checkFunction` | bool | `true` | 审计是否做功能审计（拿任务需求验证功能真能跑通） |
| `checkDocs` | bool | `true` | 审计是否查文档覆盖 |
| `styleGuide` | string | — | 可选风格指南文件（绝对或相对工作区）；存在则注入审计 prompt |

### 运行时开关（settings）

`enabled` / `guardCompletion` / `guardTurnEnd` / `maxAttempts` 注册为 **`audit-gate` 设置命名空间**
（`@deepseek-ai/dsh-settings`），可在不编辑组合、不重启的情况下热切换。改 `~/.dsh/settings.yaml`
里的 `audit-gate:` 段即生效：

```yaml
audit-gate:
  enabled: false          # 一键关闭全部自动闸门（run_audit 手动工具仍可用）
  guardTurnEnd: off       # 或只关 turn 结束闸门
  maxAttempts: 3
```

其余字段（审计模型、provider、风格/文档开关、styleGuide 等）仍属组合配置，改动需重载。

### 斜杠命令（查看 / 切换开关）

| 命令 | 作用 |
| --- | --- |
| `/audit` | 查看当前开关状态与审计配置 |
| `/audit-toggle [on\|off]` | 开/关总开关；不带参数则翻转 |

---

## 4. 行为细节

- **审计执行**：通过 `ctx.subagents.start(subagentProvider, ...)` 开一个廉价模型 subagent，
  用 `outputSchema` 索取结构化裁决（`structured_output` 工具）。subagent 继承父级工作区作为 cwd，
  自行生成并运行语言合适的检查命令，并审风格、找逻辑茬、做功能验证、查文档覆盖。
- **找茬式逻辑审计**：审计 subagent 会真正阅读源码，按「本任务改了什么」优先（git 仓库下用
  `git status --short` / `git diff` / `git diff --cached` 定位变更文件），针对性地找越界、
  空指针/类型假设、竞态、未处理异步拒绝、资源泄漏、算法/契约不符、错误处理缺口、安全问题等真实缺陷，
  每条以 `file:line` 举证；明确「宁缺毋滥」，不编造问题凑数。
- **功能审计**：审计 subagent 收到任务的真实需求上下文（最近的直接用户请求 + goal objective，由插件从
  session 提取，插件注入的系统快照不计入），按需求验证功能是否真的实现并跑通——能跑就跑起来实际用一遍
  （服务/CLI 限生命周期、用完杀掉），不能跑就核对链路是否完整接通；只抓「需求要求却没做到」的硬伤。
- **取消**：审计全程观察 `AbortSignal`（闸门 (A) 用 `exec.signal`，闸门 (B) 用 `turn-stopping` 的 `signal`），
  turn 被中止时 subagent 会被 dispose，且中止结果不会被缓存。
- **三态结果，绝不伪装通过**：审计只产出 `passed` / `failed` / `inconclusive`。`inconclusive` 覆盖
  subagent 启动失败、`run.result` 拒绝、`error`/`max-tokens`/`refusal`/异常 stopReason、结构化输出缺失
  或非法、以及**空检查列表**（无证据不构成通过）。顶层整体裁决由宿主从每条检查推导，模型输出的顶层
  `passed` 字段被移除且不被信任——模型说通过了、但它的每条检查都失败时，结果是 `failed`。
- **瞬时故障自动重试**：审计 subagent 若以 `error`/`max-tokens` 结束（API 传输抖动等基础设施问题），
  插件会间隔 1.2s 重开一次；两次都失败才判定 `inconclusive`（默认阻断，见 `onInconclusive`）。
  `refusal`（模型拒绝）不重试。
- **无进展检测**：闸门 (B) 对每次失败做「失败项+输出」指纹，若与上一轮一致（模型没实际改好），立即停止注入，避免空转。
- **结果缓存**：闸门 (A/B) 与 `run_audit` 工具共享同一个 per-agent 缓存（已完成结果 + in-flight
  single-flight：并发触发只开一个 subagent）；**任何 mutating 工具的结果——无论成功失败——都使缓存失效**
  （一条失败的命令也可能在报错前已改动文件），`turn/start` 也会清缓存。审计是 LLM 往返，缓存让
  「先 `run_audit` 通过、再 `complete`」不重复花钱。
- **goal 判定**：只有 `active` 的 goal 才让位给完成闸门 (A)；`complete`/`paused`/`blocked` 的旧 goal
  不会让后续普通任务绕过 turn 结束闸门 (B)。
- **反馈来源**：注入的消息 `source.kind = "plugin"`（非 `"user"`），不会被 goal 系统误判为「人类直接输入」。
- **生命周期**：监听器和 per-agent 状态随 fiber/agent 自动清理，无进程级残留副作用。

---

## 5. 局限与后续方向

1. **goal 的“完成前拦截”是借 `tools/pre-execute` 实现的**（无需改内核即可工作，且语义正确）。
   若希望更通用的“任何路径写 complete 都被拦”的钩子，可向 `@deepseek-ai/dsh-goal` 提议增加一个
   `goal/pre-complete` waterfall 事件。
2. **审计成本与延迟**：每次全新审计都要一次 subagent 往返（生成检查 + 执行）。廉价模型让成本可控；
   缓存缓解重复触发。若某个检查命令很慢（如全量 build），可后续把“生成检查计划”单独缓存复用。
3. **写产物类检查**（如 `build`）受部署 sandbox/文件策略约束；若部署为只读，写产物命令会被策略拒绝，
   可让审计 subagent 改用只读检查（`--noEmit`、`lint`、`test`）。
4. **结构化到文件/行号级**：审计 subagent 的裁决已经要求 `detail` 带 `file:line`；未来可进一步要求
   输出标准化的修复建议 JSON，供模型更精准修复。
5. **命令执行隔离**：目前审计 subagent 直接持有通用 `bash`，靠 prompt 约束「只读/只写临时区」。更严格的
   做法是宿主掌握受控命令执行器（argv 而非 shell 字符串、临时 worktree、安全环境变量、单命令超时、
   进程组清理、真实证据记录），并让模型只输出检查计划——见「局限与后续方向」第 3 条同层的 roadmap。
6. **审计证明（certificate）与最终 guard**：当前完成拦截借 `tools/pre-execute`，后续可引入
   `ctx.tools.guard()` 同步校验「当前工作区/任务/配置对应的审计通过证明」，做成不可被重排撤销的不变量。
7. **测试与 CI**：仓库内置 `node --test` 测试（`test/unit.test.mjs` 纯逻辑 + `test/gate.test.mjs` 事件
   状态机），覆盖三态裁决、maxAttempts 精确计数、goal phase、缓存失效、single-flight、递归保护等；
   CI 运行 `npm ci && npm run check && npm test && npm run pack:check`。

---

## 6. 目录

```
dsh-audit-gate/
├── package.json          # 包清单（peer deps：cordis / dsh-llm / dsh-tools / dsh-settings）
├── lib/index.js          # 插件本体（语言检测 + 廉价 subagent 审计执行器 + 两个 gate）
├── example.cordis.yml    # 示例组合行
└── README.md
```
