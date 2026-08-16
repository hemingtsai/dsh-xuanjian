# 让任务在完成前先通过全量审计 —— `dsh-audit-gate`

一个 DeepSeek Harness（DSH）插件：**当模型声称「任务完成」时，先对代码做全量审计与验证；验证不通过就把失败信息反馈给模型让它修复，修复后再审计，直到通过才真正结束这个 task。**

- 全量审计 = 可配置的 shell 命令套件（`lint` / `typecheck` / `test` / `build` / 任意脚本）。
- 反馈修复 = 失败输出自动回到模型当前 turn，模型继续改，改完 gate 自动重跑。
- 结束门槛 = 只有必选检查全部通过（或触达安全上限）任务才真正结束。

---

## 1. 工作原理

DSH 里「任务完成」有两个落地信号，本插件对应两个互补的闸门（gate）加一个主动验证工具（tool），
共用同一个审计执行器（带按 agent 的缓存，代码没变时不会重复全量跑）：

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
                                 │  runAuditCached(commands)
                                 ▼
                    ctx.shell.run("npm run lint" | "npx tsc --noEmit" | "npm test" ...)
```

### (A) 目标类任务的「完成闸门」（精确、保语义）

DSH 的目标系统里，模型用 `update_goal`（action `complete`）宣布完成。本插件监听
`tools/pre-execute`（waterfall，可异步），当工具是 `update_goal` 且 action 为 `complete` 时：

1. 先跑完整审计（`await`，阻塞在工具执行前）。
2. 全部必选检查通过 → `next()`，放行，goal 正常进入 `complete`。
3. 有失败 → 返回 `{ kind: "deny", reason: <失败摘要> }`，工具调用被拒绝，goal **不会被标记 complete**。

模型看到 `update_goal` 返回错误（含失败明细），继续修复，再重试 `complete`。这样 goal 的
「complete 才代表真的做完」语义被严格保留，不需要事后“反悔”已完成的 goal。

### (B) 非目标类任务的「turn 结束闸门」（通用兜底）

没有 goal 的单轮/多轮编码任务，模型直接停手结束 turn。本插件监听
`agent/turn-stopping`（serial，`await` 后才提交 turn 边界）：

1. 模型停手、turn 即将关闭时触发。
2. 按 `guardTurnEnd` 触发条件决定是否审计（默认 `modified`：仅当本 turn 改过文件）。
3. 审计通过 → 什么都不做，turn 正常关闭（任务结束）。
4. 审计失败 → 用 `agent.steer(...)` 把失败摘要注入当前 turn 的下一个 step，
   **机器会重读 inbox、再跑一个 step**，模型据此修复；修完停手又触发 `agent/turn-stopping`，自动重跑审计。
5. 循环直到通过；用 `maxAttempts` 和「无进展指纹」双重兜底，防止死循环。

> 有 active/completed goal 时，(B) 自动让位给 (A)，避免重复审计。

### (C) 主动验证工具 `run_audit`（可选，默认开启）

插件还会注册一个 `run_audit` 工具（`registerTool: true`，名字可改 `toolName`）：模型在
标记完成**之前**主动调用它跑同一套审计，拿到结构化报告（`passed` / `summary` / `results[]`）。
一方面减少“试 `complete` 被拒”的往返，另一方面结果会写入缓存——随后同一次 turn 内模型
再 `complete`，闸门 (A) 直接复用缓存，不再重复全量跑。缓存以“代码是否被改动”为失效条件：
模型一旦 `write`/`edit`/`bash`/`pwsh` 改了东西，缓存自动作废、下次重跑。

---

## 2. 安装与组合

### 作为 npm 包

```bash
# 在部署 DSH 的 node_modules 里安装（或 npm link 本地包）
npm install <this-package>
```

在预设的 `agent.cordis.yml`（或 host 组合）里加一行。它**不发布任何 service**，因此可以像
`tool-bash` 一样松散地放在 preset 里：

```yaml
- id: audit-gate
  name: dsh-audit-gate
  config:
    guardTurnEnd: modified      # off | modified | always
    scope: root                 # root | all
    maxAttempts: 5
    commands:
      - name: lint
        run: npm run lint
        timeoutMs: 120000
      - name: typecheck
        run: npx tsc --noEmit
        timeoutMs: 120000
      - name: test
        run: npm test -- --run
        timeoutMs: 300000
```

完整示例见 [`example.cordis.yml`](./example.cordis.yml)。

> 推荐放在**预设（agent 平面）**：`agent/turn-stopping`、`tools/pre-execute` 都是按 scope 向上路由的
> 事件，站在 preset 的 standing scope 上能收到其下所有 session 的事件（`@deepseek-ai/dsh-scope`
> 的 `scopeTarget`：「listener owned by an enclosing scope receives every descendant scope's events」）。
> 放在 host 平面（unscoped）同样能收到全部 agent，二选一即可。

---

## 3. 配置参考

| 键 | 类型 | 默认 | 说明 |
| --- | --- | --- | --- |
| `enabled` | bool | `true` | 总开关（**运行时热切换**） |
| `guardCompletion` | bool | `true` | 闸门 (A)：拒绝未通过审计的 `update_goal complete`（**运行时热切换**） |
| `guardTurnEnd` | `off\|modified\|always` | `modified` | 闸门 (B)：turn 结束审计的触发条件（**运行时热切换**） |
| `scope` | `root\|all` | `root` | 只对顶层 agent 生效，还是含 subagent |
| `maxAttempts` | int | `5` | (B) 每个 turn 最多注入几次修复提示（**运行时热切换**） |
| `workdir` | string | agent cwd | 审计工作目录覆盖 |
| `mutatingTools` | string[] | `write,edit,bash,pwsh` | `modified` 触发判定 + 缓存失效判定用到的“会改文件”工具名 |
| `registerTool` | bool | `true` | 是否注册 (C) 主动验证工具 |
| `toolName` | string | `run_audit` | 主动验证工具的模型可见名称 |
| `commands[]` | object[] | `[]` | 审计套件 |

### 运行时开关（settings）

`enabled` / `guardCompletion` / `guardTurnEnd` / `maxAttempts` 注册为 **`audit-gate` 设置命名空间**
（`@deepseek-ai/dsh-settings`），可在**不编辑组合、不重启**的情况下热切换，层级为
`schema 默认 ← 组合 entry ← 用户 settings 层`。改 `~/.dsh/settings.yaml` 里的 `audit-gate:` 段即生效：

```yaml
audit-gate:
  enabled: false          # 一键关闭全部自动闸门（run_audit 手动工具仍可用）
  guardTurnEnd: off       # 或只关 turn 结束闸门
  maxAttempts: 3
```

其余字段（`commands`/`scope`/`workdir`/`mutatingTools`/`registerTool`/`toolName`）仍属组合配置，改动需重载。

### 斜杠命令（查看 / 切换开关）

插件注册两个人类命令，直接改上面的 settings 命名空间、即时生效：

| 命令 | 作用 |
| --- | --- |
| `/audit` | 查看当前开关状态（`enabled` / `guardCompletion` / `guardTurnEnd` / `maxAttempts` / 套件数量） |
| `/audit-toggle [on\|off]` | 开/关总开关；不带参数则翻转 |

```
/audit             # → Audit gate: enabled: true, guardTurnEnd: modified, …
/audit-toggle off  # → 关闭自动闸门（run_audit 手动工具仍可用）
/audit-toggle      # → 再翻转一次 = 重新打开
```

> Web 端可视化开关：DSH 的设置 UI 是 slot 驱动、由每个插件自带 client 半区渲染（`settings.general.item`
> 或 `settings.section`）。当前本包提供 `/audit`、`/audit-toggle` 命令 + host 侧 settings 命名空间（热切换已就绪），
> 若还要在设置页出现可点击的开关/下拉框，需再补一个 client 半区把该命名空间渲染成 schema 表单。

每个 `commands[]` 项：

| 键 | 类型 | 默认 | 说明 |
| --- | --- | --- | --- |
| `name` | string | — | 失败摘要里显示的名称 |
| `run` | string | — | shell 命令，以 `bash -c <run>` 执行 |
| `timeoutMs` | int | executor 默认 | 单命令超时 |
| `required` | bool | `true` | 失败是否阻断完成 |
| `maxOutputChars` | int | `4000` | 反馈给模型的失败输出字符数（取尾部） |

---

## 4. 行为细节

- **审计执行**：通过 `ctx.shell.resolve(...)` + `ctx.shell.run(...)`（`@deepseek-ai/dsh-shell` 执行器 seam），
  走 DSH 的受管进程组、输出截断/溢出、超时/取消语义。`workdir` 取 `agent.session.header.cwd`。
- **取消**：审计全程观察 `AbortSignal`（闸门 (A) 用 `exec.signal`，闸门 (B) 用 `turn-stopping` 的 `signal`），
  turn 被中止时审计命令会被 kill。
- **无进展检测**：闸门 (B) 对每次失败做「失败项+输出」指纹，若与上一轮一致（模型没实际改好），立即停止注入，避免空转。
- **结果缓存**：闸门 (A/B) 与 `run_audit` 工具共享同一个 per-agent 缓存；模型代码（`write`/`edit`/`bash`/`pwsh`）一有改动
  （`dirtyVersion` 递增）缓存即失效，`turn/start` 也会清缓存。因此“先 `run_audit` 通过、再 `complete`”不会重复全量跑。
- **反馈来源**：注入的消息 `source.kind = "plugin"`（非 `"user"`），因此不会被 goal 系统的
  `hasDirectHumanInput` 误判为「人类直接输入」，不会意外获得本不该有的权限。
- **生命周期**：监听器和 per-agent 状态随 fiber/agent 自动清理，无进程级残留副作用。

---

## 5. 局限与后续方向

1. **goal 的“完成前拦截”是借 `tools/pre-execute` 实现的**（今天无需改内核即可工作，且语义正确）。
   若希望更通用的“任何路径写 complete 都被拦”的钩子，可向 `@deepseek-ai/dsh-goal` 提议增加一个
   `goal/pre-complete` waterfall 事件，插件改为监听该事件即可。
2. **写产物类检查**（如 `build`）受部署 sandbox/文件策略约束；若部署为只读，写产物命令会被策略拒绝，
   可改用只读检查（`--noEmit`、`lint`、`test`）。
3. **结构化到文件/行号级**：当前 `run_audit` 报告与 gate 反馈是「检查项 + 退出码 + 输出摘要」；
   可进一步解析 lint/test 输出为 `文件 / 行号 / 修复建议` 的结构化 JSON，供模型更精准修复。

---

## 6. 目录

```
dsh-audit-gate/
├── package.json          # 包清单
├── lib/index.js          # 插件本体（audit runner + 两个 gate）
├── example.cordis.yml    # 示例组合行
└── README.md
```
