# 交接包：VEB Idea 审稿–科学家迭代工作流

本文档供下一位接手人快速恢复上下文与继续跑后台任务。仓库根目录：`d:\Shervin\OneDrive\Desktop\breaking`（若路径不同请全文替换）。

---

## 1. 这是什么东西

- **脚本**：[`veb_idea_workflow.py`](veb_idea_workflow.py)  
- **依赖**：从 [`paper_adversarial_workflow.py`](paper_adversarial_workflow.py) 复用 `ApiConfig`、`call_chat`（OpenAI 兼容 HTTP：`POST …/chat/completions`）。
- **角色**：
  - **审稿人**：无记忆；每次请求只有 `system` + `user`（当前版方法论全文），见 `run_reviewer` / `call_chat`。
  - **科学家**：带 **最近 3 轮** 历史（磁盘上已有的 `veb_review_Vi.md` + `veb_idea_V{i+1}.md` 成对拼接），见 `history_for_current_version`。
- **与** `paper_adversarial_workflow.py` **的区别**：后者是整篇论文大纲 adversarial；本脚本是 **单文件方法论 idea** 的 V1/V2/… 版本链。

---

## 2. 网关与模型（默认）

| 项 | 值 |
|----|-----|
| Base URL | `https://new.lemonapi.site/v1` |
| 模型 | `[L]gemini-3.1-pro-preview` |
| 密钥 | **硬编码**在 `veb_idea_workflow.py` 的 `DEFAULT_LEMON_API_KEY`（约第 16 行）。也可用 `--api-key` 或环境变量 `LEMON_API_KEY` 覆盖。 |

**安全**：密钥在源码里＝**切勿把该仓库推送到公开远程**；若曾泄露，应在 Lemon 控制台轮换。

---

## 3. 工作区文件约定（每个 workspace 目录内）

| 模式 | 含义 |
|------|------|
| `veb_idea_V{n}.md` | 第 n 版方法论全文（Windows 上 `V` 大小写不敏感，但建议统一 `veb_idea_V29.md`） |
| `veb_review_V{n}.md` | 对第 n 版 idea 的审稿输出 |

**续跑规则**：若某版 `veb_review_Vn.md` 已存在则跳过审稿；若 `veb_idea_V{n+1}.md` 已存在则跳过科学家。

---

## 4. 命令行参数（必会）

```text
python -u veb_idea_workflow.py --workspace "<绝对或相对路径>" --start-version <N> --max-version <M> [--timeout-seconds 900] [--api-url …] [--model …]
```

- **`--max-version` 必须 ≥ 2**。
- **循环**：对每个 `v` 从 `start_version` 到 `max_version`：审 `Vv` →（若 `v < max`）科学家生成 `V(v+1)`；当 `v == max` 时只审 `Vmax`，不再生成下一版。

**「迭代 K 轮」**（每轮 = 审稿 + 科学家，最后一轮结束后多一次只审 `Vmax`）：

- 起点版本 `S`，要 **K 次科学家改稿**（产出 `V(S+1)…V(S+K)`），则：

```text
--start-version S --max-version (S + K)
```

**例**：从 **V29** 再跑 **15** 次改稿 → 产出到 **V44**：

```text
--start-version 29 --max-version 44
```

**首次从种子建 V1**：`--start-version 1` 且工作区内尚无 `veb_idea_V1.md` 时，会把 `--source`（默认 `veb_idea_seed.md`）复制为 `veb_idea_V1.md`。

---

## 5. 主工作目录现状（接手时请自己再列一次目录确认）

- **`veb_run_3rounds\`**：主实验线；已有 **`veb_idea_V1` … `veb_idea_V29`**（其中 V29 在磁盘上可能显示为 `veb_idea_v29.md`，一般仍可被找到）。
- **`veb_review_V29.md`**：若不存在，下次从 29 跑起会先 **审 V29**。
- **`veb_run_uni15\`**：另一条线（从方法章节复制出的 V1 起跑 15 轮）；曾与 `veb_run_3rounds` 并行，**不要混用 workspace**。
- **日志**（历次尝试，文件名供排查）：
  - `veb_run_3rounds\run_V29_to_V44.log` / `.err.log`
  - `veb_run_3rounds\run_V29_iter15.log` / `.err.log`
  - `veb_run_uni15\run_15rounds.log` / `.err.log`
- **`resume_V18_to_V45.log`** 等：历史续跑痕迹。

---

## 6. 已知问题（403 / 额度）

- 错误形态：`API HTTP error 403`，JSON 里 `pre_consume_token_quota_failed`，文案里常见 **need quota ≈ ¥10.5**、**remain ≈ ¥0.5**。
- **同一密钥**下，`ping` 级小请求可能 **200**，但 **审稿/科学家** 单次 prompt 很长，**预扣费更高**，仍可能 403。
- **Cursor 内置终端/代理环境** 与用户本机 **`C:\Users\shers` 下 PowerShell** 行为可能不一致；用户曾在本机 PowerShell 测通 `/models` 与最小 `chat/completions`，而代理里启动同一脚本仍出现过 403。
- **建议**：长跑任务在用户 **本机 PowerShell** 执行；`-u` 便于日志实时刷盘。

---

## 7. 推荐执行指令（从 V29 再跑 15 轮改稿）

在 **PowerShell** 中：

```powershell
cd "d:\Shervin\OneDrive\Desktop\breaking"
python -u veb_idea_workflow.py --workspace "d:\Shervin\OneDrive\Desktop\breaking\veb_run_3rounds" --start-version 29 --max-version 44 --timeout-seconds 900
```

后台跑（输出进日志）：

```powershell
$wd = "d:\Shervin\OneDrive\Desktop\breaking"
$ws = "$wd\veb_run_3rounds"
$log = "$ws\run_V29_iter15.log"
$err = "$ws\run_V29_iter15.err.log"
$arg = "/c cd /d `"$wd`" && set PYTHONUNBUFFERED=1 && python -u veb_idea_workflow.py --workspace `"$ws`" --start-version 29 --max-version 44 --timeout-seconds 900 1>`"$log`" 2>`"$err`""
Start-Process cmd.exe -ArgumentList $arg -WindowStyle Hidden
```

---

## 8. 如何确认「在跑 / 跑完 / 报错」

```powershell
tasklist /FI "IMAGENAME eq python.exe"
Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match 'veb_idea_workflow' } | ForEach-Object { $_.ProcessId; $_.CommandLine }
Get-Content "d:\Shervin\OneDrive\Desktop\breaking\veb_run_3rounds\run_V29_iter15.log" -Tail 20
Get-Content "d:\Shervin\OneDrive\Desktop\breaking\veb_run_3rounds\run_V29_iter15.err.log"
```

成功结束时日志末尾应有 **`Done.`** 与 **`Last idea: …\veb_idea_V44.md`**（对 29→44 这一配置而言）。

---

## 9. 辅助脚本

| 文件 | 用途 |
|------|------|
| [`scripts/verify_lemon_gateway.py`](scripts/verify_lemon_gateway.py) | 测 `/models` + 最小 chat；可选 `--e2e-mini` 临时目录冒烟 |
| [`scripts/watch_veb_workflow.ps1`](scripts/watch_veb_workflow.ps1) | 周期性把「是否有匹配的 python + 主日志最后一行」追加到 `veb_workflow_watch.log` |

---

## 10. 与本项目其它工作流的关系

- **`paper_adversarial_workflow.py`**：`论文大纲_V*.md` / `Review_Strict_V*.md` 长论文 adversarial；启动说明见各 `审稿工作流启动命令_V*.md`。
- **`ssh_paper_workflow.py`**：同步到远程 SSH 跑上述论文工作流；内含硬编码 SSH 信息，**勿外传**。
- **计划任务**：曾发现根任务 **`Night4GPUWatcherHourly`**（每小时跑 `automation\run_night_4gpu_watcher.ps1`），**已禁用**；若需恢复：`schtasks /Change /TN "Night4GPUWatcherHourly" /ENABLE`。

---

## 11. 接手人最小检查清单

1. 确认 `veb_run_3rounds\veb_idea_V29.md`（或 `veb_idea_v29.md`）存在且内容正确。  
2. 本机 PowerShell 用 `scripts/verify_lemon_gateway.py` 或文档第 6 节同类请求测通后再长跑。  
3. 执行第 7 节命令；盯 `.log` / `.err.log` 与 `veb_idea_V30.md` 是否出现。  
4. 若仅代理环境 403、本机正常，则固定在本机跑。  
5. 轮换密钥后同时改 `DEFAULT_LEMON_API_KEY` 或改用环境变量。

---

*文档生成目的：交接背景任务与 VEB idea 迭代管线；与具体对话记录无关的部分以仓库内代码为准。*
