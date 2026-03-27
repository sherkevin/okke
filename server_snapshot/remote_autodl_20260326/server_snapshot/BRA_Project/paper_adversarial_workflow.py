from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from http.client import RemoteDisconnected
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_API_URL = "https://new.lemonapi.site/v1"
DEFAULT_MODEL = "[L]gemini-3.1-pro-preview"
DEFAULT_SOURCE = "论文大纲.md"
STATE_FILE = "paper_workflow_state.json"
PROMPT_DOC = "paper_workflow_prompt_strategy.md"
COMPARISON_DOC = "Pilot_Comparison.md"
SCALE_UP_DOC = "Workflow_Scaleup_Guide.md"
SCIENTIST_MEMORY_FILE = "Scientist_Memory.md"
SCORE_TRACKER_DOC = "评分波动记录.md"
HIGH_SCORE_DOC = "高分版本记录.md"
OBSERVER_LOG_DOC = "旁观者洞察日志.md"
COGNITIVE_INSIGHT_DOC = "认知洞察文档.md"


@dataclass
class ApiConfig:
    api_url: str
    api_keys: list[str]
    model: str
    timeout_seconds: int


def read_utf8(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_utf8(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _extract_between(text: str, start_marker: str, end_marker: str) -> str | None:
    start = text.find(start_marker)
    end = text.find(end_marker)
    if start == -1 and end == -1:
        return None
    if start != -1 and end == -1:
        return text[start + len(start_marker) :].strip()
    if start == -1 or end < start:
        return None
    return text[start + len(start_marker) : end].strip()


def _fallback_extract_block(text: str, start_marker: str, end_marker: str) -> str:
    marker_key = start_marker.replace("<<<", "").replace("_START>>>", "").strip().upper()
    if marker_key == "TITLE":
        match = re.search(r"^# .+$", text, flags=re.MULTILINE)
        return match.group(0).strip() if match else text.strip()
    if marker_key == "ABSTRACT":
        try:
            return extract_first_matching_section(text, ["## Abstract", "## Abstract (摘要)"], ["## 1. Introduction"])
        except Exception:
            return text.strip()
    if marker_key == "INTRODUCTION":
        try:
            return extract_first_matching_section(text, ["## 1. Introduction"], ["## 2. Related Work", "## 3. Methodology"])
        except Exception:
            return text.strip()
    if marker_key == "METHODOLOGY":
        try:
            return extract_first_matching_section(
                text,
                ["## 3. Methodology"],
                ["## 4. Experiments", "## 4. Experimental Design", "## 4. Evaluation Protocol and Experimental Design"],
            )
        except Exception:
            return text.strip()
    if marker_key == "EXPERIMENTS":
        try:
            return extract_first_matching_section(
                text,
                ["## 4. Experiments", "## 4. Experimental Design", "## 4. Evaluation Protocol and Experimental Design"],
                [
                    "## 5. Conclusion & Limitations",
                    "## 5. Discussion",
                    "## 5. Discussion & Limitations",
                    "## 5. Discussion and Limitations",
                    "## 6. Conclusion and Future Execution",
                ],
            )
        except Exception:
            return text.strip()
    if marker_key == "CLOSING":
        try:
            return extract_first_matching_section(
                text,
                [
                    "## 5. Conclusion & Limitations",
                    "## 5. Discussion",
                    "## 5. Discussion & Limitations",
                    "## 5. Discussion and Limitations",
                    "## 6. Conclusion and Future Execution",
                ],
                ["## Works cited", "## Works Cited"],
            )
        except Exception:
            return text.strip()
    if marker_key == "REVISION_LOG":
        match = re.search(r"(?ms)^# Revision_Log_V\d+.*", text)
        return match.group(0).strip() if match else text.strip()
    if marker_key == "SCIENTIST_MEMORY":
        match = re.search(r"(?ms)^# Scientist_Memory.*", text)
        return match.group(0).strip() if match else text.strip()
    if marker_key == "REVIEW":
        match = re.search(r"(?ms)^# Review_Strict_V\d+.*", text)
        return match.group(0).strip() if match else text.strip()
    return text.strip()


def extract_block(text: str, start_marker: str, end_marker: str) -> str:
    extracted = _extract_between(text, start_marker, end_marker)
    if extracted is not None and extracted.strip():
        return extracted.strip()
    fallback = _fallback_extract_block(text, start_marker, end_marker)
    return fallback.strip()


def parse_score(review_text: str) -> float | None:
    match = re.search(r"Score:\s*([0-5](?:\.\d+)?)\s*/\s*5", review_text)
    if match:
        return float(match.group(1))
    return None


def ensure_file_absent(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")


def call_chat(api: ApiConfig, system_prompt: str, user_prompt: str) -> str:
    payload = {
        "model": api.model,
        "stream": False,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    last_error: Exception | None = None

    for api_key in api.api_keys:
        for attempt in range(1, 4):
            request = Request(
                f"{api.api_url.rstrip('/')}/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "Connection": "close",
                },
                method="POST",
            )
            try:
                with urlopen(request, timeout=api.timeout_seconds) as response:
                    body = json.loads(response.read().decode("utf-8"))
                try:
                    return body["choices"][0]["message"]["content"]
                except (KeyError, IndexError, TypeError) as exc:
                    raise RuntimeError(
                        f"Unexpected API response: {json.dumps(body, ensure_ascii=False)}"
                    ) from exc
            except HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                last_error = RuntimeError(f"API HTTP error {exc.code}: {detail}")
            except (URLError, RemoteDisconnected, ConnectionResetError) as exc:
                last_error = RuntimeError(f"API connection failed: {exc}")

            if attempt < 3:
                time.sleep(5 * attempt)

    raise RuntimeError(str(last_error) if last_error else "Unknown API failure")


def paper_path(workspace: Path, version: int) -> Path:
    return workspace / f"论文大纲_V{version}.md"


def review_path(workspace: Path, version: int) -> Path:
    return workspace / f"Review_Strict_V{version}.md"


def revision_log_path(workspace: Path, version: int) -> Path:
    return workspace / f"Revision_Log_V{version}.md"


def reviewer_system_prompt() -> str:
    return (
        "You are the lead area chair-level reviewer for ACM Multimedia. "
        "Be rigorous, skeptical, fair, and non-appeasing. "
        "Judge the paper like a real high-standard conference reviewer."
    )


def scientist_system_prompt() -> str:
    return (
        "You are a world-class AI scientist and experienced ACM MM author/reviewer. "
        "You must improve the paper from first principles without flattery. "
        "Preserve valid core ideas, revise weak claims, and make targeted improvements instead of rewriting from scratch."
    )


def build_reviewer_prompt(version: int, paper_text: str) -> str:
    return f"""
你是 ACM MM 会议的首席主审稿人。下面是论文完整文本初稿，请十分公正、严厉、尖锐地评价。

关键要求：
1. 不要讨好作者。
2. 重点关注方法论是否成立、核心创新是否明确、实验设计是否足以支撑 claims、消融和对比是否完整。
3. 需要给出清晰的打分，采用 1-5 分制，5 分最高。
4. 要指出哪些内容是可以保留的，哪些内容必须改。
5. 要给出图表、实验、消融、评测协议、失败案例分析等具体补强建议，帮助作者形成后续实验大纲。
6. 这是一个全新的评审窗口，不要假设你见过任何旧版本。
7. 不要把注意力放在排版、美观、格式或参考文献样式上，除非它们直接妨碍你理解方法或实验。
8. 作者当前实验尚未完成，因此请像强审稿人一样重点审查“实验计划是否合理”，并提出可执行的实验改进路线，而不是只因为结果未完成就给出表面化意见。
9. 这篇论文当前的合法核心不是对外部 baseline 的宏大批判，而是一个正面方法命题：如何在 decode-time 将 token-local visual evidence 注入 logits 调整，同时尽量不伤害语言结构、功能词、多 token 实体和一般推理能力。
10. 不要因为作者使用了旧术语就默认其成立；如果文中重新滑回“DoLa / VCD / OPERA 都依赖 global pooling”之类的错误 framing，请明确指出这是严重问题。
11. `DoLa`、`VCD`、`OPERA` 只应被当作竞争基线来审视，不是本文主问题定义的证据。
12. 请围绕以下三条证据链评审其闭环性：
   - Hallucination Reduction：POPE / CHAIR + AGL
   - Structure and Reasoning Preservation：MMBench / MME / MMMU(Hard)
   - Local Evidence Value：FREAK / DocVQA，以及 `TLRA_zero` vs `TLRA_MeanPool`
13. 当前 benchmark 口径是：image-centric 主合同 + bounded secondary video pilot。视频不是共主线；如果视频部分没有形成真正的时序局部证据论证，你应建议其继续降级为 secondary pilot / appendix，而不是为了 ACM MM 叙事机械保留。
14. 如果作者继续把 OCR-heavy documents 写成核心动机，但正文又承认方法在 OCR token 上常常默认不触发或旁路，请明确指出这是逻辑自相矛盾，必须收缩动机或重写数据集定位。
15. 如果作者的最佳主张只是“token-local logits intervention + VASM + fair zero-shot/calibrated split”，请鼓励其收缩 claim，而不是继续发明新的大题眼。
16. 你应默认接受当前实验现实：`TLRA_zero` 更像 viability probe，`VASM` 是当前最强正证据资产，`AdaptiveTopK` 尚未被证明显著优于 `MeanPool`，`DocVQA` 更适合作为 OCR concession 下的负控制。

请严格按下面的结构输出，并保留开始/结束标记：

<<<REVIEW_START>>>
# Review_Strict_V{version}
## Overall Score
Score: <1-5>/5

## Verdict

## Summary

## WhatShouldBeKept

## MajorWeaknesses

## SectionBySectionComments

## RequiredRevisions

## SuggestedFiguresTablesExperiments

## AcceptanceOutlook
<<<REVIEW_END>>>

下面是需要评审的论文全文：

--- PAPER V{version} START ---
{paper_text}
--- PAPER V{version} END ---
""".strip()


def build_scientist_prompt(
    current_version: int,
    next_version: int,
    paper_text: str,
    review_text: str,
    scientist_memory: str,
) -> str:
    memory_block = scientist_memory.strip() if scientist_memory.strip() else "当前还没有历史科学家记忆。"
    return f"""
你是全球顶级 AI 科学家，是 ACM MM 的资深投稿者和多年审稿人。你需要根据审稿意见对原稿进行针对性精修，目标是提升论文可接受性。

强约束：
1. 以第一性原理和会议审稿标准修改，不要讨好作者。
2. 在原稿基础上精细化微调，不要另起炉灶，不要把已经写得好的创新点和结构全部推翻。
3. 对明显缺乏证据的绝对化表达要降温、收敛、重写。
4. 对实验未跑通的内容，只能明确写成“计划实验”“待验证假设”“未来工作/实验协议”，不能伪造结果。
5. 允许补入更合理的局限性、实验计划、图表设计、消融建议、失败案例分析。
6. 这是科学家的持续工作线程，你可以参考历史科学家记忆来避免重复犯错。
7. 当前必须保留的核心只有：
   - decode-time / logits-space intervention
   - token-local / region-local visual evidence
   - VASM 这类 structure-preserving protection
   - `TLRA_zero` / `TLRA_calib` 的公平边界
8. 不要再把 `Pooling Paradox` 当成标题、摘要或引言的主题眼；如果要提到 coarse aggregation，也只能作为内部机制对照（如 `BRA_MeanPool`），不能再当作领域总批判。
9. `DoLa`、`VCD`、`OPERA` 只能作为竞争基线，不得被错误描述成依赖全局视觉池化的方法。
10. 当前 benchmark 口径不是 image+video 双主线，而是 image-centric 主合同 + bounded secondary video pilot。若视频证据不足，你必须把视频维持为次主线，而不是重新升级为强主线。
11. 不要让 OCR-heavy documents、offline-calibrated plug-in adapter、PEFT 叙事、WordNet 工程细节、EMA 细节喧宾夺主；如果这些内容开始挤掉核心方法命题，你必须主动收缩它们。
12. 如果方法对 OCR token 的有效干预尚未自洽，不要再把 DocVQA / OCR-heavy documents 写成核心动机；可以把它们降级为边界测试或条件性扩展。
13. 你应默认接受当前实验现实：`TLRA_zero` 更像 viability probe，`VASM` 是当前最强主线资产，`AdaptiveTopK` 尚未在 `FREAK` 上证明优于 `MeanPool`，`DocVQA` 主要服务于 OCR concession 的负控制论证。

请严格输出以下 3 个带标记的部分：

<<<REVISED_PAPER_START>>>
[这里放论文 V{next_version} 的完整 Markdown 正文，只放修订后的论文，不要加入解释性前言]
<<<REVISED_PAPER_END>>>

<<<REVISION_LOG_START>>>
# Revision_Log_V{next_version}
- 逐条对应 review 中的关键问题，说明你做了什么修改
- 指出你刻意保留了哪些原有优点
- 指出哪些问题仍然需要未来实验支撑
<<<REVISION_LOG_END>>>

<<<SCIENTIST_MEMORY_START>>>
# Scientist_Memory
- 总结这轮最值得保留的修改策略
- 总结接下来迭代最容易继续掉坑的点
<<<SCIENTIST_MEMORY_END>>>

下面是历史科学家记忆：
--- SCIENTIST MEMORY START ---
{memory_block}
--- SCIENTIST MEMORY END ---

下面是当前论文 V{current_version}：
--- PAPER V{current_version} START ---
{paper_text}
--- PAPER V{current_version} END ---

下面是审稿意见 Review_Strict_V{current_version}：
--- REVIEW V{current_version} START ---
{review_text}
--- REVIEW V{current_version} END ---
""".strip()


def build_scientist_compact_prompt(
    current_version: int,
    next_version: int,
    paper_text: str,
    review_text: str,
    scientist_memory: str,
) -> str:
    memory_block = scientist_memory.strip() if scientist_memory.strip() else "当前还没有历史科学家记忆。"
    return f"""
你是全球顶级 AI 科学家，是 ACM MM 的资深投稿者和多年审稿人。你现在要对论文进行“精准微调式升级”，而不是整篇推倒重写。

核心要求：
1. 必须保留原稿中成立的主线创新：decode-time / logits-space intervention、token-local visual evidence、VASM 的结构保护作用，以及 `TLRA_zero` / `TLRA_calib` 的公平边界。
2. 必须明确避免旧的错误 framing：不要再把 `Pooling Paradox` 作为标题级主问题；不要再把 `DoLa / VCD / OPERA` 错写成依赖全局视觉池化的代表方法。
3. 必须优先修复 reviewer 指出的硬伤，尤其是方法漏洞、实验闭环缺口、消融设计不足、评测协议不充分。
4. 不要捏造真实实验结果。对于未跑通的实验，明确改写成“evaluation protocol / planned experiment / experiment outline / hypothesis”。
5. 只输出关键章节替换稿，便于在原稿上做局部更新。
6. 重点升级实验章节和实验大纲，让后续真实实验可以直接按这份文稿执行。
7. 当前轮次不需要重写 Related Work；默认保留旧版 Related Work，只精修核心章节。
8. 当前 benchmark 口径是 image-centric 主合同 + bounded secondary video pilot。视频只有在你能让它服务于“token-local visual evidence 在时序场景也成立”时才应保留为次主线；否则继续降级，不得挤占图像主线。
9. 你必须把实验设计明确组织成三条证据链和五道防线：
   - Hallucination Reduction：POPE / CHAIR + AGL
   - Structure and Reasoning Preservation：MMBench / MME / MMMU(Hard) + `TLRA_no_VASM`
   - Local Evidence Value：FREAK / DocVQA + `TLRA_zero` vs `TLRA_MeanPool`
10. 如果正文开始围绕 OCR paradox、PEFT 身份、WordNet dictionary、EMA kill-switch 等工程边界无限扩写，你必须主动收缩，把它们从主叙事降回 limitation / protocol / appendix，而不是让它们接管论文。
11. 当前新版本必须比旧自动链条更像“真正可投稿的核心版本”：减少摇摆叙事，明确正面方法 claim、主实验表、关键消融、失败案例和可执行协议。
12. 你应默认接受当前实验现实：`TLRA_zero` 更像 probe、`VASM` 当前证据最强、`AdaptiveTopK` 尚未证明强于 `MeanPool`、`DocVQA` 是负控制而不是主胜点。

请严格输出以下带标记的内容：

<<<TITLE_START>>>
[新的论文标题，只保留一行，以 # 开头]
<<<TITLE_END>>>

<<<ABSTRACT_START>>>
[新的摘要块，必须从 ## Abstract 或 ## Abstract (摘要) 开始，到摘要结束]
<<<ABSTRACT_END>>>

<<<INTRODUCTION_START>>>
[新的引言块，必须从 ## 1. Introduction 开始]
<<<INTRODUCTION_END>>>

<<<METHODOLOGY_START>>>
[新的 Section 3，必须从 ## 3. Methodology 开始；重点修复 BPE 与视频模块表述，并收敛绝对化论断]
<<<METHODOLOGY_END>>>

<<<EXPERIMENTS_START>>>
[新的 Section 4，必须从 ## 4. Experiments、## 4. Experimental Design 或 ## 4. Evaluation Protocol and Experimental Design 开始；将无数据的绝对结论改成 evaluation plan / hypothesis-driven protocol]
<<<EXPERIMENTS_END>>>

<<<CLOSING_START>>>
[新的结尾块，优先从 ## 5. Discussion、## 5. Discussion and Limitations 或 ## 5. Conclusion & Limitations 开始；如果你确实拆出单独结论，也可以继续写 ## 6. Conclusion and Future Execution 和 Appendix Plan，但不要删除局限性、修订 claims、appendix plan 等需要保留的后续内容]
<<<CLOSING_END>>>

<<<REVISION_LOG_START>>>
# Revision_Log_V{next_version}
- 逐条说明你针对 review 修了什么
- 说明保留了哪些原有亮点
- 标记哪些点仍需未来实验验证
<<<REVISION_LOG_END>>>

<<<SCIENTIST_MEMORY_START>>>
# Scientist_Memory
- 本轮最有效的改稿策略
- 下一轮最需要继续盯紧的问题
<<<SCIENTIST_MEMORY_END>>>

下面是历史科学家记忆：
--- SCIENTIST MEMORY START ---
{memory_block}
--- SCIENTIST MEMORY END ---

下面是当前论文 V{current_version}：
--- PAPER V{current_version} START ---
{paper_text}
--- PAPER V{current_version} END ---

下面是审稿意见 Review_Strict_V{current_version}：
--- REVIEW V{current_version} START ---
{review_text}
--- REVIEW V{current_version} END ---
""".strip()


def build_comparison_doc(
    model: str,
    score_v1: float | None,
    score_v2: float | None,
    review_v1: str,
    review_v2: str,
    revision_log_v2: str,
) -> str:
    delta_text = "unknown"
    if score_v1 is not None and score_v2 is not None:
        delta = score_v2 - score_v1
        delta_text = f"{delta:+.1f}"

    return f"""# Pilot Comparison

## Workflow Settings
- Model: `{model}`
- Pilot scope: `V1 -> Review_V1 -> V2 -> Review_V2`
- Goal: validate that the adversarial workflow can improve paper quality without discarding the core idea

## Score Tracking
- `V1`: {score_v1 if score_v1 is not None else "not parsed"}/5
- `V2`: {score_v2 if score_v2 is not None else "not parsed"}/5
- Delta: {delta_text}

## Review V1 Snapshot
{review_v1[:2500].strip()}

## Revision Log V2 Snapshot
{revision_log_v2[:2500].strip()}

## Review V2 Snapshot
{review_v2[:2500].strip()}

## Preliminary Judgment
- If `V2` improves the score or shifts criticism from fatal flaws to actionable weaknesses, the workflow is functioning.
- If the score does not improve, inspect whether the scientist over-rewrote the draft or failed to address the harshest reviewer concerns.
- Use the next rounds to focus on claim calibration, experimental closure, fairer related-work positioning, and clearer contribution boundaries.
"""


def build_scaleup_doc(model: str) -> str:
    return f"""# Workflow Scale-Up Guide

## Current State
- Pilot model: `{model}`
- Reviewer policy: always use a fresh request context
- Scientist policy: carry forward accumulated scientist memory between rounds

## Recommended Next Loop
1. Feed the current benchmark-based latest paper to a fresh reviewer and create the matching `Review_Strict_Vx.md`
2. Use `Review_Strict_Vx.md` plus `Scientist_Memory.md` to generate `论文大纲_V(x+1).md`
3. Keep every version and never overwrite prior outputs
4. Stop when a fresh reviewer gives `4/5` or `5/5`, or after 100 rounds

## What To Optimize First
- Tone down unsupported absolute claims
- Preserve the positive core: token-local logits intervention + VASM + fair zero-shot/calibrated split
- Tighten the method-to-evaluation loop with explicit ablations and planned experiments
- Avoid reviving invalid baseline-framing arguments
- Add limitations and failure cases before the reviewer asks for them
- If the loop flatlines around `3/5`, reset from the current benchmark instead of continuing a drifted branch

## Operating Notes
- Re-run the same script with a larger `--max-version` target when you want to continue the loop
- Keep the API key outside the code and provide it via environment variable
- Review `Scientist_Memory.md` occasionally to prevent local overfitting to one reviewer style
"""


def resolve_start_paper(workspace: Path, source_name: str, start_version: int) -> Path:
    if start_version == 1:
        return freeze_v1(workspace, source_name)
    start_path = paper_path(workspace, start_version)
    if not start_path.exists():
        source_path = workspace / source_name
        if not source_path.exists():
            raise FileNotFoundError(
                f"Missing start paper for resume workflow: {start_path}. "
                "Create or copy the benchmark paper to that version before continuing."
            )
        write_utf8(start_path, read_utf8(source_path))
    return start_path


def extract_version_number(path: Path) -> int | None:
    match = re.search(r"_V(\d+)$", path.stem)
    if match:
        return int(match.group(1))
    return None


def collect_round_rows(workspace: Path) -> list[dict[str, Any]]:
    versions: set[int] = set()
    for pattern in ("论文大纲_V*.md", "Review_Strict_V*.md", "Revision_Log_V*.md"):
        for path in workspace.glob(pattern):
            version = extract_version_number(path)
            if version is not None:
                versions.add(version)

    rows: list[dict[str, Any]] = []
    previous_score: float | None = None
    for version in sorted(versions):
        paper = paper_path(workspace, version)
        review = review_path(workspace, version)
        revision_log = revision_log_path(workspace, version)
        score = parse_score(read_utf8(review)) if review.exists() else None
        delta: float | None = None
        if score is not None and previous_score is not None:
            delta = score - previous_score
        if score is not None:
            previous_score = score
        rows.append(
            {
                "version": version,
                "paper_exists": paper.exists(),
                "review_exists": review.exists(),
                "revision_log_exists": revision_log.exists(),
                "score": score,
                "delta": delta,
                "paper_path": paper,
                "review_path": review,
                "revision_log_path": revision_log,
            }
        )
    return rows


def build_score_tracker_doc(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# 评分波动记录",
        "",
        "## 说明",
        "",
        "- 本文档自动记录每一轮论文版本、审稿意见、得分与分数波动。",
        "- `delta` 表示相对于上一轮已评分版本的变化。",
        "",
        "## 评分台账",
        "",
        "| Version | Paper | Review | Revision Log | Score | Delta |",
        "|---|---|---|---|---:|---:|",
    ]
    for row in rows:
        score_text = f"{row['score']:.1f}" if row["score"] is not None else "-"
        delta_text = f"{row['delta']:+.1f}" if row["delta"] is not None else "-"
        lines.append(
            f"| V{row['version']} | {'Y' if row['paper_exists'] else 'N'} | "
            f"{'Y' if row['review_exists'] else 'N'} | {'Y' if row['revision_log_exists'] else 'N'} | "
            f"{score_text} | {delta_text} |"
        )
    return "\n".join(lines) + "\n"


def build_high_score_doc(rows: list[dict[str, Any]]) -> str:
    above_four = [row for row in rows if row["score"] is not None and row["score"] > 4.0]
    at_least_four = [row for row in rows if row["score"] is not None and row["score"] >= 4.0]
    lines = [
        "# 高分版本记录",
        "",
        "## 超过 4 分的版本",
        "",
    ]
    if above_four:
        for row in above_four:
            lines.append(
                f"- `V{row['version']}`: {row['score']:.1f}/5, "
                f"`{row['paper_path'].name}`, `{row['review_path'].name}`"
            )
    else:
        lines.append("- 当前暂无严格大于 `4/5` 的版本。")

    lines.extend(
        [
            "",
            "## 达到 4 分及以上的版本",
            "",
        ]
    )
    if at_least_four:
        for row in at_least_four:
            lines.append(
                f"- `V{row['version']}`: {row['score']:.1f}/5, "
                f"`{row['paper_path'].name}`, `{row['review_path'].name}`"
            )
    else:
        lines.append("- 当前暂无达到 `4/5` 的版本。")
    return "\n".join(lines) + "\n"


def summarize_review_themes(workspace: Path, rows: list[dict[str, Any]]) -> list[str]:
    theme_rules = {
        "Local evidence / dense spatial grounding": ["docvqa", "freak", "dense", "chartqa", "local evidence", "top-k"],
        "Phi / modality alignment": ["phi", "alignment", "unembedding", "w_{vocab}", "w_vocab"],
        "Latency / TPOT / VRAM": ["latency", "tpot", "vram", "tokens/s", "complexity"],
        "Video / temporal reasoning": ["video", "vidhalluc", "temporal", "frame", "t x h x w"],
        "VASM / syntax protection / polysemy": ["vasm", "entropy", "polysemy", "bpe", "syntax"],
    }
    counts = {theme: 0 for theme in theme_rules}
    recent_rows = [row for row in rows if row["review_exists"]][-5:]
    for row in recent_rows:
        review_text = read_utf8(row["review_path"]).lower()
        for theme, keywords in theme_rules.items():
            if any(keyword in review_text for keyword in keywords):
                counts[theme] += 1
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [f"{theme}: mentioned in {count}/5 recent reviews" for theme, count in ordered if count > 0]


def build_observer_log_doc(workspace: Path, rows: list[dict[str, Any]]) -> str:
    scored_rows = [row for row in rows if row["score"] is not None]
    best_row = max(scored_rows, key=lambda row: row["score"]) if scored_rows else None
    latest_scored = scored_rows[-1] if scored_rows else None
    last_five_scores = [f"V{row['version']}={row['score']:.1f}" for row in scored_rows[-5:]]
    themes = summarize_review_themes(workspace, rows)

    lines = [
        "# 旁观者洞察日志",
        "",
        "## 当前观察",
        "",
    ]
    if best_row:
        lines.append(f"- 当前最高分版本：`V{best_row['version']}`，`{best_row['score']:.1f}/5`。")
    if latest_scored:
        lines.append(f"- 当前最新已评分版本：`V{latest_scored['version']}`，`{latest_scored['score']:.1f}/5`。")
    if last_five_scores:
        lines.append(f"- 最近五个已评分版本：{', '.join(last_five_scores)}。")

    lines.extend(
        [
            "",
            "## 近期高频争议主题",
            "",
        ]
    )
    if themes:
        lines.extend(f"- {theme}" for theme in themes)
    else:
        lines.append("- 当前尚无足够评审文本用于提炼主题。")

    lines.extend(
        [
            "",
            "## 调整建议",
            "",
            "- 若分数连续 3 轮不提升，应优先收缩主 claim，而不是继续扩写方法分支。",
            "- 若 `Phi / alignment` 与 `latency / VRAM` 同时高频出现，说明论文已经进入“实验兑现与复杂度证明”阶段。",
            "- 若视频主题持续被点名但未带来加分，需要检查视频主线是否真正服务于 `T x H x W` 的统一论证，而不是独立扩展。",
            "- 若 OCR / DocVQA 被频繁点名且分数不升，应检查论文是否正在把一个尚未自洽的边界案例错误写成核心动机。",
        ]
    )
    return "\n".join(lines) + "\n"


def append_observer_checkpoint_if_needed(workspace: Path, version: int) -> None:
    if version % 10 != 0:
        return

    checkpoint_title = f"## 自动旁观者检查点 V{version}"
    path = workspace / COGNITIVE_INSIGHT_DOC
    existing = read_utf8(path) if path.exists() else "# 认知洞察文档\n"
    if checkpoint_title in existing:
        return

    rows = collect_round_rows(workspace)
    scored = [row for row in rows if row["score"] is not None and row["version"] <= version]
    window = scored[-10:]
    if not window:
        return

    scores = [row["score"] for row in window if row["score"] is not None]
    avg_score = sum(scores) / len(scores)
    best_score = max(scores)
    latest_score = scores[-1]
    last_five = scores[-5:]
    plateau = len(last_five) == 5 and all(score <= 3.0 for score in last_five)
    review_blob = "\n".join(read_utf8(row["review_path"]).lower() for row in window if row["review_exists"])

    insights: list[str] = []
    if plateau:
        insights.append("最近五轮持续停在 `3/5` 或以下，说明自动链路正在细化边界，但没有提升主接受度；下一窗口应优先回到 `标杆V3` 主合同，而不是继续扩写工程细节。")
    if "ocr" in review_blob or "docvqa" in review_blob:
        insights.append("OCR / DocVQA 在最近窗口继续构成逻辑风险；除非方法能在 OCR token 上真正触发介入，否则不要再把 OCR-heavy documents 写成核心动机。")
    if "video" in review_blob or "vidhalluc" in review_blob:
        insights.append("视频仍应保持 bounded secondary pilot；它可以增强 ACM MM 契合度，但不应反过来主导论文主合同。")
    if any(keyword in review_blob for keyword in ("phi", "alignment", "w_vocab", "unembedding")):
        insights.append("当前主风险已收敛到 `TLRA_zero` 是否 viable 与 `TLRA_calib` 何时必要；后续轮次应优先明确公平边界，而不是继续发明新模块。")
    if any(keyword in review_blob for keyword in ("latency", "vram", "tpot", "tokens/s", "complexity")):
        insights.append("复杂度与失败边界已经成为可信度资产；后续版本应主动写清效率审计和 out-of-candidate unrecoverability，而不是回避它们。")
    if not insights:
        insights.append("最近十轮没有出现新的理论增益，说明 benchmark 约束仍有效；继续围绕 `TLRA + VASM + fair zero/calib split + bounded video pilot` 打磨。")

    block = [
        "",
        checkpoint_title,
        "",
        f"- 检查窗口：`V{window[0]['version']}-V{window[-1]['version']}`",
        f"- 平均分：`{avg_score:.2f}/5`",
        f"- 窗口最高分：`{best_score:.1f}/5`",
        f"- 当前最新分：`{latest_score:.1f}/5`",
        "",
        "### 自动判断",
        "",
    ]
    block.extend(f"- {insight}" for insight in insights)
    block.extend(
        [
            "",
            "### 自动建议",
            "",
            "- 若下一窗口仍未突破当前高分带，应优先 benchmark reset，而不是继续沿用当前漂移叙事。",
            "- 新 benchmark 候选只应建立在更稳的主合同上，而不是建立在某个边界问题的过度扩写上。",
            "",
        ]
    )
    write_utf8(path, existing.rstrip() + "\n" + "\n".join(block))


def refresh_tracking_docs(workspace: Path) -> None:
    rows = collect_round_rows(workspace)
    write_utf8(workspace / SCORE_TRACKER_DOC, build_score_tracker_doc(rows))
    write_utf8(workspace / HIGH_SCORE_DOC, build_high_score_doc(rows))
    write_utf8(workspace / OBSERVER_LOG_DOC, build_observer_log_doc(workspace, rows))


def load_state(state_path: Path, source_path: Path, api: ApiConfig) -> dict[str, Any]:
    if state_path.exists():
        return json.loads(read_utf8(state_path))
    return {
        "source": str(source_path),
        "api_url": api.api_url,
        "model": api.model,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "runs": [],
    }


def save_state(state_path: Path, state: dict[str, Any]) -> None:
    write_utf8(state_path, json.dumps(state, ensure_ascii=False, indent=2))


def append_state_event(state: dict[str, Any], **entry: Any) -> None:
    state.setdefault("runs", []).append(
        {
            **entry,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
    )


def freeze_v1(workspace: Path, source_name: str) -> Path:
    source_path = workspace / source_name
    if not source_path.exists():
        raise FileNotFoundError(f"Source paper not found: {source_path}")
    target_path = paper_path(workspace, 1)
    if not target_path.exists():
        write_utf8(target_path, read_utf8(source_path))
    return target_path


def build_recovered_revision_log(current_version: int, next_version: int) -> str:
    return f"""# Revision_Log_V{next_version}

- Recovered existing `论文大纲_V{next_version}.md` from a previous interrupted workflow run.
- This log was reconstructed so the workflow can continue from `V{current_version}` to `V{next_version}` without overwriting the existing paper draft.
- Recommended follow-up: review the diff between `论文大纲_V{current_version}.md` and `论文大纲_V{next_version}.md`, then let later reviewer rounds validate whether the revision meaningfully improves methodology and experiment design.
"""


def strip_references_section(text: str) -> str:
    stripped = re.sub(r"\n## Works cited\s*$.*", "", text, flags=re.DOTALL)
    return stripped.strip() + "\n"


def extract_section(text: str, start_heading: str, end_heading: str | None = None) -> str:
    start_pattern = re.escape(start_heading)
    if end_heading is None:
        pattern = re.compile(start_pattern + r".*", re.MULTILINE | re.DOTALL)
    else:
        end_pattern = re.escape(end_heading)
        pattern = re.compile(start_pattern + r".*?(?=" + end_pattern + r")", re.MULTILINE | re.DOTALL)
    match = pattern.search(text)
    if not match:
        raise RuntimeError(f"Could not extract section starting with: {start_heading}")
    return match.group(0).strip()


def extract_first_matching_section(
    text: str,
    start_headings: list[str],
    end_headings: list[str] | None = None,
) -> str:
    last_error: Exception | None = None
    candidate_ends = end_headings or [None]
    for start_heading in start_headings:
        for end_heading in candidate_ends:
            try:
                return extract_section(text, start_heading, end_heading)
            except RuntimeError as exc:
                last_error = exc
    raise RuntimeError(str(last_error) if last_error else "Could not extract matching section.")


def build_reviewer_excerpt(text: str) -> str:
    clean_text = strip_references_section(text)
    title_match = re.search(r"^# .+$", clean_text, flags=re.MULTILINE)
    if not title_match:
        raise RuntimeError("Could not find paper title for reviewer excerpt.")

    sections = [
        title_match.group(0).strip(),
        extract_first_matching_section(
            clean_text,
            ["## Abstract", "## Abstract (摘要)"],
            ["## 1. Introduction"],
        ),
        extract_first_matching_section(
            clean_text,
            ["## 1. Introduction"],
            ["## 2. Related Work", "## 3. Methodology"],
        ),
        extract_first_matching_section(
            clean_text,
            ["## 3. Methodology"],
            ["## 4. Experiments", "## 4. Experimental Design", "## 4. Evaluation Protocol and Experimental Design"],
        ),
        extract_first_matching_section(
            clean_text,
            ["## 4. Experiments", "## 4. Experimental Design", "## 4. Evaluation Protocol and Experimental Design"],
            [
                "## 5. Conclusion & Limitations",
                "## 5. Discussion",
                "## 5. Discussion & Limitations",
                "## 5. Discussion and Limitations",
                "## 6. Conclusion and Future Execution",
            ],
        ),
        extract_first_matching_section(
            clean_text,
            [
                "## 5. Conclusion & Limitations",
                "## 5. Discussion",
                "## 5. Discussion & Limitations",
                "## 5. Discussion and Limitations",
                "## 6. Conclusion and Future Execution",
            ],
            [
                "# 6. Appendix",
                "## 6. Limitations",
                "## 6. Revised Claims (Pending Empirical Execution)",
                "## 7. Appendix Plan",
                "## 8. Appendix Plan",
                "## Appendix Plan (Pending Experimental Results)",
                "## Works cited",
                "## Works Cited",
            ],
        ),
        extract_first_matching_section(
            clean_text,
            ["## 6. Limitations", "## 6. Conclusion and Future Execution"],
            [
                "## 7. Revised Claims for the Paper",
                "## 8. Appendix Plan",
                "## Appendix Plan (Pending Experimental Results)",
                "## Works cited",
                "## Works Cited",
            ],
        )
        if "## 6. Limitations" in clean_text or "## 6. Conclusion and Future Execution" in clean_text
        else "",
    ]
    return "\n\n".join(section.strip() for section in sections if section.strip()) + "\n"


def build_scientist_excerpt(text: str) -> str:
    clean_text = strip_references_section(text)
    title_match = re.search(r"^# .+$", clean_text, flags=re.MULTILINE)
    if not title_match:
        raise RuntimeError("Could not find paper title for scientist excerpt.")

    sections = [
        title_match.group(0).strip(),
        extract_first_matching_section(
            clean_text,
            ["## Abstract", "## Abstract (摘要)"],
            ["## 1. Introduction"],
        ),
        extract_first_matching_section(
            clean_text,
            ["## 1. Introduction"],
            ["## 2. Related Work", "## 3. Methodology"],
        ),
        extract_first_matching_section(
            clean_text,
            ["## 3. Methodology"],
            ["## 4. Experiments", "## 4. Experimental Design", "## 4. Evaluation Protocol and Experimental Design"],
        ),
        extract_first_matching_section(
            clean_text,
            ["## 4. Experiments", "## 4. Experimental Design", "## 4. Evaluation Protocol and Experimental Design"],
            [
                "## 5. Conclusion & Limitations",
                "## 5. Discussion",
                "## 5. Discussion & Limitations",
                "## 5. Discussion and Limitations",
                "## 6. Conclusion and Future Execution",
            ],
        ),
        extract_first_matching_section(
            clean_text,
            [
                "## 5. Conclusion & Limitations",
                "## 5. Discussion",
                "## 5. Discussion & Limitations",
                "## 5. Discussion and Limitations",
                "## 6. Conclusion and Future Execution",
            ],
            [
                "# 6. Appendix",
                "## 6. Limitations",
                "## 6. Revised Claims (Pending Empirical Execution)",
                "## 7. Revised Claims for the Paper",
                "## 7. Appendix Plan",
                "## 8. Appendix Plan",
                "## Appendix Plan (Pending Experimental Results)",
                "## Works cited",
                "## Works Cited",
            ],
        ),
    ]
    if "## 6. Limitations" in clean_text or "## 6. Conclusion and Future Execution" in clean_text:
        sections.append(
            extract_first_matching_section(
                clean_text,
                ["## 6. Limitations", "## 6. Conclusion and Future Execution"],
                [
                    "## 7. Revised Claims for the Paper",
                    "## 8. Appendix Plan",
                    "## Appendix Plan (Pending Experimental Results)",
                    "## Works cited",
                    "## Works Cited",
                ],
            )
        )
    if "## 7. Revised Claims for the Paper" in clean_text:
        sections.append(
            extract_first_matching_section(
                clean_text,
                ["## 7. Revised Claims for the Paper"],
                ["## 8. Appendix Plan", "## Works cited", "## Works Cited"],
            )
        )
    if "## 8. Appendix Plan" in clean_text or "## Appendix Plan (Pending Experimental Results)" in clean_text:
        sections.append(
            extract_first_matching_section(
                clean_text,
                ["## 8. Appendix Plan", "## Appendix Plan (Pending Experimental Results)"],
                ["## Works cited", "## Works Cited"],
            )
        )
    if "## 6. Revised Claims (Pending Empirical Execution)" in clean_text:
        sections.append(
            extract_first_matching_section(
                clean_text,
                ["## 6. Revised Claims (Pending Empirical Execution)"],
                ["## 7. Appendix Plan", "## Appendix Plan (Pending Experimental Results)", "## Works cited", "## Works Cited"],
            )
        )
    if "## 7. Appendix Plan" in clean_text or "## Appendix Plan (Pending Experimental Results)" in clean_text:
        sections.append(
            extract_first_matching_section(
                clean_text,
                ["## 7. Appendix Plan", "## Appendix Plan (Pending Experimental Results)"],
                ["## Works cited", "## Works Cited"],
            )
        )
    return "\n\n".join(section.strip() for section in sections if section.strip()) + "\n"


def normalize_spacing(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


def replace_title_block(text: str, new_title: str) -> str:
    return re.sub(r"^# .+$", new_title.strip(), text, count=1, flags=re.MULTILINE)


def replace_block_between(text: str, start_regex: str, end_regex: str, replacement: str) -> str:
    pattern = re.compile(start_regex + r".*?(?=" + end_regex + r")", re.MULTILINE | re.DOTALL)
    if not pattern.search(text):
        raise RuntimeError(f"Could not find block starting with pattern: {start_regex}")
    replacement_text = replacement.strip() + "\n\n"
    return pattern.sub(lambda _: replacement_text, text, count=1)


def cleanup_meta_text(text: str) -> str:
    text = re.sub(r"(?ms)^> ```latex\s*\n.*?^> ```\s*\n?", "", text)
    banned_line_patterns = [
        r"^这是一份基于.+$",
        r"^我已无情剔除了.+$",
        r"^以下内容可直接作为论文正文使用：$",
        r"^指挥官，.+$",
        r"^\*\*\*$",
    ]
    for pattern in banned_line_patterns:
        text = re.sub(pattern, "", text, flags=re.MULTILINE)
    return normalize_spacing(text)


def apply_compact_revision(
    original_paper: str,
    title_block: str,
    abstract_block: str,
    introduction_block: str,
    methodology_block: str,
    experiments_block: str,
    closing_block: str,
) -> str:
    updated = cleanup_meta_text(original_paper)
    related_work_block = ""
    if "## 2. Related Work" in updated:
        related_work_block = extract_first_matching_section(
            updated,
            ["## 2. Related Work"],
            ["## 3. Methodology"],
        )
    references_block = ""
    if "## Works cited" in updated or "## Works Cited" in updated:
        references_block = extract_first_matching_section(
            updated,
            ["## Works cited", "## Works Cited"],
            None,
        )

    sections = [
        title_block.strip(),
        abstract_block.strip(),
        introduction_block.strip(),
    ]
    if related_work_block.strip():
        sections.append(related_work_block.strip())
    sections.extend(
        [
            methodology_block.strip(),
            experiments_block.strip(),
            closing_block.strip(),
        ]
    )
    if references_block.strip():
        sections.append(references_block.strip())
    return normalize_spacing("\n\n".join(section for section in sections if section))


def build_scientist_retry_prompt(base_prompt: str) -> str:
    return (
        base_prompt
        + "\n\n"
        + "严格格式补救要求：\n"
        + "1. 只输出要求的标记块，不要前言、解释、道歉或额外说明。\n"
        + "2. 每个 `<<<..._START>>>` 和 `<<<..._END>>>` 必须完整成对出现。\n"
        + "3. 每个 block 都必须非空。\n"
        + "4. `TITLE` block 只能包含单行 `# ...` 标题。\n"
        + "5. 如果你无法完成，也要按标记输出最小可解析内容，不允许漏 block。\n"
    )


def dump_failed_response(
    workspace: Path,
    response_kind: str,
    current_version: int,
    next_version: int,
    content: str,
) -> Path:
    out_dir = workspace / "paper_workflow_runs"
    ensure_parent(out_dir / ".keep")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"{response_kind}_v{current_version}_to_v{next_version}_{timestamp}.md"
    write_utf8(out_path, content)
    return out_path


def classify_scientist_failure(exc: Exception) -> str:
    message = str(exc).lower()
    if "api" in message:
        return "API failure"
    if "replacement block" in message or "did not contain all compact" in message:
        return "formatting failure"
    if "could not extract section" in message or "could not find block" in message or "could not find paper title" in message:
        return "section splice failure"
    return "formatting failure"


def run_reviewer(api: ApiConfig, version: int, paper_text: str) -> tuple[str, float | None]:
    reviewer_excerpt = build_reviewer_excerpt(paper_text)
    response = call_chat(api, reviewer_system_prompt(), build_reviewer_prompt(version, reviewer_excerpt))
    review_text = extract_block(response, "<<<REVIEW_START>>>", "<<<REVIEW_END>>>")
    score = parse_score(review_text)
    return review_text, score


def run_scientist(
    api: ApiConfig,
    workspace: Path,
    current_version: int,
    next_version: int,
    paper_text: str,
    review_text: str,
    scientist_memory: str,
) -> tuple[str, str, str]:
    scientist_excerpt = build_scientist_excerpt(paper_text)
    base_prompt = build_scientist_compact_prompt(
        current_version,
        next_version,
        scientist_excerpt,
        review_text,
        scientist_memory,
    )

    def parse_scientist_response(response: str) -> tuple[str, str, str]:
        title_block = extract_block(response, "<<<TITLE_START>>>", "<<<TITLE_END>>>")
        abstract_block = extract_block(response, "<<<ABSTRACT_START>>>", "<<<ABSTRACT_END>>>")
        introduction_block = extract_block(response, "<<<INTRODUCTION_START>>>", "<<<INTRODUCTION_END>>>")
        methodology_block = extract_block(response, "<<<METHODOLOGY_START>>>", "<<<METHODOLOGY_END>>>")
        experiments_block = extract_block(response, "<<<EXPERIMENTS_START>>>", "<<<EXPERIMENTS_END>>>")
        closing_block = extract_block(response, "<<<CLOSING_START>>>", "<<<CLOSING_END>>>")
        revision_log = extract_block(response, "<<<REVISION_LOG_START>>>", "<<<REVISION_LOG_END>>>")
        new_memory = extract_block(response, "<<<SCIENTIST_MEMORY_START>>>", "<<<SCIENTIST_MEMORY_END>>>")

        required_blocks = [
            title_block,
            abstract_block,
            introduction_block,
            methodology_block,
            experiments_block,
            closing_block,
        ]
        if not all(block.strip() for block in required_blocks):
            raise RuntimeError("Scientist response did not contain all compact replacement blocks.")
        if not revision_log.strip():
            revision_log = "# Revision_Log\n- No structured revision log was returned."
        if not new_memory.strip():
            new_memory = "# Scientist_Memory\n- No structured scientist memory was returned."

        revised_paper = apply_compact_revision(
            original_paper=paper_text,
            title_block=title_block,
            abstract_block=abstract_block,
            introduction_block=introduction_block,
            methodology_block=methodology_block,
            experiments_block=experiments_block,
            closing_block=closing_block,
        )
        return revised_paper, revision_log, new_memory

    response = ""
    try:
        response = call_chat(api, scientist_system_prompt(), base_prompt)
        return parse_scientist_response(response)
    except Exception as first_exc:
        if response:
            dump_failed_response(workspace, "scientist_raw_first_failure", current_version, next_version, response)
        retry_response = ""
        try:
            retry_response = call_chat(api, scientist_system_prompt(), build_scientist_retry_prompt(base_prompt))
            return parse_scientist_response(retry_response)
        except Exception as retry_exc:
            if retry_response:
                dump_failed_response(workspace, "scientist_raw_retry_failure", current_version, next_version, retry_response)
            error_label = classify_scientist_failure(retry_exc if retry_response else first_exc)
            raise RuntimeError(f"{error_label}: {retry_exc if retry_response else first_exc}")


def write_prompt_strategy_doc(path: Path, model: str, api_url: str) -> None:
    content = f"""# Paper Workflow Prompt Strategy

## Active API Settings
- API URL: `{api_url}`
- Model: `{model}`

## Reviewer Strategy
- Every reviewer call uses a fresh request context
- The reviewer is instructed to act like an ACM MM chief reviewer
- Output includes an explicit `Score: x/5` line and concrete figure/table/experiment suggestions
- The reviewer must separate what should be kept from what must be fixed
- The reviewer should focus on methodology and experimental design instead of typography or formatting
- The reviewer only receives title + Abstract + Introduction + Methodology + Experiments + Conclusion/Limitations
- The reviewer does not receive related work, appendices, or references, which saves tokens and keeps attention on the core paper
- The reviewer is explicitly told that `DoLa`, `VCD`, and `OPERA` are competitive baselines, not evidence for the paper's problem definition
- The reviewer evaluates the paper through three evidence chains:
  - hallucination reduction
  - structure/reasoning preservation
  - local-evidence value
- The reviewer also checks whether the paper stays within the active benchmark contract:
  - image-centric core claim
  - bounded secondary video pilot
  - no OCR-heavy overclaim unless the intervention is actually active there

## Scientist Strategy
- The scientist performs targeted upgrades on top of the latest draft
- The scientist must preserve valid innovation and strong structure already present in the paper
- Unsupported claims must be softened instead of defended with fabricated evidence
- Planned but unrun experiments must be clearly marked as planned work
- The scientist should turn reviewer criticism into a stronger experiment outline for future execution
- The scientist is explicitly forbidden from reviving the invalid `Pooling Paradox`-as-title-eye framing
- The scientist must preserve the active benchmark core:
  - decode-time / logits-space intervention
  - token-local visual evidence
  - VASM-based structure preservation
  - `TLRA_zero` / `TLRA_calib` fairness boundary
- The scientist must not let OCR-heavy motivation, PEFT identity debates, or over-detailed engineering mechanisms take over the main paper narrative
- The scientist should keep video as a bounded secondary pilot unless there is clear evidence that temporal locality materially strengthens the paper
- The scientist emits:
  - the full revised paper
  - a structured revision log
  - a compact scientist memory block for future rounds

## Context Policy
- Reviewer: no memory, always fresh
- Scientist: cumulative memory saved in `Scientist_Memory.md`
- All paper and review versions are immutable once written
"""
    write_utf8(path, content)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the paper adversarial polishing workflow.")
    parser.add_argument("--workspace", default=".", help="Workspace directory")
    parser.add_argument("--source", default=DEFAULT_SOURCE, help="Source paper filename")
    parser.add_argument(
        "--start-version",
        type=int,
        default=1,
        help="Existing paper version to start from. Use 1 for fresh runs, or an existing version such as 75 to resume from the current benchmark reset.",
    )
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help="Chat completion API base URL")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model ID to use")
    parser.add_argument("--api-key", default="", help="Primary API key. If omitted, use LEMON_API_KEY env var.")
    parser.add_argument(
        "--backup-api-key",
        default="",
        help="Backup API key. If omitted, use LEMON_API_KEY_BACKUP env var when available.",
    )
    parser.add_argument("--timeout-seconds", type=int, default=300, help="API timeout in seconds")
    parser.add_argument(
        "--max-version",
        type=int,
        default=2,
        help="Highest paper version to produce. Example: 2 means generate/review V1 and produce/review V2.",
    )
    args = parser.parse_args()

    api_key = args.api_key or ""
    backup_api_key = args.backup_api_key or ""
    if not api_key:
        import os

        api_key = os.environ.get("LEMON_API_KEY", "")
        backup_api_key = backup_api_key or os.environ.get("LEMON_API_KEY_BACKUP", "")
    if not api_key:
        raise SystemExit("Missing API key. Pass --api-key or set LEMON_API_KEY.")
    api_keys = [api_key]
    if backup_api_key and backup_api_key != api_key:
        api_keys.append(backup_api_key)

    workspace = Path(args.workspace).resolve()
    if args.max_version < 2:
        raise SystemExit("--max-version must be at least 2.")
    if args.start_version < 1:
        raise SystemExit("--start-version must be at least 1.")
    if args.max_version < args.start_version:
        raise SystemExit("--max-version must be greater than or equal to --start-version.")
    api = ApiConfig(
        api_url=args.api_url,
        api_keys=api_keys,
        model=args.model,
        timeout_seconds=args.timeout_seconds,
    )

    start_path = resolve_start_paper(workspace, args.source, args.start_version)
    prompt_doc_path = workspace / PROMPT_DOC
    write_prompt_strategy_doc(prompt_doc_path, api.model, api.api_url)

    state_path = workspace / STATE_FILE
    state = load_state(state_path, workspace / args.source, api)
    append_state_event(
        state,
        version=args.start_version,
        paper=str(start_path),
        status="frozen" if args.start_version == 1 else "resume_anchor",
    )
    save_state(state_path, state)
    refresh_tracking_docs(workspace)

    scientist_memory_path = workspace / SCIENTIST_MEMORY_FILE
    scientist_memory = read_utf8(scientist_memory_path) if scientist_memory_path.exists() else ""
    latest_score_v1: float | None = None
    latest_score_target: float | None = None
    latest_review_v1 = ""
    latest_review_target = ""
    latest_revision_log_target = ""

    for version in range(args.start_version, args.max_version + 1):
        current_paper_path = paper_path(workspace, version)
        if not current_paper_path.exists():
            raise FileNotFoundError(f"Missing paper version required by workflow: {current_paper_path}")
        current_paper_text = read_utf8(current_paper_path)

        current_review_path = review_path(workspace, version)
        if current_review_path.exists():
            current_review_text = read_utf8(current_review_path)
            current_score = parse_score(current_review_text)
        else:
            current_review_text, current_score = run_reviewer(api, version, current_paper_text)
            write_utf8(current_review_path, current_review_text)
            append_state_event(
                state,
                version=version,
                review=str(current_review_path),
                score=current_score,
                status="reviewed",
            )
            save_state(state_path, state)
            refresh_tracking_docs(workspace)

        append_observer_checkpoint_if_needed(workspace, version)

        if version == args.start_version:
            latest_score_v1 = current_score
            latest_review_v1 = current_review_text
        if version == args.max_version:
            latest_score_target = current_score
            latest_review_target = current_review_text
            break

        next_paper_path = paper_path(workspace, version + 1)
        next_revision_log_path = revision_log_path(workspace, version + 1)

        if next_paper_path.exists() and next_revision_log_path.exists():
            revised_paper_text = read_utf8(next_paper_path)
            revision_log_text = read_utf8(next_revision_log_path)
        elif next_paper_path.exists() and not next_revision_log_path.exists():
            revised_paper_text = read_utf8(next_paper_path)
            revision_log_text = build_recovered_revision_log(version, version + 1)
            write_utf8(next_revision_log_path, revision_log_text)
            append_state_event(
                state,
                version=version + 1,
                paper=str(next_paper_path),
                revision_log=str(next_revision_log_path),
                status="recovered",
            )
            save_state(state_path, state)
            refresh_tracking_docs(workspace)
        else:
            revised_paper_text, revision_log_text, scientist_memory = run_scientist(
                api=api,
                workspace=workspace,
                current_version=version,
                next_version=version + 1,
                paper_text=current_paper_text,
                review_text=current_review_text,
                scientist_memory=scientist_memory,
            )
            if next_paper_path.exists() or next_revision_log_path.exists():
                raise FileExistsError(
                    f"Refusing partial overwrite for version {version + 1}. "
                    "Delete incomplete outputs or continue from a clean state."
                )
            write_utf8(next_paper_path, revised_paper_text)
            write_utf8(next_revision_log_path, revision_log_text)
            write_utf8(scientist_memory_path, scientist_memory)
            append_state_event(
                state,
                version=version + 1,
                paper=str(next_paper_path),
                revision_log=str(next_revision_log_path),
                scientist_memory=str(scientist_memory_path),
                status="revised",
            )
            save_state(state_path, state)
            refresh_tracking_docs(workspace)

        if version + 1 == args.max_version:
            latest_revision_log_target = revision_log_text

    comparison_path = workspace / COMPARISON_DOC
    scaleup_path = workspace / SCALE_UP_DOC
    write_utf8(
        comparison_path,
        build_comparison_doc(
            model=api.model,
            score_v1=latest_score_v1,
            score_v2=latest_score_target,
            review_v1=latest_review_v1,
            review_v2=latest_review_target,
            revision_log_v2=latest_revision_log_target,
        ),
    )
    write_utf8(scaleup_path, build_scaleup_doc(api.model))
    refresh_tracking_docs(workspace)

    print("Workflow completed successfully.")
    print(f"Start paper: {start_path}")
    print(f"Start review: {review_path(workspace, args.start_version)}")
    print(f"Target paper: {paper_path(workspace, args.max_version)}")
    print(f"Target review: {review_path(workspace, args.max_version)}")
    print(f"Comparison: {comparison_path}")
    print(f"Scale-up guide: {scaleup_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pragma: no cover - CLI guard
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
