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


def extract_block(text: str, start_marker: str, end_marker: str) -> str:
    start = text.find(start_marker)
    end = text.find(end_marker)
    if start == -1 or end == -1 or end < start:
        return text.strip()
    return text[start + len(start_marker) : end].strip()


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
1. 必须保留原稿中成立的主线创新：对 pooling paradox 的批判、对语言先验与视觉锚定失配的关注、BRA 的核心直觉。
2. 必须优先修复 reviewer 指出的硬伤，尤其是方法漏洞、实验闭环缺口、消融设计不足、评测协议不充分。
3. 不要捏造真实实验结果。对于未跑通的实验，明确改写成“evaluation protocol / planned experiment / experiment outline / hypothesis”。
4. 只输出关键章节替换稿，便于在原稿上做局部更新。
5. 重点升级实验章节和实验大纲，让后续真实实验可以直接按这份文稿执行。

请严格输出以下带标记的内容：

<<<TITLE_START>>>
[新的论文标题，只保留一行，以 # 开头]
<<<TITLE_END>>>

<<<ABSTRACT_START>>>
[新的摘要块，必须从 ## Abstract (摘要) 开始，到摘要结束]
<<<ABSTRACT_END>>>

<<<RELATED_WORK_START>>>
[新的 Section 2，必须从 ## 2. Related Work 开始]
<<<RELATED_WORK_END>>>

<<<METHODOLOGY_START>>>
[新的 Section 3，必须从 ## 3. Methodology 开始；重点修复 BPE 与视频模块表述，并收敛绝对化论断]
<<<METHODOLOGY_END>>>

<<<EXPERIMENTS_START>>>
[新的 Section 4，必须从 ## 4. Experiments 开始；将无数据的绝对结论改成 evaluation plan / hypothesis-driven protocol]
<<<EXPERIMENTS_END>>>

<<<CONCLUSION_APPENDIX_START>>>
[新的 Section 5 和 Appendix 引导块，必须从 ## 5. Conclusion & Limitations 开始，并在块内包含 # 6. Appendix]
<<<CONCLUSION_APPENDIX_END>>>

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
1. Feed `论文大纲_V2.md` to a fresh reviewer and create `Review_Strict_V2.md`
2. Use `Review_Strict_V2.md` plus `Scientist_Memory.md` to generate `论文大纲_V3.md`
3. Keep every version and never overwrite prior outputs
4. Stop when a fresh reviewer gives `4/5` or `5/5`, or after 100 rounds

## What To Optimize First
- Tone down unsupported absolute claims
- Replace combative related-work language with precise, evidence-based comparison
- Tighten the method-to-evaluation loop with explicit ablations and planned experiments
- Add limitations and failure cases before the reviewer asks for them

## Operating Notes
- Re-run the same script with a larger `--max-version` target when you want to continue the loop
- Keep the API key outside the code and provide it via environment variable
- Review `Scientist_Memory.md` occasionally to prevent local overfitting to one reviewer style
"""


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


def strip_references_section(text: str) -> str:
    stripped = re.sub(r"\n## Works cited\s*$.*", "", text, flags=re.DOTALL)
    return stripped.strip() + "\n"


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
    related_work_block: str,
    methodology_block: str,
    experiments_block: str,
    conclusion_appendix_block: str,
) -> str:
    updated = cleanup_meta_text(original_paper)
    updated = replace_title_block(updated, title_block)
    updated = replace_block_between(
        updated,
        r"^## Abstract \(摘要\)\s*$",
        r"^## 1\. Introduction\s*$",
        abstract_block,
    )
    updated = replace_block_between(
        updated,
        r"^## 2\. Related Work\s*$",
        r"^## 3\. Methodology\s*$",
        related_work_block,
    )
    updated = replace_block_between(
        updated,
        r"^## 3\. Methodology\s*$",
        r"^## 4\. Experiments\s*$",
        methodology_block,
    )
    updated = replace_block_between(
        updated,
        r"^## 4\. Experiments\s*$",
        r"^## 5\. Conclusion & Limitations\s*$",
        experiments_block,
    )
    updated = replace_block_between(
        updated,
        r"^## 5\. Conclusion & Limitations\s*$",
        r"^## A\. Implementation Details & Algorithmic Complexity\s*$",
        conclusion_appendix_block,
    )
    return normalize_spacing(updated)


def run_reviewer(api: ApiConfig, version: int, paper_text: str) -> tuple[str, float | None]:
    paper_without_refs = strip_references_section(paper_text)
    response = call_chat(api, reviewer_system_prompt(), build_reviewer_prompt(version, paper_without_refs))
    review_text = extract_block(response, "<<<REVIEW_START>>>", "<<<REVIEW_END>>>")
    score = parse_score(review_text)
    return review_text, score


def run_scientist(
    api: ApiConfig,
    current_version: int,
    next_version: int,
    paper_text: str,
    review_text: str,
    scientist_memory: str,
) -> tuple[str, str, str]:
    paper_without_refs = strip_references_section(paper_text)
    response = call_chat(
        api,
        scientist_system_prompt(),
        build_scientist_compact_prompt(
            current_version,
            next_version,
            paper_without_refs,
            review_text,
            scientist_memory,
        ),
    )
    title_block = extract_block(response, "<<<TITLE_START>>>", "<<<TITLE_END>>>")
    abstract_block = extract_block(response, "<<<ABSTRACT_START>>>", "<<<ABSTRACT_END>>>")
    related_work_block = extract_block(response, "<<<RELATED_WORK_START>>>", "<<<RELATED_WORK_END>>>")
    methodology_block = extract_block(response, "<<<METHODOLOGY_START>>>", "<<<METHODOLOGY_END>>>")
    experiments_block = extract_block(response, "<<<EXPERIMENTS_START>>>", "<<<EXPERIMENTS_END>>>")
    conclusion_appendix_block = extract_block(
        response,
        "<<<CONCLUSION_APPENDIX_START>>>",
        "<<<CONCLUSION_APPENDIX_END>>>",
    )
    revision_log = extract_block(response, "<<<REVISION_LOG_START>>>", "<<<REVISION_LOG_END>>>")
    new_memory = extract_block(response, "<<<SCIENTIST_MEMORY_START>>>", "<<<SCIENTIST_MEMORY_END>>>")

    required_blocks = [
        title_block,
        abstract_block,
        related_work_block,
        methodology_block,
        experiments_block,
        conclusion_appendix_block,
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
        related_work_block=related_work_block,
        methodology_block=methodology_block,
        experiments_block=experiments_block,
        conclusion_appendix_block=conclusion_appendix_block,
    )
    return revised_paper, revision_log, new_memory


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
- The reviewer does not receive the references section, which saves tokens and keeps attention on the core paper

## Scientist Strategy
- The scientist performs targeted upgrades on top of the latest draft
- The scientist must preserve valid innovation and strong structure already present in the paper
- Unsupported claims must be softened instead of defended with fabricated evidence
- Planned but unrun experiments must be clearly marked as planned work
- The scientist should turn reviewer criticism into a stronger experiment outline for future execution
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
    api = ApiConfig(
        api_url=args.api_url,
        api_keys=api_keys,
        model=args.model,
        timeout_seconds=args.timeout_seconds,
    )

    v1_path = freeze_v1(workspace, args.source)
    prompt_doc_path = workspace / PROMPT_DOC
    write_prompt_strategy_doc(prompt_doc_path, api.model, api.api_url)

    state_path = workspace / STATE_FILE
    state = load_state(state_path, workspace / args.source, api)
    append_state_event(state, version=1, paper=str(v1_path), status="frozen")
    save_state(state_path, state)

    scientist_memory_path = workspace / SCIENTIST_MEMORY_FILE
    scientist_memory = read_utf8(scientist_memory_path) if scientist_memory_path.exists() else ""
    latest_score_v1: float | None = None
    latest_score_target: float | None = None
    latest_review_v1 = ""
    latest_review_target = ""
    latest_revision_log_target = ""

    for version in range(1, args.max_version + 1):
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

        if version == 1:
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
        else:
            revised_paper_text, revision_log_text, scientist_memory = run_scientist(
                api=api,
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

    print("Workflow completed successfully.")
    print(f"V1 paper: {v1_path}")
    print(f"Review V1: {review_path(workspace, 1)}")
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
