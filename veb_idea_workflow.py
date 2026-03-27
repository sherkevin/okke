from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from paper_adversarial_workflow import ApiConfig, call_chat, read_utf8, write_utf8

DEFAULT_API_URL = "https://new.lemonapi.site/v1"
DEFAULT_MODEL = "[L]gemini-3.1-pro-preview"
DEFAULT_SEED = "veb_idea_seed.md"
DEFAULT_IDEA_PATTERN = "veb_idea_V{version}.md"
DEFAULT_REVIEW_PATTERN = "veb_review_V{version}.md"
# Primary key; fallback chain: --api-key > LEMON_API_KEY env > this default > backup below.
DEFAULT_LEMON_API_KEY = "sk-36bWqzTxOqCP1HFFf03fFdoktSGItMZUZyxJcZHMTMcOp4rq"
DEFAULT_LEMON_API_KEY_BACKUP = "sk-2gHrK5lYydj4mRl9PqP1NA6Qd9r4RSYPMZYHGRSVJb600HyI"

REVIEWER_SYSTEM = (
    "你是 ACM MM 会议的的首席主审稿人。"
    "以下是我论文的idea，请公正严厉刻薄地评价我的论文研究内容，"
    "尖锐地指出其中所有的缺点和不足，不要任何讨好我的想法，只给出意见和评价，不要有其他内容，"
    "并给出你本次 review 的打分，满分是五分，下面是论文idea："
)

SCIENTIST_SYSTEM = (
    "你是全球顶级 AI 科学家，是 ACM MM 的资深投稿者和多年审稿人。"
    "你需要根据审稿意见对原稿进行针对性精修，不要给出多余回答，直接给出优化后的完整方法论。"
)


@dataclass
class RoundSnapshot:
    """One completed reviewer→scientist leg: review text and the methodology the scientist produced next."""

    review: str
    methodology: str


@dataclass(frozen=True)
class WorkflowLayout:
    idea_pattern: str
    review_pattern: str


def _pattern_to_regex(pattern: str, version: int) -> re.Pattern[str]:
    escaped = re.escape(pattern)
    escaped = escaped.replace(r"\{version\}", str(version))
    return re.compile(rf"^{escaped}$", flags=re.IGNORECASE)


def _resolve_existing_path(workspace: Path, pattern: str, version: int) -> Path | None:
    regex = _pattern_to_regex(pattern, version)
    for child in workspace.iterdir():
        if child.is_file() and regex.match(child.name):
            return child
    return None


def idea_path(workspace: Path, version: int, layout: WorkflowLayout) -> Path:
    return workspace / layout.idea_pattern.format(version=version)


def resolve_idea_path(workspace: Path, version: int, layout: WorkflowLayout) -> Path:
    return _resolve_existing_path(workspace, layout.idea_pattern, version) or idea_path(workspace, version, layout)


def review_path(workspace: Path, version: int, layout: WorkflowLayout) -> Path:
    return workspace / layout.review_pattern.format(version=version)


def resolve_review_path(workspace: Path, version: int, layout: WorkflowLayout) -> Path:
    return _resolve_existing_path(workspace, layout.review_pattern, version) or review_path(workspace, version, layout)


def parse_review_score(text: str) -> float | None:
    patterns = [
        r"Score:\s*([0-5](?:\.\d+)?)\s*/\s*5",
        r"打分[：:]\s*([0-5](?:\.\d+)?)\s*(?:分|/|\s|$)",
        r"评分[：:]\s*([0-5](?:\.\d+)?)",
        r"([0-5](?:\.\d+)?)\s*/\s*5\s*分",
    ]
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            return float(m.group(1))
    return None


def strip_optional_fence(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        lines = t.split("\n")
        if len(lines) >= 2 and lines[0].startswith("```"):
            end = None
            for i in range(1, len(lines)):
                if lines[i].strip() == "```":
                    end = i
                    break
            if end is not None:
                return "\n".join(lines[1:end]).strip()
    return t


def build_reviewer_user(idea_text: str) -> str:
    return idea_text.strip()


def build_scientist_user(
    history: list[RoundSnapshot],
    current_review: str,
    current_methodology: str,
) -> str:
    parts: list[str] = []
    if history:
        parts.append("## 近三轮历史对话（供你保持上下文，按时间顺序）\n")
        for i, snap in enumerate(history[-3:], start=1):
            parts.append(f"### 历史第{i}轮\n")
            parts.append("**审稿意见：**\n")
            parts.append(snap.review.strip())
            parts.append("\n\n**该轮之后修订得到的完整方法论：**\n")
            parts.append(snap.methodology.strip())
            parts.append("\n\n")
    parts.append("## 本轮审稿意见\n")
    parts.append(current_review.strip())
    parts.append("\n\n## 当前待修订的完整方法论\n")
    parts.append(current_methodology.strip())
    parts.append(
        "\n\n---\n请直接输出**优化后的完整方法论**正文（Markdown），不要前言、不要后记、不要解释。"
    )
    return "".join(parts)


def run_reviewer(api: ApiConfig, idea_text: str) -> str:
    """Single-shot review: each call uses only system+user with current idea (no chat history)."""
    return call_chat(api, REVIEWER_SYSTEM, build_reviewer_user(idea_text))


def run_scientist(
    api: ApiConfig,
    history: list[RoundSnapshot],
    current_review: str,
    current_methodology: str,
) -> str:
    user = build_scientist_user(history, current_review, current_methodology)
    raw = call_chat(api, SCIENTIST_SYSTEM, user)
    return strip_optional_fence(raw)


def history_for_current_version(workspace: Path, current_version: int, layout: WorkflowLayout) -> list[RoundSnapshot]:
    """Completed legs: (review for Vi, methodology V{i+1}) for i < current_version; keep last 3."""
    snaps: list[RoundSnapshot] = []
    for i in range(1, current_version):
        rp = resolve_review_path(workspace, i, layout)
        nxt = resolve_idea_path(workspace, i + 1, layout)
        if rp.exists() and nxt.exists():
            snaps.append(RoundSnapshot(read_utf8(rp), read_utf8(nxt)))
    return snaps[-3:]


def resolve_source_path(workspace: Path, source_name: str) -> Path:
    source = Path(source_name)
    if source.is_absolute():
        return source
    return workspace / source


def ensure_v1_from_seed(workspace: Path, source_name: str, layout: WorkflowLayout) -> None:
    v1 = resolve_idea_path(workspace, 1, layout)
    if v1.exists():
        return
    src = resolve_source_path(workspace, source_name)
    if not src.exists():
        raise FileNotFoundError(f"Missing seed idea file: {src}. Create it or pass --source.")
    write_utf8(idea_path(workspace, 1, layout), read_utf8(src))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="VEB idea loop: stateless reviewer + scientist with last-3-round memory."
    )
    parser.add_argument("--workspace", default=".", help="Workspace directory")
    parser.add_argument(
        "--source",
        default=DEFAULT_SEED,
        help=f"Seed markdown for V1 if veb_idea_V1.md is missing (default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--idea-pattern",
        default=DEFAULT_IDEA_PATTERN,
        help="Idea filename pattern, e.g. 'veb_idea_V{version}.md' or 'idea_v{version}.md'",
    )
    parser.add_argument(
        "--review-pattern",
        default=DEFAULT_REVIEW_PATTERN,
        help="Review filename pattern, e.g. 'veb_review_V{version}.md' or 'idea_review_v{version}.md'",
    )
    parser.add_argument("--start-version", type=int, default=1, help="First idea version to process")
    parser.add_argument(
        "--max-version",
        type=int,
        default=3,
        help="Last idea version to reach (each step: review Vv then scientist -> V(v+1))",
    )
    parser.add_argument("--api-url", default=DEFAULT_API_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api-key", default="", help="Or set LEMON_API_KEY")
    parser.add_argument("--backup-api-key", default="")
    parser.add_argument("--timeout-seconds", type=int, default=300)
    args = parser.parse_args()

    api_key = (
        args.api_key.strip()
        or os.environ.get("LEMON_API_KEY", "").strip()
        or DEFAULT_LEMON_API_KEY
    )
    backup = (
        args.backup_api_key
        or os.environ.get("LEMON_API_KEY_BACKUP", "")
        or DEFAULT_LEMON_API_KEY_BACKUP
    ).strip()
    keys = [api_key]
    if backup and backup != api_key:
        keys.append(backup)

    workspace = Path(args.workspace).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    layout = WorkflowLayout(
        idea_pattern=args.idea_pattern,
        review_pattern=args.review_pattern,
    )

    if args.max_version < 2:
        raise SystemExit("--max-version must be at least 2.")
    if args.start_version < 1 or args.start_version > args.max_version:
        raise SystemExit("--start-version must be in [1, max-version].")

    if args.start_version == 1:
        ensure_v1_from_seed(workspace, args.source, layout)

    api = ApiConfig(
        api_url=args.api_url,
        api_keys=keys,
        model=args.model,
        timeout_seconds=args.timeout_seconds,
    )

    for v in range(args.start_version, args.max_version + 1):
        ip = resolve_idea_path(workspace, v, layout)
        if not ip.exists():
            raise FileNotFoundError(f"Missing {ip}")

        idea_text = read_utf8(ip)
        rp = resolve_review_path(workspace, v, layout)

        if rp.exists():
            review_text = read_utf8(rp)
        else:
            print(f"[reviewer] idea V{v} …")
            review_text = run_reviewer(api, idea_text)
            write_utf8(review_path(workspace, v, layout), review_text)
            score = parse_review_score(review_text)
            print(
                f"[reviewer] wrote {review_path(workspace, v, layout).name}"
                + (f" score={score}/5" if score is not None else "")
            )

        if v >= args.max_version:
            break

        next_p = resolve_idea_path(workspace, v + 1, layout)
        if next_p.exists():
            print(f"[scientist] skip existing {next_p.name}")
            continue

        hist = history_for_current_version(workspace, v, layout)
        print(f"[scientist] V{v} -> V{v + 1} (history_rounds={len(hist)}) …")
        new_idea = run_scientist(api, hist, review_text, idea_text)
        write_utf8(idea_path(workspace, v + 1, layout), new_idea)
        print(f"[scientist] wrote {idea_path(workspace, v + 1, layout).name}")

    print("Done.")
    print(f"Last idea: {idea_path(workspace, args.max_version, layout)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
