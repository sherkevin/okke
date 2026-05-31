from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TOKEN_PATTERNS = [
    (re.compile(r"sk-[A-Za-z0-9_-]{20,}"), "sk-REDACTED_OPENAI_TOKEN"),
    (re.compile(r"github_pat_[A-Za-z0-9_]{20,}"), "github_pat_REDACTED_TOKEN"),
    (re.compile(r"gh[pousr]_[A-Za-z0-9_]{20,}"), "gh_REDACTED_TOKEN"),
]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def redact_text(text: str) -> str:
    redacted = text
    for pattern, replacement in TOKEN_PATTERNS:
        redacted = pattern.sub(replacement, redacted)
    return redacted


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                records.append(
                    {
                        "timestamp": None,
                        "type": "parse_error",
                        "line_number": line_number,
                        "preview": redact_text(line[:500]),
                    }
                )
                continue
            record["_line_number"] = line_number
            records.append(record)
    return records


def compact_text(text: str, limit: int = 2400) -> dict[str, Any]:
    text = redact_text(text or "")
    encoded = text.encode("utf-8", errors="replace")
    if len(text) <= limit:
        preview = text
    else:
        head = text[: limit // 2]
        tail = text[-limit // 2 :]
        preview = f"{head}\n\n...[truncated {len(text) - limit} chars; sha256={sha256_bytes(encoded)}]...\n\n{tail}"
    return {
        "text": preview,
        "chars": len(text),
        "lines": text.count("\n") + (1 if text else 0),
        "sha256": sha256_bytes(encoded),
        "truncated": len(text) > limit,
    }


def content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        for key in ("text", "input_text", "output_text"):
            value = content.get(key)
            if isinstance(value, str):
                return value
        return json.dumps(content, ensure_ascii=False, sort_keys=True)
    if isinstance(content, list):
        chunks: list[str] = []
        for part in content:
            if isinstance(part, dict):
                for key in ("text", "input_text", "output_text"):
                    value = part.get(key)
                    if isinstance(value, str):
                        chunks.append(value)
                        break
                else:
                    chunks.append(json.dumps(part, ensure_ascii=False, sort_keys=True))
            else:
                chunks.append(str(part))
        return "\n".join(chunks)
    return str(content)


def compact_event(record: dict[str, Any]) -> tuple[dict[str, Any], dict[str, int]]:
    stats = {
        "encrypted_reasoning_omitted": 0,
        "tool_outputs_summarized": 0,
        "custom_outputs_summarized": 0,
        "unknown_payloads_summarized": 0,
    }
    event_type = record.get("type")
    payload = record.get("payload") if isinstance(record.get("payload"), dict) else {}
    compact: dict[str, Any] = {
        "timestamp": record.get("timestamp"),
        "type": event_type,
        "line_number": record.get("_line_number"),
    }

    if event_type == "session_meta":
        compact["session"] = {
            key: payload.get(key)
            for key in ("id", "timestamp", "cwd", "originator", "cli_version")
            if key in payload
        }
        return compact, stats

    if event_type == "turn_context":
        compact["context_keys"] = sorted(payload.keys())
        compact["payload_sha256"] = sha256_bytes(
            json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8", errors="replace")
        )
        if payload.get("cwd"):
            compact["cwd"] = payload.get("cwd")
        return compact, stats

    if event_type == "response_item":
        item_type = payload.get("type")
        compact["item_type"] = item_type
        if payload.get("call_id"):
            compact["call_id"] = payload.get("call_id")
        if item_type == "message":
            compact["role"] = payload.get("role")
            compact["content"] = compact_text(content_to_text(payload.get("content")), limit=6000)
        elif item_type == "function_call":
            compact["name"] = payload.get("name")
            compact["arguments"] = compact_text(str(payload.get("arguments", "")), limit=3500)
        elif item_type == "function_call_output":
            compact["output"] = compact_text(str(payload.get("output", "")), limit=3500)
            stats["tool_outputs_summarized"] += 1
        elif item_type == "custom_tool_call":
            compact["name"] = payload.get("name")
            compact["input"] = compact_text(str(payload.get("input", "")), limit=3500)
        elif item_type == "custom_tool_call_output":
            compact["output"] = compact_text(str(payload.get("output", "")), limit=3500)
            stats["custom_outputs_summarized"] += 1
        elif item_type == "reasoning":
            compact["summary"] = payload.get("summary") or []
            compact["omitted"] = ["encrypted_content"]
            if payload.get("encrypted_content"):
                stats["encrypted_reasoning_omitted"] += 1
        else:
            safe_payload = {
                key: value
                for key, value in payload.items()
                if key not in {"encrypted_content", "content", "output", "input", "arguments"}
            }
            compact["payload_summary"] = safe_payload
            compact["payload_sha256"] = sha256_bytes(
                json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8", errors="replace")
            )
            stats["unknown_payloads_summarized"] += 1
        return compact, stats

    if event_type == "event_msg":
        compact["event"] = {
            key: payload.get(key)
            for key in ("type", "status", "call_id", "turn_id", "name")
            if key in payload
        }
        for key in ("stdout", "stderr"):
            if isinstance(payload.get(key), str) and payload.get(key):
                compact[key] = compact_text(payload[key], limit=2400)
                stats["tool_outputs_summarized"] += 1
        if isinstance(payload.get("changes"), dict):
            compact["changed_files"] = sorted(payload["changes"].keys())
        return compact, stats

    if event_type == "parse_error":
        compact["preview"] = record.get("preview")
        return compact, stats

    compact["payload_keys"] = sorted(payload.keys())
    compact["payload_sha256"] = sha256_bytes(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8", errors="replace")
    )
    stats["unknown_payloads_summarized"] += 1
    return compact, stats


def dedupe_key(compact: dict[str, Any]) -> str:
    comparable = {key: value for key, value in compact.items() if key not in {"timestamp", "line_number"}}
    data = json.dumps(comparable, ensure_ascii=False, sort_keys=True)
    return sha256_bytes(data.encode("utf-8", errors="replace"))


def render_transcript(session_id: str, summary: dict[str, Any], compact_events: list[dict[str, Any]]) -> str:
    lines = [
        f"# Clean Transcript: {session_id}",
        "",
        f"- Generated at UTC: {summary['generated_at_utc']}",
        f"- Source file: `{summary['source_file']}`",
        f"- Source SHA256: `{summary['source_sha256']}`",
        f"- Source bytes: {summary['source_bytes']}",
        f"- Compact events: {summary['compact_events']} / original events {summary['original_events']}",
        f"- Duplicates removed: {summary['duplicates_removed']}",
        f"- Encrypted reasoning payloads omitted: {summary['encrypted_reasoning_omitted']}",
        f"- Tool/custom outputs summarized: {summary['tool_outputs_summarized'] + summary['custom_outputs_summarized']}",
        "",
        "This transcript is a cleaned reading layer. The complete redacted raw jsonl remains under `codex_chat_history/redacted_raw`.",
        "",
    ]

    for event in compact_events:
        timestamp = event.get("timestamp") or "no-timestamp"
        if event.get("type") == "session_meta":
            lines.extend([f"## {timestamp} Session", "", f"```json\n{json.dumps(event.get('session', {}), ensure_ascii=False, indent=2)}\n```", ""])
            continue
        if event.get("type") == "response_item" and event.get("item_type") == "message":
            role = (event.get("role") or "message").upper()
            text = event.get("content", {}).get("text", "")
            lines.extend([f"## {timestamp} {role}", "", text, ""])
            continue
        if event.get("type") == "response_item" and event.get("item_type") == "function_call":
            name = event.get("name") or "tool"
            args = event.get("arguments", {}).get("text", "")
            lines.extend([f"## {timestamp} TOOL CALL `{name}`", "", "```json", args, "```", ""])
            continue
        if event.get("type") == "response_item" and event.get("item_type") == "function_call_output":
            output = event.get("output", {})
            lines.extend(
                [
                    f"## {timestamp} TOOL OUTPUT `{event.get('call_id', '')}`",
                    "",
                    f"- chars: {output.get('chars')}",
                    f"- lines: {output.get('lines')}",
                    f"- sha256: `{output.get('sha256')}`",
                    "",
                    "```text",
                    output.get("text", ""),
                    "```",
                    "",
                ]
            )
            continue
        if event.get("type") == "response_item" and event.get("item_type") == "custom_tool_call":
            name = event.get("name") or "custom_tool"
            text = event.get("input", {}).get("text", "")
            lines.extend([f"## {timestamp} CUSTOM TOOL `{name}`", "", "```text", text, "```", ""])
            continue
        if event.get("type") == "event_msg":
            event_name = event.get("event", {}).get("type") or "event"
            details = json.dumps(event.get("event", {}), ensure_ascii=False, sort_keys=True)
            lines.extend([f"## {timestamp} EVENT `{event_name}`", "", f"`{details}`", ""])
            continue
    return "\n".join(lines).rstrip() + "\n"


def clean_session(session_dir: Path, output_root: Path) -> dict[str, Any] | None:
    jsonl_files = sorted(session_dir.glob("*.jsonl"))
    if not jsonl_files:
        return None
    source = jsonl_files[0]
    session_id = session_dir.name
    records = read_jsonl(source)
    source_bytes = source.read_bytes()
    compact_events: list[dict[str, Any]] = []
    seen: set[str] = set()
    totals = {
        "encrypted_reasoning_omitted": 0,
        "tool_outputs_summarized": 0,
        "custom_outputs_summarized": 0,
        "unknown_payloads_summarized": 0,
    }
    duplicates_removed = 0

    for record in records:
        compact, stats = compact_event(record)
        for key, value in stats.items():
            totals[key] += value
        key = dedupe_key(compact)
        if key in seen:
            duplicates_removed += 1
            continue
        seen.add(key)
        compact_events.append(compact)

    output_dir = output_root / session_id
    output_dir.mkdir(parents=True, exist_ok=True)
    compact_path = output_dir / "compact_events.jsonl"
    transcript_path = output_dir / "transcript.md"
    summary_path = output_dir / "summary.json"

    with compact_path.open("w", encoding="utf-8", newline="\n") as handle:
        for event in compact_events:
            handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")

    summary: dict[str, Any] = {
        "session_id": session_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_file": str(source),
        "source_bytes": len(source_bytes),
        "source_sha256": sha256_bytes(source_bytes),
        "original_events": len(records),
        "compact_events": len(compact_events),
        "duplicates_removed": duplicates_removed,
        "compact_events_bytes": compact_path.stat().st_size,
        "transcript_file": str(transcript_path),
        **totals,
    }
    transcript_path.write_text(render_transcript(session_id, summary, compact_events), encoding="utf-8", newline="\n")
    summary["transcript_bytes"] = transcript_path.stat().st_size
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Create cleaned, deduplicated Codex chat-history artifacts.")
    parser.add_argument("--raw-root", default="codex_chat_history/redacted_raw")
    parser.add_argument("--out-root", default="codex_chat_history/cleaned")
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    out_root = Path(args.out_root)
    if not raw_root.exists():
        raise SystemExit(f"raw root does not exist: {raw_root}")
    out_root.mkdir(parents=True, exist_ok=True)

    summaries = []
    for session_dir in sorted(path for path in raw_root.iterdir() if path.is_dir()):
        summary = clean_session(session_dir, out_root)
        if summary:
            summaries.append(summary)

    index_lines = [
        "# Cleaned Codex Chat History",
        "",
        "Use these files for fast handoff. The complete redacted raw jsonl remains in `../redacted_raw`.",
        "",
        "| Session | Original events | Compact events | Duplicates removed | Raw bytes | Compact bytes | Transcript bytes |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        sid = summary["session_id"]
        index_lines.append(
            f"| `{sid}` | {summary['original_events']} | {summary['compact_events']} | "
            f"{summary['duplicates_removed']} | {summary['source_bytes']} | "
            f"{summary['compact_events_bytes']} | {summary['transcript_bytes']} |"
        )
    index_lines.append("")
    index_lines.append("Coverage note: cleaned files summarize long tool outputs and omit encrypted reasoning blobs; full redacted raw logs are kept for exact reconstruction.")
    (out_root / "README.md").write_text("\n".join(index_lines) + "\n", encoding="utf-8", newline="\n")
    (out_root / "index.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"cleaned_sessions={len(summaries)}")
    for summary in summaries:
        print(
            f"{summary['session_id']}\toriginal={summary['original_events']}\t"
            f"compact={summary['compact_events']}\tduplicates={summary['duplicates_removed']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
