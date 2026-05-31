#!/usr/bin/env python3
"""
Aggregate Cursor usage-events CSV: by model, by day, top requests by Total Tokens.
Usage: python cursor_usage_aggregate.py [path/to/usage-events.csv]
Default CSV: D:/chromedown/usage-events-2026-03-23.csv
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path


def parse_int(s: str) -> int:
    s = (s or "").strip().strip('"')
    if not s:
        return 0
    return int(s)


def main() -> int:
    default = Path(r"D:\chromedown\usage-events-2026-03-23.csv")
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else default
    if not csv_path.is_file():
        print(f"File not found: {csv_path}", file=sys.stderr)
        return 1

    rows: list[dict[str, str]] = []
    with csv_path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    # Per-model aggregates
    by_model: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "requests": 0,
            "total_tokens": 0,
            "cache_read": 0,
            "input_no_cache": 0,
            "input_cache_write": 0,
            "output": 0,
        }
    )
    # Per-day (UTC date)
    by_day: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "requests": 0,
            "total_tokens": 0,
            "cache_read": 0,
        }
    )

    parsed: list[dict] = []
    for row in rows:
        model = (row.get("Model") or "").strip()
        kind = (row.get("Kind") or "").strip()
        max_mode = (row.get("Max Mode") or "").strip()
        date_s = (row.get("Date") or "").strip().strip('"')
        day = date_s[:10] if len(date_s) >= 10 else date_s

        cw = parse_int(row.get("Input (w/ Cache Write)", "0"))
        nc = parse_int(row.get("Input (w/o Cache Write)", "0"))
        cr = parse_int(row.get("Cache Read", "0"))
        out = parse_int(row.get("Output Tokens", "0"))
        total = parse_int(row.get("Total Tokens", "0"))

        m = by_model[model]
        m["requests"] += 1
        m["total_tokens"] += total
        m["cache_read"] += cr
        m["input_no_cache"] += nc
        m["input_cache_write"] += cw
        m["output"] += out

        d = by_day[day]
        d["requests"] += 1
        d["total_tokens"] += total
        d["cache_read"] += cr

        parsed.append(
            {
                "Date": date_s,
                "day": day,
                "Kind": kind,
                "Model": model,
                "Max Mode": max_mode,
                "Cache Read": cr,
                "Input_nc": nc,
                "Output": out,
                "Total": total,
            }
        )

    # Sort by total tokens descending for top requests
    by_total = sorted(parsed, key=lambda x: x["Total"], reverse=True)
    top_n = 25

    out_lines: list[str] = []
    out_lines.append(f"Cursor usage aggregate: {csv_path}")
    out_lines.append(f"Rows: {len(rows)}")
    out_lines.append("")

    out_lines.append("=== By Model (sorted by total_tokens desc) ===")
    for model, agg in sorted(
        by_model.items(), key=lambda kv: kv[1]["total_tokens"], reverse=True
    ):
        out_lines.append(
            f"{model}\trequests={agg['requests']}\ttotal_tokens={agg['total_tokens']:,}\t"
            f"cache_read={agg['cache_read']:,}\tinput_nc={agg['input_no_cache']:,}\t"
            f"cache_write={agg['input_cache_write']:,}\toutput={agg['output']:,}"
        )
    out_lines.append("")

    out_lines.append("=== By Day UTC (sorted by date) ===")
    for day in sorted(by_day.keys()):
        agg = by_day[day]
        out_lines.append(
            f"{day}\trequests={agg['requests']}\ttotal_tokens={agg['total_tokens']:,}\t"
            f"cache_read={agg['cache_read']:,}"
        )
    out_lines.append("")

    out_lines.append(f"=== Top {top_n} requests by Total Tokens ===")
    out_lines.append("Total\tCacheRead\tInNoCache\tModel\tMaxMode\tKind\tDate")
    for r in by_total[:top_n]:
        out_lines.append(
            f"{r['Total']:,}\t{r['Cache Read']:,}\t{r['Input_nc']:,}\t"
            f"{r['Model']}\t{r['Max Mode']}\t{r['Kind']}\t{r['Date']}"
        )
    out_lines.append("")

    max_mode_yes = sum(1 for r in parsed if r["Max Mode"].lower() == "yes")
    out_lines.append(f"Max Mode Yes count: {max_mode_yes} / {len(parsed)}")
    out_lines.append(
        "Pattern note: large Total with large Cache Read => long thread / repeated context; "
        "large Input_nc => big fresh prompt or tool payload in that turn."
    )

    report = "\n".join(out_lines)
    print(report)

    report_path = Path(__file__).resolve().parent.parent / "cursor_usage_aggregate_report.txt"
    report_path.write_text(report, encoding="utf-8")
    print(f"\nWrote: {report_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
