#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def append_tsv_row(path: str | Path, header: list[str], row: dict[str, Any]) -> None:
    tsv_path = Path(path)
    tsv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = tsv_path.exists()
    with open(tsv_path, "a", encoding="utf-8") as fh:
        if not exists:
            fh.write("\t".join(header) + "\n")
        fh.write("\t".join(str(row.get(col, "")) for col in header) + "\n")


def append_jsonl(path: str | Path, payload: dict[str, Any]) -> None:
    jsonl_path = Path(path)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with open(jsonl_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
