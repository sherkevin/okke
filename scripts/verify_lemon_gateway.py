#!/usr/bin/env python3
"""Verify Lemon OpenAI-compatible gateway + optional paper adversarial workflow.

Reads API key from (first match):
  - --api-key
  - LEMON_API_KEY environment variable
  - first line of file from --api-key-file

Exit codes: 0 success, 1 probe/workflow failure, 2 missing key, 3 argparse error.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_API_URL = "https://new.lemonapi.site/v1"
DEFAULT_MODEL = "[L]gemini-3.1-pro-preview"

# Minimal paper so build_reviewer_excerpt / scientist excerpt paths succeed.
MINIMAL_PAPER = """# Test Paper for Gateway Smoke

## Abstract

Short abstract for connectivity test.

## 1. Introduction

Brief introduction.

## 3. Methodology

Minimal methodology.

## 4. Experiments

Planned experiments only.

## 5. Conclusion & Limitations

Limitations noted.

## 6. Limitations

Extra limitations for excerpt parsing.

## Works cited

- Placeholder reference.
"""


def load_api_key(args: argparse.Namespace) -> str:
    if args.api_key.strip():
        return args.api_key.strip()
    env = os.environ.get("LEMON_API_KEY", "").strip()
    if env:
        return env
    if args.api_key_file:
        p = Path(args.api_key_file)
        if not p.is_file():
            raise SystemExit(f"api-key-file not found: {p}")
        line = p.read_text(encoding="utf-8").splitlines()[0].strip()
        if not line:
            raise SystemExit("api-key-file is empty")
        return line
    return ""


def probe_models(api_url: str, api_key: str, timeout: int) -> int:
    req = urllib.request.Request(
        f"{api_url.rstrip('/')}/models",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as response:
        body = json.loads(response.read().decode("utf-8"))
    n = len(body.get("data", []))
    elapsed = time.time() - t0
    print(f"[models] OK count={n} elapsed_sec={elapsed:.2f} http={response.status}")
    return n


def probe_chat(api_url: str, api_key: str, model: str, timeout: int) -> None:
    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": "You reply with exactly: OK"},
            {"role": "user", "content": "ping"},
        ],
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{api_url.rstrip('/')}/chat/completions",
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Connection": "close",
        },
        method="POST",
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as response:
        body = json.loads(response.read().decode("utf-8"))
    elapsed = time.time() - t0
    try:
        content = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Unexpected chat response: {json.dumps(body, ensure_ascii=False)[:800]}") from exc
    preview = (content or "").strip().replace("\n", " ")[:120]
    print(f"[chat] OK elapsed_sec={elapsed:.2f} preview={preview!r}")


def run_e2e_mini(repo_root: Path, api_url: str, model: str, api_key: str, timeout: int) -> None:
    workflow = repo_root / "paper_adversarial_workflow.py"
    if not workflow.is_file():
        raise SystemExit(f"paper_adversarial_workflow.py not found: {workflow}")
    with tempfile.TemporaryDirectory(prefix="lemon_e2e_") as tmp:
        ws = Path(tmp)
        (ws / "论文大纲.md").write_text(MINIMAL_PAPER, encoding="utf-8")
        cmd = [
            sys.executable,
            str(workflow),
            "--workspace",
            str(ws),
            "--source",
            "论文大纲.md",
            "--start-version",
            "1",
            "--max-version",
            "2",
            "--api-url",
            api_url,
            "--model",
            model,
            "--api-key",
            api_key,
            "--timeout-seconds",
            str(timeout),
        ]
        print("[e2e] running:", " ".join(cmd[:6]), "...")
        proc = subprocess.run(cmd, cwd=str(repo_root), capture_output=True, text=True, timeout=timeout * 20)
        out = (proc.stdout or "") + (proc.stderr or "")
        if proc.returncode != 0:
            raise SystemExit(f"[e2e] workflow failed rc={proc.returncode}\n{out[-4000:]}")
        print("[e2e] OK workflow completed")
        v2 = ws / "论文大纲_V2.md"
        r1 = ws / "Review_Strict_V1.md"
        if not v2.is_file() or not r1.is_file():
            raise SystemExit(f"[e2e] missing outputs v2={v2.exists()} review={r1.exists()}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify Lemon API gateway and optional workflow.")
    parser.add_argument("--api-url", default=os.environ.get("LEMON_API_URL", DEFAULT_API_URL))
    parser.add_argument("--model", default=os.environ.get("LEMON_MODEL", DEFAULT_MODEL))
    parser.add_argument("--api-key", default="", help="Primary key (else env or --api-key-file)")
    parser.add_argument("--api-key-file", default="", help="Path to text file; first line = key")
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument("--skip-chat", action="store_true")
    parser.add_argument("--e2e-mini", action="store_true", help="Run paper_adversarial_workflow V1->V2 in a temp dir")
    args = parser.parse_args()

    key = load_api_key(args)
    if not key:
        print(
            "Missing API key. Set LEMON_API_KEY, pass --api-key, or --api-key-file.",
            file=sys.stderr,
        )
        return 2

    try:
        probe_models(args.api_url, key, args.timeout_seconds)
        if not args.skip_chat:
            probe_chat(args.api_url, key, args.model, args.timeout_seconds)
        if args.e2e_mini:
            repo = Path(__file__).resolve().parent.parent
            run_e2e_mini(repo, args.api_url, args.model, key, args.timeout_seconds)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:800]
        print(f"HTTP {exc.code}: {detail}", file=sys.stderr)
        return 1
    except (urllib.error.URLError, RuntimeError, OSError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
