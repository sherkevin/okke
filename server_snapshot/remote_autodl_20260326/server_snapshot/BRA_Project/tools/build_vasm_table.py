#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bra_vasm import save_vasm_table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, help="Tokenizer/model path")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    out = save_vasm_table(tokenizer, args.output)
    print(f"Saved VASM table to {out}")


if __name__ == "__main__":
    main()
