#!/usr/bin/env python3

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.eval_utils import append_jsonl, load_completed_ids, load_jsonl

try:
    import openai
except ImportError:
    openai = None

INPUT_COST_PER_1M = 2.50
OUTPUT_COST_PER_1M = 10.0

PROMPT_TEMPLATE = """You are an AI assistant evaluating the quality of a model's response to a visual question.

Question: {question}
Ground Truth Answer: {gt_answer}
Model's Response: {prediction}

Please evaluate the model's response by comparing it with the ground truth. Consider:
1. Factual accuracy: Does the response contain hallucinated objects or incorrect facts?
2. Completeness: Does it capture the key information from the ground truth?
3. Relevance: Is the response relevant to the question?

Rate the response on a scale of 1-6:
6: Completely accurate and comprehensive
5: Mostly accurate with minor omissions
4: Partially accurate, captures main points
3: Some correct elements but significant errors or hallucinations
2: Mostly incorrect with major hallucinations
1: Completely wrong or entirely hallucinated

Return your evaluation as JSON: {{"score": <int 1-6>, "reason": "<brief explanation>"}}"""


def extract_json(text: str) -> dict:
    text = text.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if match:
        text = match.group(1).strip()
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        return json.loads(match.group(0))
    raise ValueError(f"No JSON object found in: {text[:200]}")


def call_judge(client, model: str, prompt: str, max_retries: int = 3) -> tuple[dict, int, int]:
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            content = resp.choices[0].message.content
            parsed = extract_json(content)
            score = int(parsed.get("score", 0))
            if not 1 <= score <= 6:
                score = max(1, min(6, score))
            reason = parsed.get("reason", "")
            inp = resp.usage.prompt_tokens if resp.usage else 0
            out = resp.usage.completion_tokens if resp.usage else 0
            return {"score": score, "reason": reason}, inp, out
        except (openai.APIError, openai.APIConnectionError, openai.RateLimitError) as e:
            if attempt == max_retries - 1:
                raise
            wait = 2 ** attempt
            time.sleep(wait)
        except (json.JSONDecodeError, ValueError) as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("Unexpected retry exhaustion")


def main():
    p = argparse.ArgumentParser(description="MMHal-Bench GPT-4 Judge Evaluation")
    p.add_argument("--predictions", type=str, required=True, help="JSONL predictions file")
    p.add_argument("--output", type=str, required=True, help="Output JSONL for scored results")
    p.add_argument("--judge_model", type=str, default="gpt-4o")
    p.add_argument("--limit", type=int, default=0, help="Score only first N samples (0=all)")
    p.add_argument("--dry_run", action="store_true", help="Print first prompt without API call")
    args = p.parse_args()

    key = os.environ.get("OPENAI_API_KEY")
    if not key and not args.dry_run:
        print("Error: OPENAI_API_KEY environment variable not set")
        sys.exit(1)
    if openai is None and not args.dry_run:
        print("Error: openai package not installed (pip install openai)")
        sys.exit(1)

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    completed = load_completed_ids(args.output)
    if completed:
        print(f"[Resume] Found {len(completed)} completed — skipping.")

    samples = load_jsonl(args.predictions)
    if args.limit > 0:
        samples = samples[: args.limit]
    pending = [s for s in samples if s.get("question_id") not in completed]
    print(f"[Dataset] Total={len(samples)} | Pending={len(pending)}")

    if args.dry_run:
        if pending:
            s = pending[0]
            prompt = PROMPT_TEMPLATE.format(
                question=s.get("question", ""),
                gt_answer=s.get("gt_answer", ""),
                prediction=s.get("prediction", ""),
            )
            print("--- First prompt (dry run) ---")
            print(prompt)
            print("--- End prompt ---")
        return

    if not pending:
        print("[Done] All samples scored.")
        all_results = load_jsonl(args.output)
        _print_summary(all_results, 0, 0)
        return

    client = openai.OpenAI(api_key=key)
    total_in, total_out = 0, 0

    for idx, sample in enumerate(pending):
        qid = sample.get("question_id", "")
        prompt = PROMPT_TEMPLATE.format(
            question=sample.get("question", ""),
            gt_answer=sample.get("gt_answer", ""),
            prediction=sample.get("prediction", ""),
        )
        result, inp, out = call_judge(client, args.judge_model, prompt)
        total_in += inp
        total_out += out

        record = dict(sample)
        record["gpt4_score"] = result["score"]
        record["gpt4_reason"] = result["reason"]
        append_jsonl(args.output, record)

        if (idx + 1) % 10 == 0 or (idx + 1) == len(pending):
            print(f"[{len(completed) + idx + 1}/{len(samples)}] qid={qid} score={result['score']}")

    all_results = load_jsonl(args.output)
    _print_summary(all_results, total_in, total_out)


def _print_summary(results: list[dict], total_in: int, total_out: int):
    if not results:
        return
    scores = [r["gpt4_score"] for r in results if "gpt4_score" in r]
    mean_score = sum(scores) / max(len(scores), 1)
    print("\n" + "=" * 60)
    print(f"Mean score: {mean_score:.2f} (n={len(scores)})")

    type_scores: dict[str, list[int]] = {}
    for r in results:
        qt = r.get("question_type", "unknown")
        if "gpt4_score" in r:
            type_scores.setdefault(qt, []).append(r["gpt4_score"])
    per_type = {qt: sum(s) / max(len(s), 1) for qt, s in type_scores.items()}
    print("Per-type mean:")
    for qt, avg in sorted(per_type.items()):
        print(f"  {qt}: {avg:.2f} (n={len(type_scores[qt])})")

    cost = (total_in / 1e6 * INPUT_COST_PER_1M) + (total_out / 1e6 * OUTPUT_COST_PER_1M)
    print(f"\nTokens: input={total_in}, output={total_out}")
    print(f"Estimated cost: ${cost:.4f}")


if __name__ == "__main__":
    main()
