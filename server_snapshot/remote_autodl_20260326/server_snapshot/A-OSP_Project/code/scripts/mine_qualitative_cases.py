"""
A-OSP Qualitative Case Miner — 成功/失败案例挖掘机
====================================================
自动对比 Base 与 A-OSP 的推理结果，挖掘论文正文和附录所需的定性案例：

  ✦ Success Case (附录 B)：Base 幻觉 + A-OSP 纠正，且 AGL 未缩短
  ✦ Failure Case (附录 D)：Base 正确 + A-OSP 误杀（过度正则化）

同时支持 POPE (Yes/No 判别式) 和 MMHal-Bench (开放式生成) 两种格式。

输出：
  1. 终端摘要
  2. Markdown 文件（直接可贴入论文）
  3. JSON 结构化数据

用法：
  python mine_qualitative_cases.py \
      --base_pope  logs/eval_results/base_pope_50_real_results.jsonl \
      --aosp_pope  logs/eval_results/aosp_pope_50_real_results.jsonl \
      --base_mmhal logs/eval_results/base_mmhal_50_results.jsonl \
      --aosp_mmhal logs/eval_results/aosp_mmhal_50_results.jsonl \
      --output_dir logs/qualitative_cases
"""

import argparse
import json
import re
from pathlib import Path
from typing import Optional


PROJECT_ROOT = Path("/root/autodl-tmp/A-OSP_Project")


def load_jsonl(path: str) -> list:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def normalize_yesno(text: str) -> Optional[str]:
    """将 POPE 的自由文本回答归一化为 'yes' / 'no' / None。"""
    t = text.strip().lower().rstrip(".")
    if t in ("yes", "yes,", "yes."):
        return "yes"
    if t in ("no", "no,", "no."):
        return "no"
    if t.startswith("yes"):
        return "yes"
    if t.startswith("no"):
        return "no"
    return None


# ────────────────────────────────────────────────────────────────
# POPE 案例挖掘
# ────────────────────────────────────────────────────────────────

def mine_pope_cases(base_records: list, aosp_records: list) -> dict:
    """
    对比 POPE 判别式结果。
    Success: base 错 + aosp 对
    Failure: base 对 + aosp 错
    """
    base_map = {r["question_id"]: r for r in base_records}
    aosp_map = {r["question_id"]: r for r in aosp_records}

    common_ids = sorted(set(base_map) & set(aosp_map))

    success, failure, both_correct, both_wrong = [], [], [], []

    for qid in common_ids:
        b = base_map[qid]
        a = aosp_map[qid]

        gt = b["ground_truth"].strip().lower()
        b_pred = normalize_yesno(b["prediction"])
        a_pred = normalize_yesno(a["prediction"])

        b_correct = (b_pred == gt)
        a_correct = (a_pred == gt)

        entry = {
            "question_id": qid,
            "image": b.get("image", ""),
            "question": b["question"],
            "ground_truth": gt,
            "base_prediction": b["prediction"],
            "aosp_prediction": a["prediction"],
            "base_correct": b_correct,
            "aosp_correct": a_correct,
            "base_agl": b.get("generation_length", 0),
            "aosp_agl": a.get("generation_length", 0),
            "aosp_interventions": a.get("intervention_count", 0),
        }

        if not b_correct and a_correct:
            success.append(entry)
        elif b_correct and not a_correct:
            failure.append(entry)
        elif b_correct and a_correct:
            both_correct.append(entry)
        else:
            both_wrong.append(entry)

    return {
        "benchmark": "POPE",
        "total": len(common_ids),
        "both_correct": len(both_correct),
        "both_wrong": len(both_wrong),
        "success": success,
        "failure": failure,
    }


# ────────────────────────────────────────────────────────────────
# MMHal-Bench 案例挖掘
# ────────────────────────────────────────────────────────────────

def semantic_match_score(prediction: str, gt_answer: str) -> float:
    """
    简单的词袋语义匹配分数。
    提取 GT 中的关键实体词，计算在 prediction 中出现的比例。
    """
    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "in", "on", "at",
        "to", "of", "and", "or", "it", "its", "this", "that", "with",
        "for", "from", "by", "has", "have", "had", "be", "been", "being",
        "image", "picture", "photo", "shows", "depicts", "there",
    }

    def extract_keywords(text):
        words = re.findall(r"[a-zA-Z]+", text.lower())
        return set(w for w in words if w not in stop_words and len(w) > 2)

    gt_kw = extract_keywords(gt_answer)
    if not gt_kw:
        return 0.5

    pred_kw = extract_keywords(prediction)
    overlap = gt_kw & pred_kw
    return len(overlap) / len(gt_kw)


def mine_mmhal_cases(base_records: list, aosp_records: list) -> dict:
    """
    对比 MMHal-Bench 开放式生成结果。
    Success: base 语义匹配差 + aosp 语义匹配好 + aosp AGL 未显著缩短
    Failure: base 语义匹配好 + aosp 语义匹配差
    """
    base_map = {r["question_id"]: r for r in base_records}
    aosp_map = {r["question_id"]: r for r in aosp_records}

    common_ids = sorted(set(base_map) & set(aosp_map))

    success, failure, neutral = [], [], []

    for qid in common_ids:
        b = base_map[qid]
        a = aosp_map[qid]

        gt = b.get("gt_answer", "")
        b_score = semantic_match_score(b["prediction"], gt)
        a_score = semantic_match_score(a["prediction"], gt)

        b_agl = b.get("generation_length", len(b["prediction"].split()))
        a_agl = a.get("generation_length", len(a["prediction"].split()))
        agl_ratio = a_agl / max(b_agl, 1)

        entry = {
            "question_id": qid,
            "image_id": b.get("image_id", ""),
            "question": b["question"],
            "question_type": b.get("question_type", ""),
            "question_topic": b.get("question_topic", ""),
            "gt_answer": gt,
            "base_prediction": b["prediction"],
            "aosp_prediction": a["prediction"],
            "base_semantic_score": round(b_score, 3),
            "aosp_semantic_score": round(a_score, 3),
            "score_delta": round(a_score - b_score, 3),
            "base_agl": b_agl,
            "aosp_agl": a_agl,
            "agl_ratio": round(agl_ratio, 2),
            "aosp_interventions": a.get("intervention_count", 0),
        }

        if a_score > b_score + 0.1 and agl_ratio > 0.7:
            success.append(entry)
        elif b_score > a_score + 0.1:
            failure.append(entry)
        else:
            neutral.append(entry)

    success.sort(key=lambda x: x["score_delta"], reverse=True)
    failure.sort(key=lambda x: x["score_delta"])

    return {
        "benchmark": "MMHal-Bench",
        "total": len(common_ids),
        "neutral": len(neutral),
        "success": success,
        "failure": failure,
    }


# ────────────────────────────────────────────────────────────────
# Markdown 格式化
# ────────────────────────────────────────────────────────────────

def format_pope_case_md(case: dict, case_type: str) -> str:
    icon = "**[SUCCESS]**" if case_type == "success" else "**[FAILURE]**"
    lines = [
        f"### {icon} POPE Q{case['question_id']}",
        f"- **Image**: `{case['image']}`",
        f"- **Question**: {case['question']}",
        f"- **Ground Truth**: {case['ground_truth']}",
        f"- **Base**: {case['base_prediction']} "
        f"({'correct' if case['base_correct'] else 'WRONG'})",
        f"- **A-OSP**: {case['aosp_prediction']} "
        f"({'correct' if case['aosp_correct'] else 'WRONG'})",
        f"- **Interventions**: {case['aosp_interventions']}",
        "",
    ]
    return "\n".join(lines)


def format_mmhal_case_md(case: dict, case_type: str) -> str:
    icon = "**[SUCCESS]**" if case_type == "success" else "**[FAILURE]**"
    lines = [
        f"### {icon} MMHal Q{case['question_id']} "
        f"({case['question_type']}/{case['question_topic']})",
        f"- **Image ID**: `{case['image_id']}`",
        f"- **Question**: {case['question']}",
        f"- **Ground Truth**: {case['gt_answer']}",
        "",
        f"| | Prediction | Semantic Score | AGL |",
        f"|---|---|---|---|",
        f"| Base | {case['base_prediction'][:120]}{'...' if len(case['base_prediction']) > 120 else ''} "
        f"| {case['base_semantic_score']:.3f} | {case['base_agl']} |",
        f"| A-OSP | {case['aosp_prediction'][:120]}{'...' if len(case['aosp_prediction']) > 120 else ''} "
        f"| {case['aosp_semantic_score']:.3f} | {case['aosp_agl']} |",
        "",
        f"- **Score Delta**: {case['score_delta']:+.3f} | "
        f"**AGL Ratio**: {case['agl_ratio']:.2f} | "
        f"**Interventions**: {case['aosp_interventions']}",
        "",
    ]
    return "\n".join(lines)


def generate_report(pope_result: dict, mmhal_result: dict,
                    output_dir: Path) -> Path:
    """生成完整的 Markdown 报告。"""
    output_dir.mkdir(parents=True, exist_ok=True)

    lines = [
        "# A-OSP Qualitative Case Analysis",
        "",
        "> Auto-generated by `mine_qualitative_cases.py`",
        "> For Appendix B (Extended Case Studies) and Appendix D (Failure Cases)",
        "",
    ]

    # ── 统计摘要 ─────────────────────────────────────────
    lines.append("## Summary Statistics")
    lines.append("")

    if pope_result:
        lines.append(f"### POPE ({pope_result['total']} samples)")
        lines.append(f"- Both correct: {pope_result['both_correct']}")
        lines.append(f"- Both wrong: {pope_result['both_wrong']}")
        lines.append(f"- **Success (Base wrong → A-OSP correct): "
                     f"{len(pope_result['success'])}**")
        lines.append(f"- **Failure (Base correct → A-OSP wrong): "
                     f"{len(pope_result['failure'])}**")
        lines.append("")

    if mmhal_result:
        lines.append(f"### MMHal-Bench ({mmhal_result['total']} samples)")
        lines.append(f"- Neutral (similar performance): {mmhal_result['neutral']}")
        lines.append(f"- **Success (A-OSP > Base, AGL preserved): "
                     f"{len(mmhal_result['success'])}**")
        lines.append(f"- **Failure (A-OSP < Base): "
                     f"{len(mmhal_result['failure'])}**")
        lines.append("")

    # ── Success Cases ─────────────────────────────────────
    lines.append("---")
    lines.append("## Appendix B: Success Cases (A-OSP Wins)")
    lines.append("")

    if pope_result and pope_result["success"]:
        lines.append("### POPE Success Cases")
        lines.append("")
        for c in pope_result["success"][:10]:
            lines.append(format_pope_case_md(c, "success"))

    if mmhal_result and mmhal_result["success"]:
        lines.append("### MMHal-Bench Success Cases")
        lines.append("")
        for c in mmhal_result["success"][:10]:
            lines.append(format_mmhal_case_md(c, "success"))

    # ── Failure Cases ─────────────────────────────────────
    lines.append("---")
    lines.append("## Appendix D: Failure Cases (A-OSP Over-regularization)")
    lines.append("")

    if pope_result and pope_result["failure"]:
        lines.append("### POPE Failure Cases")
        lines.append("")
        lines.append("> These cases warrant investigation: "
                     "check if original images contain **extremely small objects** "
                     "or **severe occlusion** that might trigger false positives.")
        lines.append("")
        for c in pope_result["failure"][:10]:
            lines.append(format_pope_case_md(c, "failure"))

    if mmhal_result and mmhal_result["failure"]:
        lines.append("### MMHal-Bench Failure Cases")
        lines.append("")
        lines.append("> These cases suggest A-OSP may over-penalize when "
                     "visual confidence is inherently low. "
                     "Cross-reference with image characteristics.")
        lines.append("")
        for c in mmhal_result["failure"][:10]:
            lines.append(format_mmhal_case_md(c, "failure"))

    # ── 写入 ─────────────────────────────────────────────
    md_path = output_dir / "qualitative_cases.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    json_path = output_dir / "qualitative_cases.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "pope": pope_result,
            "mmhal": mmhal_result,
        }, f, indent=2, ensure_ascii=False)

    return md_path


# ────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="A-OSP 定性案例挖掘机")
    parser.add_argument("--base_pope", default=None,
                        help="Base POPE JSONL 路径")
    parser.add_argument("--aosp_pope", default=None,
                        help="A-OSP POPE JSONL 路径")
    parser.add_argument("--base_mmhal", default=None,
                        help="Base MMHal-Bench JSONL 路径")
    parser.add_argument("--aosp_mmhal", default=None,
                        help="A-OSP MMHal-Bench JSONL 路径")
    parser.add_argument("--output_dir", default=str(
        PROJECT_ROOT / "logs" / "qualitative_cases"),
        help="输出目录")
    args = parser.parse_args()

    eval_dir = PROJECT_ROOT / "logs" / "eval_results"
    if not args.base_pope and (eval_dir / "base_pope_50_real_results.jsonl").exists():
        args.base_pope = str(eval_dir / "base_pope_50_real_results.jsonl")
    if not args.aosp_pope and (eval_dir / "aosp_pope_50_real_results.jsonl").exists():
        args.aosp_pope = str(eval_dir / "aosp_pope_50_real_results.jsonl")
    if not args.base_mmhal and (eval_dir / "base_mmhal_50_results.jsonl").exists():
        args.base_mmhal = str(eval_dir / "base_mmhal_50_results.jsonl")
    if not args.aosp_mmhal and (eval_dir / "aosp_mmhal_50_results.jsonl").exists():
        args.aosp_mmhal = str(eval_dir / "aosp_mmhal_50_results.jsonl")

    print("=" * 60)
    print("  A-OSP Qualitative Case Miner")
    print("=" * 60)

    pope_result = None
    mmhal_result = None

    if args.base_pope and args.aosp_pope:
        print(f"\n[POPE] 加载数据...")
        base_pope = load_jsonl(args.base_pope)
        aosp_pope = load_jsonl(args.aosp_pope)
        print(f"  Base: {len(base_pope)} 条 | A-OSP: {len(aosp_pope)} 条")
        pope_result = mine_pope_cases(base_pope, aosp_pope)
        print(f"  Success: {len(pope_result['success'])} | "
              f"Failure: {len(pope_result['failure'])} | "
              f"Both OK: {pope_result['both_correct']} | "
              f"Both Wrong: {pope_result['both_wrong']}")
    else:
        print("\n[POPE] 未找到数据，跳过。")

    if args.base_mmhal and args.aosp_mmhal:
        print(f"\n[MMHal] 加载数据...")
        base_mmhal = load_jsonl(args.base_mmhal)
        aosp_mmhal = load_jsonl(args.aosp_mmhal)
        print(f"  Base: {len(base_mmhal)} 条 | A-OSP: {len(aosp_mmhal)} 条")
        mmhal_result = mine_mmhal_cases(base_mmhal, aosp_mmhal)
        print(f"  Success: {len(mmhal_result['success'])} | "
              f"Failure: {len(mmhal_result['failure'])} | "
              f"Neutral: {mmhal_result['neutral']}")
    else:
        print("\n[MMHal] 未找到数据，跳过。")

    if pope_result or mmhal_result:
        md_path = generate_report(pope_result, mmhal_result,
                                  Path(args.output_dir))
        print(f"\n{'=' * 60}")
        print(f"  报告已生成:")
        print(f"    Markdown → {md_path}")
        print(f"    JSON     → {md_path.with_suffix('.json')}")
        print(f"{'=' * 60}")
    else:
        print("\n[WARN] 无可用数据，未生成报告。")


if __name__ == "__main__":
    main()
