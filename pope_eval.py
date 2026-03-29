import argparse
import json
import os
import random
from pathlib import Path
from statistics import mean

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm

from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from torchvision.utils import save_image

from pope_loader import POPEDataSet
from minigpt4.common.dist_utils import get_rank
from minigpt4.models import load_preprocess

from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry

# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *
from chord.anchor_cache import AnchorCache
from chord.knowledge_kernel_evaluator import build_anchor_weight_result, build_knowledge_kernel_result_from_cache
from chord.query_formulation import build_model_query, extract_anchor_query



MODEL_EVAL_CONFIG_PATH = {
    "minigpt4": "eval_configs/minigpt4_eval.yaml",
    "instructblip": "eval_configs/instructblip_eval.yaml",
    "lrv_instruct": "eval_configs/lrv_instruct_eval.yaml",
    "shikra": "eval_configs/shikra_eval.yaml",
    "llava-1.5": "eval_configs/llava-1.5_eval.yaml",
}

POPE_PATH = {
    "random": "pope_coco/coco_pope_random.json",
    "popular": "pope_coco/coco_pope_popular.json",
    "adversarial": "pope_coco/coco_pope_adversarial.json",
}

INSTRUCTION_TEMPLATE = {
    "minigpt4": "###Human: <Img><ImageHere></Img> <question> ###Assistant:",
    "instructblip": "<ImageHere><question>",
    "lrv_instruct": "###Human: <Img><ImageHere></Img> <question> ###Assistant:",
    "shikra": "USER: <im_start><ImageHere><im_end> <question> ASSISTANT:",
    "llava-1.5": "USER: <ImageHere> <question> ASSISTANT:"
}


def parse_args():
    parser = argparse.ArgumentParser(description="POPE-Adv evaluation on LVLMs.")
    parser.add_argument("--model", type=str, help="model")
    parser.add_argument("--pope-type", type=str, help="model")
    # parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    parser.add_argument("--data_path", type=str, default="COCO_2014/val2014/", help="data path")
    parser.add_argument("--pope-path", type=str, default=None, help="optional explicit POPE json path override")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--num_workers", type=int, default=2, help="num workers")
    parser.add_argument("--limit", type=int, default=None, help="optional limit for effect-oriented smoke evaluation")

    parser.add_argument("--beam", type=int)
    parser.add_argument("--sample", action='store_true')
    parser.add_argument("--max-new-tokens", type=int, default=10)
    parser.add_argument("--scale_factor", type=float, default=50)
    parser.add_argument("--threshold", type=int, default=15)
    parser.add_argument("--num_attn_candidates", type=int, default=5)
    parser.add_argument("--penalty_weights", type=float, default=1.0)
    parser.add_argument("--llava-ckpt", type=str, default=None, help="optional local merged checkpoint path for llava-1.5")
    parser.add_argument("--llava-proc-path", type=str, default=None, help="optional local CLIP processor path for llava-1.5")
    parser.add_argument("--output-json", type=str, default=None, help="optional path to save metrics and decoded outputs")
    parser.add_argument("--chord-enable", action="store_true", help="enable CHORD current-term reranking on top of official OPERA")
    parser.add_argument("--anchor-cache-jsonl", type=str, default=None, help="offline anchor cache produced by precompute_pope_anchor_cache.py")
    parser.add_argument("--alpha-anchor", type=float, default=0.5, help="soft anchor gain applied to matched visual tokens")
    parser.add_argument("--lambda-cur", type=float, default=0.0, help="current-step CHORD bonus weight")
    parser.add_argument("--lambda-fut", type=float, default=0.0, help="future-rollout CHORD bonus weight")
    parser.add_argument("--lambda-txt", type=float, default=1.0, help="future-rollout text penalty")
    parser.add_argument("--zero-anchor-penalty", type=float, default=0.0, help="extra penalty applied when the knowledge-kernel support is zero")
    parser.add_argument("--future-horizon", type=int, default=0, help="future rollout horizon")
    parser.add_argument("--future-topk", type=int, default=0, help="number of candidate branches to probe into the future")
    parser.add_argument("--tau-abort", type=float, default=0.0, help="abort rollout when step-1 visual ratio falls below this threshold; 0 disables the gate")
    parser.add_argument("--attention-last-n-layers", type=int, default=2, help="number of trailing attention layers to average")
    parser.add_argument("--attention-head-reduce", type=str, default="mean", help="how to reduce attention heads")
    parser.add_argument("--detector-box-threshold", type=float, default=0.25)
    parser.add_argument("--detector-text-threshold", type=float, default=0.2)
    parser.add_argument("--max-boxes", type=int, default=8)
    parser.add_argument("--anchor-query-mode", type=str, default="raw", help="how to convert raw POPE questions into detector/relevance queries")
    parser.add_argument("--model-query-suffix", type=str, default="", help="optional suffix appended to the model-facing question only")
    args = parser.parse_args()
    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


def print_acc(pred_list, label_list):
    pos = 1
    neg = 0
    yes_ratio = pred_list.count(1) / len(pred_list)
    # unknown_ratio = pred_list.count(2) / len(pred_list)

    TP, TN, FP, FN = 0, 0, 0, 0
    for pred, label in zip(pred_list, label_list):
        if pred == pos and label == pos:
            TP += 1
        elif pred == pos and label == neg:
            FP += 1
        elif pred == neg and label == neg:
            TN += 1
        elif pred == neg and label == pos:
            FN += 1

    print('TP\tFP\tTN\tFN\t')
    print('{}\t{}\t{}\t{}'.format(TP, FP, TN, FN))

    precision = float(TP) / float(TP + FP)
    recall = float(TP) / float(TP + FN)
    f1 = 2*precision*recall / (precision + recall)
    acc = (TP + TN) / (TP + TN + FP + FN)
    print('Accuracy: {}'.format(acc))
    print('Precision: {}'.format(precision))
    print('Recall: {}'.format(recall))
    print('F1 score: {}'.format(f1))
    print('Yes ratio: {}'.format(yes_ratio))


def compute_metrics(pred_list, label_list):
    pos = 1
    neg = 0
    yes_ratio = pred_list.count(1) / len(pred_list)

    TP, TN, FP, FN = 0, 0, 0, 0
    for pred, label in zip(pred_list, label_list):
        if pred == pos and label == pos:
            TP += 1
        elif pred == pos and label == neg:
            FP += 1
        elif pred == neg and label == neg:
            TN += 1
        elif pred == neg and label == pos:
            FN += 1

    precision = float(TP) / float(TP + FP)
    recall = float(TP) / float(TP + FN)
    f1 = 2 * precision * recall / (precision + recall)
    acc = (TP + TN) / (TP + TN + FP + FN)
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "yes_ratio": yes_ratio,
        "tp": TP,
        "tn": TN,
        "fp": FP,
        "fn": FN,
    }


def _count_nested_true(values) -> int:
    total = 0
    for row in values or []:
        total += sum(1 for item in row if item)
    return total


def _count_nested_items(values) -> int:
    return sum(len(row) for row in (values or []))


def summarize_run_outputs(run_outputs: list[dict]) -> dict:
    anchor_logs = [item.get("anchor_diagnostics") for item in run_outputs if item.get("anchor_diagnostics") is not None]
    decode_logs = [diag for item in run_outputs for diag in item.get("decode_diagnostics", [])]

    anchor_cache_hits = sum(1 for log in anchor_logs if log.get("cache_hit"))
    kernel_cache_hits = sum(1 for log in anchor_logs if log.get("kernel_cache_hit"))
    used_fallback_count = sum(1 for log in anchor_logs if log.get("used_fallback"))
    zero_relevance_count = sum(1 for log in anchor_logs if log.get("zero_relevance"))
    positive_relevance_counts = [int(log.get("num_positive_relevance", 0)) for log in anchor_logs]
    token_weight_maxes = [float(log["token_weight_stats"]["max"]) for log in anchor_logs if log.get("token_weight_stats")]

    v_anchor_zero_count = 0
    v_anchor_total = 0
    skipped_no_anchor_total = 0
    skipped_no_anchor_candidates = 0
    future_failed_total = 0
    future_candidate_total = 0
    future_aborted_total = 0

    for diag in decode_logs:
        v_anchor_rows = diag.get("v_anchor") or []
        for row in v_anchor_rows:
            v_anchor_total += len(row)
            v_anchor_zero_count += sum(1 for value in row if float(value) == 0.0)
        skipped_no_anchor_total += _count_nested_true(diag.get("future_skipped_no_anchor"))
        skipped_no_anchor_candidates += _count_nested_items(diag.get("future_skipped_no_anchor"))
        future_failed_total += _count_nested_true(diag.get("future_failed"))
        future_candidate_total += _count_nested_items(diag.get("future_failed"))
        future_aborted_total += _count_nested_true(diag.get("future_aborted_tau"))

    return {
        "num_samples": len(run_outputs),
        "num_anchor_samples": len(anchor_logs),
        "anchor_cache_hit_rate": (anchor_cache_hits / len(anchor_logs)) if anchor_logs else 0.0,
        "kernel_cache_hit_rate": (kernel_cache_hits / len(anchor_logs)) if anchor_logs else 0.0,
        "used_fallback_rate": (used_fallback_count / len(anchor_logs)) if anchor_logs else 0.0,
        "zero_relevance_rate": (zero_relevance_count / len(anchor_logs)) if anchor_logs else 0.0,
        "mean_positive_relevance_anchors": mean(positive_relevance_counts) if positive_relevance_counts else 0.0,
        "mean_token_weight_max": mean(token_weight_maxes) if token_weight_maxes else 0.0,
        "v_anchor_zero_measure_rate": (v_anchor_zero_count / v_anchor_total) if v_anchor_total else 0.0,
        "v_anchor_zero_gate_rate": (skipped_no_anchor_total / skipped_no_anchor_candidates) if skipped_no_anchor_candidates else 0.0,
        "future_failed_total": future_failed_total,
        "future_failed_rate": (future_failed_total / future_candidate_total) if future_candidate_total else 0.0,
        "future_aborted_total": future_aborted_total,
    }


def _override_llava_runtime_paths(cfg, args) -> None:
    if args.model != "llava-1.5":
        return
    if args.llava_ckpt:
        cfg.model_cfg.merged_ckpt = args.llava_ckpt
    if args.llava_proc_path:
        preprocess = cfg.get_config().preprocess
        if hasattr(preprocess, "vis_processor"):
            if hasattr(preprocess.vis_processor, "train"):
                preprocess.vis_processor.train.proc_type = args.llava_proc_path
            if hasattr(preprocess.vis_processor, "eval"):
                preprocess.vis_processor.eval.proc_type = args.llava_proc_path


def _load_anchor_cache(args) -> AnchorCache | None:
    if not args.chord_enable:
        return None
    if args.anchor_cache_jsonl is None:
        raise ValueError("EKKO requires --anchor-cache-jsonl so knowledge-kernel lookup stays fully offline.")
    return AnchorCache.from_jsonl(args.anchor_cache_jsonl)


def recorder(out, pred_list):
    NEG_WORDS = ["No", "not", "no", "NO"]
    for line in out:

        line = line.replace('.', '')
        line = line.replace(',', '')
        words = line.split(' ')
        if any(word in NEG_WORDS for word in words) or any(word.endswith("n't") for word in words):
            pred_list.append(0)
        else:
            pred_list.append(1)
    
    return pred_list




def main():

    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    args.cfg_path = MODEL_EVAL_CONFIG_PATH[args.model]
    args.pope_path = args.pope_path or POPE_PATH[args.pope_type]
    cfg = Config(args)
    _override_llava_runtime_paths(cfg, args)
    anchor_cache = _load_anchor_cache(args)

    setup_seeds(cfg)
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

    # ========================================
    #             Model Initialization
    # ========================================
    print('Initializing Model')

    model_config = cfg.model_cfg
    model_config.device_8bit = args.gpu_id
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to(device)
    model.eval()
    vis_processors, txt_processors = load_preprocess(cfg.get_config().preprocess)
    # vis_processors.do_normalize = False
    print(vis_processors["eval"].transform)
    print("Done!")

    # load pope data
    pope_dataset = POPEDataSet(
        pope_path=args.pope_path, 
        data_path=args.data_path, 
        trans=vis_processors["eval"]
    )
    if args.limit is not None:
        pope_dataset = torch.utils.data.Subset(pope_dataset, range(min(args.limit, len(pope_dataset))))
    pope_loader = torch.utils.data.DataLoader(
        pope_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        drop_last=False
    )

    print ("load data finished")


    print("Start eval...")
    pred_list, pred_list_s, label_list = [], [], []
    run_outputs = []
    if args.chord_enable and args.batch_size != 1:
        raise ValueError("The current CHORD-on-EKKO path supports batch_size=1 only.")
    for batch_id, data in tqdm(enumerate(pope_loader), total=len(pope_loader)):
        image = data["image"]
        raw_queries = data["query"]
        image_paths = data.get("image_path")
        image_sizes = data.get("image_size")
        label = data["label"]
        label_list = label_list + list(label)

        template = INSTRUCTION_TEMPLATE[args.model]
        model_queries = [build_model_query(q, suffix=args.model_query_suffix) for q in raw_queries]
        prompts = [template.replace("<question>", q) for q in model_queries]

        image = image.to(device)
        label = torch.Tensor(label).to(device)
        chord_visual_token_weights = None
        anchor_log = None
        decode_diagnostics = []
        if args.chord_enable:
            sample_image_path = image_paths[0]
            if isinstance(sample_image_path, bytes):
                sample_image_path = sample_image_path.decode("utf-8")
            if isinstance(image_sizes, list):
                sample_image_size = tuple(int(x[0]) if isinstance(x, torch.Tensor) else int(x) for x in image_sizes)
            else:
                sample_image_size = tuple(int(x) for x in image_sizes[0])
            sample_anchor_query = extract_anchor_query(raw_queries[0], mode=args.anchor_query_mode)
            cached = anchor_cache.get(
                image_path=sample_image_path,
                query=raw_queries[0],
                anchor_query=sample_anchor_query,
                box_threshold=args.detector_box_threshold,
                text_threshold=args.detector_text_threshold,
                max_boxes=args.max_boxes,
            )
            cached_anchors = [] if cached is None else cached.anchors
            if cached is not None and cached.membership is not None and cached.relevance is not None:
                anchor_result = build_knowledge_kernel_result_from_cache(
                    cached_entry=cached,
                    alpha_anchor=args.alpha_anchor,
                )
            else:
                anchor_result = build_anchor_weight_result(
                    anchors=cached_anchors,
                    query=sample_anchor_query,
                    image_size=sample_image_size,
                    grid_size=(24, 24),
                    alpha_anchor=args.alpha_anchor,
                )
            chord_visual_token_weights = anchor_result.token_weights.unsqueeze(0).to(device)
            positive_relevance = int(torch.count_nonzero(anchor_result.relevance > 0).item())
            anchor_log = {
                "cache_hit": cached is not None,
                "kernel_cache_hit": bool(cached is not None and cached.membership is not None and cached.relevance is not None),
                "image_path": sample_image_path,
                "image_size": sample_image_size,
                "raw_query": raw_queries[0],
                "model_query": model_queries[0],
                "anchor_query": sample_anchor_query,
                "num_cached_anchors": len(cached_anchors),
                "used_fallback": anchor_result.used_fallback,
                "num_positive_relevance": positive_relevance,
                "zero_relevance": positive_relevance == 0,
                "token_weight_stats": {
                    "min": float(anchor_result.token_weights.min().item()),
                    "max": float(anchor_result.token_weights.max().item()),
                    "mean": float(anchor_result.token_weights.mean().item()),
                },
                "relevance_stats": {
                    "max": float(anchor_result.relevance.max().item()) if anchor_result.relevance.numel() else 0.0,
                    "mean": float(anchor_result.relevance.mean().item()) if anchor_result.relevance.numel() else 0.0,
                },
            }

        with torch.inference_mode():
            with torch.no_grad():
                out = model.generate(
                    {"image": image, "prompt":prompts}, 
                    use_nucleus_sampling=args.sample, 
                    num_beams=args.beam,
                    max_new_tokens=args.max_new_tokens,
                    output_attentions=True,
                    opera_decoding=True,
                    scale_factor=args.scale_factor,
                    threshold=args.threshold,
                    num_attn_candidates=args.num_attn_candidates,
                    penalty_weights=args.penalty_weights,
                    chord_visual_token_weights=chord_visual_token_weights,
                    chord_lambda_cur=args.lambda_cur if args.chord_enable else 0.0,
                    chord_lambda_fut=args.lambda_fut if args.chord_enable else 0.0,
                    chord_lambda_txt=args.lambda_txt,
                    chord_zero_anchor_penalty=args.zero_anchor_penalty if args.chord_enable else 0.0,
                    chord_future_horizon=args.future_horizon if args.chord_enable else 0,
                    chord_future_topk=args.future_topk if args.chord_enable else 0,
                    chord_tau_abort=args.tau_abort,
                    chord_attention_last_n_layers=args.attention_last_n_layers,
                    chord_attention_head_reduce=args.attention_head_reduce,
                    chord_diagnostics=decode_diagnostics,
                )
                pred_list = recorder(out, pred_list)
                for line in out:
                    print(line)
                run_outputs.append(
                    {
                        "batch_idx": batch_id,
                        "query": list(raw_queries),
                        "model_query": model_queries,
                        "label": list(label.cpu().tolist()),
                        "answer": out,
                        "chord_enabled": args.chord_enable,
                        "anchor_diagnostics": anchor_log,
                        "decode_diagnostics": decode_diagnostics,
                    }
                )

    print("[{}, {}]===============================================".format(args.scale_factor, args.num_attn_candidates))
    if len(pred_list) != 0:
        print_acc(pred_list, label_list)
    if len(pred_list_s) != 0:
        print_acc(pred_list_s, label_list)
    if args.output_json:
        payload = {
            "model": args.model,
            "pope_type": args.pope_type,
            "pope_path": args.pope_path,
            "limit": args.limit,
            "metrics": compute_metrics(pred_list, label_list),
            "diagnostics_summary": summarize_run_outputs(run_outputs),
            "outputs": run_outputs,
            "chord": {
                "enabled": args.chord_enable,
                "anchor_cache_jsonl": args.anchor_cache_jsonl,
                "alpha_anchor": args.alpha_anchor,
                "lambda_cur": args.lambda_cur,
                "lambda_fut": args.lambda_fut,
                "lambda_txt": args.lambda_txt,
                "zero_anchor_penalty": args.zero_anchor_penalty,
                "future_horizon": args.future_horizon,
                "future_topk": args.future_topk,
                "tau_abort": args.tau_abort,
                "attention_last_n_layers": args.attention_last_n_layers,
                "attention_head_reduce": args.attention_head_reduce,
                "detector_box_threshold": args.detector_box_threshold,
                "detector_text_threshold": args.detector_text_threshold,
                "max_boxes": args.max_boxes,
                "anchor_query_mode": args.anchor_query_mode,
                "model_query_suffix": args.model_query_suffix,
            },
        }
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")








if __name__ == "__main__":
    main()
