import argparse
import json
import os
import random
from pathlib import Path

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
from chord.anchor_builder import build_anchor_weight_result
from chord.config import CHORDConfig
from chord.detector_client import GroundingDinoClient

# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *



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
    parser.add_argument("--cuda-visible-devices", type=str, default=None, help="comma-separated GPU list to expose")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    parser.add_argument("--data_path", type=str, default="COCO_2014/val2014/", help="data path")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--num_workers", type=int, default=2, help="num workers")

    parser.add_argument("--beam", type=int)
    parser.add_argument("--sample", action='store_true')
    parser.add_argument("--max-new-tokens", type=int, default=10)
    parser.add_argument("--scale_factor", type=float, default=50)
    parser.add_argument("--threshold", type=int, default=15)
    parser.add_argument("--num_attn_candidates", type=int, default=5)
    parser.add_argument("--penalty_weights", type=float, default=1.0)
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument("--chord-enable", action="store_true")
    parser.add_argument("--alpha-anchor", type=float, default=None)
    parser.add_argument("--lambda-cur", type=float, default=None)
    parser.add_argument("--lambda-fut", type=float, default=None)
    parser.add_argument("--lambda-txt", type=float, default=None)
    parser.add_argument("--future-horizon", type=int, default=None)
    parser.add_argument("--future-topk", type=int, default=None)
    parser.add_argument("--detector-box-threshold", type=float, default=None)
    parser.add_argument("--detector-text-threshold", type=float, default=None)
    parser.add_argument("--max-boxes", type=int, default=None)
    parser.add_argument("--grounding-dino-path", type=str, default="/root/autodl-tmp/BRA_Project/models/grounding-dino-base")
    parser.add_argument("--detector-python", type=str, default="/root/miniconda3/bin/python")
    parser.add_argument("--detector-device", type=str, default="cpu")
    parser.add_argument("--chord-log-jsonl", type=str, default=None)
    parser.add_argument("--llava-ckpt", type=str, default="/root/autodl-tmp/BRA_Project/models/llava-1.5-7b-hf")
    parser.add_argument("--llava-proc-path", type=str, default="/root/chiro_assets/clip-vit-large-patch14-336")
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


def compute_metrics(pred_list, label_list):
    pos = 1
    neg = 0
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

    precision = float(TP) / float(TP + FP) if (TP + FP) else 0.0
    recall = float(TP) / float(TP + FN) if (TP + FN) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    acc = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) else 0.0
    yes_ratio = pred_list.count(1) / len(pred_list) if pred_list else 0.0
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


def decode_candidate_ids(model, candidate_id_rows):
    if not hasattr(model, "llama_tokenizer"):
        return candidate_id_rows
    decoded_rows = []
    for row in candidate_id_rows:
        decoded_rows.append(
            [
                model.llama_tokenizer.decode([token_id], skip_special_tokens=False)
                for token_id in row
            ]
        )
    return decoded_rows




def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices or str(args.gpu_id)
    os.environ["CHORD_PRIMARY_GPU"] = str(args.gpu_id)
    visible_gpu_count = len((args.cuda_visible_devices or str(args.gpu_id)).split(","))
    if visible_gpu_count > 1:
        os.environ["CHORD_HEADROOM_GPU"] = "0" if args.gpu_id != 0 else "1"
        os.environ["CHORD_FORCE_INFER_AUTO_DEVICE_MAP"] = "0"
        os.environ.setdefault("CHORD_PRIMARY_GPU_LAYER_COUNT", "16")
    else:
        os.environ["CHORD_HEADROOM_GPU"] = str(args.gpu_id)
        os.environ.setdefault("CHORD_FORCE_INFER_AUTO_DEVICE_MAP", "1")
    chord_cfg = CHORDConfig.from_args(args)

    args.cfg_path = MODEL_EVAL_CONFIG_PATH[args.model]
    args.pope_path = POPE_PATH[args.pope_type]
    cfg = Config(args)
    if args.model == "llava-1.5":
        cfg.model_cfg.merged_ckpt = args.llava_ckpt
        cfg.get_config().preprocess.vis_processor.train.proc_type = args.llava_proc_path
        cfg.get_config().preprocess.vis_processor.eval.proc_type = args.llava_proc_path

    setup_seeds(cfg)
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_id)
        device = torch.device(f"cuda:{args.gpu_id}")
    else:
        device = torch.device("cpu")

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

    detector_client = None
    if chord_cfg.enabled:
        if args.batch_size != 1:
            raise ValueError("The first CHORD POPE wiring path currently supports batch_size=1 only.")
        detector_client = GroundingDinoClient(
            python_executable=chord_cfg.detector_python,
            server_script=str(Path(__file__).resolve().parent / "chord" / "detector_server.py"),
            model_path=chord_cfg.grounding_dino_path,
            device=args.detector_device,
        )

    try:
        for batch_id, data in tqdm(enumerate(pope_loader), total=len(pope_loader)):
            image = data["image"]
            raw_queries = data["query"]
            image_paths = data.get("image_path")
            image_sizes = data.get("image_size")
            label = data["label"]
            label_list = label_list + list(label)

            template = INSTRUCTION_TEMPLATE[args.model]
            prompts = [template.replace("<question>", q) for q in raw_queries]

            image = image.to(device)
            label = torch.Tensor(label).to(device)

            chord_visual_token_weights = None
            anchor_log = None
            decode_diagnostics = []
            if chord_cfg.enabled:
                sample_image_path = image_paths[0]
                if isinstance(sample_image_path, bytes):
                    sample_image_path = sample_image_path.decode("utf-8")
                if isinstance(image_sizes, list):
                    sample_image_size = tuple(int(x[0]) if isinstance(x, torch.Tensor) else int(x) for x in image_sizes)
                else:
                    sample_image_size = tuple(int(x) for x in image_sizes[0])
                anchors = detector_client.detect(
                    image_path=sample_image_path,
                    query=raw_queries[0],
                    box_threshold=chord_cfg.detector_box_threshold,
                    text_threshold=chord_cfg.detector_text_threshold,
                    max_boxes=chord_cfg.max_boxes,
                )
                anchor_result = build_anchor_weight_result(
                    anchors=anchors,
                    query=raw_queries[0],
                    image_size=sample_image_size,
                    grid_size=(24, 24),
                    alpha_anchor=chord_cfg.alpha_anchor,
                )
                chord_visual_token_weights = anchor_result.token_weights.unsqueeze(0).to(device)
                anchor_log = {
                    "image_path": sample_image_path,
                    "image_size": sample_image_size,
                    "anchors": [
                        {
                            "box": list(anchor.box),
                            "confidence": anchor.confidence,
                            "phrase": anchor.phrase,
                        }
                        for anchor in anchors
                    ],
                    "relevance": anchor_result.relevance.tolist(),
                    "token_weight_stats": {
                        "min": float(anchor_result.token_weights.min().item()),
                        "max": float(anchor_result.token_weights.max().item()),
                        "mean": float(anchor_result.token_weights.mean().item()),
                    },
                    "used_fallback": anchor_result.used_fallback,
                }

            with torch.inference_mode():
                with torch.no_grad():
                    out = model.generate(
                        {"image": image, "prompt": prompts},
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
                        chord_lambda_cur=chord_cfg.lambda_cur if chord_cfg.enabled else 0.0,
                        chord_lambda_fut=chord_cfg.lambda_fut if chord_cfg.enabled else 0.0,
                        chord_lambda_txt=chord_cfg.lambda_txt,
                        chord_future_horizon=chord_cfg.future_horizon if chord_cfg.enabled else 0,
                        chord_future_topk=chord_cfg.future_topk if chord_cfg.enabled else 0,
                        chord_diagnostics=decode_diagnostics,
                    )
                    pred_list = recorder(out, pred_list)
                    for line in out:
                        print(line)

            decoded_diagnostics = []
            for step in decode_diagnostics:
                step_copy = dict(step)
                if "candidate_token_ids_before" in step_copy:
                    step_copy["candidate_tokens_before"] = decode_candidate_ids(model, step_copy["candidate_token_ids_before"])
                if "candidate_token_ids_after" in step_copy:
                    step_copy["candidate_tokens_after"] = decode_candidate_ids(model, step_copy["candidate_token_ids_after"])
                decoded_diagnostics.append(step_copy)

            run_record = {
                "batch_idx": batch_id,
                "query": list(raw_queries),
                "label": list(label.cpu().tolist()),
                "answer": out,
                "chord_enabled": chord_cfg.enabled,
                "anchor_diagnostics": anchor_log,
                "decode_diagnostics": decoded_diagnostics,
            }
            run_outputs.append(run_record)

            if chord_cfg.diagnostics_jsonl is not None:
                Path(chord_cfg.diagnostics_jsonl).parent.mkdir(parents=True, exist_ok=True)
                with Path(chord_cfg.diagnostics_jsonl).open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(run_record, ensure_ascii=False) + "\n")
    finally:
        if detector_client is not None:
            detector_client.close()

    print("[{}, {}]===============================================".format(args.scale_factor, args.num_attn_candidates))
    if len(pred_list) != 0:
        print_acc(pred_list, label_list)
    if len(pred_list_s) != 0:
        print_acc(pred_list_s, label_list)

    if args.output_json:
        payload = {
            "model": args.model,
            "pope_type": args.pope_type,
            "metrics": compute_metrics(pred_list, label_list),
            "outputs": run_outputs,
            "chord": {
                "enabled": chord_cfg.enabled,
                "alpha_anchor": chord_cfg.alpha_anchor,
                "lambda_cur": chord_cfg.lambda_cur,
                "lambda_fut": chord_cfg.lambda_fut,
                "lambda_txt": chord_cfg.lambda_txt,
                "future_horizon": chord_cfg.future_horizon,
                "future_topk": chord_cfg.future_topk,
                "detector_box_threshold": chord_cfg.detector_box_threshold,
                "detector_text_threshold": chord_cfg.detector_text_threshold,
                "max_boxes": chord_cfg.max_boxes,
            },
        }
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")








if __name__ == "__main__":
    main()
