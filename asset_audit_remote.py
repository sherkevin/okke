import json
import os
from pathlib import Path

import paramiko


HOST = "connect.westc.seetacloud.com"
PORT = 47559
USER = "root"
PASSWORD = "aMNIL2fW6aoV"


REMOTE_SCRIPT = r"""
import json
import os
from pathlib import Path

root = Path("/root/autodl-tmp/BRA_Project")
datasets = root / "datasets"
models = root / "models"
logs = root / "logs"

targets = {
    "project_root": root,
    "models_root": models,
    "datasets_root": datasets,
    "logs_root": logs,
    "qwen3_vl_8b": models / "Qwen3-VL-8B-Instruct",
    "qwen3_vl_2b": models / "Qwen3-VL-2B-Instruct",
    "coco_images": datasets / "coco2014" / "val2014",
    "coco_annotations": datasets / "coco2014" / "annotations",
    "pope": datasets / "POPE",
    "chair_instances": datasets / "coco2014" / "annotations" / "instances_val2014.json",
    "mmbench": datasets / "MMBench",
    "mme": datasets / "MME",
    "mmmu": datasets / "MMMU",
    "freak": datasets / "FREAK",
    "docvqa": datasets / "DocVQA",
    "lora_root": root / "lora",
    "outputs_root": root / "outputs",
    "baselines_root": root / "baselines",
}

def summarize_path(path: Path):
    exists = path.exists()
    info = {
        "path": str(path),
        "exists": exists,
        "type": "missing",
    }
    if not exists:
        return info
    if path.is_file():
        info["type"] = "file"
        try:
            info["size_bytes"] = path.stat().st_size
        except Exception:
            pass
        return info
    info["type"] = "dir"
    try:
        entries = sorted(os.listdir(path))
        info["entry_count"] = len(entries)
        info["sample_entries"] = entries[:20]
    except Exception as e:
        info["list_error"] = str(e)
    return info

report = {name: summarize_path(path) for name, path in targets.items()}

# Add a few specific probes.
extra = {}

for key in ("coco_images", "pope", "mmbench", "mme", "mmmu", "freak", "docvqa", "lora_root"):
    path = targets[key]
    if path.exists() and path.is_dir():
        try:
            count = 0
            for _, dirs, files in os.walk(path):
                count += len(files)
            extra[key + "_file_count"] = count
        except Exception as e:
            extra[key + "_file_count_error"] = str(e)

# Search for likely LoRA checkpoints.
lora_hits = []
for search_root in [root, root / "outputs", root / "checkpoints", root / "lora"]:
    if search_root.exists():
        for cur_root, dirs, files in os.walk(search_root):
            name = os.path.basename(cur_root).lower()
            if "lora" in name or "adapter" in name:
                lora_hits.append(cur_root)
            for f in files:
                fl = f.lower()
                if fl in {"adapter_config.json", "adapter_model.bin", "adapter_model.safetensors"}:
                    lora_hits.append(os.path.join(cur_root, f))
        if len(lora_hits) > 50:
            break
extra["lora_hits"] = sorted(set(lora_hits))[:50]

# Probe renamed / hf-style dataset folders in more detail.
hf_targets = [
    datasets / "MMBench_EN_hf",
    datasets / "MME_hf",
    datasets / "MMMU_hf",
    datasets / "FREAK_hf",
    datasets / "HallusionBench_hf",
]
hf_details = {}
for path in hf_targets:
    item = {"exists": path.exists(), "path": str(path)}
    if path.exists():
        item["type"] = "dir" if path.is_dir() else "file"
        if path.is_dir():
            try:
                entries = sorted(os.listdir(path))
                item["entry_count"] = len(entries)
                item["sample_entries"] = entries[:20]
                file_count = 0
                for _, _, files in os.walk(path):
                    file_count += len(files)
                item["file_count"] = file_count
            except Exception as e:
                item["error"] = str(e)
    hf_details[path.name] = item

docvqa_hits = []
for cur_root, dirs, files in os.walk(datasets):
    base = os.path.basename(cur_root).lower()
    if "docvqa" in base:
        docvqa_hits.append(cur_root)
    for f in files:
        if "docvqa" in f.lower():
            docvqa_hits.append(os.path.join(cur_root, f))
extra["hf_details"] = hf_details
extra["docvqa_hits"] = sorted(set(docvqa_hits))[:50]

calib_hits = []
keywords = ("calib", "phi", "matrix", "adapter", "lora", "tlra")
for search_root in [root, models]:
    if search_root.exists():
        for cur_root, dirs, files in os.walk(search_root):
            base = os.path.basename(cur_root).lower()
            if any(k in base for k in keywords):
                calib_hits.append(cur_root)
            for f in files:
                fl = f.lower()
                if any(k in fl for k in keywords) and (fl.endswith(".pt") or fl.endswith(".bin") or fl.endswith(".safetensors") or fl.endswith(".json")):
                    calib_hits.append(os.path.join(cur_root, f))
extra["calib_or_adapter_hits"] = sorted(set(calib_hits))[:100]

print(json.dumps({"targets": report, "extra": extra}, ensure_ascii=False, indent=2))
"""


def main():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username=USER, password=PASSWORD, timeout=20)
    stdin, stdout, stderr = client.exec_command(
        f'export PATH="/root/miniconda3/bin:$PATH"; /root/miniconda3/bin/python - <<\'PY\'\n{REMOTE_SCRIPT}\nPY',
        timeout=120,
    )
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    client.close()
    if err.strip():
        print("STDERR:")
        print(err)
    print(out)


if __name__ == "__main__":
    main()
