from __future__ import annotations

import json
import shlex
from dataclasses import dataclass
from pathlib import Path

import paramiko


HOST = "connect.westc.seetacloud.com"
PORT = 47559
USER = "root"
PASSWORD = "aMNIL2fW6aoV"

PROJECT = "/root/autodl-tmp/BRA_Project"
PSI_CHECKPOINT = f"{PROJECT}/models/uniground_v6/psi_univ_clip_large_coco_val2014_v1.pt"
EXTERNAL_ENCODER = f"{PROJECT}/models/clip-vit-large-patch14"
PROJECTOR_CHECKPOINT = f"{PROJECT}/models/V_matrix.pt"

UNIGROUND_METHODS = [
    "base",
    "tlra_internal_zero",
    "tlra_internal_calib",
    "external_global_prior",
    "uniground",
    "uniground_no_gate",
    "uniground_no_abstain",
    "uniground_global_only",
    "uniground_region_only",
]


@dataclass
class LaunchSpec:
    gpu: int
    name: str
    log_path: str
    command_body: str


def connect() -> paramiko.SSHClient:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username=USER, password=PASSWORD, timeout=45)
    return client


def run(client: paramiko.SSHClient, command: str, timeout: int = 60) -> tuple[str, str, int]:
    stdin, stdout, stderr = client.exec_command(command, timeout=timeout)
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    code = stdout.channel.recv_exit_status()
    return out, err, code


def quote(cmd: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def build_uniground_cmd(model: str, dataset: str, output_dir: str, mini_test: int, chair_max_new_tokens: int = 384) -> str:
    cmd = [
        "/root/miniconda3/bin/python",
        f"{PROJECT}/run_uniground_eval.py",
        "--model",
        model,
        "--dataset",
        dataset,
        "--mini_test",
        str(mini_test),
        "--pope_split",
        "random",
        "--chair_max_new_tokens",
        str(chair_max_new_tokens),
        "--external_encoder",
        EXTERNAL_ENCODER,
        "--external_device",
        "cpu",
        "--psi_checkpoint",
        PSI_CHECKPOINT,
        "--projector_checkpoint",
        PROJECTOR_CHECKPOINT,
        "--output_dir",
        output_dir,
        "--methods",
        *UNIGROUND_METHODS,
    ]
    return quote(cmd)


def build_bra_cmd(dataset: str, bra_method: str, output_path: str, n_samples: int) -> str:
    cmd = [
        "/root/miniconda3/bin/python",
        f"{PROJECT}/bra_eval_matrix.py",
        "--model",
        "qwen3vl2b",
        "--dataset",
        dataset,
        "--n_samples",
        str(n_samples),
        "--bra_method",
        bra_method,
        "--output",
        output_path,
    ]
    return quote(cmd)


def gpu_probe(client: paramiko.SSHClient) -> dict:
    py = r"""
import json, subprocess
query = "index,name,memory.used,memory.total,utilization.gpu"
gpu_cmd = ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"]
proc_cmd = ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv,noheader,nounits"]
gpus = []
for line in subprocess.check_output(gpu_cmd, text=True).strip().splitlines():
    idx, name, mem_used, mem_total, util = [part.strip() for part in line.split(",")]
    gpus.append({
        "index": int(idx),
        "name": name,
        "memory_used_mb": int(mem_used),
        "memory_total_mb": int(mem_total),
        "utilization_gpu_pct": int(util),
    })
apps_raw = subprocess.check_output(proc_cmd, text=True).strip()
apps = [line.strip() for line in apps_raw.splitlines() if line.strip()]
print(json.dumps({"gpus": gpus, "apps": apps}, ensure_ascii=False))
"""
    out, err, code = run(client, f"/root/miniconda3/bin/python -c {shlex.quote(py)}", timeout=60)
    if code != 0:
        raise RuntimeError(f"GPU probe failed:\nSTDOUT:\n{out}\nSTDERR:\n{err}")
    return json.loads(out.strip())


def launch_detached(client: paramiko.SSHClient, spec: LaunchSpec) -> dict:
    remote = (
        "source /etc/network_turbo >/dev/null 2>&1 || true; "
        "export PATH=/root/miniconda3/bin:$PATH; "
        f"mkdir -p {PROJECT}/logs/uniground_v6 {PROJECT}/logs/uniground_v6/second_host_qwen4b "
        f"{PROJECT}/logs/uniground_v6/second_host_qwen2b {PROJECT}/logs/uniground_v6/qwen2b_table2; "
        f"cd {PROJECT}; "
        f"CUDA_VISIBLE_DEVICES={spec.gpu} nohup /bin/bash -lc {shlex.quote(spec.command_body)} "
        f"> {shlex.quote(spec.log_path)} 2>&1 < /dev/null & echo $!"
    )
    out, err, code = run(client, remote, timeout=60)
    if code != 0:
        raise RuntimeError(f"Launch failed for {spec.name}:\nSTDOUT:\n{out}\nSTDERR:\n{err}")
    return {"name": spec.name, "gpu": spec.gpu, "pid": out.strip(), "log_path": spec.log_path}


def main() -> None:
    gpu2_body = " && ".join(
        [
            build_uniground_cmd(
                model="qwen3-vl-4b",
                dataset="pope",
                output_dir=f"{PROJECT}/logs/uniground_v6/second_host_qwen4b",
                mini_test=200,
            ),
            build_uniground_cmd(
                model="qwen3-vl-4b",
                dataset="chair",
                output_dir=f"{PROJECT}/logs/uniground_v6/second_host_qwen4b",
                mini_test=150,
                chair_max_new_tokens=384,
            ),
        ]
    )
    gpu0_body = " && ".join(
        [
            build_uniground_cmd(
                model="qwen3-vl-2b",
                dataset="pope",
                output_dir=f"{PROJECT}/logs/uniground_v6/second_host_qwen2b",
                mini_test=200,
            ),
            build_bra_cmd(
                dataset="mmbench",
                bra_method="tlra_full",
                output_path="logs/uniground_v6/qwen2b_table2/mmbench_tlra_full_qwen3vl2b.json",
                n_samples=300,
            ),
            build_bra_cmd(
                dataset="mmbench",
                bra_method="tlra_no_vasm",
                output_path="logs/uniground_v6/qwen2b_table2/mmbench_tlra_no_vasm_qwen3vl2b.json",
                n_samples=300,
            ),
            build_bra_cmd(
                dataset="mme",
                bra_method="tlra_full",
                output_path="logs/uniground_v6/qwen2b_table2/mme_tlra_full_qwen3vl2b.json",
                n_samples=300,
            ),
            build_bra_cmd(
                dataset="mme",
                bra_method="tlra_no_vasm",
                output_path="logs/uniground_v6/qwen2b_table2/mme_tlra_no_vasm_qwen3vl2b.json",
                n_samples=300,
            ),
        ]
    )

    specs = [
        LaunchSpec(
            gpu=2,
            name="gpu2_qwen4b_pope_then_chair_full_matrix",
            log_path=f"{PROJECT}/logs/uniground_v6/gpu2_qwen4b_pope_then_chair_full_matrix.log",
            command_body=gpu2_body,
        ),
        LaunchSpec(
            gpu=0,
            name="gpu0_qwen2b_pope_then_mmbench_mme",
            log_path=f"{PROJECT}/logs/uniground_v6/gpu0_qwen2b_pope_then_mmbench_mme.log",
            command_body=gpu0_body,
        ),
    ]

    client = connect()
    try:
        probe = gpu_probe(client)
        launches = [launch_detached(client, spec) for spec in specs]
    finally:
        client.close()

    print(json.dumps({"probe": probe, "launches": launches}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
