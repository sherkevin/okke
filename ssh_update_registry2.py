import paramiko

HOST = 'connect.westd.seetacloud.com'
PORT = 23427
USER = 'root'
PASS = 'aMNIL2fW6aoV'

registry_content = r"""# SHARED DATA REGISTRY
> BRA Project — Centralized index of all datasets, models, logs, and experiment records.
> Last updated: 2026-03-21 21:44 CST

---

## 1. Datasets

| Dataset | Path | Notes |
|---------|------|-------|
| COCO val2014 images | `/root/autodl-tmp/BRA_Project/datasets/coco2014/val2014/` | ~40k images |
| COCO annotations | `/root/autodl-tmp/BRA_Project/datasets/coco2014/annotations/` | instances_val2014.json, captions_val2014.json |
| POPE splits | `/root/autodl-tmp/BRA_Project/datasets/POPE/output/coco/` | random / popular / adversarial |

## 2. Models

| Model | Path | Notes |
|-------|------|-------|
| Qwen3-VL-8B-Instruct | `/root/autodl-tmp/BRA_Project/models/Qwen3-VL-8B-Instruct/` | Primary eval model |
| Qwen3-VL-2B-Instruct | `/root/autodl-tmp/BRA_Project/models/Qwen3-VL-2B-Instruct/` | Dev / smoke test |

## 3. Baseline Source Repos

| Baseline | Path | Cloned |
|----------|------|--------|
| VCD (DAMO-NLP-SG) | `/root/autodl-tmp/BRA_Project/baselines/VCD/` | 2026-03-21 via network_turbo |
| OPERA (shikiw) | `/root/autodl-tmp/BRA_Project/baselines/OPERA/` | 2026-03-21 via network_turbo |

## 4. Core Code

| File | Path | Description |
|------|------|-------------|
| BRA operator | `/root/autodl-tmp/BRA_Project/bra_operator.py` | BRA algorithm (monkey-patch + forward hook) |
| Baseline processors | `/root/autodl-tmp/BRA_Project/baseline_processors.py` | VCD / OPERA / DoLa as LogitsProcessor |
| Eval pipeline | `/root/autodl-tmp/BRA_Project/run_eval_pipeline.py` | Unified --method --dataset --mini_test runner |
| BRA smoke test | `/root/autodl-tmp/BRA_Project/bra_smoke_test.py` | 20-sample AGL deviation test |

---

## 5. Experiment Logs

### Experiment #1: Mini-Test 50 Baseline Evaluation (2026-03-21)

**Config**: Qwen3-VL-8B, RTX 5090 (32GB), greedy decoding, max_new_tokens=128, mini_test=50

**Local report**: `experiment_logs/minitest_50_baseline_eval_20260321.md`

#### JSON Log Paths

| Method | Dataset | Server Path |
|--------|---------|-------------|
| base | POPE | `/root/autodl-tmp/BRA_Project/logs/minitest/base_pope_20260321_202446.json` |
| vcd | POPE | `/root/autodl-tmp/BRA_Project/logs/minitest/vcd_pope_20260321_202828.json` |
| opera | POPE | `/root/autodl-tmp/BRA_Project/logs/minitest/opera_pope_20260321_204435.json` |
| base | CHAIR | `/root/autodl-tmp/BRA_Project/logs/minitest/base_chair_20260321_213138.json` |
| vcd | CHAIR | `/root/autodl-tmp/BRA_Project/logs/minitest/vcd_chair_20260321_213639.json` |
| opera | CHAIR | `/root/autodl-tmp/BRA_Project/logs/minitest/opera_chair_20260321_213946.json` |

#### POPE Results (random split, 50 samples)

| Method | Accuracy | F1     | AGL   | ITL (ms/tok) | Peak VRAM (GB) | Errors |
|--------|----------|--------|-------|--------------|----------------|--------|
| Base   | 0.88     | 0.8636 | 86.34 | 26.98        | 17.648         | 0      |
| VCD    | 0.88     | 0.8636 | 77.04 | 55.30        | 17.704         | 0      |
| OPERA  | 0.8889   | 0.8750 | 83.22 | 302.89       | 17.449         | 14     |

#### CHAIR Results (50 images)

| Method | CHAIR-s | CHAIR-i | AGL   | ITL (ms/tok) | Peak VRAM (GB) | Errors |
|--------|---------|---------|-------|--------------|----------------|--------|
| Base   | 0.1333  | 0.2879  | 128.0 | 21.95        | 17.681         | 0      |
| VCD    | 0.1197  | 0.2869  | 128.0 | 43.14        | 17.765         | 0      |
| OPERA  | 0.1319  | 0.2984  | 128.0 | 26.29        | 18.104         | 0      |

#### Base vs VCD Delta (JSON)

```json
{
  "comparison": "Base vs VCD (50 mini-test samples)",
  "pope": {
    "base_agl": 86.34, "vcd_agl": 77.04, "agl_delta_pct": -10.77,
    "base_itl_ms": 26.98, "vcd_itl_ms": 55.30, "itl_delta_pct": +104.97,
    "accuracy_delta": 0.0
  },
  "chair": {
    "base_agl": 128.0, "vcd_agl": 128.0, "agl_delta_pct": 0.0,
    "base_itl_ms": 21.95, "vcd_itl_ms": 43.14, "itl_delta_pct": +96.54,
    "chair_s_delta": -0.0136, "chair_i_delta": -0.0010
  }
}
```

#### Key Findings

1. VCD doubles ITL (~2x) due to shadow forward pass with diffusion-noised KV cache
2. VCD reduces POPE AGL by 10.8% — model becomes more conservative
3. OPERA impractical on Qwen3-VL: requires eager attn → 11.2x ITL, 28% OOM rate on POPE
4. CHAIR AGL saturated at max_new_tokens=128 for all methods
5. OPERA's `output_attentions` flag stripped by `model.generate()`, limiting penalty effectiveness

---

*Add new experiments below this line.*
"""

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(HOST, port=PORT, username=USER, password=PASS, timeout=15)
sftp = client.open_sftp()

remote_path = "/root/autodl-tmp/BRA_Project/SHARED_DATA_REGISTRY.md"
with sftp.open(remote_path, 'w') as f:
    f.write(registry_content)

sftp.close()

stdin, stdout, stderr = client.exec_command(f"wc -l {remote_path}")
print(f"Registry updated: {stdout.read().decode().strip()}")
client.close()
