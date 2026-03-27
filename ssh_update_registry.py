import paramiko

HOST = 'connect.westd.seetacloud.com'
PORT = 23427
USER = 'root'
PASS = 'aMNIL2fW6aoV'

registry_content = r"""# SHARED DATA REGISTRY
> Auto-updated by evaluation pipeline

## Mini-Test Evaluation Logs (--mini_test 50)

### JSON Log Paths

| Method | Dataset | Log File |
|--------|---------|----------|
| base   | POPE    | `/root/autodl-tmp/BRA_Project/logs/minitest/base_pope_20260321_202446.json` |
| vcd    | POPE    | `/root/autodl-tmp/BRA_Project/logs/minitest/vcd_pope_20260321_202828.json` |
| opera  | POPE    | `/root/autodl-tmp/BRA_Project/logs/minitest/opera_pope_20260321_204435.json` |
| base   | CHAIR   | `/root/autodl-tmp/BRA_Project/logs/minitest/base_chair_20260321_213138.json` |
| vcd    | CHAIR   | `/root/autodl-tmp/BRA_Project/logs/minitest/vcd_chair_20260321_213639.json` |
| opera  | CHAIR   | `/root/autodl-tmp/BRA_Project/logs/minitest/opera_chair_20260321_213946.json` |

---

### Full Comparison Table (50 samples, Qwen3-VL-8B-Instruct, RTX 5090)

#### POPE (random split)

| Method | Accuracy | F1     | AGL   | ITL (ms/tok) | Peak VRAM (GB) | Errors |
|--------|----------|--------|-------|--------------|----------------|--------|
| Base   | 0.88     | 0.8636 | 86.34 | 26.98        | 17.648         | 0      |
| VCD    | 0.88     | 0.8636 | 77.04 | 55.30        | 17.704         | 0      |
| OPERA  | 0.8889   | 0.875  | 83.22 | 302.89       | 17.449         | 14     |

#### CHAIR (50 images, "Please describe this image in detail.")

| Method | CHAIR-s | CHAIR-i | AGL   | ITL (ms/tok) | Peak VRAM (GB) | Errors |
|--------|---------|---------|-------|--------------|----------------|--------|
| Base   | 0.1333  | 0.2879  | 128.0 | 21.95        | 17.681         | 0      |
| VCD    | 0.1197  | 0.2869  | 128.0 | 43.14        | 17.765         | 0      |
| OPERA  | 0.1319  | 0.2984  | 128.0 | 26.29        | 18.104         | 0      |

---

### Base vs VCD Delta Summary (JSON)

```json
{
  "comparison": "Base vs VCD on 50 mini-test samples",
  "pope": {
    "base_agl": 86.34,
    "vcd_agl": 77.04,
    "agl_delta": "-10.77%  (VCD makes model less verbose)",
    "base_itl_ms": 26.98,
    "vcd_itl_ms": 55.30,
    "itl_delta": "+104.97%  (VCD ~2x slower due to shadow forward pass)",
    "accuracy_delta": "0.0  (no change on this small sample)"
  },
  "chair": {
    "base_agl": 128.0,
    "vcd_agl": 128.0,
    "agl_delta": "0.0%  (both hit max_new_tokens=128)",
    "base_itl_ms": 21.95,
    "vcd_itl_ms": 43.14,
    "itl_delta": "+96.54%  (VCD ~2x slower)",
    "chair_s_delta": "-0.0136  (VCD slightly lower sentence hallucination)",
    "chair_i_delta": "-0.0010  (negligible object hallucination difference)"
  }
}
```

---

### Key Findings

1. **VCD doubles latency**: ITL increases ~2x on both POPE and CHAIR due to the mandatory shadow forward pass per token (diffusion-noised image KV cache).
2. **VCD reduces POPE verbosity**: AGL drops from 86.34 to 77.04 (-10.8%), confirming VCD makes the model more concise / conservative on yes/no questions.
3. **OPERA is impractical on Qwen3-VL**: With `attn_implementation='eager'` (required for attention weight capture), ITL explodes to 302.89 ms/tok (11.2x base). 14 out of 50 POPE samples hit OOM. FlashAttention/SDPA is incompatible with OPERA's attention-monitoring mechanism.
4. **CHAIR AGL saturated**: All methods hit `max_new_tokens=128` for detailed descriptions, so CHAIR AGL differences are masked. Increase `max_new_tokens` for differentiated AGL measurement.
5. **OPERA attention hook limitation**: The `output_attentions` flag is stripped by `model.generate()`, so the OPERA hook captures weights only when the attention module returns them by default in eager mode. The penalty effect is minimal in this configuration.

---

### Datasets

| Dataset | Path |
|---------|------|
| COCO val2014 images | `/root/autodl-tmp/BRA_Project/datasets/coco2014/val2014/` |
| COCO annotations    | `/root/autodl-tmp/BRA_Project/datasets/coco2014/annotations/` |
| POPE splits         | `/root/autodl-tmp/BRA_Project/datasets/POPE/output/coco/` |

### Model

| Model | Path |
|-------|------|
| Qwen3-VL-8B-Instruct | `/root/autodl-tmp/BRA_Project/models/Qwen3-VL-8B-Instruct/` |

### Baseline Source Repos

| Baseline | Path |
|----------|------|
| VCD  | `/root/autodl-tmp/BRA_Project/baselines/VCD/` |
| OPERA | `/root/autodl-tmp/BRA_Project/baselines/OPERA/` |
"""

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(HOST, port=PORT, username=USER, password=PASS, timeout=15)
sftp = client.open_sftp()

remote_path = "/root/autodl-tmp/BRA_Project/SHARED_DATA_REGISTRY.md"
with sftp.open(remote_path, 'w') as f:
    f.write(registry_content)

sftp.close()

# Verify
stdin, stdout, stderr = client.exec_command(f"wc -l {remote_path} && head -5 {remote_path}")
print(stdout.read().decode())
client.close()
print("SHARED_DATA_REGISTRY.md updated successfully.")
