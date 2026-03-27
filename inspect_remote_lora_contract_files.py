import paramiko


HOST = "connect.westc.seetacloud.com"
PORT = 47559
USER = "root"
PASSWORD = "aMNIL2fW6aoV"


REMOTE_PY = r"""
from pathlib import Path

targets = [
    Path("/root/autodl-tmp/BRA_Project/BASE_LORA_MATCHED_BUDGET_CONTRACT_20260322.md"),
    Path("/root/autodl-tmp/BRA_Project/configs/base_lora_matched_budget_qwen3vl2b_seed1.json"),
    Path("/root/autodl-tmp/BRA_Project/configs/base_lora_matched_budget_qwen3vl2b_seed2.json"),
    Path("/root/autodl-tmp/BRA_Project/configs/base_lora_matched_budget_qwen3vl2b_seed3.json"),
    Path("/root/autodl-tmp/BRA_Project/configs/base_lora_matched_budget_qwen3vl2b_seed3_precheck.json"),
    Path("/root/autodl-tmp/BRA_Project/logs/v3_engineer_a/phi_candidates.txt"),
    Path("/root/autodl-tmp/BRA_Project/logs/v3_engineer_a/phi_freeze_decision.txt"),
]

for path in targets:
    print("=" * 80)
    print(path)
    print("=" * 80)
    if not path.exists():
        print("MISSING")
        continue
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    for line in lines[:120]:
        print(line)
"""


def main():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username=USER, password=PASSWORD, timeout=45)
    cmd = f'export PATH="/root/miniconda3/bin:$PATH"; /root/miniconda3/bin/python - <<\'PY\'\n{REMOTE_PY}\nPY'
    stdin, stdout, stderr = client.exec_command(cmd, timeout=120)
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    client.close()
    print(out)
    if err.strip():
        print("=== STDERR ===")
        print(err)


if __name__ == "__main__":
    main()
