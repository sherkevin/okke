import paramiko


HOST = "connect.westc.seetacloud.com"
PORT = 47559
USER = "root"
PASSWORD = "aMNIL2fW6aoV"


REMOTE_BASH = r"""
source /etc/network_turbo >/dev/null 2>&1 || true
export PATH="/root/miniconda3/bin:$PATH"
cd /root/autodl-tmp/BRA_Project
mkdir -p logs/v3_engineer_b

CUDA_VISIBLE_DEVICES=3 nohup bash -lc '
cd /root/autodl-tmp/BRA_Project
export PATH="/root/miniconda3/bin:$PATH"
python bra_eval_matrix.py \
  --model qwen3vl2b \
  --dataset docvqa \
  --n_samples 20 \
  --bra_method tlra_full \
  --output logs/v3_engineer_b/docvqa_smoke_tlra_full.json
' > logs/v3_engineer_b/docvqa_smoke_tlra_full.log 2>&1 &
echo "launched: docvqa_smoke"

CUDA_VISIBLE_DEVICES=2 nohup bash -lc '
cd /root/autodl-tmp/BRA_Project
export PATH="/root/miniconda3/bin:$PATH"
python bra_eval_matrix.py \
  --model qwen3vl2b \
  --dataset freak \
  --n_samples 300 \
  --bra_method tlra_meanpool \
  --output logs/v3_engineer_b/freak_tlra_meanpool.json ;
python bra_eval_matrix.py \
  --model qwen3vl2b \
  --dataset freak \
  --n_samples 300 \
  --bra_method tlra_adaptivetopk \
  --output logs/v3_engineer_b/freak_tlra_adaptivetopk.json
' > logs/v3_engineer_b/freak_parity.log 2>&1 &
echo "launched: freak_parity"

sleep 2
ps -eo pid,etimes,cmd | grep -E 'bra_eval_matrix.py' | grep -v grep || true
"""


def main():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username=USER, password=PASSWORD, timeout=45)
    stdin, stdout, stderr = client.exec_command(REMOTE_BASH, timeout=120)
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    client.close()
    print(out)
    if err.strip():
        print("=== STDERR ===")
        print(err)


if __name__ == "__main__":
    main()
