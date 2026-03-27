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
  --dataset video_mme \
  --n_samples 32 \
  --bra_method tlra_full \
  --output logs/v3_engineer_b/video_mme_tlra_full_smoke.json
' > logs/v3_engineer_b/video_mme_smoke.log 2>&1 &
echo "launched: video_mme_smoke"
sleep 2
ps -eo pid,etimes,cmd | grep -E 'bra_eval_matrix.py --model qwen3vl2b --dataset video_mme' | grep -v grep || true
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
