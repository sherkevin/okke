import paramiko


HOST = "connect.westc.seetacloud.com"
PORT = 47559
USER = "root"
PASSWORD = "aMNIL2fW6aoV"


REMOTE_BASH = r"""
echo "=== PROCESS ==="
ps -eo pid,etimes,%cpu,%mem,cmd | grep -E 'bra_eval_matrix.py --model qwen3vl2b --dataset docvqa --n_samples 200 --bra_method tlra_no_vasm' | grep -v grep || true
echo "=== LOG STAT ==="
stat /root/autodl-tmp/BRA_Project/logs/v3_engineer_b/docvqa_contract_ready.log 2>/dev/null || true
echo "=== JSON STAT ==="
stat /root/autodl-tmp/BRA_Project/logs/v3_engineer_b/docvqa_tlra_no_vasm_rerun.json 2>/dev/null || true
echo "=== LOG TAIL ==="
tail -n 40 /root/autodl-tmp/BRA_Project/logs/v3_engineer_b/docvqa_contract_ready.log 2>/dev/null || true
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
