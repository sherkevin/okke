import paramiko
import time

HOST = 'connect.westd.seetacloud.com'
PORT = 23427
USER = 'root'
PASS = 'aMNIL2fW6aoV'

def ssh_run(cmd, timeout=30):
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, port=PORT, username=USER, password=PASS, timeout=15)
    stdin, stdout, stderr = c.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode('utf-8', errors='replace')
    err = stderr.read().decode('utf-8', errors='replace')
    c.close()
    return out, err

# Check if another process is using the GPU
out, _ = ssh_run("nvidia-smi")
print("=== GPU status ===")
print(out)

# Check if there's a competing process
out, _ = ssh_run("ps aux | grep -E 'python|qwen' | grep -v grep")
print("\n=== Python processes ===")
print(out if out.strip() else "(none)")

# Check existing POPE JSON contents
for method in ['base', 'vcd', 'opera']:
    out, _ = ssh_run(f"cat /root/autodl-tmp/BRA_Project/logs/minitest/{method}_pope_*.json 2>/dev/null")
    print(f"\n=== {method}_pope JSON ===")
    print(out[:500] if out else "(not found)")
