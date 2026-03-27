import paramiko

def ssh_run(cmd, timeout=30):
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect('connect.westd.seetacloud.com', port=23427, username='root', password='aMNIL2fW6aoV', timeout=15)
    stdin, stdout, stderr = c.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode('utf-8', errors='replace')
    c.close()
    return out

base = "/root/autodl-tmp/BRA_Project/baselines"

# Find OPERA's core penalty mechanism
print("=== OPERA key files ===")
print(ssh_run(f"find {base}/OPERA/ -maxdepth 3 -name '*.py' | grep -v transformers | grep -v __pycache__"))

print("\n=== OPERA pope_eval.py (look for penalty/opera logic) ===")
print(ssh_run(f"grep -n 'penalty\\|opera\\|beam\\|rollback\\|attention\\|logits_processor\\|stopping' {base}/OPERA/pope_eval.py 2>/dev/null | head -30"))

print("\n=== OPERA opera_decoding search ===")
print(ssh_run(f"find {base}/OPERA/ -name '*.py' -exec grep -l 'opera\\|penalty\\|retrospect\\|over.trust' {{}} \\; 2>/dev/null | grep -v __pycache__ | grep -v transformers"))
