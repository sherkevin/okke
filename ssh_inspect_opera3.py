import paramiko

def ssh_run(cmd, timeout=30):
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect('connect.westd.seetacloud.com', port=23427, username='root', password='aMNIL2fW6aoV', timeout=15)
    stdin, stdout, stderr = c.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode('utf-8', errors='replace')
    c.close()
    return out

f = "/root/autodl-tmp/BRA_Project/baselines/OPERA/transformers-4.29.2/src/transformers/generation/utils.py"

# Get the actual penalty computation (lines around 3410-3470)
print("=== OPERA penalty block (lines 3400-3470) ===")
print(ssh_run(f"sed -n '3400,3470p' {f}"))
