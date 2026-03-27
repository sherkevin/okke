import paramiko

def ssh_run(cmd):
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect('connect.westd.seetacloud.com', port=23427, username='root', password='aMNIL2fW6aoV', timeout=15)
    stdin, stdout, stderr = c.exec_command(cmd, timeout=120)
    out = stdout.read().decode('utf-8', errors='replace')
    err = stderr.read().decode('utf-8', errors='replace')
    c.close()
    return out, err

cmd = r"""
export PATH="/root/miniconda3/bin:$PATH"
export HF_ENDPOINT="https://hf-mirror.com"
cd /root/autodl-tmp/BRA_Project/baselines/

echo '=== Cloning VCD ==='
git clone --depth 1 https://github.com/DAMO-NLP-SG/VCD.git 2>&1 | tail -3

echo '=== Cloning OPERA ==='
git clone --depth 1 https://github.com/shikiw/OPERA.git 2>&1 | tail -3

echo '=== Verify ==='
ls -la VCD/ OPERA/ 2>&1 | head -20
"""

out, err = ssh_run(cmd)
print(out)
if err.strip():
    print(f"\n--- STDERR ---\n{err[:2000]}")
