import paramiko

def ssh_run(cmd):
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect('connect.westd.seetacloud.com', port=23427, username='root', password='aMNIL2fW6aoV', timeout=15)
    stdin, stdout, stderr = c.exec_command(cmd, timeout=180)
    out = stdout.read().decode('utf-8', errors='replace')
    err = stderr.read().decode('utf-8', errors='replace')
    c.close()
    return out, err

cmd = r"""
export PATH="/root/miniconda3/bin:$PATH"
cd /root/autodl-tmp/BRA_Project/baselines/

echo '=== Cloning VCD via ghproxy ==='
git clone --depth 1 https://ghproxy.com/https://github.com/DAMO-NLP-SG/VCD.git 2>&1 | tail -5

echo '=== Cloning OPERA via ghproxy ==='
git clone --depth 1 https://ghproxy.com/https://github.com/shikiw/OPERA.git 2>&1 | tail -5

echo '=== Verify ==='
ls -d VCD/ OPERA/ 2>&1
"""

out, err = ssh_run(cmd)
print(out)
if err.strip():
    print(f"STDERR: {err[:1000]}")
