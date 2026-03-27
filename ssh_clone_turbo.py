import paramiko

def ssh_run(cmd, timeout=180):
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect('connect.westd.seetacloud.com', port=23427, username='root', password='aMNIL2fW6aoV', timeout=15)
    stdin, stdout, stderr = c.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode('utf-8', errors='replace')
    err = stderr.read().decode('utf-8', errors='replace')
    c.close()
    return out, err

cmd = r"""
source /etc/network_turbo
export PATH="/root/miniconda3/bin:$PATH"
cd /root/autodl-tmp/BRA_Project/baselines/

echo '=== Cloning VCD ==='
rm -rf VCD 2>/dev/null
git clone --depth 1 https://github.com/DAMO-NLP-SG/VCD.git 2>&1 | tail -5

echo '=== Cloning OPERA ==='
rm -rf OPERA 2>/dev/null
git clone --depth 1 https://github.com/shikiw/OPERA.git 2>&1 | tail -5

echo '=== Verify ==='
ls -d VCD/ OPERA/ 2>&1
echo '=== VCD key files ==='
find VCD/ -maxdepth 2 -name "*.py" | head -15
echo '=== OPERA key files ==='
find OPERA/ -maxdepth 2 -name "*.py" | head -15
"""

out, err = ssh_run(cmd)
print(out)
if err.strip():
    print(f"\n--- STDERR ---\n{err[:2000]}")
