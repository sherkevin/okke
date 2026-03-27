import paramiko

def ssh_run(cmd):
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect('connect.westd.seetacloud.com', port=23427, username='root', password='aMNIL2fW6aoV', timeout=15)
    stdin, stdout, stderr = c.exec_command(cmd, timeout=60)
    out = stdout.read().decode('utf-8', errors='replace')
    err = stderr.read().decode('utf-8', errors='replace')
    c.close()
    return out, err

cmd = r"""
export PATH="/root/miniconda3/bin:$PATH"

echo '=== baselines dir ==='
ls -la ~/autodl-tmp/BRA_Project/baselines/ 2>/dev/null

echo '=== datasets dir ==='
ls -la ~/autodl-tmp/BRA_Project/datasets/ 2>/dev/null

echo '=== POPE data? ==='
find ~/autodl-tmp/BRA_Project/datasets/ -iname "*pope*" -type f 2>/dev/null | head -10

echo '=== CHAIR / COCO annotations? ==='
find ~/autodl-tmp/BRA_Project/datasets/ -iname "*chair*" -o -iname "*caption*" -o -iname "*instances*" 2>/dev/null | head -10

echo '=== COCO val2014 images ==='
ls ~/autodl-tmp/BRA_Project/datasets/coco2014/ 2>/dev/null | head -10
ls ~/autodl-tmp/BRA_Project/datasets/coco2014/val2014/ 2>/dev/null | head -5
NIMG=$(ls ~/autodl-tmp/BRA_Project/datasets/coco2014/val2014/*.jpg 2>/dev/null | wc -l)
echo "Total val2014 images: $NIMG"

echo '=== existing python scripts ==='
find ~/autodl-tmp/BRA_Project/ -maxdepth 1 -name "*.py" 2>/dev/null

echo '=== logs dir ==='
ls -la ~/autodl-tmp/BRA_Project/logs/ 2>/dev/null

echo '=== GPU status ==='
nvidia-smi 2>/dev/null | head -12 || echo "No GPU"

echo '=== disk ==='
df -h /root/autodl-tmp

echo '=== bra_operator.py exists? ==='
head -5 ~/autodl-tmp/BRA_Project/bra_operator.py 2>/dev/null

echo '=== existing SHARED_DATA_REGISTRY? ==='
cat ~/autodl-tmp/BRA_Project/SHARED_DATA_REGISTRY.md 2>/dev/null || echo "(not found)"

echo '=== HF datasets downloaded ==='
ls ~/autodl-tmp/BRA_Project/datasets/ 2>/dev/null
"""

out, err = ssh_run(cmd)
print(out[:8000])
if err.strip():
    print(f"\n--- STDERR (first 1000) ---\n{err[:1000]}")
