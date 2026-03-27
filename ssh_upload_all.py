import paramiko
import os

HOST = 'connect.westd.seetacloud.com'
PORT = 23427
USER = 'root'
PASS = 'aMNIL2fW6aoV'
REMOTE_DIR = '/root/autodl-tmp/BRA_Project/'

LOCAL_DIR = os.path.dirname(os.path.abspath(__file__))
FILES = [
    'baseline_processors.py',
    'run_eval_pipeline.py',
    'bra_operator.py',
]

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(HOST, port=PORT, username=USER, password=PASS, timeout=15)

sftp = client.open_sftp()
for fname in FILES:
    local = os.path.join(LOCAL_DIR, fname)
    remote = REMOTE_DIR + fname
    print(f"Uploading {fname} -> {remote}")
    sftp.put(local, remote)
    stat = sftp.stat(remote)
    print(f"  OK  size={stat.st_size} bytes")
sftp.close()

# Create logs/minitest dir
stdin, stdout, stderr = client.exec_command('mkdir -p /root/autodl-tmp/BRA_Project/logs/minitest')
stdout.read()

# Verify imports
verify = r"""
export PATH="/root/miniconda3/bin:$PATH"
cd /root/autodl-tmp/BRA_Project
python3 -c "
import sys; sys.path.insert(0, '.')
from baseline_processors import VCDLogitsProcessor, OPERALogitsProcessor, DoLaLogitsProcessor
print('baseline_processors: OK')
import ast
ast.parse(open('run_eval_pipeline.py').read())
print('run_eval_pipeline: syntax OK')
from bra_operator import BRAOperator, BRAConfig
print('bra_operator: OK')
print('ALL IMPORTS PASSED')
" 2>&1
"""
stdin, stdout, stderr = client.exec_command(verify, timeout=30)
print("\n" + stdout.read().decode())

client.close()
print("Upload complete.")
