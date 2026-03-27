import paramiko
import os
import time

HOST = 'connect.westd.seetacloud.com'
PORT = 23427
USER = 'root'
PASS = 'aMNIL2fW6aoV'
REMOTE = '/root/autodl-tmp/BRA_Project/'

LOCAL_DIR = os.path.dirname(os.path.abspath(__file__))

# Step 1: upload updated files
print("=== UPLOADING ===")
client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(HOST, port=PORT, username=USER, password=PASS, timeout=15)
sftp = client.open_sftp()
for f in ['baseline_processors.py', 'run_eval_pipeline.py', 'bra_operator.py']:
    sftp.put(os.path.join(LOCAL_DIR, f), REMOTE + f)
    print(f"  uploaded {f}")
sftp.close()
client.close()

# Step 2: launch the full evaluation via screen
print("\n=== LAUNCHING EVALUATION ===")
client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(HOST, port=PORT, username=USER, password=PASS, timeout=15)

run_script = r"""
export PATH="/root/miniconda3/bin:$PATH"
cd /root/autodl-tmp/BRA_Project
mkdir -p logs/minitest

echo "[$(date)] Starting evaluation pipeline" | tee logs/minitest/run_all.log

echo "=== [1/6] base + pope ===" | tee -a logs/minitest/run_all.log
python3 run_eval_pipeline.py --method base --dataset pope --mini_test 50 2>&1 | tee -a logs/minitest/run_all.log

echo "=== [2/6] vcd + pope ===" | tee -a logs/minitest/run_all.log
python3 run_eval_pipeline.py --method vcd --dataset pope --mini_test 50 2>&1 | tee -a logs/minitest/run_all.log

echo "=== [3/6] opera + pope ===" | tee -a logs/minitest/run_all.log
python3 run_eval_pipeline.py --method opera --dataset pope --mini_test 50 2>&1 | tee -a logs/minitest/run_all.log

echo "=== [4/6] base + chair ===" | tee -a logs/minitest/run_all.log
python3 run_eval_pipeline.py --method base --dataset chair --mini_test 50 2>&1 | tee -a logs/minitest/run_all.log

echo "=== [5/6] vcd + chair ===" | tee -a logs/minitest/run_all.log
python3 run_eval_pipeline.py --method vcd --dataset chair --mini_test 50 2>&1 | tee -a logs/minitest/run_all.log

echo "=== [6/6] opera + chair ===" | tee -a logs/minitest/run_all.log
python3 run_eval_pipeline.py --method opera --dataset chair --mini_test 50 2>&1 | tee -a logs/minitest/run_all.log

echo "[$(date)] ALL DONE" | tee -a logs/minitest/run_all.log
echo "=== JSON outputs ===" | tee -a logs/minitest/run_all.log
ls -la logs/minitest/*.json | tee -a logs/minitest/run_all.log
"""

# Launch in screen so it survives SSH disconnect
screen_cmd = f'screen -dmS eval_run bash -c \'{run_script}\''
stdin, stdout, stderr = client.exec_command(screen_cmd, timeout=10)
stdout.read()
time.sleep(2)

# Verify screen started
stdin, stdout, stderr = client.exec_command('screen -ls 2>&1')
print(stdout.read().decode())

client.close()
print("Evaluation launched in screen 'eval_run'. Monitor with:")
print("  ssh root@... 'tail -f /root/autodl-tmp/BRA_Project/logs/minitest/run_all.log'")
