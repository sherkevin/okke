import paramiko
import time

HOST = 'connect.westd.seetacloud.com'
PORT = 23427
USER = 'root'
PASS = 'aMNIL2fW6aoV'

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(HOST, port=PORT, username=USER, password=PASS, timeout=15)

run_script = r"""
source /etc/network_turbo 2>/dev/null
export PATH="/root/miniconda3/bin:$PATH"
cd /root/autodl-tmp/BRA_Project
mkdir -p logs/minitest

echo "[$(date)] Starting CHAIR evaluations" | tee logs/minitest/run_chair.log

echo "=== [1/3] base + chair ===" | tee -a logs/minitest/run_chair.log
python3 run_eval_pipeline.py --method base --dataset chair --mini_test 50 2>&1 | tee -a logs/minitest/run_chair.log

echo "=== [2/3] vcd + chair ===" | tee -a logs/minitest/run_chair.log
python3 run_eval_pipeline.py --method vcd --dataset chair --mini_test 50 2>&1 | tee -a logs/minitest/run_chair.log

echo "=== [3/3] opera + chair ===" | tee -a logs/minitest/run_chair.log
python3 run_eval_pipeline.py --method opera --dataset chair --mini_test 50 2>&1 | tee -a logs/minitest/run_chair.log

echo "[$(date)] CHAIR ALL DONE" | tee -a logs/minitest/run_chair.log
ls -la logs/minitest/*.json | tee -a logs/minitest/run_chair.log
"""

screen_cmd = f'screen -dmS chair_eval bash -c \'{run_script}\''
stdin, stdout, stderr = client.exec_command(screen_cmd, timeout=10)
stdout.read()
time.sleep(2)

stdin, stdout, stderr = client.exec_command('screen -ls 2>&1')
print(stdout.read().decode())
client.close()
print("CHAIR evaluation launched in screen 'chair_eval'")
