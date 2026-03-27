import paramiko
import os
import json

HOST = 'connect.westd.seetacloud.com'
PORT = 23427
USER = 'root'
PASS = 'aMNIL2fW6aoV'

local_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'experiment_logs', 'json')
os.makedirs(local_dir, exist_ok=True)

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(HOST, port=PORT, username=USER, password=PASS, timeout=15)
sftp = client.open_sftp()

remote_dir = '/root/autodl-tmp/BRA_Project/logs/minitest/'
files = sftp.listdir(remote_dir)
json_files = [f for f in files if f.endswith('.json')]

for f in sorted(json_files):
    remote_path = remote_dir + f
    local_path = os.path.join(local_dir, f)
    sftp.get(remote_path, local_path)
    with open(local_path) as fh:
        data = json.load(fh)
    method = data.get('method', '?')
    dataset = data.get('dataset', '?')
    print(f"  Downloaded {f}  ({method}/{dataset})")

sftp.close()
client.close()
print(f"\nAll {len(json_files)} JSON files saved to {local_dir}")
