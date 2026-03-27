import paramiko

def ssh_run(cmd, timeout=15):
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect('connect.westd.seetacloud.com', port=23427, username='root', password='aMNIL2fW6aoV', timeout=15)
    stdin, stdout, stderr = c.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode('utf-8', errors='replace')
    c.close()
    return out

print("=== Last 30 lines of run_chair.log ===")
print(ssh_run("tail -30 /root/autodl-tmp/BRA_Project/logs/minitest/run_chair.log 2>/dev/null"))

print("\n=== All JSON files ===")
print(ssh_run("ls -la /root/autodl-tmp/BRA_Project/logs/minitest/*.json 2>/dev/null"))

print("\n=== GPU ===")
print(ssh_run("nvidia-smi 2>/dev/null | head -12"))

print("\n=== screen alive? ===")
print(ssh_run("screen -ls 2>&1 | grep chair"))
