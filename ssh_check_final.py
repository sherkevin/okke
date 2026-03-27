import paramiko

def ssh_run(cmd, timeout=15):
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect('connect.westd.seetacloud.com', port=23427, username='root', password='aMNIL2fW6aoV', timeout=15)
    stdin, stdout, stderr = c.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode('utf-8', errors='replace')
    c.close()
    return out

print("=== All JSON files ===")
print(ssh_run("ls -la /root/autodl-tmp/BRA_Project/logs/minitest/*.json 2>/dev/null"))

print("\n=== Last 80 lines of run_all.log ===")
print(ssh_run("tail -80 /root/autodl-tmp/BRA_Project/logs/minitest/run_all.log 2>/dev/null"))

print("\n=== screen sessions ===")
print(ssh_run("screen -ls 2>&1"))
