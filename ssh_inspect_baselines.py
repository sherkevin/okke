import paramiko

def ssh_run(cmd, timeout=30):
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect('connect.westd.seetacloud.com', port=23427, username='root', password='aMNIL2fW6aoV', timeout=15)
    stdin, stdout, stderr = c.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode('utf-8', errors='replace')
    c.close()
    return out

base = "/root/autodl-tmp/BRA_Project/baselines"

print("=== VCD vcd_sample.py (core logic) ===")
print(ssh_run(f"cat {base}/VCD/vcd_utils/vcd_sample.py"))

print("\n\n=== VCD vcd_add_noise.py ===")
print(ssh_run(f"cat {base}/VCD/vcd_utils/vcd_add_noise.py"))
