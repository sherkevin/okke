import paramiko, os

key_path = os.path.expanduser(r"~\.ssh\id_ed25519_autodl.pub")
key = open(key_path).read().strip()

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("connect.westd.seetacloud.com", port=23427, username="root", password="aMNIL2fW6aoV", timeout=30)

cmd = f'mkdir -p ~/.ssh && echo "{key}" >> ~/.ssh/authorized_keys && chmod 700 ~/.ssh && chmod 600 ~/.ssh/authorized_keys && echo KEY_DEPLOYED'
stdin, stdout, stderr = c.exec_command(cmd, timeout=15)
print(stdout.read().decode())
err = stderr.read().decode()
if err:
    print("STDERR:", err)
c.close()
