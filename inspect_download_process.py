import paramiko


HOST = "connect.westc.seetacloud.com"
PORT = 47559
USER = "root"
PASSWORD = "aMNIL2fW6aoV"


REMOTE_BASH = r"""
export PATH="/root/miniconda3/bin:$PATH"
echo "=== ps ==="
ps -p 8985 -o pid,etimes,pcpu,pmem,stat,cmd || true
echo "=== cwd ==="
readlink -f /proc/8985/cwd 2>/dev/null || true
echo "=== cmdline ==="
tr '\0' ' ' </proc/8985/cmdline 2>/dev/null || true
echo
echo "=== open files tail ==="
lsof -p 8985 2>/dev/null | tail -n 30 || true
"""


def main():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username=USER, password=PASSWORD, timeout=45)
    stdin, stdout, stderr = client.exec_command(REMOTE_BASH, timeout=120)
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    client.close()
    print(out)
    if err.strip():
        print("=== STDERR ===")
        print(err)


if __name__ == "__main__":
    main()
