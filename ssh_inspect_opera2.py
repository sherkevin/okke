import paramiko

def ssh_run(cmd, timeout=30):
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect('connect.westd.seetacloud.com', port=23427, username='root', password='aMNIL2fW6aoV', timeout=15)
    stdin, stdout, stderr = c.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode('utf-8', errors='replace')
    c.close()
    return out

base = "/root/autodl-tmp/BRA_Project/baselines/OPERA"

# OPERA's core is patched into transformers beam_search. Find it.
print("=== OPERA beam_search patch location ===")
print(ssh_run(f"find {base}/transformers-4.29.2/ -name 'utils.py' -path '*/generation/*' 2>/dev/null"))

print("\n=== OPERA's generation utils (opera_decoding section) ===")
print(ssh_run(f"grep -n 'opera_decoding\\|penalty_weights\\|over.trust\\|retrospect\\|key_position\\|selected_beam' {base}/transformers-4.29.2/src/transformers/generation/utils.py 2>/dev/null | head -40"))

# Get the core opera penalty logic
print("\n=== OPERA penalty core (around key_position / penalty lines) ===")
print(ssh_run(f"grep -n -A3 'penalty_weights\\|key_position' {base}/transformers-4.29.2/src/transformers/generation/utils.py 2>/dev/null | head -60"))
