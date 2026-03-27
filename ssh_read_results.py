import paramiko
import json

def ssh_run(cmd, timeout=15):
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect('connect.westd.seetacloud.com', port=23427, username='root', password='aMNIL2fW6aoV', timeout=15)
    stdin, stdout, stderr = c.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode('utf-8', errors='replace')
    c.close()
    return out

files = [
    "base_pope", "vcd_pope", "opera_pope",
    "base_chair", "vcd_chair", "opera_chair",
]

results = {}
for prefix in files:
    raw = ssh_run(f"cat /root/autodl-tmp/BRA_Project/logs/minitest/{prefix}_*.json 2>/dev/null")
    if raw.strip():
        data = json.loads(raw)
        results[prefix] = data
        print(f"\n{'='*60}")
        print(f"  {prefix}")
        print(f"{'='*60}")
        for k, v in data.items():
            if k not in ("sample_captions", "timestamp", "errors"):
                print(f"  {k:25s} = {v}")
        if data.get("n_errors", 0) > 0:
            print(f"  ** {data['n_errors']} errors **")

print("\n\n" + "="*70)
print("  COMPARISON TABLE")
print("="*70)
print(f"{'Method':<10} {'Dataset':<8} {'Accuracy/CHAIR-s':>16} {'F1/CHAIR-i':>12} {'AGL':>8} {'ITL(ms)':>10} {'VRAM(GB)':>10} {'Errors':>8}")
print("-"*70)
for prefix in files:
    d = results.get(prefix, {})
    method = d.get("method", "?")
    dataset = d.get("dataset", "?")
    if dataset == "pope":
        score1 = f"Acc={d.get('accuracy','?')}"
        score2 = f"F1={d.get('f1','?')}"
    else:
        score1 = f"Cs={d.get('chair_s','?')}"
        score2 = f"Ci={d.get('chair_i','?')}"
    agl = d.get("agl", "?")
    itl = d.get("itl_ms_per_token", "?")
    vram = d.get("peak_vram_gb", "?")
    errs = d.get("n_errors", 0)
    print(f"{method:<10} {dataset:<8} {score1:>16} {score2:>12} {agl:>8} {itl:>10} {vram:>10} {errs:>8}")
