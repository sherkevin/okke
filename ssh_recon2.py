import paramiko

def ssh_run(cmd):
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect('connect.westd.seetacloud.com', port=23427, username='root', password='aMNIL2fW6aoV', timeout=15)
    stdin, stdout, stderr = c.exec_command(cmd, timeout=30)
    out = stdout.read().decode('utf-8', errors='replace')
    c.close()
    return out

# Check where COCO images actually are
print("=== COCO image locations ===")
print(ssh_run("ls ~/autodl-tmp/BRA_Project/datasets/coco2014/val2014/ 2>/dev/null | head -5; echo '---'; ls ~/autodl-tmp/BRA_Project/datasets/coco2014/*.jpg 2>/dev/null | wc -l; echo 'imgs in coco2014/'; ls ~/autodl-tmp/BRA_Project/datasets/coco2014/val2014/*.jpg 2>/dev/null | wc -l; echo 'imgs in val2014/'"))

# Check POPE data format
print("\n=== POPE data format ===")
print(ssh_run("head -3 ~/autodl-tmp/BRA_Project/datasets/POPE/output/coco/coco_pope_random.json"))

# Check COCO annotations
print("\n=== COCO annotations ===")
print(ssh_run("ls -la ~/autodl-tmp/BRA_Project/datasets/coco2014/annotations/"))

# Check captions format
print("\n=== Captions format (first entry) ===")
print(ssh_run("python3 -c \"import json; d=json.load(open('/root/autodl-tmp/BRA_Project/datasets/coco2014/annotations/captions_val2014.json')); print('keys:', list(d.keys())); print('images[0]:', d['images'][0]); print('annotations[0]:', d['annotations'][0])\" 2>&1"))

# Check instances format for CHAIR
print("\n=== Instances categories (first 5) ===")
print(ssh_run("python3 -c \"import json; d=json.load(open('/root/autodl-tmp/BRA_Project/datasets/coco2014/annotations/instances_val2014.json')); print('categories[:5]:', d['categories'][:5]); print('n_images:', len(d['images'])); print('n_annotations:', len(d['annotations']))\" 2>&1"))

# Check val2014 unzipped status
print("\n=== val2014.zip status ===")
print(ssh_run("ls -lh ~/autodl-tmp/BRA_Project/datasets/coco2014/val2014.zip 2>/dev/null; ls ~/autodl-tmp/BRA_Project/datasets/coco2014/unzip.log 2>/dev/null; cat ~/autodl-tmp/BRA_Project/datasets/coco2014/unzip.log 2>/dev/null | tail -3"))
