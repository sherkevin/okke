import paramiko

def ssh_run(cmd):
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect('connect.westd.seetacloud.com', port=23427, username='root', password='aMNIL2fW6aoV', timeout=15)
    stdin, stdout, stderr = c.exec_command(cmd, timeout=30)
    out = stdout.read().decode('utf-8', errors='replace')
    c.close()
    return out

cmd = r"""
export PATH="/root/miniconda3/bin:$PATH"

echo '=== count val2014 images ==='
find /root/autodl-tmp/BRA_Project/datasets/coco2014/val2014/ -name "*.jpg" 2>/dev/null | wc -l

echo '=== POPE image ref check ==='
python3 -c "
import json
lines = open('/root/autodl-tmp/BRA_Project/datasets/POPE/output/coco/coco_pope_random.json').readlines()
d = json.loads(lines[0])
print('POPE image field:', d['image'])
print('question:', d['text'])
print('label:', d['label'])
import os
# check if image exists in val2014
p1 = '/root/autodl-tmp/BRA_Project/datasets/coco2014/val2014/' + d['image']
print('exists in val2014/?', os.path.exists(p1))
" 2>&1

echo '=== COCO categories ==='
python3 -c "
import json
d = json.load(open('/root/autodl-tmp/BRA_Project/datasets/coco2014/annotations/instances_val2014.json'))
cats = {c['id']: c['name'] for c in d['categories']}
print('n_categories:', len(cats))
print('first 10:', list(cats.values())[:10])
print('n_images:', len(d['images']))
" 2>&1

echo '=== POPE total lines ==='
wc -l /root/autodl-tmp/BRA_Project/datasets/POPE/output/coco/*.json
"""

print(ssh_run(cmd))
