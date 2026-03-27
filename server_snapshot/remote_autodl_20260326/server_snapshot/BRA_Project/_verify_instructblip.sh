#!/bin/bash
MODEL=/root/autodl-tmp/BRA_Project/models/instructblip-vicuna-7b

echo "=== model.safetensors.index.json ==="
python3 -c "
import json
idx = json.load(open('$MODEL/model.safetensors.index.json'))
print('  total_size:', idx.get('metadata',{}).get('total_size', 'N/A'))
shards = sorted(set(idx['weight_map'].values()))
print('  shards:', shards)
print('  total weights:', len(idx['weight_map']))
" 2>/dev/null || echo "  no index file"

echo ""
echo "=== config.json ==="
python3 -c "
import json
cfg = json.load(open('$MODEL/config.json'))
print('  model_type:', cfg.get('model_type','?'))
arch = cfg.get('architectures',['?'])
print('  architectures:', arch)
# check language model
lm = cfg.get('language_model',{})
if lm:
    print('  language_model.model_type:', lm.get('model_type','?'))
    print('  language_model.num_hidden_layers:', lm.get('num_hidden_layers','?'))
    print('  language_model.hidden_size:', lm.get('hidden_size','?'))
    print('  language_model.num_attention_heads:', lm.get('num_attention_heads','?'))
    print('  language_model.vocab_size:', lm.get('vocab_size','?'))
    intermediate = lm.get('intermediate_size','?')
    print('  language_model.intermediate_size:', intermediate)
" 2>/dev/null || echo "  cannot parse config"

echo ""
echo "=== All shard sizes ==="
ls -lh $MODEL/model-*.safetensors 2>/dev/null

echo ""
echo "=== Quick load test (no GPU, just metadata) ==="
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
python3 -c "
import torch
from safetensors import safe_open
import os

shard1 = '$MODEL/model-00001-of-00004.safetensors'
print('Opening shard 1...')
with safe_open(shard1, framework='pt', device='cpu') as f:
    keys = list(f.keys())
    print(f'  Keys in shard1: {len(keys)}')
    print(f'  First 5 keys: {keys[:5]}')
    print(f'  Last 5 keys: {keys[-5:]}')
    # detect model size from weight shapes
    for k in keys:
        if 'embed_tokens.weight' in k:
            shape = f.get_tensor(k).shape
            print(f'  embed_tokens shape: {shape}  -> vocab_size={shape[0]}, hidden={shape[1]}')
        if 'layers.0.self_attn.q_proj.weight' in k:
            shape = f.get_tensor(k).shape
            print(f'  q_proj shape: {shape}')
        if 'layers.31' in k and 'self_attn.q_proj' in k:
            print(f'  has layer 31 (32-layer = 7B)')
        if 'layers.39' in k and 'self_attn.q_proj' in k:
            print(f'  has layer 39 (40-layer = 13B)')
" 2>&1 | head -30
