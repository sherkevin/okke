#!/usr/bin/env python3
"""
Zero-Vision Modality Stripping: Extract S_text-only for Qwen3-VL-8B.

V3.5 paradigm shift: use pure text prompts (no pixel_values) to extract
the language gravity well subspace. This completely eliminates visual prior
residuals by design (pixel_values=None → 100% language autoregressive inertia).
"""

import os, sys, gc, json, torch, numpy as np

MODEL_PATH = "/root/autodl-tmp/A-OSP_Project/models/Qwen3-VL-8B-Instruct"
OUTPUT_PATH = "/root/autodl-tmp/A-OSP_Project/models/V_text_only_q3.pt"
LOG_DIR = "/root/autodl-tmp/A-OSP_Project/logs/extract_logs"
os.makedirs(LOG_DIR, exist_ok=True)

K = 20          # subspace dimension
N_PROMPTS = 50  # reduced for speed (200 is ideal but 50 is sufficient for stable SVD at K=20)
# For Qwen3-VL-8B with 36 layers, L-4 = layer 32 (0-indexed)
# V3.5: also supports layer 29 for video MeanPool alignment
TARGET_LAYER = 32

# 200 high-diversity text prompts covering: entities, abstract concepts,
# spatial relations, temporal descriptions, quantity judgments, etc.
PROMPTS_200 = [
    # --- Objects & Entities ---
    "Describe a golden retriever playing in a park.",
    "What does a modern laptop computer look like?",
    "Tell me about a red double-decker bus in London.",
    "Describe the appearance of a grand piano.",
    "What does a sunflower field look like in summer?",
    "Describe a crowded subway car during rush hour.",
    "Tell me about a lighthouse at night.",
    "What does a sushi restaurant look like inside?",
    "Describe a vintage motorcycle from the 1960s.",
    "Tell me about a fruit market with colorful displays.",
    "Describe a snowy mountain peak at dawn.",
    "What does a library with tall bookshelves look like?",
    "Tell me about an aquarium with tropical fish.",
    "Describe a construction site with cranes.",
    "What does a bakery smell and look like?",
    "Describe a medieval castle with towers.",
    "Tell me about a busy airport terminal.",
    "What does a garden greenhouse look like?",
    "Describe a chemistry laboratory setup.",
    "Tell me about a football stadium during a match.",
    # --- Abstract Concepts ---
    "Explain what freedom means.",
    "What is the concept of democracy?",
    "Describe the notion of infinity.",
    "What does justice mean in modern society?",
    "Explain the concept of entropy in thermodynamics.",
    "What is artificial intelligence?",
    "Describe the concept of time travel.",
    "What is consciousness?",
    "Explain what beauty means philosophically.",
    "What is the meaning of life according to existentialism?",
    # --- Spatial Relations ---
    "Describe the spatial arrangement of furniture in a living room.",
    "How are the planets arranged in our solar system?",
    "Describe the layout of a typical supermarket.",
    "What is the spatial relationship between Paris and London?",
    "Describe how books are organized in a library.",
    "What is the arrangement of keys on a piano?",
    "Describe the layout of a typical kitchen.",
    "How are seats arranged in a movie theater?",
    "Describe the position of stars in the Big Dipper.",
    "What is the spatial relationship between the heart and lungs?",
    # --- Temporal Descriptions ---
    "Describe what happens during a typical morning routine.",
    "Tell me about the sequence of seasons in a year.",
    "What happens during the process of making bread?",
    "Describe the lifecycle of a butterfly.",
    "What are the steps in the scientific method?",
    "Describe what happens during a thunderstorm.",
    "Tell me about the process of photosynthesis.",
    "What happens during a solar eclipse?",
    "Describe the process of digestion.",
    "Tell me about the steps to bake a chocolate cake.",
    # --- Quantity & Numerical ---
    "How many bones are in the human body?",
    "What percentage of Earth's surface is water?",
    "How many languages are spoken worldwide?",
    "What is the population of China?",
    "How many chambers does the human heart have?",
    "What is the speed of light in kilometers per second?",
    "How many stars are in the Milky Way approximately?",
    "What percentage of the atmosphere is nitrogen?",
    "How many keys does a standard piano have?",
    "What is the boiling point of water in Celsius?",
    # --- Comparisons ---
    "Compare the size of an elephant and a mouse.",
    "What is larger, the Pacific Ocean or the Atlantic?",
    "Compare the speed of a cheetah and a human.",
    "Which is older, the Great Wall of China or the Colosseum?",
    "Compare the density of gold and water.",
    "Which continent has more countries, Africa or Europe?",
    "Compare the brightness of the sun and the moon.",
    "Which is faster, sound or light?",
    "Compare the lifespan of a tortoise and a mayfly.",
    "Which is deeper, the Pacific or the Atlantic Ocean?",
    # --- Scientific Facts ---
    "What causes rainbows to form?",
    "How does a black hole form?",
    "What is the process of nuclear fission?",
    "How do vaccines work in the immune system?",
    "What causes the tides on Earth?",
    "How does sonar work?",
    "What is the Doppler effect?",
    "How do plants produce oxygen?",
    "What causes earthquakes?",
    "How does GPS navigation work?",
    # --- Historical Events ---
    "Describe the significance of the moon landing in 1969.",
    "What happened during the French Revolution?",
    "Describe the invention of the printing press.",
    "What was the significance of the Berlin Wall falling?",
    "Describe the Industrial Revolution.",
    "What happened during World War II?",
    "Describe the discovery of penicillin.",
    "What was the significance of the Magna Carta?",
    "Describe the Renaissance period.",
    "What happened during the American Civil War?",
    # --- Cultural Questions ---
    "What is the significance of Chinese New Year?",
    "Describe the traditions of Thanksgiving.",
    "What is the significance of the Olympic Games?",
    "Describe the cultural importance of music.",
    "What is the significance of the Eiffel Tower to France?",
    "Describe the traditions of a Japanese tea ceremony.",
    "What is the cultural significance of storytelling?",
    "Describe the traditions of the Day of the Dead.",
    "What is the significance of the Hajj pilgrimage?",
    "Describe the cultural impact of Shakespeare.",
    # --- Food & Cuisine ---
    "Describe the ingredients in a classic Italian pasta.",
    "What are the main components of a sushi platter?",
    "Describe the process of making sourdough bread.",
    "What are the key spices in Indian curry?",
    "Describe a traditional French breakfast.",
    "What makes Japanese ramen unique?",
    "Describe the ingredients in a Greek salad.",
    "What is the difference between a cappuccino and a latte?",
    "Describe the process of making chocolate.",
    "What are the key ingredients in a traditional paella?",
    # --- Nature & Environment ---
    "Describe the ecosystem of a coral reef.",
    "What is the importance of rainforests?",
    "Describe the water cycle.",
    "What causes the Northern Lights?",
    "Describe the life of a great white shark.",
    "What is the significance of wetlands?",
    "Describe the process of glacier formation.",
    "What causes the changing of autumn leaves?",
    "Describe the ecosystem of a desert.",
    "What is the importance of bees in agriculture?",
    # --- Technology ---
    "How does the internet work?",
    "Describe the components of a smartphone.",
    "How does machine learning work?",
    "Describe the process of 3D printing.",
    "How does solar energy work?",
    "Describe the components of a computer processor.",
    "How does Bluetooth technology work?",
    "Describe the process of DNA sequencing.",
    "How do self-driving cars navigate?",
    "Describe the components of a wind turbine.",
    # --- Health & Medicine ---
    "Describe the symptoms of the common cold.",
    "How does the immune system fight infections?",
    "Describe the process of blood clotting.",
    "What is the function of the liver?",
    "Describe the symptoms of diabetes.",
    "How does anesthesia work?",
    "Describe the process of wound healing.",
    "What is the function of the kidneys?",
    "Describe the difference between bacteria and viruses.",
    "How does the heart pump blood?",
    # --- Philosophy & Ethics ---
    "What is the trolley problem in ethics?",
    "Describe the philosophy of utilitarianism.",
    "What is the meaning of free will?",
    "Describe the concept of the social contract.",
    "What is moral relativism?",
    "Describe Plato's theory of forms.",
    "What is the philosophy of mind?",
    "Describe the concept of nihilism.",
    "What is the difference between ethics and morality?",
    "Describe the philosophy of Stoicism.",
    # --- Sports & Activities ---
    "Describe the rules of chess.",
    "What are the key skills in basketball?",
    "Describe the process of training for a marathon.",
    "What are the rules of tennis?",
    "Describe the techniques in swimming.",
    "What makes a good soccer team strategy?",
    "Describe the process of learning to ski.",
    "What are the key techniques in photography?",
    "Describe the rules of cricket.",
    "What are the skills needed for rock climbing?",
    # --- Architecture & Design ---
    "Describe the features of Gothic architecture.",
    "What makes modern minimalist design unique?",
    "Describe the features of Art Deco architecture.",
    "What are the principles of good interior design?",
    "Describe the features of a traditional Japanese house.",
    "What makes Baroque architecture distinctive?",
    "Describe the features of a sustainable eco-house.",
    "What are the principles of urban planning?",
    "Describe the features of Neoclassical architecture.",
    "What makes brutalist architecture controversial?",
    # --- Music & Arts ---
    "Describe the elements of jazz music.",
    "What makes classical ballet unique?",
    "Describe the process of painting a watercolor.",
    "What are the elements of hip-hop culture?",
    "Describe the features of Impressionist painting.",
    "What makes flamenco dancing distinctive?",
    "Describe the elements of a symphony orchestra.",
    "What are the features of street art?",
    "Describe the process of sculpting in marble.",
    "What makes opera a unique art form?",
    # --- Language & Communication ---
    "What are the main language families in the world?",
    "Describe the process of language acquisition in children.",
    "What makes sign language unique?",
    "Describe the features of poetry.",
    "What is the importance of translation?",
    "Describe the evolution of writing systems.",
    "What makes a speech persuasive?",
    "Describe the features of a good narrative.",
    "What is the role of metaphor in language?",
    "Describe the features of scientific writing.",
]

print(f"=== Zero-Vision S_text-only Extraction for Qwen3-VL-8B ===")
print(f"N_PROMPTS={N_PROMPTS}, K={K}, TARGET_LAYER={TARGET_LAYER}")
print(f"pixel_values=None → 100% language inertia, zero visual residual")
print()

# Load model
print("[1/4] Loading Qwen3-VL-8B model...")
from transformers import AutoConfig, AutoProcessor
from transformers import Qwen3VLForConditionalGeneration

cfg = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
arch = getattr(cfg, 'architectures', [''])[0] if hasattr(cfg, 'architectures') else ''
# Qwen3VL stores hidden_size inside text_config
D = getattr(cfg, 'hidden_size', None) or getattr(getattr(cfg, 'text_config', cfg), 'hidden_size', 4096)
n_layers = getattr(cfg, 'num_hidden_layers', None) or getattr(getattr(cfg, 'text_config', cfg), 'num_hidden_layers', 36)
print(f"  Architecture: {arch}, hidden_size={D}, num_layers={n_layers}")

# Load model with bfloat16
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
    trust_remote_code=True,
)
model.eval()
processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)

print(f"  Loaded. hidden_size={D}, target layer={TARGET_LAYER}")

# Register hook to capture hidden states at target layer
collected_hidden = []

def get_hidden_hook(module, inp, out):
    """Capture the hidden state output at target layer (text-only, no vision)."""
    if isinstance(out, tuple):
        h = out[0]  # [batch, seq_len, D]
    else:
        h = out     # bare tensor for Qwen3 + FA2
    # For text-only generation, take mean over sequence positions
    h_mean = h[0].mean(dim=0)  # [D]
    collected_hidden.append(h_mean.float().detach().cpu())

# Find target layer - probe actual architecture
def get_decoder_layers(m):
    """Find decoder layers across different Qwen-VL architectures."""
    # Qwen3-VL: model.model.language_model.layers
    if hasattr(m, 'model') and hasattr(m.model, 'language_model') and hasattr(m.model.language_model, 'layers'):
        return m.model.language_model.layers
    # Qwen3-VL alt path: model.language_model.layers
    if hasattr(m, 'language_model') and hasattr(m.language_model, 'layers'):
        return m.language_model.layers
    # Qwen2-VL / Qwen2.5-VL: model.model.layers
    if hasattr(m, 'model') and hasattr(m.model, 'layers'):
        return m.model.layers
    raise AttributeError(f"Cannot find decoder layers in {type(m).__name__}")

try:
    layers = get_decoder_layers(model)
    hook = layers[TARGET_LAYER].register_forward_hook(get_hidden_hook)
    print(f"[2/4] Hook registered at layer[{TARGET_LAYER}] (total: {len(layers)})")
except Exception as e:
    print(f"  ERROR: {e}")
    sys.exit(1)

# Run text-only inference (NO pixel_values)
print(f"[3/4] Running {N_PROMPTS} text-only prompts (pixel_values=None)...")
from qwen_vl_utils import process_vision_info

for i, prompt_text in enumerate(PROMPTS_200[:N_PROMPTS]):
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt_text}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = processor(
        text=[text],
        images=None,   # ← KEY: zero-vision modality stripping
        videos=None,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        _ = model.generate(
            **inputs,
            max_new_tokens=30,
            do_sample=False,
        )

    del inputs
    if i % 20 == 0:
        print(f"  Processed {i}/{N_PROMPTS} prompts, collected={len(collected_hidden)}")

hook.remove()
print(f"[3/4] Done. Collected {len(collected_hidden)} hidden states.")

# SVD decomposition
print("[4/4] Running SVD to extract S_text-only subspace...")
H = torch.stack(collected_hidden, dim=0)  # [N, D]
H_mean = H.mean(dim=0, keepdim=True)      # [1, D]
R = H - H_mean                             # centered residual matrix

U, S, Vt = torch.linalg.svd(R, full_matrices=False)  # Vt: [min(N,D), D]
V_bias = Vt[:K, :]  # top-K right singular vectors: [K, D]

evr = (S[:K]**2).sum() / (S**2).sum()
evr = evr.item()
print(f"  EVR (top {K} components): {evr:.4f}")

# Compute L_prior (mean L2 norm of projections onto bias subspace)
proj = R @ V_bias.T          # [N, K]
L_prior = torch.sqrt((proj**2).sum(dim=1)).mean().item()
print(f"  L_prior: {L_prior:.4f}")
print(f"  Top-3 singular values: {S[:3].tolist()}")

# Save
result = {
    "V_bias": V_bias,
    "H_mean": H_mean,
    "singular_values": S[:K],
    "evr": evr,
    "K": K,
    "D": D,
    "N": N_PROMPTS,
    "L_prior": L_prior,
    "layer_idx": TARGET_LAYER,
    "model": "Qwen3-VL-8B-Instruct",
    "tag": "S_text_only_zero_vision",
    "extraction_method": "zero_vision_modality_stripping",
    "top3_sigma": S[:3].tolist(),
}
torch.save(result, OUTPUT_PATH)
print(f"\n✓ Saved V_text_only_q3.pt → {OUTPUT_PATH}")
print(f"  Shape: {list(V_bias.shape)}, EVR={evr:.4f}, L_prior={L_prior:.4f}")

# Cleanup
del model, processor, H, R, U, S, Vt
gc.collect()
torch.cuda.empty_cache()
print("EXTRACTION COMPLETE")
