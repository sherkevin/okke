from transformers import AutoProcessor


processor = AutoProcessor.from_pretrained("/root/autodl-tmp/BRA_Project/models/Qwen3-VL-4B-Instruct")
tokenizer = getattr(processor, "tokenizer", processor)

variants = ["yes", " yes", "Yes", " Yes", "no", " no", "No", " No"]
for variant in variants:
    token_ids = tokenizer.encode(variant, add_special_tokens=False)
    decoded = tokenizer.decode(token_ids, skip_special_tokens=False)
    print(repr(variant), token_ids, repr(decoded))
