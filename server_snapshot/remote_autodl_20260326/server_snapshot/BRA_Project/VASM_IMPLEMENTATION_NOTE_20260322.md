# VASM Implementation Note

## Current Implementation Status

The current `VASM` implementation is an **offline vocabulary-level mask**, not an online linguistic parser.

Concretely, the runtime path is:

1. Build or load a precomputed `gamma_by_id` table over the tokenizer vocabulary.
2. During decoding, look up candidate token IDs in that table.
3. Apply BPE continuation inheritance by reusing the normalized root-token decision for continuation subwords.

Code anchors:

- `bra_vasm.py`
- `bra_logits_processor.py`

## What VASM Is

`VASM` is currently:

- a **pure precomputed vocab lookup**
- plus **BPE continuation inheritance**
- plus lightweight tokenizer-family heuristics (`qwen` / `llama` / generic)

The table is constructed once from tokenizer vocabulary items and cached in memory. At decode time, the mask is applied by direct ID lookup.

## What VASM Is Not

`VASM` is **not** currently:

- an online POS tagger
- an online dependency parser
- a runtime WordNet query loop
- a per-step lexical analyzer over generated text

There is **no online POS tagger in the loop**. All masking decisions used during generation are resolved from the precomputed vocabulary table and BPE inheritance rule.

## Practical Paper Wording

For paper revision, the safest accurate wording is:

`VASM is currently implemented as an offline precomputed vocabulary mask with BPE continuation inheritance. It does not invoke an online POS tagger or any runtime lexical parser during decoding.`

## Boundary Note

Because the implementation is vocabulary-level and heuristic, its behavior is sensitive to tokenizer segmentation. This is why suffix-collapse and collateral-damage audits remain necessary companion assets rather than optional diagnostics.
