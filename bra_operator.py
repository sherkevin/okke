"""
Legacy Qwen-specific BRA operator shim.

Kept for backward compatibility with older scripts that expect:

    bra = BRAOperator(model, cfg, tokenizer=...)
    model.generate(...)

Internally this delegates to the canonical logits-processor implementation.
"""

from __future__ import annotations

from bra_logits_processor import BRAConfig
from bra_operator_multi import BRAOperatorMulti, Qwen3VLAdapter


class BRAOperator(BRAOperatorMulti):
    def __init__(self, model, config: BRAConfig | None = None, tokenizer=None):
        super().__init__(
            model=model,
            config=config,
            tokenizer=tokenizer,
            adapter=Qwen3VLAdapter(),
        )
