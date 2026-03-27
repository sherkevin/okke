# Review_Strict_V101
## Overall Score
Score: 3/5

## Verdict
Borderline / Major Revision. The pre-registered experimental protocol is exceptionally rigorous, highly self-aware, and refreshingly falsifiable. However, the theoretical execution contains a fatal mathematical flaw in the evidence aggregation (Softmax), a highly questionable claim regarding GPU hardware physics (the relative size and read-cost of $W_{calib}$ vs $B$), and a complete absence of the training formulation for the calibrated head. The protocol is strong, but the method itself needs critical structural corrections before these experiments are run.

## Summary
The paper proposes Token-Local Resonance Anchoring (TLRA), a late-fusion decode-time intervention to mitigate entity-centric hallucinations in MLLMs. It projects visual states into vocabulary logit biases using a trained linear head (`TLRA_calib`), applies a precomputed BPE-level mask (VASM) to constrain interventions to physical nouns, and claims to navigate High Bandwidth Memory (HBM) bottlenecks via dynamic sliced-matmul. The current draft presents a detailed, pre-registered evaluation protocol rather than completed empirical results.

## WhatShouldBeKept
1. **The Evaluation Protocol:** Tables 1 through 4 represent an incredibly high standard for MLLM evaluation. Retain the exact structure.
2. **The Matched Baselines:** Testing against a matched-parameter `Base + LoRA` and a `Global_Visual_State` uniform pooling control is exactly the right way to isolate the value of decode-time spatial routing.
3. **The Vulnerability Audits:** The Hijacking CDF, the Action/Verb Hallucination stress test, and the BPE Entity Coverage metrics are excellent and must remain in the final paper. They expose the hard boundaries of your method perfectly.

## MajorWeaknesses

**1. The Hardware Physics Claim is Mathematically Flawed ($W_{calib}$ vs $B$)**
You claim that statically precomputing $B \in \mathbb{R}^{N_v \times V}$ causes an HBM uncoalesced read bottleneck for HD images ($N_v \ge 3000$) because $B$ exceeds 1.2GB. You propose dynamically slicing $W_{calib} \in \mathbb{R}^{D \times V}$ as the solution: $B_{step} = X_v \cdot W_{calib}[:, \mathcal{C}_t]$. 
Let's do the math. Assuming fp16 (2 bytes per element) and $V = 150,000$:
- Size of $B$ (at $N_v=3000$): $3000 \times 150,000 \times 2 = 900$ MB.
- Size of $W_{calib}$ (assuming standard LLaVA-style projector dimension $D=4096$): $4096 \times 150,000 \times 2 \approx 1.22$ GB.
$W_{calib}$ is **larger** than $B$. Slicing columns $\mathcal{C}_t$ from $W_{calib}$ requires arbitrary uncoalesced reads across a *larger* matrix residing in the exact same HBM. You trade a 900MB uncoalesced read for a 1.22GB uncoalesced read, plus you add the FLOPs of doing the matmul $X_v \cdot W_{calib}[:, \mathcal{C}_t]$ at every single decoding step. Unless you are proposing a highly specific memory transposition or custom Triton kernel that you haven't mentioned, your claim that dynamic sliced-matmul "navigates the HBM wall" better than static precomputation is false. 

**2. Fatal Logic in Evidence Aggregation (The Softmax Bug)**
In Section 3.2, you define relative evidence as $\hat S(c) = \text{Softmax}\big(S_{raw}(c) / \tau_{evidence} \mid c \in \mathcal{C}_t\big)$. 
Using a Softmax over the candidate set $\mathcal{C}_t$ is structurally dangerous. If the model hallucinates and *all* top-$M$ candidates are visually unsupported (e.g., raw scores are all near zero), the Softmax will forcefully normalize them so they sum to 1.0. One of the hallucinated tokens will receive a high $\hat S(c)$ near 1.0. Looking at your penalty equation: $(1 - \hat S(c))$, this means the penalty becomes $0$ for that token. You are mathematically granting a free pass to a hallucinated token simply because it is the "best of the worst." Grounding evidence must be **absolute**, not relative to the local candidate window. This must be changed to an independent Sigmoid or clamped linear activation.

**3. Missing Training Formulation for `TLRA_calib`**
You describe `TLRA_calib` as a "lightweight linear projection... trained on a bounded conceptual-caption subset." This is a massive hand-wave. What is the loss function? Is it cross-entropy on the next-token prediction for VASM-masked tokens? Is it a contrastive loss against global visual states? What is the size of the "bounded subset"? Since `TLRA_calib` introduces new parameters, its training mechanics are not an implementation detail; they are the core identity of the method. Without this, the method is irreproducible.

**4. `TLRA_zero` is a Theatrical Strawman**
You dedicate significant space to `TLRA_zero`, openly admitting it is a "negative control" that is "hypothesized to fail" due to the geometric anisotropy of the MLP projector output vs. discrete text embeddings. Proving that an MLP output does not align with a static embedding matrix via dot-product is trivial—they are completely different latent spaces optimized for different layers. While intellectually honest, dressing this up as a "Geometric Isotropy Probe" is theatrical. State it in one sentence as an obvious property of the architecture, drop `TLRA_zero` from the main tables, and save the space for the missing training details of `TLRA_calib`.

## SectionBySectionComments
- **Abstract & Intro:** Clearly defines the scope. Explicitly owning the "Noun-Bias" is a strong rhetorical move.
- **Section 3.1:** Remove `TLRA_zero` as a formal variant. It is a distraction. Replace this space with the exact training objective, dataset size, and hyperparameters for $W_{calib}$.
- **Section 3.2:** Re-evaluate the static vs. dynamic memory argument. If $N_v < D$, static precomputation is strictly superior for memory bandwidth. If $N_v > D$ (e.g., ultra-HD), dynamic might win, but you must map out the exact crossover point based on $D$, not just assert it. 
- **Section 3.3:** Replace Softmax with Sigmoid or bounded ReLU.
- **Section 3.4:** VASM string-to-BPE logic is solid, but what happens when a noun is tokenized into multiple subwords (e.g., "refrigerator" -> "refrig", "erator")? Does VASM mask *all* subwords, or just the first? If just the first, the autoregressive decay will destroy the rest of the word. Clarify the subword continuation masking strategy.

## RequiredRevisions
1. **Rewrite Section 3.1** to completely formalize the training objective, data composition, and optimization strategy for $W_{calib}$.
2. **Fix the Softmax bug** in Equation 1. Replace it with an absolute confidence metric (e.g., Sigmoid) so that universally bad candidate windows are universally penalized.
3. **Correct the Hardware Analysis in 3.2 & 4.4.** Calculate and state the exact matrix sizes for $B$ and $W_{calib}$ given your specific architecture's $D$ and $V$. If dynamic slicing does not actually save memory bandwidth, remove the claim and simply report the empirical latency difference.
4. **Clarify VASM Subword Handling.** Explain how VASM handles multi-token entities to prevent mid-word truncation.

## SuggestedFiguresTablesExperiments
- **Top-M Ablation (Table/Graph):** The latency of uncoalesced reads heavily depends on $M$ (the size of $\mathcal{C}_t$). Add a throughput vs. $M$ curve ($M \in \{10, 50, 100, 500\}$).
- **Subword Autoregressive Check:** Add a qualitative failure case analyzing what happens when TLRA intervenes on the first BPE token of an entity, but the base model's internal state fails to complete the suffix.
- **Training Overhead Table:** Specify the exact compute hours and GPU requirements to calibrate $W_{calib}$.

## AcceptanceOutlook
The experimental protocol is one of the most intellectually honest and rigorous frameworks I have seen submitted to this track. However, the theoretical method *inside* that framework has critical bugs (the Softmax relative scoring, the flawed HBM size claims, the missing loss function). If you fix the math, correct the hardware claims, and execute the protocol exactly as written, this will be a highly impactful, boundary-pushing paper. If you fail to fix the structural math or provide the training formulation, it will be rejected regardless of the empirical results.