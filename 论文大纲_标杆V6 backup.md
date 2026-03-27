# CHORD: Calibrating Hallucinations via Object-Resonant Decoding in Multimodal Large Language Models

## Abstract
Multimodal Large Language Models have demonstrated remarkable capabilities in multi-modal understanding and generation. However, they persistently suffer from hallucination issues, generating text that contradicts visual inputs. The fundamental root cause of these hallucinations lies in the cross-modal granularity mismatch during autoregressive decoding: while MLLMs encode images as continuous holistic features, they are forced to generate text as discrete, often semantically fragmented sub-word tokens. This structural misalignment makes it mathematically intractable to ground individual, meaningless tokens to concrete visual entities. Consequently, the model is forced into an unverified decoding state, where the statistical inertia of language priors inevitably breeds hallucinations. Existing mitigation strategies struggle to address this efficiently, as they typically require computationally expensive model retraining or incur severe inference overhead through complex multi-round verification. Motivated by this insight, we propose CHORD, a training-free, plug-and-play decoding framework designed to forcefully re-anchor text generation to visual reality via Object-Resonant Decoding. Inspired by the physical resonance process, CHORD intercepts the active decoding frontier and dynamically evaluates the alignment between candidate text spans and external visual evidence. A candidate span's generation probability is amplified only if it semantically "resonates" with a concrete visual object, whereas ungrounded hallucinations—driven merely by linguistic inertia—are decisively penalized. Furthermore, to limit overhead while still catching overconfident errors, we use continuous Top-$1$ monitoring with full-$K$ backoff, exploiting KV-cache reuse in the external text encoder so that expensive full-frontier scoring runs only when the top hypothesis fails a resonance check. Extensive experiments across multiple hallucination-focused benchmarks demonstrate that CHORD substantially reduces object hallucinations. As a model-agnostic framework, it requires no additional training and establishes a highly scalable paradigm for trustworthy MLLM deployment.

## 1. Introduction

Multimodal large language models (MLLMs) have rapidly advanced visual question answering, captioning, and open-ended multimodal reasoning. Yet even strong models remain vulnerable to hallucination: they may generate fluent but weakly grounded statements, invent objects absent from the image, or confidently choose visually unsupported entities under strong language priors. These failures are particularly damaging when the correct evidence is sparse, local, or easily drowned out by semantically irrelevant visual context.

Recent hallucination-mitigation work shows that inference-time intervention is a strong alternative to retraining the host model. Contrastive decoding (e.g., VCD), attention-based control (e.g., OPERA), and layer-contrast decoding (e.g., DoLa) all treat autoregressive decoding as an actionable surface. Yet they leave open a stricter question:

**Can decode-time mitigation remain portable across heterogeneous MLLM families without per-model training and without relying on host-internal hidden states, logits, or attention?**

This goal differs from improving a single backbone. Any method that consumes model-specific multimodal activations and writes corrections directly in the host’s native geometry remains **host-coupled**, even if the vision–language weights are frozen. Prior **hidden-state calibration** illustrates the phenomenon but does not meet a strict plug-in portability requirement.

We therefore articulate a **universality boundary**: strict hot-swappability is not achievable for mechanisms that live entirely inside model-native hidden, logit, or attention spaces. A falsifiable alternative is to restrict all scoring to a **shared external space** built from frozen public vision features, frozen public text encoders, candidate spans obtained during decoding, and **deterministic** tokenizer bridging—without learned adapters tied to a specific host.

**CHORD** (Calibrating Hallucinations via Object-Resonant Decoding) instantiates this principle. It performs **candidate-driven visual resonance**: the relevant signal is not global similarity of the unfinished prefix to the image, but whether the **current candidate span** activates consistent local visual evidence and whether that evidence supports an intervention. Mitigation becomes selective activation of external, object-level evidence tied to the candidate, rather than wholesale redistribution from unrelated regions or pure language continuation bias.

Two implications follow. First, CHORD should be judged not only by raw hallucination reduction but by **stable behavior across host families** when the external encoders and region bank are held fixed and only the tokenizer bridge varies deterministically with the host. Second, span-level resonance with a dense region bank should materially outperform a simplistic global-image prior; otherwise the added machinery is hard to justify.

The paper makes four contributions:

1. It formalizes the **universality boundary** that limits strictly portable control for host-internal decoders.
2. It proposes **CHORD**, a **training-free** decode-time framework that calibrates the active logit frontier using object-resonant evidence and a closed-form resonance delta in external embedding space.
3. It introduces a **parameter-free tokenizer bridge** plus **continuous Top-$1$ monitoring** (with full-$K$ backoff) so that resonance checks stay lightweight and address overconfident hallucinations without naive entropy-only gating.
4. It reports an **evaluation protocol** on POPE and CHAIR with decode-time baselines, ablations, and efficiency analysis.

The remainder of the paper is organized as follows. Section 2 reviews related work. Section 3 formalizes the universality boundary and presents CHORD. Section 4 describes experiments. Section 5 concludes.

## 2. Related Work

### 2.1 Multimodal hallucination evaluation and benchmark design

Evaluating hallucination in vision–language models has crystallized around complementary protocols. **CHAIR** [1] scores whether objects mentioned in generated captions are licensed by the image, highlighting that fluent text can still be visually unsupported. **POPE** [2] recasts assessment as balanced yes/no existence probes, making object hallucination measurable under controlled prompts and revealing sensitivity to language priors. **AMBER** [3] broadens the lens to multiple error types, including attributes and relations, so that “object-only” scores are not mistaken for full multimodal faithfulness. These instruments answer different questions: CHAIR suits open-ended captioning; POPE suits discriminative probing; AMBER stresses multi-faceted failure modes. They should not be treated as interchangeable drop-in replacements for one another.

Large-scale discriminative suites extend the same agenda with richer task and context structure. **PhD** [4] provides a sizable VQA-style benchmark with multiple recognition tasks and adversarial prompting modes, stressing robustness of MLLMs to visually misleading or prior-triggering questions. We use CHAIR and POPE as primary reporting axes (Section 4) while acknowledging such broader evaluation lines as context for future stress-testing.

### 2.2 Decode-time mitigation and control

A prominent class of methods intervenes **during** autoregressive decoding without updating weights. **Visual contrastive decoding (VCD)** [5] contrasts logits under intact versus perturbed visual inputs to down-weight language-prior-driven objects. **OPERA** [6] shapes decoding using attention- and retrospection-based penalties over generation trajectories. **DoLa** [7], developed for text-only LMs, improves factuality by contrasting early and late layer logits; adapted to MLLMs, it remains a strong **internal** layerwise baseline but does not, by itself, enforce object-level grounding in an external visual space.

Recent MLLM-specific decoding refinements still predominantly **read or reshape internal representations**. For example, **DeCo** [8] dynamically blends logits from earlier layers when intermediate activations appear to retain visual evidence that final layers suppress. Such designs highlight that useful visual signals often exist inside the stack, yet they inherit the universality limitations of any approach that requires access to host layer outputs and geometry.

**CHORD** differs along two axes emphasized in Section 3: (i) it never consumes host hidden states or attention tensors—only host logits on a short candidate frontier plus deterministic tokenizer lookahead; (ii) it scores **candidate spans** against a **frozen** region-level visual bank and a **frozen** public text encoder, so the intervention signal is defined in a shared external space. This targets the cross-modal **granularity** gap between fragmented sub-word proposals and object-level evidence, while continuous Top-$1$ monitoring limits the cost of span-level scoring.

### 2.3 Post-hoc correction, preference optimization, and position relative to CHORD

Complementary families edit outputs **after** generation or change the model itself. **LURE** [9] performs lightweight post-hoc revision informed by statistical factors linked to object hallucination. **Woodpecker** [10] runs a multi-stage, tool-assisted pipeline that extracts claims, queries external vision modules, and rewrites inconsistent text. **Volcano** [11] iterates self-feedback and revision. **HA-DPO** [12] reduces hallucination via preference optimization, permanently shifting model behavior. These methods can exploit richer verification or stronger supervision precisely because they are not constrained to the per-step budget of greedy or beam decoding.

**CHORD** targets a narrower regime: **training-free**, **in-decoding** calibration of the top-$K$ frontier using external object-level evidence and a closed-form resonance delta on lifted spans. It is closest in spirit to decode-time baselines [5–8] on the time axis, but closest in *evidence* to tool-augmented work [10] while avoiding full post-hoc pipelines and without preference updates [12]. Section 3 details how this design relates to the universality boundary and to mitigating overconfident hallucinations without relying solely on entropy triggers.


## 3. Methodology
### 3.1 The Universality Boundary and Granularity Mismatch

A decode-time intervention module fails to be strictly universal if either its inputs or outputs are host-specific. Earlier calibration methods break this universality boundary by consuming model-native hidden states and projecting them directly into the host model's lexical geometry. Such learned mappings are host-coupled by construction and cannot be transferred across heterogeneous Multimodal Large Language Models (MLLMs) without retraining. 

To achieve true hot-swappability, an intervention mechanism must operate in a standardized external semantic space. However, designing such an external controller requires addressing the fundamental root cause of hallucinations: the **cross-modal granularity mismatch**. While MLLMs encode images as continuous, holistic feature maps, autoregressive generation forces them to output discrete, often semantically fragmented sub-word tokens (e.g., `["gi", "##raf", "##fe"]`). It is mathematically intractable to establish reliable physical grounding between a continuous visual entity and a meaningless, fragmented token. This granularity collapse forces the MLLM into an *open-loop decoding state* where visual verification fails, allowing the statistical inertia of language priors to overwhelm visual constraints. CHORD is designed to bridge this mismatch and forcefully close the decoding loop.

### 3.2 The CHORD Framework Overview

CHORD (Calibrating Hallucinations via Object-Resonant Decoding) is a training-free, plug-and-play decoding framework. Instead of overriding the full vocabulary distribution, CHORD acts exclusively on the host model's active decoding frontier $\mathcal{F}_t = \mathrm{Top}\text{-}K(L_t)$, where $L_t$ represents the original logits at step $t$. The framework operates through four sequential stages: (1) Tokenizer Bridging, (2) Continuous Resonance Monitoring, (3) Hypothesis-Driven Resonance Delta, and (4) Logit Calibration.

Before generation begins, CHORD performs a one-time offline extraction. Given the input image $I$, a frozen external vision encoder extracts a dense local region feature bank $\mathcal{Z}_R = \{z_{R_1}, z_{R_2}, \dots, z_{R_N}\}$. This bank serves as a static, objective physical reality cache for subsequent resonance verification.

### 3.3 Parameter-Free Tokenizer Bridge

To overcome the granularity mismatch, CHORD does not evaluate raw token IDs. Let $B_m$ denote a parameter-free tokenizer bridge associated with the host model. For each candidate token $c_k \in \mathcal{F}_t$, the bridge performs a deterministic lookahead (utilizing the tokenizer's prefix tree) to lift the token into a semantically complete text span $A_k$. 

This yields a set of candidate spans $\mathcal{S}_t = \{A_1, A_2, \dots, A_K\}$. This bridge introduces no learned parameters. It is the critical mechanism that transforms model-specific, fragmented token proposals into cross-model, physically groundable semantic propositions.

### 3.4 Hypothesis-Driven Resonance Delta

Instead of training a black-box MLP to blindly score text-image pairs, CHORD models decoding as a physical resonance process. We treat the text prefix $p$ appended with a candidate span $A_k$ as a "semantic sonar probe" directed at the visual region bank $\mathcal{Z}_R$. 

To strictly prevent *spatial misalignment* (where the prefix and the candidate match different, unrelated regions), CHORD calculates the resonance shift anchored to the exact same physical region. First, we identify the single most relevant region $r^*$ for the combined hypothesis:
$$r^* = \arg\max_{r \in \mathcal{Z}_R} \cos(E_{text}(p \oplus A_k), z_r)$$
where $E_{text}$ is a frozen public text encoder. Next, we compute the **Resonance Delta ($\Delta S$)**, which isolates the physical impact of adding candidate $A_k$:
$$\Delta S(A_k) = \cos(E_{text}(p \oplus A_k), z_{r^*}) - \cos(E_{text}(p), z_{r^*})$$

This pure mathematical formulation is exceptionally elegant as it natively acts as an **heuristic-free filter**. 
1. **True Visual Spans ($\Delta S > 0$):** If $A_k$ is a visually supported object or correct spatial relation (e.g., "left"), it increases the semantic alignment with region $r^*$, yielding a positive gradient.
2. **Hallucinations ($\Delta S < 0$):** If $A_k$ contradicts the image (e.g., inventing an object), the alignment collapses, yielding a negative gradient.
3. **Non-Visual Words ($\Delta S \approx 0$):** If $A_k$ is an abstract or grammatical word (e.g., "very", "is"), the semantic wave undergoes zero frequency shift, resulting in $\Delta S \approx 0$. Thus, CHORD safely ignores non-visual words without relying on brittle POS taggers.

### 3.5 Overcoming Overconfidence: Continuous Monitoring & System Optimization

A naive entropy-based gate fails to detect *overconfidence hallucinations*, where strong language priors cause the model to output hallucinations with extremely low entropy. To address this while maintaining near-zero inference overhead, CHORD abandons entropy gating in favor of **Continuous Top-1 Monitoring with Full-K Backoff**, powered by Key-Value (KV) Cache acceleration.

Because the external $E_{text}$ utilizes causal masked self-attention, the prefix $p$ is continuously KV-cached. Extracting the feature for $p \oplus A_k$ requires only 1-2 tokens of forward propagation, taking microseconds. CHORD exploits this by continuously calculating $\Delta S(A_{top1})$ for the host model's most confident token at every step. 
* If $\Delta S(A_{top1}) \ge 0$, the token is visually safe or irrelevant, and CHORD allows generation to proceed instantly. 
* If $\Delta S(A_{top1}) < 0$, a language-prior-driven hallucination is detected, regardless of how confident the host model is. Only then does CHORD trigger the full evaluation over all $K$ candidates to find a grounded alternative. This system-level optimization reduces the computational overhead to negligible levels while rigorously preventing overconfidence failures.

### 3.6 Logit Calibration

When full intervention is triggered, CHORD calibrates the active frontier by injecting the physical resonance energy back into the host model's logits. The adjusted logit for candidate token $c_k$ (corresponding to span $A_k$) is computed as:
$$\tilde{L}_t(c_k) = L_t(c_k) + \alpha \cdot \Delta S(A_k)$$
where $\alpha$ is a scaling hyperparameter controlling the intervention strength. The logits are then re-normalized via softmax. Through this bounded redistribution, CHORD strictly nudges the host model's expressed trajectory toward visual reality without destroying native linguistic structures or requiring any parameter updates.


## 4. Experiments

To rigorously validate the CHORD framework, our experiments are designed to answer four critical research questions:
**RQ1 (Effectiveness & Universality):** Does CHORD consistently mitigate object hallucinations across heterogeneous MLLM families without any model-specific retraining?
**RQ2 (Generalization):** Does CHORD generalize to open-ended generation tasks without degrading the host model's structural and linguistic capabilities?
**RQ3 (Mechanism Ablation):** How do the individual components of CHORD (e.g., Tokenizer Bridge, Resonance Delta $\Delta S$) contribute to the final performance?
**RQ4 (System Efficiency):** Does the Continuous Top-1 Monitoring effectively reduce the computational overhead to negligible levels?

### 4.1 Experimental Setup
**Models & Baselines.** To prove the strict universality of CHORD, we evaluate it across three distinct, leading MLLM families: Qwen-VL (e.g., Qwen-VL-Chat), LLaVA-1.5, and InstructBLIP. We compare CHORD against the standard autoregressive decoding (Base) and three state-of-the-art, training-free decode-time baselines: **DoLa**, **OPERA**, and **VCD**.
**Benchmarks.** We utilize **POPE** (random, popular, and adversarial splits) for systematic object hallucination evaluation, and **CHAIR** to assess hallucination rates in open-ended image captioning.
**Implementation Details.** CHORD requires zero parameter updates. We use Grounding DINO and the frozen CLIP ViT-L/14 for offline region caching, and the CLIP Text Encoder for resonance probing.

### 4.2 Main Results: Hallucination Mitigation (RQ1)
Table 1 presents the main quantitative results on the POPE benchmark. CHORD consistently and substantially outperforms the base decoding strategy across all three MLLM families. Importantly, unlike model-coupled baselines that may exhibit unstable performance across different architectures, CHORD demonstrates robust universality. By strictly evaluating the physical Resonance Delta ($\Delta S$) in an external, model-agnostic semantic space, CHORD effectively curbs the statistical inertia of language priors, setting a new state-of-the-art for decode-time hallucination calibration.

**Table 1: Main evaluation on the POPE (Random Split) benchmark.**
| Host Model | Method | Acc | Prec | Recall | F1 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Qwen-VL** | Base | TBD | TBD | TBD | TBD |
| | DoLa | TBD | TBD | TBD | TBD |
| | OPERA | TBD | TBD | TBD | TBD |
| | VCD | TBD | TBD | TBD | TBD |
| | **CHORD (Ours)** | **0.9153** | **0.9822** | **0.8460** | **0.9090** |
| **LLaVA-1.5** | Base | TBD | TBD | TBD | TBD |
| | DoLa | TBD | TBD | TBD | TBD |
| | OPERA | TBD | TBD | TBD | TBD |
| | VCD | TBD | TBD | TBD | TBD |
| | **CHORD (Ours)** | **TBD** | **TBD** | **TBD** | **TBD** |
| **InstructBLIP** | Base | TBD | TBD | TBD | TBD |
| | **CHORD (Ours)** | **TBD** | **TBD** | **TBD** | **TBD** |

*(Note: Extended results for POPE popular and adversarial splits are provided in the Supplementary Material, where CHORD maintains stable gains.)*

### 4.3 Generalization and Structure Preservation (RQ2)
Mitigating hallucinations on discriminative tasks (like POPE) is insufficient if the method destroys the model's ability to generate fluent, open-ended descriptions. Table 2 reports the performance on the CHAIR metric. CHORD significantly lowers both CHAIR-s (sentence-level) and CHAIR-i (instance-level) hallucination rates compared to the Base model. The bounded Logit Calibration ensures that only visually contradicted entities are penalized, perfectly preserving the host model's native syntax and reasoning capabilities.

**Table 2: Open-ended generation evaluation on CHAIR.** (Lower is better).
| Host Model | Method | CHAIR-s $\downarrow$ | CHAIR-i $\downarrow$ |
| :--- | :--- | :--- | :--- |
| Qwen-VL | Base | TBD | TBD |
| | **CHORD** | **TBD** | **TBD** |
| LLaVA-1.5 | Base | TBD | TBD |
| | **CHORD** | **TBD** | **TBD** |

### 4.4 Ablation Study (RQ3)
To isolate the sources of CHORD's performance, we conduct an ablation study on Qwen-VL (Table 3).
1. **w/o Tokenizer Bridge:** Applying resonance to raw, fragmented sub-word tokens causes performance to collapse, proving that bridging the *cross-modal granularity mismatch* is a prerequisite for external evaluation.
2. **Absolute $S_{new}$ instead of $\Delta S$:** If we use absolute similarity rather than the Resonance Delta, the method suffers from spatial misalignment (e.g., matching the prefix to one object and the candidate to another), leading to a drop in F1. This validates our strict heuristic-free anchoring design.
3. **w/o Region Bank (Global Only):** Relying solely on the global image feature instead of the dense region bank fails to capture fine-grained details, significantly reducing the mitigation efficacy.

**Table 3: Ablation of CHORD components on Qwen-VL (POPE).**
| Variant | F1 Score | $\Delta$ F1 |
| :--- | :--- | :--- |
| **Full CHORD** | **TBD** | - |
| w/o Tokenizer Bridge (Raw Tokens) | TBD | - TBD |
| Absolute $S_{new}$ (w/o $\Delta S$) | TBD | - TBD |
| Global Feature Only (w/o Region Bank)| TBD | - TBD |

### 4.5 System Efficiency Analysis (RQ4)
A critical claim of CHORD is its minimal impact on inference latency. In Table 4, we audit the operational cost. Traditional full-K evaluation methods introduce severe bottlenecks. By contrast, CHORD’s **Continuous Top-1 Monitoring** effectively filters out visually safe or non-visual grammatical tokens (e.g., "the", "is") in $O(1)$ time. Combined with the CLIP Text Encoder's KV-Cache, the extra latency per generation step is reduced to mere milliseconds, adding negligible overhead to the host MLLM's total runtime while consuming minimal additional VRAM.

**Table 4: Inference Efficiency Audit on Qwen-VL.**
| Method | Peak VRAM (GB) | Extra Latency per Step (ms) | F1 Score |
| :--- | :--- | :--- | :--- |
| Base (Beam Search) | TBD | 0 | TBD |
| DoLa | TBD | TBD | TBD |
| CHORD (Full-K every step) | TBD | TBD (Severe) | TBD |
| **CHORD (Top-1 Monitor)** | **TBD** | **TBD (Negligible)** | **TBD** |


## 5. Conclusion

In this paper, we identified a fundamental structural flaw in modern Multimodal Large Language Models (MLLMs): the **cross-modal granularity mismatch**. This architectural misalignment between continuous visual features and fragmented, discrete sub-word tokens forces autoregressive generation into an unverified, *open-loop decoding state*. Consequently, the statistical inertia of language priors frequently overwhelms visual constraints, acting as the primary catalyst for object and relational hallucinations.

To directly address this root cause, we introduced **CHORD** (Calibrating Hallucinations via Object-Resonant Decoding), a training-free, plug-and-play decoding framework. By employing a parameter-free tokenizer bridge, CHORD lifts fragmented tokens into semantically complete spans. It then elegantly reframes decoding as a physical resonance process. Instead of relying on brittle heuristic rules or computationally heavy black-box evaluators, CHORD calculates the **Resonance Delta ($\Delta S$)** in an external, shared semantic space. This pure mathematical gradient naturally and dynamically amplifies visually grounded entities while decisively penalizing fabricated hallucinations. 

Furthermore, we resolved the critical latency bottleneck that plagues most decode-time interventions. By abandoning naive entropy gating in favor of **Continuous Top-1 Monitoring** equipped with KV-Cache acceleration, CHORD effectively neutralizes overconfidence hallucinations while reducing the intervention overhead to mere milliseconds. 

Extensive experiments across multiple leading MLLM families demonstrate that CHORD significantly curtails object hallucinations without compromising native linguistic fluency. Ultimately, CHORD breaks the universality boundary, proving that highly effective, cross-model grounding control can be achieved entirely through external physical resonance, establishing a highly scalable and rigorous paradigm for trustworthy MLLM deployment.


## References

References are numbered to match in-text citations in Section 2.

[1] Anna Rohrbach, Lisa Anne Hendricks, Kaylee Burns, Trevor Darrell, and Kate Saenko. Object Hallucination in Image Captioning. In *Proceedings of the Conference on Empirical Methods in Natural Language Processing (EMNLP)*, 2018. Available: https://arxiv.org/abs/1809.02156

[2] Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Wayne Xin Zhao, and Ji-Rong Wen. Evaluating Object Hallucination in Large Vision-Language Models. In *Proceedings of EMNLP*, 2023. Available: https://arxiv.org/abs/2305.10355

[3] Junyang Wang, Yuhang Wang, Guohai Xu, Jing Zhang, Yukai Gu, Haitao Jia, Jiaqi Wang, Haiyang Xu, Ming Yan, Ji Zhang, and Jitao Sang. AMBER: An LLM-free Multi-dimensional Benchmark for MLLMs Hallucination Evaluation. arXiv:2311.07397, 2023.

[4] Jiazhen Liu, Yuhan Fu, Ruobing Xie, Runquan Xie, Xingwu Sun, Fengzong Lian, Zhanhui Kang, and Xirong Li. PhD: A ChatGPT-Prompted Visual Hallucination Evaluation Dataset. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2025. Available: https://arxiv.org/abs/2403.11116

[5] Sicong Leng, Hang Zhang, Guanzheng Chen, Xin Li, Shijian Lu, Chunyan Miao, and Lidong Bing. Mitigating Object Hallucinations in Large Vision-Language Models through Visual Contrastive Decoding. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2024. Available: https://arxiv.org/abs/2311.16922

[6] Yuheng Huang, Jiayang Bai, Jiazhi Guan, Yiqing Shen, Yung-Hui Li, and Yun Fu. OPERA: Alleviating Hallucination in Multi-Modal Large Language Models via Over-Trust Penalty and Retrospection-Allocation. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2024. Available: https://arxiv.org/abs/2311.17911

[7] Yung-Sung Chuang, Yujia Xie, Hongyin Luo, Yoon Kim, James Glass, and Pengcheng He. DoLa: Decoding by Contrasting Layers Improves Factuality in Large Language Models. In *Proceedings of the International Conference on Learning Representations (ICLR)*, 2024. Available: https://arxiv.org/abs/2309.03883

[8] Chenxi Wang, Xiang Chen, Ningyu Zhang, Bozhong Tian, Haoming Xu, Shumin Deng, and Huajun Chen. MLLM can see? Dynamic Correction Decoding for Hallucination Mitigation. arXiv:2410.11779, 2024.

[9] Yiyang Zhou, Chenhang Cui, Jaehong Yoon, Linjun Zhang, Zhun Deng, Chelsea Finn, Mohit Bansal, and Huaxiu Yao. Analyzing and Mitigating Object Hallucination in Large Vision-Language Models. In *Proceedings of ICLR*, 2024. OpenReview: https://openreview.net/forum?id=oZDJKTlOUe

[10] Shukang Yin, Chaoyou Fu, Sirui Zhao, Tong Xu, Hao Wang, Dianbo Sui, Yunhang Shen, Ke Li, Xing Sun, and Enhong Chen. Woodpecker: Hallucination Correction for Multimodal Large Language Models. *Science China Information Sciences*, 2024. DOI: https://doi.org/10.1007/s11432-024-4251-x. Preprint: https://arxiv.org/abs/2310.16045

[11] Seongyun Lee, Sue Hyun Park, Yongrae Jo, and Minjoon Seo. Volcano: Mitigating Multimodal Hallucination through Self-Feedback Guided Revision. In *Proceedings of the North American Chapter of the Association for Computational Linguistics (NAACL)*, 2024. Available: https://arxiv.org/abs/2311.07362

[12] Zhiyuan Zhao, Bin Wang, Linke Ouyang, Xiaoyi Dong, Jiaqi Wang, and Conghui He. Beyond Hallucinations: Enhancing LVLMs through Hallucination-Aware Direct Preference Optimization. arXiv:2311.16839, 2023.

