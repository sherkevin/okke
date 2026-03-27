# Revision_Log_V64
- 修订了 "Training-free" 的不准确表述，基于审稿人意见，将方法正确定位为 "Base-Model-Frozen Plug-in"，并大方承认引入了 CC3M 带来的 external data asymmetry。
- 彻底删除了 Intro 和实验中关于 DocVQA、document reading、complex charts 的表述，确保与 VASM 放弃 OCR 任务的逻辑自洽；将评测替换为 Spatial Relations 和 Object HalBench。
- 明确了 `TLRA_calib` 的 InfoNCE 训练机制，补充了 noun-phrase extraction + dense feature matching 来防止 bag-of-words collapse。
- 在 3.4 节正面回应了 WordNet fallback 的多义词（Polysemy）问题，提出 "safety-first delegation" 策略。
- 补充了 `TLRA_MeanPool` 必须使用相同 $\Phi_{calib}$ 的实验硬性约束。
- 补充了 Table 1 (AGL std dev)、Figure 1 ($N_v=4096$)、Figure 3 (显示被 mask 掉的 raw score) 的具体作图要求。
- 坚决保留了原有的“正向命题（Positive Proposition）”框架、控制变量法（AGL/PPL）和 $O(M \times N_{active})$ 效率声明。