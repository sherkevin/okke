# 3 Methodology
This section presents the two core phases of **ViHallu**:
1) Visual variation image generation, which employs advanced LLMs and controllable text-to-image (T2I) models to produce high-quality variation images;
2) Visual instruction generation, which constructs high-quality instruction-tuning data based on the generated image pairs.

## 3.1 Visual Variation Image Generation
The visual variation image generation pipeline consists of three key modules: original image captioning and segmentation mask extraction, caption editing via concept substitution, and visual variation image generation with quality evaluation.

### Image Caption and Segmentation Mask
We use **Tag2Text** [10] and **MobileSAM** [42] to obtain captions and segmentation masks from input images.
- Tag2Text extracts image tags and guides a vision-language model (VLM) to generate detailed image captions.
- MobileSAM provides strong zero-shot segmentation ability and outputs precise object segmentation masks, which lay the foundation for controllable visual variation.

### Caption Editing
We propose an LLM-based caption editing mechanism to generate modified captions. The edited captions only differ from the original ones in specified object categories or attributes, while keeping all other contextual content unchanged. The target modification can be applied to individual objects or background scenes.

We design a dedicated prompt template to guide the editing process and use **DeepSeek-chat V2** [7] for caption generation. This mechanism creates counterfactual combinations by inserting objects into low-frequency scenes (e.g., “a herd of zebras grazing in an open field of sandy desert”), which breaks spurious correlations between commonly co-occurring objects and forces the model to rely on visual evidence instead of statistical priors.

### Visual Variation Image Generation
We use **ControlNet++** [27] as the controllable T2I model, taking segmentation masks and edited captions as dual inputs. The masks preserve the spatial layout and object positions, while the modified captions guide local content changes, ensuring structural consistency and natural fusion of altered regions.

For quality control, we adopt **VQAScore** [17] to evaluate image–text alignment. Generated images with scores below 0.6 are filtered out, and only high-quality samples are kept. The original images and the filtered variation images together form the final image dataset.

## 3.2 Visual Instruction Generation
The visual instruction generation phase builds fine-grained QA instruction data through three modules: detailed description and object tag generation, question–answer generation, and multi-model quality assessment.

### Detailed Description and Object Tag Generation
We first use a large vision-language model (LVLM) to generate detailed image descriptions. However, these descriptions may contain hallucinations and often miss minor objects in complex scenes. To ensure full object coverage, we apply **Grounded-SAM** [13,23,30] for object detection and segmentation, extracting complete object tags to supplement the textual descriptions.

### Question-Answer Generation
We use **DeepSeek-chat V2** to generate questions based on the LVLM descriptions and object tags. Hallucinated content in the descriptions naturally leads to questions about non-existent entities (e.g., “Are there trees in the field?” when trees are hallucinated), which target typical hallucination patterns of LVLMs. Object tags ensure all actual objects are covered.

For accurate answers, we employ **InternVL2.5** [5] to generate ground-truth responses based on real image content, rather than the potentially noisy descriptions.

### Quality Assessment
We conduct quality evaluation using an ensemble of expert LVLMs, including LLaVA-1.5 [19], MiniCPM-V 2.6 [38], and mPLUG-OWL3 [39]. Each model judges whether the QA pair aligns with the image content and outputs a binary decision (“Yes” for correct, “No” for misaligned).

We retain QA pairs approved by at least two expert models and discard those rejected by two or more. This majority-voting mechanism ensures the reliability of the final ViHallu-Instruction dataset.