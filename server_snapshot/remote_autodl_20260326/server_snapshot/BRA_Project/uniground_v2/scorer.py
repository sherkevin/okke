from __future__ import annotations

import torch
import torch.nn.functional as F

from bra_universal_plugin import UniversalObservation, UniversalPluginOutput


class HardcodedUniversalScorer:
    """
    Deterministic fallback scorer used for end-to-end pipeline validation.

    This is intentionally non-learned: it combines cosine agreements in the
    frozen external space so we can validate the full runtime before a trained
    Psi_univ checkpoint is ready.
    """

    def __call__(
        self,
        observation: UniversalObservation,
        candidate_embeddings: torch.Tensor,
        prefix_embedding: torch.Tensor,
    ) -> UniversalPluginOutput:
        candidates = F.normalize(candidate_embeddings.float(), dim=-1)
        prefix = F.normalize(prefix_embedding.float(), dim=-1)
        if prefix.ndim == 2 and prefix.shape[0] == 1:
            prefix = prefix.expand(candidates.shape[0], -1)

        image = observation.image_embedding.float()
        if image.ndim == 1:
            image = image.unsqueeze(0)
        if image.shape[0] == 1:
            image = image.expand(candidates.shape[0], -1)
        image = F.normalize(image, dim=-1)

        global_sim = F.cosine_similarity(image, candidates, dim=-1)
        prefix_sim = F.cosine_similarity(prefix, candidates, dim=-1)

        if observation.region_embeddings is None or observation.region_embeddings.numel() == 0:
            region_sim = global_sim
        else:
            regions = observation.region_embeddings.float()
            if regions.ndim == 3:
                regions = regions.mean(dim=1)
            regions = F.normalize(regions, dim=-1)
            if regions.ndim == 1:
                regions = regions.unsqueeze(0)
            region_sim = (candidates @ regions.T).max(dim=-1).values

        anchor = torch.maximum(global_sim, region_sim)
        support = 4.0 * anchor + 0.5 * prefix_sim
        contradiction = 4.0 * (0.35 - anchor) - 0.25 * prefix_sim
        abstain = 6.0 * (0.20 - anchor)
        return UniversalPluginOutput(
            support=support,
            contradiction=contradiction,
            abstain=abstain,
        )
