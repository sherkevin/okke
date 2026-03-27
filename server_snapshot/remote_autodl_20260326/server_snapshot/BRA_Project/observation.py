from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable, Optional

from PIL import Image

from bra_universal_plugin import UniversalObservation
from uniground_v2.regions import RegionProposal, RegionProposalCacheBuilder


@dataclass
class ObservationCacheResult:
    observation: UniversalObservation
    cache_info: dict[str, object]


def build_v2_observation_cache(
    encoder,
    image: Image.Image,
    proposals: Optional[Iterable[RegionProposal]] = None,
    mode: str = "grid_regions",
) -> ObservationCacheResult:
    start = time.perf_counter()
    builder = RegionProposalCacheBuilder(encoder=encoder)
    observation = builder.build(image=image, proposals=proposals, mode=mode)
    region_embeddings = observation.region_embeddings
    cache_info = {
        "cache_stage": "pre_generate",
        "region_mode": observation.metadata.get("region_mode", mode),
        "region_count": int(region_embeddings.shape[0]) if region_embeddings is not None else 0,
        "precompute_ms": round((time.perf_counter() - start) * 1000.0, 4),
        "uses_detector_proposals": bool(proposals),
    }
    merged_metadata = dict(observation.metadata)
    merged_metadata.update(cache_info)
    observation.metadata = merged_metadata
    return ObservationCacheResult(observation=observation, cache_info=cache_info)


def build_v2_observation(
    encoder,
    image: Image.Image,
    proposals: Optional[Iterable[RegionProposal]] = None,
    mode: str = "grid_regions",
) -> UniversalObservation:
    return build_v2_observation_cache(
        encoder=encoder,
        image=image,
        proposals=proposals,
        mode=mode,
    ).observation
