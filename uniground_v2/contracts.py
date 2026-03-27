from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from bra_universal_plugin import UniversalObservation


@dataclass
class RetrievalResult:
    observation: UniversalObservation
    metadata: dict[str, Any]
