from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CuriosityTarget:
    concept: str
    context: dict = field(default_factory=dict)
