from __future__ import annotations

from enum import Enum
from typing import Any, Callable


class CognitionPhase(str, Enum):
    PERCEIVE = "perceive"
    REMEMBER = "remember"
    INQUIRE = "inquire"
    LEARN = "learn"


class CognitionLoop:
    """Minimal cognition loop shim for Nexus wiring."""

    def __init__(self, storage) -> None:
        self.storage = storage
        self.on_phase: Callable[[CognitionPhase, str], None] | None = None
        self.on_inquiry: Callable[[Any], Any] | None = None

    def think(self, problem: dict) -> dict:
        if self.on_phase:
            self.on_phase(CognitionPhase.PERCEIVE, "perceive")
            self.on_phase(CognitionPhase.REMEMBER, "remember")
            self.on_phase(CognitionPhase.LEARN, "learn")
        return {"status": "ok", "problem": problem}
