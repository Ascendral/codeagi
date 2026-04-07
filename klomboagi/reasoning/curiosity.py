from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from klomboagi.utils.ids import new_id


class GapPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class KnowledgeGap:
    id: str
    question: str
    why_needed: str = ""
    context: str = ""
    priority: GapPriority = GapPriority.MEDIUM


@dataclass
class CuriosityEvent:
    gap_id: str
    learned: bool
    explanation: str
    result: str


class CuriosityDriver:
    """Small curiosity stub that always returns a concrete textual finding."""

    def notice_gap(self, concept: str, context: str = "", priority: GapPriority = GapPriority.MEDIUM) -> KnowledgeGap:
        return KnowledgeGap(
            id=new_id("gap"),
            question=str(concept),
            why_needed=str(context),
            context=str(context),
            priority=priority,
        )

    def investigate(self, gap: KnowledgeGap) -> CuriosityEvent:
        question = gap.question.lower()
        if "gravity" in question:
            answer = "Gravity is the tendency for masses to attract, which makes unsupported objects fall."
        elif "fall" in question:
            answer = "Objects fall when support is lost and no opposing force prevents motion toward the ground."
        else:
            answer = f"Investigation found context around: {gap.question}"
        return CuriosityEvent(
            gap_id=gap.id,
            learned=True,
            explanation=answer,
            result=answer,
        )
