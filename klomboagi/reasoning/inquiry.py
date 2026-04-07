from __future__ import annotations

from dataclasses import dataclass


@dataclass
class KnowledgeGap:
    id: str
    question: str
    why_needed: str = ""


class InquiryEngine:
    """Minimal inquiry registry for gap resolution."""

    def __init__(self, storage) -> None:
        self.storage = storage
        self._resolved: dict[str, dict[str, str]] = {}

    def resolve(self, gap_id: str, answer: str, source: str) -> None:
        self._resolved[gap_id] = {"answer": answer, "source": source}
