from __future__ import annotations


class StructuralComparator:
    """Placeholder structural comparator used by the connective prototypes."""

    def __init__(self, abstraction_engine) -> None:
        self.abstraction = abstraction_engine

    def similarity(self, left: dict, right: dict) -> float:
        left_tokens = set(str(left).lower().split())
        right_tokens = set(str(right).lower().split())
        if not left_tokens or not right_tokens:
            return 0.0
        return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)
