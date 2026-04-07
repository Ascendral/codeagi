from __future__ import annotations

from dataclasses import dataclass


def _tokens(text: str) -> set[str]:
    return {token.strip(".,!?-_'\"").lower() for token in str(text).split() if token.strip(".,!?-_'\"")}


@dataclass
class CausalEdge:
    cause: str
    effect: str
    strength: float = 0.6
    confidence: float = 0.5


class CausalGraph:
    def __init__(self) -> None:
        self.nodes: set[str] = set()
        self.edges: list[CausalEdge] = []

    def add_edge(self, cause: str, effect: str, strength: float = 0.6, confidence: float = 0.5) -> CausalEdge:
        edge = CausalEdge(cause=cause, effect=effect, strength=strength, confidence=confidence)
        self.nodes.add(cause)
        self.nodes.add(effect)
        self.edges.append(edge)
        return edge

    def get_causes(self, concept: str) -> list[CausalEdge]:
        concept_tokens = _tokens(concept)
        return [edge for edge in self.edges if concept_tokens & _tokens(edge.effect)]

    def get_effects(self, concept: str) -> list[CausalEdge]:
        concept_tokens = _tokens(concept)
        return [edge for edge in self.edges if concept_tokens & _tokens(edge.cause)]


class CausalModel:
    """Small causal memory used by the connective tests."""

    def __init__(self, storage) -> None:
        self.storage = storage
        self.graph = CausalGraph()

    def learn_from_episode(self, episode: dict) -> list[CausalEdge]:
        outcome = str(episode.get("outcome", "unknown"))
        actions = episode.get("actions", [])
        edges: list[CausalEdge] = []
        if outcome == "unknown":
            return edges

        for action in actions:
            for candidate in (action.get("target"), action.get("type")):
                if candidate:
                    edges.append(self.graph.add_edge(str(candidate), outcome, strength=0.7, confidence=0.6))

        description = str(episode.get("description", ""))
        if description:
            key_phrase = " ".join(list(_tokens(description))[:3])
            if key_phrase:
                edges.append(self.graph.add_edge(key_phrase, outcome, strength=0.55, confidence=0.45))

        return edges

    def explain_outcome(self, question: str) -> list[tuple[str, float, float]]:
        return [(edge.cause, edge.strength, edge.confidence) for edge in self.graph.get_causes(question)]

    def predict_outcome(self, cause: str) -> list[tuple[str, float, float]]:
        return [(edge.effect, edge.strength, edge.confidence) for edge in self.graph.get_effects(cause)]

    def what_should_i_test(self) -> dict | None:
        if not self.graph.edges:
            return None
        edge = self.graph.edges[-1]
        return {"cause": edge.cause, "effect": edge.effect, "confidence": edge.confidence}
