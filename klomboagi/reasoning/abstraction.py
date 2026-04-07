from __future__ import annotations

from dataclasses import dataclass, field

from klomboagi.utils.ids import new_id


def _tokenize(text: str) -> set[str]:
    stop = {"the", "and", "for", "with", "from", "that", "this", "into", "over", "under"}
    return {
        token.strip(".,!?-_'\"").lower()
        for token in str(text).split()
        if len(token.strip(".,!?-_'\"")) >= 3 and token.strip(".,!?-_'\"").lower() not in stop
    }


@dataclass
class Abstraction:
    id: str
    name: str
    schema: list[dict] = field(default_factory=list)
    invariants: list[str] = field(default_factory=list)
    variables: list[str] = field(default_factory=list)
    source_episodes: list[str] = field(default_factory=list)
    instance_count: int = 0
    confidence: float = 0.5

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "schema": self.schema,
            "invariants": self.invariants,
            "variables": self.variables,
            "source_episodes": self.source_episodes,
            "instance_count": self.instance_count,
            "confidence": self.confidence,
        }


class AbstractionEngine:
    """Very small structural abstraction helper for the connective layer."""

    def __init__(self, storage) -> None:
        self.storage = storage
        self.abstractions: dict[str, Abstraction] = {}

    def abstract(self, episodes: list[dict]) -> Abstraction | None:
        if len(episodes) < 2:
            return None

        desc_tokens = [_tokenize(ep.get("description", "")) for ep in episodes]
        common_tokens = set.intersection(*desc_tokens) if desc_tokens else set()
        action_sets = [
            {str(action.get("type", "")).lower() for action in ep.get("actions", []) if action.get("type")}
            for ep in episodes
        ]
        common_actions = set.intersection(*action_sets) if action_sets else set()
        outcomes = {str(ep.get("outcome", "")).lower() for ep in episodes if ep.get("outcome")}
        common_outcome = outcomes.pop() if len(outcomes) == 1 else ""

        invariants = [f"token:shared={token}" for token in sorted(common_tokens)[:3]]
        invariants.extend(f"action:operation={action}" for action in sorted(common_actions)[:2])
        if common_outcome and common_outcome != "unknown":
            invariants.append(f"outcome:state={common_outcome}")

        name_parts = sorted(common_actions)[:1] + sorted(common_tokens)[:2]
        name = "pattern:" + "+".join(name_parts) if name_parts else "pattern:episode_group"

        abstraction = Abstraction(
            id=new_id("abs"),
            name=name,
            schema=[{"role": "action", "type": "operation"}, {"role": "outcome", "type": "state"}],
            invariants=invariants,
            variables=[str(ep.get("domain", "unknown")) for ep in episodes[:3]],
            source_episodes=[str(ep.get("id", "unknown")) for ep in episodes],
            instance_count=len(episodes),
            confidence=min(0.9, 0.45 + len(episodes) * 0.1),
        )
        self.abstractions[abstraction.id] = abstraction
        return abstraction

    def match(self, episode: dict) -> list[tuple[dict, float]]:
        episode_tokens = _tokenize(episode.get("description", ""))
        episode_tokens.add(str(episode.get("domain", "")).lower())
        episode_tokens.add(str(episode.get("outcome", "")).lower())
        for action in episode.get("actions", []):
            episode_tokens.add(str(action.get("type", "")).lower())
            episode_tokens.add(str(action.get("target", "")).lower())

        matches: list[tuple[dict, float]] = []
        for abstraction in self.abstractions.values():
            abstraction_tokens = _tokenize(abstraction.name)
            abstraction_tokens |= _tokenize(" ".join(abstraction.invariants))
            abstraction_tokens |= _tokenize(" ".join(abstraction.variables))
            if not abstraction_tokens:
                continue
            overlap = len(episode_tokens & abstraction_tokens)
            score = overlap / max(1, len(abstraction_tokens))
            if score > 0:
                matches.append((abstraction.to_dict(), min(1.0, score + abstraction.confidence * 0.1)))

        matches.sort(key=lambda item: item[1], reverse=True)
        return matches
