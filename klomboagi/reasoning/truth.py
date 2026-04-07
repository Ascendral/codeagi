from __future__ import annotations

from dataclasses import dataclass, field

from klomboagi.utils.time import utc_now


def w2c(weight: float) -> float:
    weight = max(0.0, float(weight))
    return weight / (weight + 1.0) if weight > 0 else 0.0


def _c2w(confidence: float) -> float:
    confidence = max(0.0, min(0.999999, float(confidence)))
    if confidence == 0:
        return 0.0
    return confidence / (1.0 - confidence)


@dataclass
class TruthValue:
    frequency: float = 0.5
    confidence: float = 0.0

    @property
    def expectation(self) -> float:
        return self.confidence * (self.frequency - 0.5) + 0.5

    def is_positive(self) -> bool:
        return self.frequency >= 0.5

    def to_dict(self) -> dict[str, float]:
        return {"frequency": self.frequency, "confidence": self.confidence}

    @staticmethod
    def from_dict(data: dict) -> "TruthValue":
        return TruthValue(
            frequency=float(data.get("frequency", 0.5)),
            confidence=float(data.get("confidence", 0.0)),
        )

    @staticmethod
    def from_single_observation(positive: bool) -> "TruthValue":
        return TruthValue(1.0 if positive else 0.0, w2c(1))


@dataclass
class EvidenceStamp:
    evidence_ids: list[int] = field(default_factory=list)
    created_at: str = field(default_factory=utc_now)

    @staticmethod
    def new(evidence_id: int) -> "EvidenceStamp":
        return EvidenceStamp(evidence_ids=[int(evidence_id)], created_at=utc_now())

    def overlaps(self, other: "EvidenceStamp") -> bool:
        return bool(set(self.evidence_ids) & set(other.evidence_ids))

    def to_dict(self) -> dict:
        return {"evidence_ids": self.evidence_ids, "created_at": self.created_at}

    @staticmethod
    def from_dict(data: dict) -> "EvidenceStamp":
        return EvidenceStamp(
            evidence_ids=[int(v) for v in data.get("evidence_ids", [])],
            created_at=str(data.get("created_at", utc_now())),
        )


@dataclass
class Belief:
    statement: str
    truth: TruthValue
    stamp: EvidenceStamp
    subject: str = ""
    predicate: str = ""
    source: str = ""

    def to_dict(self) -> dict:
        return {
            "statement": self.statement,
            "truth": self.truth.to_dict(),
            "stamp": self.stamp.to_dict(),
            "subject": self.subject,
            "predicate": self.predicate,
            "source": self.source,
        }

    @staticmethod
    def from_dict(data: dict) -> "Belief":
        return Belief(
            statement=str(data.get("statement", "")),
            truth=TruthValue.from_dict(data.get("truth", {})),
            stamp=EvidenceStamp.from_dict(data.get("stamp", {})),
            subject=str(data.get("subject", "")),
            predicate=str(data.get("predicate", "")),
            source=str(data.get("source", "")),
        )

    def revise_with(self, other: "Belief") -> "Belief | None":
        if self.stamp.overlaps(other.stamp):
            return None
        merged_ids = sorted(set(self.stamp.evidence_ids + other.stamp.evidence_ids))
        return Belief(
            statement=self.statement,
            truth=revision(self.truth, other.truth),
            stamp=EvidenceStamp(evidence_ids=merged_ids, created_at=utc_now()),
            subject=self.subject or other.subject,
            predicate=self.predicate or other.predicate,
            source=self.source or other.source,
        )


def revision(a: TruthValue, b: TruthValue) -> TruthValue:
    wa = _c2w(a.confidence)
    wb = _c2w(b.confidence)
    total = wa + wb
    if total <= 0:
        return TruthValue((a.frequency + b.frequency) / 2.0, 0.0)
    return TruthValue(
        frequency=(wa * a.frequency + wb * b.frequency) / total,
        confidence=w2c(total),
    )


def analogy(pattern: TruthValue, fit: TruthValue) -> TruthValue:
    return TruthValue(
        frequency=(pattern.frequency + fit.frequency) / 2.0,
        confidence=max(0.0, min(0.99, pattern.confidence * fit.confidence)),
    )


def negation(value: TruthValue) -> TruthValue:
    return TruthValue(frequency=1.0 - value.frequency, confidence=value.confidence)


def temporal_projection(value: TruthValue, time_distance: float = 1.0, decay_rate: float = 0.1) -> TruthValue:
    decay = max(0.0, 1.0 - float(time_distance) * float(decay_rate))
    return TruthValue(frequency=value.frequency, confidence=value.confidence * decay)
