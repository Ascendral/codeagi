"""
Transfer Cortex — cross-domain pattern recognition and prediction.

The shower-to-garage engine.

When the system learns that wet tile causes slipping in the bathroom,
and wet linoleum causes slipping in the kitchen, the AbstractionEngine
forms a pattern: "liquid on smooth surface = reduced friction."

The Transfer Cortex takes that pattern and does three things:
1. REGISTERS it as transferable across domains
2. SCANS new situations for matches in domains it hasn't seen before
3. GENERATES predictions: "This garage floor with oil matches the slip pattern"
4. CONFIRMS or REFUTES predictions to strengthen or weaken the pattern

The output is not an alarm. It's awareness:
"Heads up — this situation rhymes with something I've seen before."

Uses NARS truth values for principled confidence tracking:
- analogy() inference for cross-domain transfer
- revision() for combining evidence across confirmations
- temporal_projection() for decaying stale predictions

No LLM. Pure structural pattern matching + evidential reasoning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from klomboagi.reasoning.abstraction import AbstractionEngine, Abstraction
from klomboagi.reasoning.comparator import StructuralComparator
from klomboagi.reasoning.causal import CausalModel
from klomboagi.reasoning.truth import (
    TruthValue, Belief, EvidenceStamp,
    revision, analogy, negation, temporal_projection,
    w2c,
)
from klomboagi.storage.manager import StorageManager
from klomboagi.storage.runtime_state import JsonStateStore
from klomboagi.utils.time import utc_now
from klomboagi.utils.ids import new_id


# ── Data Structures ──

@dataclass
class TransferablePattern:
    """A structural pattern that has been seen across multiple domains."""
    id: str
    abstraction_id: str              # Links to AbstractionEngine's abstraction
    abstract_principle: str          # Human-readable: "liquid on smooth surface = friction loss"
    invariant_roles: list[str]       # The structural skeleton: ["liquid_agent", "smooth_surface", "outcome"]
    source_domains: list[str]        # Where we've seen this: ["bathroom", "kitchen"]
    confidence: TruthValue           # NARS truth value — grows with cross-domain confirmation
    instance_count: int = 0
    last_transferred: str = ""
    predictions_made: int = 0
    predictions_confirmed: int = 0
    predictions_refuted: int = 0
    created_at: str = ""
    updated_at: str = ""

    def confirmation_rate(self) -> float:
        total = self.predictions_confirmed + self.predictions_refuted
        if total == 0:
            return 0.0
        return self.predictions_confirmed / total

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "abstraction_id": self.abstraction_id,
            "abstract_principle": self.abstract_principle,
            "invariant_roles": self.invariant_roles,
            "source_domains": self.source_domains,
            "confidence": self.confidence.to_dict(),
            "instance_count": self.instance_count,
            "last_transferred": self.last_transferred,
            "predictions_made": self.predictions_made,
            "predictions_confirmed": self.predictions_confirmed,
            "predictions_refuted": self.predictions_refuted,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @staticmethod
    def from_dict(d: dict) -> TransferablePattern:
        return TransferablePattern(
            id=d["id"],
            abstraction_id=d["abstraction_id"],
            abstract_principle=d["abstract_principle"],
            invariant_roles=d.get("invariant_roles", []),
            source_domains=d.get("source_domains", []),
            confidence=TruthValue.from_dict(d.get("confidence", {})),
            instance_count=d.get("instance_count", 0),
            last_transferred=d.get("last_transferred", ""),
            predictions_made=d.get("predictions_made", 0),
            predictions_confirmed=d.get("predictions_confirmed", 0),
            predictions_refuted=d.get("predictions_refuted", 0),
            created_at=d.get("created_at", ""),
            updated_at=d.get("updated_at", ""),
        )


@dataclass
class TransferPrediction:
    """A prediction that a known pattern applies in a new domain."""
    id: str
    pattern_id: str
    source_domain: str               # Where we learned it
    target_domain: str               # Where we think it applies
    prediction: str                  # "slip risk exists in garage"
    truth_value: TruthValue          # Computed via NARS analogy()
    status: str = "pending"          # "pending", "confirmed", "refuted"
    created_at: str = ""
    resolved_at: str | None = None
    resolution_note: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "pattern_id": self.pattern_id,
            "source_domain": self.source_domain,
            "target_domain": self.target_domain,
            "prediction": self.prediction,
            "truth_value": self.truth_value.to_dict(),
            "status": self.status,
            "created_at": self.created_at,
            "resolved_at": self.resolved_at,
            "resolution_note": self.resolution_note,
        }

    @staticmethod
    def from_dict(d: dict) -> TransferPrediction:
        return TransferPrediction(
            id=d["id"],
            pattern_id=d["pattern_id"],
            source_domain=d["source_domain"],
            target_domain=d["target_domain"],
            prediction=d["prediction"],
            truth_value=TruthValue.from_dict(d.get("truth_value", {})),
            status=d.get("status", "pending"),
            created_at=d.get("created_at", ""),
            resolved_at=d.get("resolved_at"),
            resolution_note=d.get("resolution_note", ""),
        )


# ── Transfer Cortex ──

class TransferCortex:
    """
    Cross-domain pattern recognition and prediction engine.

    Sits between the AbstractionEngine (which forms patterns from episodes)
    and the rest of the brain (which acts on those patterns).

    When a new abstraction is formed or strengthened, the cortex:
    1. Registers it as a transferable pattern with its source domains
    2. Watches for new episodes that match the pattern in NEW domains
    3. Generates predictions using NARS analogy() inference
    4. Tracks whether predictions come true to build/erode confidence

    The result: the system doesn't just learn "wet floor = slip."
    It learns the abstract principle and recognizes it in new contexts
    it has never encountered before.
    """

    def __init__(
        self,
        storage: StorageManager,
        abstraction_engine: AbstractionEngine,
        comparator: StructuralComparator,
        causal_model: CausalModel,
    ) -> None:
        self.storage = storage
        self.abstraction = abstraction_engine
        self.comparator = comparator
        self.causal = causal_model

        # Own storage — new files, doesn't touch existing
        self._patterns_store = JsonStateStore(
            storage.paths.long_term_root / "memory" / "connective" / "transfer_index.json",
            storage.manifest_store,
            "connective.transfer_index",
        )
        self._predictions_store = JsonStateStore(
            storage.paths.long_term_root / "memory" / "connective" / "predictions.json",
            storage.manifest_store,
            "connective.predictions",
        )

        # In-memory state loaded from storage
        self.patterns: dict[str, TransferablePattern] = {}
        self.predictions: list[TransferPrediction] = []
        self._evidence_counter = 0
        self._load()

    # ── Persistence ──

    def _load(self) -> None:
        raw_patterns = self._patterns_store.load(default={})
        for pid, pdata in raw_patterns.items():
            self.patterns[pid] = TransferablePattern.from_dict(pdata)

        raw_predictions = self._predictions_store.load(default=[])
        self.predictions = [TransferPrediction.from_dict(p) for p in raw_predictions]

    def save(self) -> None:
        self._patterns_store.save({pid: p.to_dict() for pid, p in self.patterns.items()})
        self._predictions_store.save([p.to_dict() for p in self.predictions])

    def _next_evidence_id(self) -> int:
        self._evidence_counter += 1
        return self._evidence_counter

    # ── Core: Register an abstraction as transferable ──

    def register_abstraction(self, abstraction: Abstraction | dict, domain: str) -> TransferablePattern:
        """
        When a new abstraction is formed, register it for cross-domain transfer.

        If we already have this pattern, add the domain and strengthen confidence.
        If it's new, create a TransferablePattern with initial truth value.
        """
        if isinstance(abstraction, Abstraction):
            abs_id = abstraction.id
            abs_name = abstraction.name
            invariants = abstraction.invariants
            abs_confidence = abstraction.confidence
        else:
            abs_id = abstraction["id"]
            abs_name = abstraction.get("name", "unknown")
            invariants = abstraction.get("invariants", [])
            abs_confidence = abstraction.get("confidence", 0.5)

        # Check if we already track this pattern
        existing = self.patterns.get(abs_id)
        if existing:
            if domain not in existing.source_domains:
                # New domain for existing pattern — this is a confirmed transfer!
                existing.source_domains.append(domain)
                existing.instance_count += 1
                # Strengthen confidence via revision with new evidence
                new_evidence = TruthValue.from_single_observation(positive=True)
                existing.confidence = revision(existing.confidence, new_evidence)
                existing.updated_at = utc_now()

                self.storage.event_log.append(
                    "nexus.transfer.domain_added",
                    {"pattern_id": abs_id, "new_domain": domain,
                     "total_domains": len(existing.source_domains)},
                )
            else:
                # Same domain, just more evidence
                existing.instance_count += 1
                existing.confidence = TruthValue(
                    existing.confidence.frequency,
                    min(0.99, existing.confidence.confidence + 0.02),
                )
                existing.updated_at = utc_now()

            self.save()
            return existing

        # New pattern
        now = utc_now()
        principle = self._derive_principle(abs_name, invariants)
        roles = self._extract_roles(invariants)

        pattern = TransferablePattern(
            id=abs_id,
            abstraction_id=abs_id,
            abstract_principle=principle,
            invariant_roles=roles,
            source_domains=[domain],
            confidence=TruthValue(frequency=abs_confidence, confidence=w2c(1)),
            instance_count=1,
            created_at=now,
            updated_at=now,
        )
        self.patterns[abs_id] = pattern

        self.storage.event_log.append(
            "nexus.transfer.pattern_registered",
            {"pattern_id": abs_id, "principle": principle, "domain": domain},
        )

        self.save()
        return pattern

    # ── Core: Scan a new episode for cross-domain matches ──

    def scan_for_transfers(self, episode: dict) -> list[TransferPrediction]:
        """
        Given a new episode, check if any known patterns apply in a new domain.

        Uses AbstractionEngine.match() to find fitting patterns. If the episode's
        domain is NOT already in the pattern's source_domains, this is a potential
        cross-domain transfer — generate a prediction.
        """
        if not self.patterns:
            return []

        episode_domain = self._extract_domain(episode)
        matches = self.abstraction.match(episode)
        new_predictions = []

        for matched_abstraction, fit_score in matches:
            abs_id = matched_abstraction["id"]
            pattern = self.patterns.get(abs_id)

            if pattern is None:
                continue

            # Only generate prediction if this is a NEW domain for this pattern
            if episode_domain in pattern.source_domains:
                continue

            # Don't predict if fit is too weak
            if fit_score < 0.4:
                continue

            # Generate prediction using NARS analogy
            prediction = self._generate_prediction(pattern, episode_domain, fit_score)
            new_predictions.append(prediction)

        if new_predictions:
            self.predictions.extend(new_predictions)
            self.save()

        return new_predictions

    def _generate_prediction(
        self,
        pattern: TransferablePattern,
        target_domain: str,
        fit_score: float,
    ) -> TransferPrediction:
        """
        Create a transfer prediction using NARS analogy() inference.

        Logic:
        - Pattern works in source domains (pattern.confidence)
        - New episode fits the pattern's structure (fit_score)
        - Therefore, pattern likely works in target domain (analogy truth value)

        analogy(A→B, B↔C) = A→C
        Here: A = "situations with this structure", B = "known domains", C = "target domain"
        """
        # The pattern's track record in its known domains
        pattern_tv = pattern.confidence

        # The structural fit between the new episode and the pattern
        fit_tv = TruthValue(frequency=fit_score, confidence=w2c(1))

        # Analogy inference: pattern applies in source, source structure ≈ target structure
        predicted_tv = analogy(pattern_tv, fit_tv)

        source_domain = pattern.source_domains[0] if pattern.source_domains else "unknown"

        now = utc_now()
        pred = TransferPrediction(
            id=new_id("tpred"),
            pattern_id=pattern.id,
            source_domain=source_domain,
            target_domain=target_domain,
            prediction=(
                f"Pattern '{pattern.abstract_principle}' (seen in {', '.join(pattern.source_domains)}) "
                f"likely applies in {target_domain}"
            ),
            truth_value=predicted_tv,
            status="pending",
            created_at=now,
        )

        pattern.predictions_made += 1
        pattern.last_transferred = now
        pattern.updated_at = now

        self.storage.event_log.append(
            "nexus.transfer.prediction_generated",
            {
                "prediction_id": pred.id,
                "pattern_id": pattern.id,
                "source_domains": pattern.source_domains,
                "target_domain": target_domain,
                "truth_value": predicted_tv.to_dict(),
                "fit_score": fit_score,
            },
        )

        return pred

    # ── Core: Confirm or refute a prediction ──

    def confirm_prediction(self, prediction_id: str, outcome: bool) -> dict:
        """
        Close the loop on a transfer prediction.

        If confirmed: boost the pattern's confidence via revision().
        The pattern now covers one more domain.

        If refuted: weaken confidence. Return a question for investigation:
        "Why didn't pattern X apply in domain Y?"
        """
        pred = None
        for p in self.predictions:
            if p.id == prediction_id:
                pred = p
                break

        if pred is None:
            return {"error": f"Prediction {prediction_id} not found"}

        pattern = self.patterns.get(pred.pattern_id)
        if pattern is None:
            return {"error": f"Pattern {pred.pattern_id} not found"}

        now = utc_now()
        pred.resolved_at = now

        if outcome:
            pred.status = "confirmed"
            pred.resolution_note = f"Pattern confirmed in {pred.target_domain}"
            pattern.predictions_confirmed += 1

            # Add the new domain to the pattern
            if pred.target_domain not in pattern.source_domains:
                pattern.source_domains.append(pred.target_domain)

            # Strengthen via revision with positive evidence
            new_evidence = TruthValue.from_single_observation(positive=True)
            pattern.confidence = revision(pattern.confidence, new_evidence)

            self.storage.event_log.append(
                "nexus.transfer.prediction_confirmed",
                {"prediction_id": pred.id, "pattern_id": pattern.id,
                 "target_domain": pred.target_domain},
            )

            self.save()
            return {
                "status": "confirmed",
                "pattern_confidence": pattern.confidence.to_dict(),
                "domains": pattern.source_domains,
            }

        else:
            pred.status = "refuted"
            pred.resolution_note = f"Pattern did NOT apply in {pred.target_domain}"
            pattern.predictions_refuted += 1

            # Weaken via revision with negative evidence
            new_evidence = TruthValue.from_single_observation(positive=False)
            pattern.confidence = revision(pattern.confidence, new_evidence)

            # Generate investigation question
            investigation_question = (
                f"Why didn't the pattern '{pattern.abstract_principle}' "
                f"(which works in {', '.join(pattern.source_domains)}) "
                f"apply in {pred.target_domain}? "
                f"What's different about {pred.target_domain}?"
            )

            self.storage.event_log.append(
                "nexus.transfer.prediction_refuted",
                {"prediction_id": pred.id, "pattern_id": pattern.id,
                 "target_domain": pred.target_domain,
                 "investigation_needed": investigation_question},
            )

            self.save()
            return {
                "status": "refuted",
                "pattern_confidence": pattern.confidence.to_dict(),
                "investigation_question": investigation_question,
            }

    # ── Awareness: "Heads up" interface ──

    def get_awareness(self, episode: dict) -> list[dict]:
        """
        Given a current situation, return contextual awareness memos.

        Not alarms. Not halt signals. Just: "this rhymes with something
        I've seen before. Here's what happened in those cases."

        Returns a list of awareness memos sorted by relevance.
        """
        if not self.patterns:
            return []

        episode_domain = self._extract_domain(episode)
        matches = self.abstraction.match(episode)
        memos = []

        for matched_abstraction, fit_score in matches:
            abs_id = matched_abstraction["id"]
            pattern = self.patterns.get(abs_id)
            if pattern is None:
                continue

            # Only surface awareness if we have real confidence
            if pattern.confidence.expectation < 0.3:
                continue

            # Build the awareness memo
            is_new_domain = episode_domain not in pattern.source_domains
            domains_str = ", ".join(pattern.source_domains)

            if is_new_domain:
                message = (
                    f"This situation in '{episode_domain}' matches the pattern "
                    f"'{pattern.abstract_principle}' seen in {domains_str}. "
                    f"This is a new domain for this pattern."
                )
                caution = (
                    f"In {domains_str}, this pattern led to predictable outcomes. "
                    f"Be aware that similar dynamics may apply here."
                )
            else:
                message = (
                    f"This situation matches the known pattern "
                    f"'{pattern.abstract_principle}' seen across {domains_str}."
                )
                caution = "Proceed with awareness — this is familiar territory."

            # Use causal model to add predicted outcomes
            predicted_outcomes = self.causal.predict_outcome(pattern.abstract_principle)
            if predicted_outcomes:
                top_effect, prob, conf = predicted_outcomes[0]
                caution += f" Previously associated with: {top_effect} (p={prob:.0%})."

            memos.append({
                "pattern_name": pattern.abstract_principle,
                "pattern_id": pattern.id,
                "relevance": fit_score,
                "confidence": pattern.confidence.to_dict(),
                "expectation": pattern.confidence.expectation,
                "message": message,
                "suggested_caution": caution,
                "domains_seen_in": pattern.source_domains,
                "is_new_domain": is_new_domain,
                "confirmation_rate": pattern.confirmation_rate(),
            })

        # Sort by relevance × confidence expectation
        memos.sort(key=lambda m: m["relevance"] * m["expectation"], reverse=True)
        return memos

    # ── Lateral exploration: look AROUND a concept ──

    def lateral_neighbors(self, concept: str) -> dict:
        """
        Look around a concept, not just at it.

        What patterns share structural roles with this concept?
        What other domains have the same shape?
        What's adjacent, similar, contradictory?
        """
        neighbors = {
            "shared_patterns": [],      # Patterns this concept appears in
            "related_domains": set(),   # Domains where similar structures exist
            "adjacent_roles": [],       # Other roles that co-occur with this concept
            "causal_neighborhood": {
                "causes": [],           # What causes this
                "effects": [],          # What this causes
            },
        }

        concept_lower = concept.lower()

        # 1. Find patterns this concept participates in
        for pid, pattern in self.patterns.items():
            principle_lower = pattern.abstract_principle.lower()
            roles_str = " ".join(pattern.invariant_roles).lower()

            if concept_lower in principle_lower or concept_lower in roles_str:
                neighbors["shared_patterns"].append({
                    "pattern": pattern.abstract_principle,
                    "pattern_id": pid,
                    "domains": pattern.source_domains,
                    "confidence": pattern.confidence.to_dict(),
                })
                neighbors["related_domains"].update(pattern.source_domains)

                # Roles that co-occur with this concept
                for role in pattern.invariant_roles:
                    if concept_lower not in role.lower():
                        neighbors["adjacent_roles"].append(role)

        # 2. Causal neighborhood
        causes = self.causal.graph.get_causes(concept)
        effects = self.causal.graph.get_effects(concept)

        neighbors["causal_neighborhood"]["causes"] = [
            {"cause": e.cause, "strength": e.strength, "confidence": e.confidence}
            for e in causes[:5]
        ]
        neighbors["causal_neighborhood"]["effects"] = [
            {"effect": e.effect, "strength": e.strength, "confidence": e.confidence}
            for e in effects[:5]
        ]

        # 3. Convert set to list for serialization
        neighbors["related_domains"] = list(neighbors["related_domains"])

        return neighbors

    # ── Pending predictions management ──

    def get_pending_predictions(self) -> list[TransferPrediction]:
        """Return all predictions awaiting confirmation."""
        return [p for p in self.predictions if p.status == "pending"]

    def check_predictions_against_episode(self, episode: dict) -> list[dict]:
        """
        Check if a new episode confirms or refutes any pending predictions.

        Looks at the episode's domain and outcome to see if any pending
        predictions about that domain have come true.
        """
        domain = self._extract_domain(episode)
        outcome = episode.get("outcome", episode.get("status", "unknown"))
        success = episode.get("success", outcome in ("completed", "success", "passed"))

        results = []
        for pred in self.predictions:
            if pred.status != "pending":
                continue
            if pred.target_domain != domain:
                continue

            # This episode is in the domain we predicted about — check outcome
            result = self.confirm_prediction(pred.id, success)
            results.append(result)

        return results

    def decay_predictions(self, time_distance: float = 1.0, decay_rate: float = 0.3) -> None:
        """Decay confidence in old pending predictions. Stale predictions matter less."""
        for pred in self.predictions:
            if pred.status == "pending":
                pred.truth_value = temporal_projection(pred.truth_value, time_distance, decay_rate)
        self.save()

    # ── Helpers ──

    def _extract_domain(self, episode: dict) -> str:
        """Extract the domain from an episode. Falls back to description keywords."""
        if "domain" in episode:
            return str(episode["domain"])
        description = episode.get("description", episode.get("mission", ""))
        if isinstance(description, dict):
            description = description.get("description", "")
        # Use first 3 significant words as domain proxy
        words = [w.lower().strip(".,!?") for w in str(description).split()
                 if len(w) > 3 and w.lower() not in ("the", "and", "for", "with", "from", "that", "this")]
        return "_".join(words[:3]) if words else "unknown"

    def _derive_principle(self, name: str, invariants: list[str]) -> str:
        """Derive a human-readable principle from invariant structure."""
        if not invariants:
            return name

        # Extract the key role:value pairs from invariants
        parts = []
        for inv in invariants[:4]:
            if "=" in inv:
                role_type, value = inv.split("=", 1)
                role = role_type.split(":")[0]
                parts.append(f"{role}={value}")
            else:
                parts.append(inv)

        return " + ".join(parts) if parts else name

    def _extract_roles(self, invariants: list[str]) -> list[str]:
        """Extract structural roles from invariant descriptions."""
        roles = []
        for inv in invariants:
            if ":" in inv:
                role = inv.split(":")[0]
                roles.append(role)
        return roles

    # ── Stats ──

    def stats(self) -> dict:
        total_patterns = len(self.patterns)
        multi_domain = sum(1 for p in self.patterns.values() if len(p.source_domains) > 1)
        pending = sum(1 for p in self.predictions if p.status == "pending")
        confirmed = sum(1 for p in self.predictions if p.status == "confirmed")
        refuted = sum(1 for p in self.predictions if p.status == "refuted")

        return {
            "total_patterns": total_patterns,
            "multi_domain_patterns": multi_domain,
            "total_predictions": len(self.predictions),
            "pending_predictions": pending,
            "confirmed_predictions": confirmed,
            "refuted_predictions": refuted,
            "confirmation_rate": confirmed / max(confirmed + refuted, 1),
        }
