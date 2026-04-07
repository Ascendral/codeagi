"""
Cognitive Nexus — the connective tissue of the brain.

This is the central hub that wires KlomboAGI's subsystems together.
Each subsystem already works on its own:
- CausalModel knows cause and effect
- TruthValue tracks evidential confidence
- CuriosityDriver seeks knowledge
- InquiryEngine identifies gaps
- AbstractionEngine finds patterns
- AttentionController decides what to focus on
- StructuralComparator transfers across domains

The Nexus makes them talk to each other. It hooks into existing callbacks
and routes information between systems so the brain works as a whole.

Five behaviors emerge from the wiring:
1. DEEP PURSUIT — recursive investigation that follows knowledge threads
2. CORRECTNESS OBSESSION — cross-check everything, refuse low confidence
3. LATERAL EXPLORATION — look around a subject, not just at it
4. SPONTANEOUS PREDICTION — hypothesize outcomes from known patterns
5. UNIFIED LEARNING — every experience flows through all systems

The shower-to-garage insight lives here:
- Experience water on tile → causal model learns slip risk
- See same structure in kitchen → abstraction engine forms pattern
- Enter garage with oil on floor → transfer cortex recognizes the shape
- Nexus surfaces awareness: "heads up, this rhymes with the slip pattern"

No LLM. Pure integration of existing algorithmic systems.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from klomboagi.reasoning.abstraction import AbstractionEngine
from klomboagi.reasoning.attention import AttentionController, Drive
from klomboagi.reasoning.causal import CausalModel
from klomboagi.reasoning.cognition_loop import CognitionLoop, CognitionPhase
from klomboagi.reasoning.comparator import StructuralComparator
from klomboagi.reasoning.curiosity import CuriosityDriver
from klomboagi.core.guided_curiosity import CuriosityTarget
from klomboagi.reasoning.inquiry import InquiryEngine, KnowledgeGap
from klomboagi.reasoning.self_eval import SelfEvaluator
from klomboagi.reasoning.truth import (
    TruthValue, Belief, EvidenceStamp,
    revision, negation, temporal_projection,
    w2c,
)
from klomboagi.storage.manager import StorageManager
from klomboagi.storage.runtime_state import JsonStateStore
from klomboagi.utils.time import utc_now
from klomboagi.utils.ids import new_id

from klomboagi.connective.transfer_cortex import TransferCortex, TransferPrediction
from klomboagi.connective.investigation import InvestigationEngine, InvestigationThread


# ── Data Structures ──

@dataclass
class AwarenessMemo:
    """
    A 'heads up' — not an alarm, just contextual awareness.

    "This situation looks like something I've seen before.
    In those cases, here's what happened. Proceed with awareness."
    """
    pattern_name: str
    relevance: float                 # How closely this situation matches (0-1)
    truth_value: TruthValue          # How confident we are in the pattern
    message: str                     # What the pattern is about
    suggested_caution: str           # What to be aware of
    domains_seen_in: list[str]       # Where we've seen this before
    is_new_domain: bool = False      # Is this a domain we haven't seen the pattern in before?

    def to_dict(self) -> dict:
        return {
            "pattern_name": self.pattern_name,
            "relevance": self.relevance,
            "truth_value": self.truth_value.to_dict(),
            "expectation": self.truth_value.expectation,
            "message": self.message,
            "suggested_caution": self.suggested_caution,
            "domains_seen_in": self.domains_seen_in,
            "is_new_domain": self.is_new_domain,
        }


# ── Cognitive Nexus ──

class CognitiveNexus:
    """
    The connective tissue of KlomboAGI's brain.

    Wires into existing subsystem callbacks. Routes information between
    systems. Orchestrates the five emergent behaviors.

    Usage:
        storage = StorageManager.bootstrap()
        nexus = CognitiveNexus(storage)

        # The nexus hooks into the cognition loop automatically.
        # Use think() for reasoning problems:
        state = nexus.think({"description": "solve this"})

        # Use learn_experience() after any episode:
        result = nexus.learn_experience(episode)

        # Check for awareness before acting:
        memos = nexus.get_awareness(situation)

        # Call tick() each runtime cycle:
        nexus.tick()
    """

    def __init__(self, storage: StorageManager) -> None:
        self.storage = storage

        # ── Existing subsystems (same storage, shared state) ──
        self.abstraction = AbstractionEngine(storage)
        self.causal = CausalModel(storage)
        self.curiosity = CuriosityDriver()
        self.inquiry = InquiryEngine(storage)
        self.comparator = StructuralComparator(self.abstraction)
        self.attention = AttentionController()
        self.evaluator = SelfEvaluator()
        self.cognition_loop = CognitionLoop(storage)

        # ── New connective modules ──
        self.transfer_cortex = TransferCortex(
            storage, self.abstraction, self.comparator, self.causal,
        )
        self.investigation = InvestigationEngine(
            storage, self.causal, self.curiosity, self.inquiry, self.abstraction,
        )

        # ── Shared belief store ──
        # All findings, assertions, and cross-checks land here as NARS Beliefs
        self.beliefs: dict[str, Belief] = {}
        self._beliefs_store = JsonStateStore(
            storage.paths.long_term_root / "memory" / "connective" / "beliefs.json",
            storage.manifest_store,
            "connective.beliefs",
        )

        # Evidence counter for unique stamps
        self._evidence_counter = 0

        # Load persisted beliefs
        self._load_beliefs()

        # Share belief store with investigation engine
        self.investigation.belief_store = self.beliefs

        # ── Wire callbacks ──
        self._wire_callbacks()

    # ── Initialization ──

    def _wire_callbacks(self) -> None:
        """Hook into existing subsystem callbacks."""
        # React to cognition loop phase transitions
        self.cognition_loop.on_phase = self._on_phase

        # Deep investigation instead of single-shot gap handling
        self.cognition_loop.on_inquiry = self._on_inquiry

    def _load_beliefs(self) -> None:
        """Load persisted beliefs from storage."""
        raw = self._beliefs_store.load(default={})
        for key, bdata in raw.items():
            try:
                self.beliefs[key] = Belief.from_dict(bdata)
            except (KeyError, TypeError):
                continue

    def _save_beliefs(self) -> None:
        """Persist beliefs to storage."""
        self._beliefs_store.save({
            key: belief.to_dict() for key, belief in self.beliefs.items()
        })

    def _next_evidence_id(self) -> int:
        self._evidence_counter += 1
        return self._evidence_counter

    # ═══════════════════════════════════════════════════════
    # BEHAVIOR 1: DEEP PURSUIT — recursive knowledge seeking
    # ═══════════════════════════════════════════════════════

    def pursue(
        self,
        question: str,
        context: str = "",
        max_depth: int = 3,
        confidence_threshold: float = 0.5,
    ) -> InvestigationThread:
        """
        Deep knowledge seeking. Opens an investigation thread that
        follows the knowledge chain recursively.

        One answer leads to the next question. Cross-checks against
        the causal model, abstraction engine, and belief store.
        Won't accept low-confidence answers.
        """
        thread = self.investigation.pursue(
            question, context, max_depth, confidence_threshold,
        )

        # Register all findings as beliefs
        for finding in thread.findings:
            belief_key = f"pursuit:{question[:40]}:{finding.question[:40]}"
            self.beliefs[belief_key] = finding.belief

        # Boost curiosity drive — we just learned something
        if thread.findings:
            self.attention.drives.satisfy(Drive.CURIOSITY, 0.2)

        self._save_beliefs()
        return thread

    # ═══════════════════════════════════════════════════════
    # BEHAVIOR 2: CORRECTNESS OBSESSION — cross-check everything
    # ═══════════════════════════════════════════════════════

    def assert_belief(
        self,
        statement: str,
        source: str,
        frequency: float = 1.0,
        confidence: float = 0.5,
    ) -> Belief:
        """
        Register a belief and cross-check it against what we already know.

        If a contradicting belief exists with high confidence, opens an
        investigation to resolve the contradiction.

        If confidence is below threshold, opens an investigation to
        strengthen it.

        Returns the belief (possibly revised if corroborating evidence found).
        """
        stamp = EvidenceStamp.new(self._next_evidence_id())
        new_belief = Belief(
            statement=statement,
            truth=TruthValue(frequency, confidence),
            stamp=stamp,
            source=source,
        )

        # Check for existing beliefs about the same subject
        statement_words = set(statement.lower().split())
        for key, existing in list(self.beliefs.items()):
            existing_words = set(existing.statement.lower().split())
            overlap = len(statement_words & existing_words)

            if overlap < 3:
                continue

            # Check for contradiction
            if self._beliefs_contradict(new_belief, existing):
                # Contradiction found — boost coherence drive
                self.attention.on_contradiction()

                # If existing belief is strong, investigate
                if existing.truth.confidence > 0.5:
                    self.storage.event_log.append(
                        "nexus.belief.contradiction",
                        {
                            "new": statement[:100],
                            "existing": existing.statement[:100],
                            "existing_confidence": existing.truth.confidence,
                        },
                    )
                    # Open investigation to resolve
                    self.pursue(
                        f"Contradiction: '{statement[:80]}' vs '{existing.statement[:80]}' — which is correct?",
                        context="Contradictory beliefs detected",
                        max_depth=2,
                    )

            elif not new_belief.stamp.overlaps(existing.stamp):
                # Corroborating evidence — revise to combine
                revised = new_belief.revise_with(existing)
                if revised:
                    new_belief = revised

        # Store the belief
        belief_key = f"assert:{statement[:60]}"
        self.beliefs[belief_key] = new_belief

        # If confidence is too low, investigate to strengthen
        if new_belief.truth.confidence < 0.3:
            self.pursue(
                f"What evidence supports: '{statement}'?",
                context="Low confidence belief — seeking corroboration",
                max_depth=2,
                confidence_threshold=0.4,
            )

        self._save_beliefs()
        return new_belief

    def _beliefs_contradict(self, a: Belief, b: Belief) -> bool:
        """Simple contradiction check between two beliefs."""
        a_words = set(a.statement.lower().split())
        b_words = set(b.statement.lower().split())

        # Check for negation patterns
        negation_words = {"not", "no", "never", "doesn't", "don't", "won't", "cannot", "isn't", "aren't"}

        a_negated = bool(a_words & negation_words)
        b_negated = bool(b_words & negation_words)

        # One negated and one not, with substantial word overlap = contradiction
        content_overlap = len((a_words - negation_words) & (b_words - negation_words))
        if a_negated != b_negated and content_overlap >= 3:
            return True

        # Opposite truth values with high confidence on both
        if (a.truth.is_positive() != b.truth.is_positive()
                and a.truth.confidence > 0.3 and b.truth.confidence > 0.3):
            return True

        return False

    # ═══════════════════════════════════════════════════════
    # BEHAVIOR 3: LATERAL EXPLORATION — look around the subject
    # ═══════════════════════════════════════════════════════

    def explore_laterally(self, concept: str) -> dict:
        """
        Look AROUND a concept, not just AT it.

        Returns:
        - adjacent_concepts: what's structurally nearby
        - shared_patterns: what abstractions include this concept
        - contradictions: beliefs that conflict about this concept
        - related_domains: where similar structures appear
        - causal_neighborhood: what this causes and what causes it
        """
        result = {
            "concept": concept,
            "adjacent_concepts": [],
            "shared_patterns": [],
            "contradictions": [],
            "related_domains": [],
            "causal_neighborhood": {"causes": [], "effects": []},
            "related_beliefs": [],
        }

        concept_lower = concept.lower()
        concept_words = set(concept_lower.split())

        # 1. Transfer cortex lateral neighbors (patterns, domains, causal)
        cortex_neighbors = self.transfer_cortex.lateral_neighbors(concept)
        result["shared_patterns"] = cortex_neighbors.get("shared_patterns", [])
        result["related_domains"] = cortex_neighbors.get("related_domains", [])
        result["adjacent_concepts"] = cortex_neighbors.get("adjacent_roles", [])
        result["causal_neighborhood"] = cortex_neighbors.get("causal_neighborhood", {
            "causes": [], "effects": [],
        })

        # 2. Find beliefs related to this concept
        for key, belief in self.beliefs.items():
            belief_words = set(belief.statement.lower().split())
            overlap = len(concept_words & belief_words)
            if overlap >= 1 and belief.truth.confidence > 0.1:
                result["related_beliefs"].append({
                    "statement": belief.statement,
                    "truth": belief.truth.to_dict(),
                    "source": belief.source,
                })

        # 3. Check for contradictions among related beliefs
        related = result["related_beliefs"]
        for i, a in enumerate(related):
            for b in related[i + 1:]:
                a_belief = Belief(
                    statement=a["statement"],
                    truth=TruthValue.from_dict(a["truth"]),
                    stamp=EvidenceStamp.new(0),
                    source=a.get("source", ""),
                )
                b_belief = Belief(
                    statement=b["statement"],
                    truth=TruthValue.from_dict(b["truth"]),
                    stamp=EvidenceStamp.new(0),
                    source=b.get("source", ""),
                )
                if self._beliefs_contradict(a_belief, b_belief):
                    result["contradictions"].append({
                        "belief_a": a["statement"],
                        "belief_b": b["statement"],
                    })

        # 4. Use abstraction engine to find structurally similar things
        probe = {"description": concept, "actions": [], "outcome": "unknown"}
        matches = self.abstraction.match(probe)
        for matched_abs, score in matches[:5]:
            if score > 0.3:
                # The variables in this abstraction are "things like this concept"
                variables = matched_abs.get("variables", [])
                for var in variables:
                    if var not in result["adjacent_concepts"]:
                        result["adjacent_concepts"].append(var)

        return result

    # ═══════════════════════════════════════════════════════
    # BEHAVIOR 4: SPONTANEOUS PREDICTION — hypothesize outcomes
    # ═══════════════════════════════════════════════════════

    def generate_predictions(self) -> list[TransferPrediction]:
        """
        Spontaneous hypothesis generation.

        Scans recent episodes, applies known patterns to new domains,
        generates predictions with truth values, registers them for
        later confirmation.

        This is the "I bet if I do A, then B will happen" behavior.
        Not reactive. Proactive.
        """
        # Load recent episodes
        episodes = self.storage.load_json("episodes", default=[])
        if not episodes:
            return []

        # Scan recent episodes for cross-domain transfer opportunities
        recent = episodes[-10:]  # Last 10 episodes
        all_predictions = []

        for episode in recent:
            predictions = self.transfer_cortex.scan_for_transfers(episode)
            all_predictions.extend(predictions)

        # Also generate causal predictions from the causal graph
        experiment = self.causal.what_should_i_test()
        if experiment:
            # Register as a curiosity target
            self.attention.add_concept(
                f"test:{experiment['cause']}->{experiment['effect']}",
                experiment,
                priority=0.6,
                relevant_drives=[Drive.PREDICTION, Drive.CURIOSITY],
            )
            self.attention.drives.boost(Drive.PREDICTION, 0.2)

        if all_predictions:
            self.storage.event_log.append(
                "nexus.predictions.generated",
                {"count": len(all_predictions)},
            )

        return all_predictions

    # ═══════════════════════════════════════════════════════
    # BEHAVIOR 5: UNIFIED LEARNING — every experience flows through all systems
    # ═══════════════════════════════════════════════════════

    def learn_experience(self, episode: dict) -> dict:
        """
        The unified learning pipeline. Every experience flows through:

        1. Causal model update (existing) — learn cause-effect relationships
        2. Abstraction attempt (existing) — form/strengthen structural patterns
        3. Transfer cortex registration (new) — register patterns for cross-domain use
        4. Transfer prediction scan (new) — does this match a known pattern in a new domain?
        5. Prediction confirmation (new) — did any past predictions come true?
        6. Awareness generation (new) — surface contextual warnings
        7. Investigation trigger (new) — if something surprising, dig deeper
        8. Belief store update (new) — register/revise beliefs with NARS truth values
        9. Attention update (new) — boost/satisfy drives based on what happened
        """
        result = {
            "episode_id": episode.get("id", "unknown"),
            "causal_edges": 0,
            "abstraction": None,
            "transfer_predictions": [],
            "prediction_confirmations": [],
            "awareness_memos": [],
            "investigations_opened": 0,
            "beliefs_updated": 0,
        }

        # 1. Causal model update
        edges = self.causal.learn_from_episode(episode)
        result["causal_edges"] = len(edges)

        # 2. Abstraction attempt
        episodes = self.storage.load_json("episodes", default=[])
        if len(episodes) >= 2:
            recent = episodes[-5:]
            abstraction = self.abstraction.abstract(recent)
            if abstraction:
                result["abstraction"] = {
                    "id": abstraction.id,
                    "name": abstraction.name,
                    "confidence": abstraction.confidence,
                }

                # 3. Register with transfer cortex
                domain = self.transfer_cortex._extract_domain(episode)
                self.transfer_cortex.register_abstraction(abstraction, domain)

        # 4. Scan for cross-domain transfers
        predictions = self.transfer_cortex.scan_for_transfers(episode)
        result["transfer_predictions"] = [p.to_dict() for p in predictions]

        # 5. Check if any past predictions are confirmed/refuted
        confirmations = self.transfer_cortex.check_predictions_against_episode(episode)
        result["prediction_confirmations"] = confirmations

        for conf in confirmations:
            if conf.get("status") == "confirmed":
                self.attention.on_prediction_correct()
            elif conf.get("status") == "refuted":
                # Refutation generates investigation question
                inv_q = conf.get("investigation_question")
                if inv_q:
                    self.pursue(inv_q, context="Transfer prediction refuted", max_depth=2)
                    result["investigations_opened"] += 1

        # 6. Awareness generation
        memos = self.transfer_cortex.get_awareness(episode)
        result["awareness_memos"] = memos

        # 7. Investigation trigger — if outcome was surprising
        success = episode.get("success", True)
        if not success:
            description = episode.get("description", "unknown task")
            self.pursue(
                f"Why did '{description}' fail?",
                context=f"Episode failed: {episode.get('outcome', 'unknown')}",
                max_depth=2,
            )
            result["investigations_opened"] += 1

        # 8. Belief store update
        outcome = episode.get("outcome", "unknown")
        description = episode.get("description", "")
        if description and outcome != "unknown":
            belief = self.assert_belief(
                statement=f"Task '{description[:80]}' resulted in '{outcome}'",
                source="episode",
                frequency=1.0 if success else 0.0,
                confidence=w2c(1),
            )
            result["beliefs_updated"] += 1

        # 9. Attention update
        if success:
            self.attention.drives.satisfy(Drive.COMPETENCE, 0.2)
        else:
            self.attention.drives.boost(Drive.CURIOSITY, 0.3)
            self.attention.on_gap_found()

        self._save_beliefs()
        return result

    # ═══════════════════════════════════════════════════════
    # AWARENESS INTERFACE — "heads up"
    # ═══════════════════════════════════════════════════════

    def get_awareness(self, situation: dict) -> list[AwarenessMemo]:
        """
        Given a current situation, return contextual awareness memos.

        Not alarms. Not halt signals. Just: "this rhymes with
        something I've seen before."
        """
        raw_memos = self.transfer_cortex.get_awareness(situation)

        memos = []
        for raw in raw_memos:
            memo = AwarenessMemo(
                pattern_name=raw["pattern_name"],
                relevance=raw["relevance"],
                truth_value=TruthValue.from_dict(raw["confidence"]),
                message=raw["message"],
                suggested_caution=raw["suggested_caution"],
                domains_seen_in=raw["domains_seen_in"],
                is_new_domain=raw.get("is_new_domain", False),
            )
            memos.append(memo)

        return memos

    # ═══════════════════════════════════════════════════════
    # THINKING INTERFACE — pass-through to cognition loop
    # ═══════════════════════════════════════════════════════

    def think(self, problem: dict) -> Any:
        """
        Think about a problem using the full cognition loop.

        The Nexus's callbacks are wired in, so the cognition loop
        automatically gets deep investigation, awareness, and
        cross-domain transfer.
        """
        return self.cognition_loop.think(problem)

    # ═══════════════════════════════════════════════════════
    # CALLBACK IMPLEMENTATIONS
    # ═══════════════════════════════════════════════════════

    def _on_phase(self, phase: CognitionPhase, message: str) -> None:
        """
        React to cognition loop phase transitions.

        - After LEARN: register abstractions, generate predictions
        - After PERCEIVE: provide awareness for the current problem
        - During REMEMBER: boost prediction drive
        """
        if phase == CognitionPhase.LEARN:
            # After learning, generate spontaneous predictions
            self.generate_predictions()
            self.attention.drives.satisfy(Drive.CURIOSITY, 0.1)

        elif phase == CognitionPhase.PERCEIVE:
            # Boost coherence — we're starting a new problem
            self.attention.drives.boost(Drive.COMPETENCE, 0.3)

        elif phase == CognitionPhase.REMEMBER:
            # We're searching memory — boost prediction drive
            self.attention.drives.boost(Drive.PREDICTION, 0.2)

        elif phase == CognitionPhase.INQUIRE:
            # Knowledge gap phase — boost curiosity
            self.attention.on_gap_found()

    def _on_inquiry(self, gap: KnowledgeGap) -> Any:
        """
        Replace single-shot gap handling with deep investigation.

        Instead of just noting the gap and moving on, the Nexus opens
        a full investigation thread that follows the knowledge chain.
        """
        thread = self.pursue(
            question=gap.question,
            context=gap.why_needed,
            max_depth=2,
            confidence_threshold=0.4,
        )

        if thread.best_finding:
            best = thread.best_finding
            # Resolve the original inquiry gap
            self.inquiry.resolve(gap.id, best.answer, f"investigation:{thread.id}")
            return best.answer

        return None

    # ═══════════════════════════════════════════════════════
    # TICK — called each runtime cycle
    # ═══════════════════════════════════════════════════════

    def tick(self) -> None:
        """
        Called each runtime cycle.

        - Decays attention priorities and drives
        - Projects belief confidence over time (old beliefs fade)
        - Decays stale prediction confidence
        """
        # Attention decay
        self.attention.tick()

        # Belief temporal projection — old beliefs lose confidence
        for key, belief in self.beliefs.items():
            # Mild decay — beliefs don't vanish quickly
            belief.truth = temporal_projection(belief.truth, time_distance=0.1, decay_rate=0.05)

        # Prediction decay
        self.transfer_cortex.decay_predictions(time_distance=0.1, decay_rate=0.1)

    # ═══════════════════════════════════════════════════════
    # STATS
    # ═══════════════════════════════════════════════════════

    def stats(self) -> dict:
        return {
            "beliefs": len(self.beliefs),
            "attention": self.attention.stats(),
            "transfer_cortex": self.transfer_cortex.stats(),
            "investigation": self.investigation.stats(),
            "causal_nodes": len(self.causal.graph.nodes),
            "causal_edges": len(self.causal.graph.edges),
        }
