"""
Investigation Engine — deep recursive inquiry.

When the system hits a knowledge gap, it doesn't just note it and move on.
It follows the thread. Each answer raises new questions. It cross-checks
against the causal model, the abstraction engine, and existing beliefs.
It refuses to accept low-confidence answers.

The four follow-up strategies:
1. CAUSAL DRILL: "What causes THAT?" — follow the causal chain deeper
2. BOUNDARY PROBE: "When does this NOT apply?" — find the limits
3. EVIDENCE STRENGTHEN: "What else supports this?" — build confidence
4. LATERAL CONNECT: "What is related to this?" — explore the neighborhood

Each finding becomes a Belief (from truth.py) with proper truth values
and evidence stamps, so nothing gets double-counted.

Max 3 follow-ups per answer. Capped by max_depth.

No LLM. Uses existing CuriosityDriver for sensing, CausalModel for
cross-checking, AbstractionEngine for pattern matching.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from klomboagi.reasoning.abstraction import AbstractionEngine
from klomboagi.reasoning.causal import CausalModel
from klomboagi.reasoning.curiosity import CuriosityDriver, KnowledgeGap as CuriosityGap, GapPriority
from klomboagi.reasoning.inquiry import InquiryEngine
from klomboagi.reasoning.truth import (
    TruthValue, Belief, EvidenceStamp,
    revision, w2c,
)
from klomboagi.storage.manager import StorageManager
from klomboagi.storage.runtime_state import JsonStateStore
from klomboagi.utils.time import utc_now
from klomboagi.utils.ids import new_id


# ── Data Structures ──

@dataclass
class InvestigationQuestion:
    """A question in the investigation stack."""
    question: str
    context: str
    depth: int                              # How deep in the chain
    parent_question: str | None = None      # What question spawned this
    strategy: str = "initial"               # "initial", "causal_drill", "boundary_probe",
                                            # "evidence_strengthen", "lateral_connect"

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "context": self.context,
            "depth": self.depth,
            "parent_question": self.parent_question,
            "strategy": self.strategy,
        }


@dataclass
class InvestigationFinding:
    """An answer found during investigation."""
    question: str
    answer: str
    belief: Belief                          # NARS belief with truth value + evidence stamp
    source: str                             # "causal_model", "abstraction_match", "curiosity_search", "cross_check"
    follow_ups: list[str] = field(default_factory=list)  # Sub-questions this answer raised
    depth: int = 0

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "belief": self.belief.to_dict(),
            "source": self.source,
            "follow_ups": self.follow_ups,
            "depth": self.depth,
        }


@dataclass
class InvestigationThread:
    """A full investigation — from root question to all findings."""
    id: str
    root_question: str
    context: str
    question_stack: list[InvestigationQuestion] = field(default_factory=list)
    findings: list[InvestigationFinding] = field(default_factory=list)
    max_depth: int = 3
    confidence_threshold: float = 0.5       # Won't accept below this
    status: str = "active"                  # "active", "satisfied", "exhausted"
    created_at: str = ""
    completed_at: str | None = None

    @property
    def best_finding(self) -> InvestigationFinding | None:
        """The highest-confidence finding."""
        if not self.findings:
            return None
        return max(self.findings, key=lambda f: f.belief.truth.expectation)

    @property
    def is_satisfied(self) -> bool:
        """Do we have a finding that meets our confidence threshold?"""
        best = self.best_finding
        return best is not None and best.belief.truth.confidence >= self.confidence_threshold

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "root_question": self.root_question,
            "context": self.context,
            "findings": [f.to_dict() for f in self.findings],
            "pending_questions": len(self.question_stack),
            "max_depth": self.max_depth,
            "confidence_threshold": self.confidence_threshold,
            "status": self.status,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
        }


# ── Investigation Engine ──

class InvestigationEngine:
    """
    Deep recursive inquiry engine.

    When a knowledge gap is found, instead of noting it and moving on,
    the engine opens an investigation thread that:
    1. Checks existing knowledge (causal model, abstractions, beliefs)
    2. If confidence is too low, uses CuriosityDriver to seek new info
    3. Cross-checks answers from multiple sources via NARS revision()
    4. Generates follow-up questions (max 3 per answer)
    5. Continues until satisfied or max_depth reached
    """

    def __init__(
        self,
        storage: StorageManager,
        causal_model: CausalModel,
        curiosity_driver: CuriosityDriver,
        inquiry_engine: InquiryEngine,
        abstraction_engine: AbstractionEngine,
    ) -> None:
        self.storage = storage
        self.causal = causal_model
        self.curiosity = curiosity_driver
        self.inquiry = inquiry_engine
        self.abstraction = abstraction_engine

        # Own storage
        self._investigations_store = JsonStateStore(
            storage.paths.long_term_root / "memory" / "connective" / "investigations.json",
            storage.manifest_store,
            "connective.investigations",
        )

        # Evidence counter for unique stamps
        self._evidence_counter = 0

        # External belief store reference (set by Nexus)
        self.belief_store: dict[str, Belief] = {}

    def _next_evidence_id(self) -> int:
        self._evidence_counter += 1
        return self._evidence_counter

    # ── Main entry points ──

    def open_investigation(
        self,
        question: str,
        context: str = "",
        max_depth: int = 3,
        confidence_threshold: float = 0.5,
    ) -> InvestigationThread:
        """Start a new investigation thread."""
        thread = InvestigationThread(
            id=new_id("inv"),
            root_question=question,
            context=context,
            max_depth=max_depth,
            confidence_threshold=confidence_threshold,
            created_at=utc_now(),
        )

        # Push the root question
        thread.question_stack.append(InvestigationQuestion(
            question=question,
            context=context,
            depth=0,
            strategy="initial",
        ))

        self.storage.event_log.append(
            "nexus.investigation.opened",
            {"thread_id": thread.id, "question": question},
        )

        return thread

    def pursue(
        self,
        question: str,
        context: str = "",
        max_depth: int = 3,
        confidence_threshold: float = 0.5,
    ) -> InvestigationThread:
        """
        Full investigation: open a thread and step until satisfied or exhausted.
        This is the main entry point for deep inquiry.
        """
        thread = self.open_investigation(question, context, max_depth, confidence_threshold)

        # Step until done
        max_steps = max_depth * 4 + 5  # Safety cap
        steps = 0
        while thread.status == "active" and steps < max_steps:
            finding = self.step(thread)
            if finding is None:
                break
            steps += 1

        # Finalize
        if thread.is_satisfied:
            thread.status = "satisfied"
        elif not thread.question_stack:
            thread.status = "exhausted"

        thread.completed_at = utc_now()

        self.storage.event_log.append(
            "nexus.investigation.completed",
            {
                "thread_id": thread.id,
                "status": thread.status,
                "findings_count": len(thread.findings),
                "best_confidence": (
                    thread.best_finding.belief.truth.confidence
                    if thread.best_finding else 0.0
                ),
            },
        )

        self._save_thread(thread)
        return thread

    def step(self, thread: InvestigationThread) -> InvestigationFinding | None:
        """
        One step of investigation:
        1. Pop a question from the stack
        2. Check existing knowledge
        3. If confidence too low, seek new knowledge
        4. Cross-check and create a Belief
        5. Generate follow-ups
        6. Push follow-ups onto the stack
        """
        if not thread.question_stack:
            thread.status = "exhausted"
            return None

        if thread.is_satisfied:
            thread.status = "satisfied"
            return None

        # Pop the highest-priority question (end of stack = highest priority)
        question = thread.question_stack.pop()

        # 1. Check existing knowledge
        answer, existing_confidence = self._check_existing_knowledge(
            question.question, question.context
        )

        source = "existing_knowledge"

        # 2. If confidence is too low, seek new knowledge
        if existing_confidence < thread.confidence_threshold:
            new_answer, new_confidence = self._seek_new_knowledge(
                question.question, question.context
            )
            if new_answer and new_confidence > 0:
                if answer and existing_confidence > 0:
                    # Cross-check: combine existing + new via revision
                    combined_tv = self._cross_check(
                        existing_answer=answer,
                        existing_confidence=existing_confidence,
                        new_answer=new_answer,
                        new_confidence=new_confidence,
                    )
                    answer = f"{answer} [cross-checked: {new_answer}]"
                    source = "cross_check"
                else:
                    answer = new_answer
                    combined_tv = TruthValue(
                        frequency=new_confidence,
                        confidence=w2c(1),
                    )
                    source = "curiosity_search"
            elif answer:
                combined_tv = TruthValue(
                    frequency=existing_confidence,
                    confidence=w2c(max(1, int(existing_confidence * 5))),
                )
            else:
                # Couldn't find anything
                answer = f"Unable to determine: {question.question}"
                combined_tv = TruthValue(frequency=0.5, confidence=0.0)
                source = "unknown"
        else:
            combined_tv = TruthValue(
                frequency=existing_confidence,
                confidence=w2c(max(1, int(existing_confidence * 5))),
            )

        # 3. Create a Belief from this finding
        stamp = EvidenceStamp.new(self._next_evidence_id())
        belief = Belief(
            statement=f"RE: {question.question} → {answer[:200]}",
            truth=combined_tv,
            stamp=stamp,
            subject=question.question,
            predicate="answers",
            source=source,
        )

        # 4. Generate follow-up questions if we're not at max depth
        follow_ups = []
        if question.depth < thread.max_depth and answer and combined_tv.confidence > 0:
            follow_ups = self._generate_follow_ups(
                question.question, answer, question.context, question.depth
            )
            # Push follow-ups onto the stack
            thread.question_stack.extend(follow_ups)

        # 5. Create finding
        finding = InvestigationFinding(
            question=question.question,
            answer=answer or "No answer found",
            belief=belief,
            source=source,
            follow_ups=[fq.question for fq in follow_ups],
            depth=question.depth,
        )
        thread.findings.append(finding)

        # Register belief
        belief_key = f"inv:{thread.id}:{question.question[:50]}"
        self.belief_store[belief_key] = belief

        return finding

    # ── Knowledge checking ──

    def _check_existing_knowledge(self, question: str, context: str) -> tuple[str | None, float]:
        """
        Check what we already know.
        Sources: causal model, abstraction matches, existing beliefs.
        Returns (answer, confidence) or (None, 0).
        """
        best_answer = None
        best_confidence = 0.0

        # 1. Check causal model — can it explain or predict?
        explanations = self.causal.explain_outcome(question)
        if explanations:
            top_cause, strength, confidence = explanations[0]
            if confidence > best_confidence:
                best_answer = f"Causal analysis: '{top_cause}' causes this (strength={strength:.2f})"
                best_confidence = confidence

        predictions = self.causal.predict_outcome(question)
        if predictions:
            top_effect, prob, confidence = predictions[0]
            if confidence > best_confidence:
                best_answer = f"Causal prediction: leads to '{top_effect}' (p={prob:.0%})"
                best_confidence = confidence

        # 2. Check abstraction matches — does this fit a known pattern?
        episode_proxy = {"description": question, "actions": [], "outcome": context}
        matches = self.abstraction.match(episode_proxy)
        if matches:
            best_abs, fit_score = matches[0]
            if fit_score > best_confidence:
                best_answer = (
                    f"Matches abstraction '{best_abs.get('name', 'unknown')}' "
                    f"(fit={fit_score:.0%}, instances={best_abs.get('instance_count', 0)})"
                )
                best_confidence = fit_score

        # 3. Check existing beliefs
        question_words = set(question.lower().split())
        for key, belief in self.belief_store.items():
            belief_words = set(belief.statement.lower().split())
            overlap = len(question_words & belief_words)
            if overlap >= 2 and belief.truth.expectation > best_confidence:
                best_answer = f"Known belief: {belief.statement} ({belief.truth})"
                best_confidence = belief.truth.expectation

        return best_answer, best_confidence

    def _seek_new_knowledge(self, question: str, context: str) -> tuple[str | None, float]:
        """
        Use CuriosityDriver to seek new information about a question.
        Returns (answer, confidence) or (None, 0).
        """
        # Notice the gap and investigate it
        gap = self.curiosity.notice_gap(
            concept=question,
            context=context,
            priority=GapPriority.HIGH,
        )

        event = self.curiosity.investigate(gap)
        if event and event.learned:
            # Estimate confidence based on the quality of the result
            result_len = len(event.result.strip()) if event.result else 0
            confidence = min(0.8, 0.3 + (result_len / 500))  # Longer, more detailed = higher confidence
            return event.explanation, confidence

        return None, 0.0

    def _cross_check(
        self,
        existing_answer: str,
        existing_confidence: float,
        new_answer: str,
        new_confidence: float,
    ) -> TruthValue:
        """
        Cross-check two answers via NARS revision().
        Combines independent evidence without double-counting.
        """
        tv_existing = TruthValue(
            frequency=existing_confidence,
            confidence=w2c(max(1, int(existing_confidence * 3))),
        )
        tv_new = TruthValue(
            frequency=new_confidence,
            confidence=w2c(1),
        )
        return revision(tv_existing, tv_new)

    # ── Follow-up generation ──

    def _generate_follow_ups(
        self,
        question: str,
        answer: str,
        context: str,
        current_depth: int,
    ) -> list[InvestigationQuestion]:
        """
        Generate follow-up questions using four strategies.
        Max 3 follow-ups per answer.
        """
        follow_ups = []
        next_depth = current_depth + 1

        # Strategy 1: CAUSAL DRILL — "What causes THAT?"
        # Follow the causal chain one step deeper
        answer_words = [w.strip(".,!?'\"") for w in answer.split() if len(w) > 3]
        causal_targets = answer_words[:3]  # Use key words from the answer
        if causal_targets:
            target = " ".join(causal_targets)
            follow_ups.append(InvestigationQuestion(
                question=f"What causes or leads to '{target}'?",
                context=f"Following causal chain from: {question}",
                depth=next_depth,
                parent_question=question,
                strategy="causal_drill",
            ))

        # Strategy 2: BOUNDARY PROBE — "When does this NOT apply?"
        if len(follow_ups) < 3:
            follow_ups.append(InvestigationQuestion(
                question=f"Under what conditions does this NOT hold: '{answer[:100]}'?",
                context=f"Probing boundaries of: {question}",
                depth=next_depth,
                parent_question=question,
                strategy="boundary_probe",
            ))

        # Strategy 3: LATERAL CONNECT — "What else is related?"
        if len(follow_ups) < 3:
            follow_ups.append(InvestigationQuestion(
                question=f"What is structurally similar or related to '{question}'?",
                context=f"Exploring lateral connections from: {answer[:100]}",
                depth=next_depth,
                parent_question=question,
                strategy="lateral_connect",
            ))

        # We skip EVIDENCE STRENGTHEN at shallow depths to save investigation budget
        # It's less informative than the other three strategies

        return follow_ups[:3]  # Hard cap at 3

    # ── Persistence ──

    def _save_thread(self, thread: InvestigationThread) -> None:
        """Save a completed investigation to long-term storage."""
        all_investigations = self._investigations_store.load(default=[])
        all_investigations.append(thread.to_dict())
        # Keep last 200 investigations
        if len(all_investigations) > 200:
            all_investigations = all_investigations[-200:]
        self._investigations_store.save(all_investigations)

    def load_past_investigations(self, limit: int = 50) -> list[dict]:
        """Load recent investigations for context."""
        all_investigations = self._investigations_store.load(default=[])
        return all_investigations[-limit:]

    # ── Stats ──

    def stats(self) -> dict:
        past = self._investigations_store.load(default=[])
        satisfied = sum(1 for inv in past if inv.get("status") == "satisfied")
        exhausted = sum(1 for inv in past if inv.get("status") == "exhausted")
        total_findings = sum(len(inv.get("findings", [])) for inv in past)

        return {
            "total_investigations": len(past),
            "satisfied": satisfied,
            "exhausted": exhausted,
            "satisfaction_rate": satisfied / max(satisfied + exhausted, 1),
            "total_findings": total_findings,
            "avg_findings_per_investigation": total_findings / max(len(past), 1),
        }
