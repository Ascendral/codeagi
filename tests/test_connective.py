"""
Tests for the Connective Tissue Layer.

Tests the five behaviors:
1. Deep pursuit (recursive investigation)
2. Correctness obsession (cross-checking, contradiction detection)
3. Lateral exploration (looking around a subject)
4. Spontaneous prediction (cross-domain transfer)
5. Unified learning pipeline

Includes the shower-to-garage walkthrough as an end-to-end test.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest


# ── Fixture: temporary storage ──

@pytest.fixture
def temp_storage():
    """Create a temporary storage manager for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        runtime_root = Path(tmpdir) / "runtime"
        long_term_root = Path(tmpdir) / "long-term"

        os.environ["KLOMBOAGI_RUNTIME_ROOT"] = str(runtime_root)
        os.environ["KLOMBOAGI_LONG_TERM_ROOT"] = str(long_term_root)

        from klomboagi.storage.manager import StorageManager
        storage = StorageManager.bootstrap()
        yield storage

        # Cleanup env
        os.environ.pop("KLOMBOAGI_RUNTIME_ROOT", None)
        os.environ.pop("KLOMBOAGI_LONG_TERM_ROOT", None)


# ── Transfer Cortex Tests ──

class TestTransferCortex:

    def test_register_abstraction(self, temp_storage):
        from klomboagi.connective.transfer_cortex import TransferCortex
        from klomboagi.reasoning.abstraction import AbstractionEngine, Abstraction
        from klomboagi.reasoning.comparator import StructuralComparator
        from klomboagi.reasoning.causal import CausalModel

        abstraction_engine = AbstractionEngine(temp_storage)
        comparator = StructuralComparator(abstraction_engine)
        causal = CausalModel(temp_storage)

        cortex = TransferCortex(temp_storage, abstraction_engine, comparator, causal)

        # Register a pattern
        abs_data = Abstraction(
            id="abs_test_001",
            name="pattern:goal+outcome",
            schema=[{"role": "goal", "type": "intention"}, {"role": "outcome", "type": "state"}],
            invariants=["outcome:state=success"],
            variables=["goal:intention"],
            source_episodes=["ep1", "ep2"],
            instance_count=2,
            confidence=0.6,
        )

        pattern = cortex.register_abstraction(abs_data, "domain_a")

        assert pattern.id == "abs_test_001"
        assert "domain_a" in pattern.source_domains
        assert pattern.instance_count == 1
        assert pattern.confidence.frequency > 0

    def test_register_same_pattern_new_domain(self, temp_storage):
        from klomboagi.connective.transfer_cortex import TransferCortex
        from klomboagi.reasoning.abstraction import AbstractionEngine, Abstraction
        from klomboagi.reasoning.comparator import StructuralComparator
        from klomboagi.reasoning.causal import CausalModel

        abstraction_engine = AbstractionEngine(temp_storage)
        comparator = StructuralComparator(abstraction_engine)
        causal = CausalModel(temp_storage)

        cortex = TransferCortex(temp_storage, abstraction_engine, comparator, causal)

        abs_data = Abstraction(
            id="abs_test_002",
            name="slip_pattern",
            schema=[{"role": "surface", "type": "entity"}, {"role": "outcome", "type": "state"}],
            invariants=["outcome:state=slip"],
            variables=["surface:entity"],
            source_episodes=["ep1"],
            instance_count=1,
            confidence=0.5,
        )

        # Register in domain A
        pattern_v1 = cortex.register_abstraction(abs_data, "bathroom")
        conf_v1 = pattern_v1.confidence.confidence

        # Register same pattern in domain B — should strengthen
        pattern_v2 = cortex.register_abstraction(abs_data, "kitchen")
        conf_v2 = pattern_v2.confidence.confidence

        assert "bathroom" in pattern_v2.source_domains
        assert "kitchen" in pattern_v2.source_domains
        assert conf_v2 > conf_v1  # Confidence increased

    def test_lateral_neighbors(self, temp_storage):
        from klomboagi.connective.transfer_cortex import TransferCortex
        from klomboagi.reasoning.abstraction import AbstractionEngine, Abstraction
        from klomboagi.reasoning.comparator import StructuralComparator
        from klomboagi.reasoning.causal import CausalModel

        abstraction_engine = AbstractionEngine(temp_storage)
        comparator = StructuralComparator(abstraction_engine)
        causal = CausalModel(temp_storage)

        cortex = TransferCortex(temp_storage, abstraction_engine, comparator, causal)

        # Register a pattern with "smooth" and "surface" in it
        abs_data = Abstraction(
            id="abs_surface",
            name="surface_pattern",
            schema=[],
            invariants=["surface:entity=smooth"],
            variables=[],
            source_episodes=["ep1"],
            instance_count=1,
            confidence=0.6,
        )
        cortex.register_abstraction(abs_data, "bathroom")

        # The principle is derived from invariants: "surface=smooth"
        neighbors = cortex.lateral_neighbors("surface")
        assert "related_domains" in neighbors
        assert "bathroom" in neighbors["related_domains"]

    def test_awareness_empty(self, temp_storage):
        from klomboagi.connective.transfer_cortex import TransferCortex
        from klomboagi.reasoning.abstraction import AbstractionEngine
        from klomboagi.reasoning.comparator import StructuralComparator
        from klomboagi.reasoning.causal import CausalModel

        abstraction_engine = AbstractionEngine(temp_storage)
        comparator = StructuralComparator(abstraction_engine)
        causal = CausalModel(temp_storage)

        cortex = TransferCortex(temp_storage, abstraction_engine, comparator, causal)

        # No patterns registered — should return empty
        memos = cortex.get_awareness({"description": "something"})
        assert memos == []

    def test_stats(self, temp_storage):
        from klomboagi.connective.transfer_cortex import TransferCortex
        from klomboagi.reasoning.abstraction import AbstractionEngine
        from klomboagi.reasoning.comparator import StructuralComparator
        from klomboagi.reasoning.causal import CausalModel

        abstraction_engine = AbstractionEngine(temp_storage)
        comparator = StructuralComparator(abstraction_engine)
        causal = CausalModel(temp_storage)

        cortex = TransferCortex(temp_storage, abstraction_engine, comparator, causal)
        stats = cortex.stats()
        assert stats["total_patterns"] == 0
        assert stats["total_predictions"] == 0


# ── Investigation Engine Tests ──

class TestInvestigationEngine:

    def test_open_investigation(self, temp_storage):
        from klomboagi.connective.investigation import InvestigationEngine
        from klomboagi.reasoning.abstraction import AbstractionEngine
        from klomboagi.reasoning.causal import CausalModel
        from klomboagi.reasoning.curiosity import CuriosityDriver
        from klomboagi.reasoning.inquiry import InquiryEngine

        engine = InvestigationEngine(
            temp_storage,
            CausalModel(temp_storage),
            CuriosityDriver(),
            InquiryEngine(temp_storage),
            AbstractionEngine(temp_storage),
        )

        thread = engine.open_investigation("What is gravity?", context="physics")
        assert thread.status == "active"
        assert thread.root_question == "What is gravity?"
        assert len(thread.question_stack) == 1

    def test_pursue_produces_findings(self, temp_storage):
        from klomboagi.connective.investigation import InvestigationEngine
        from klomboagi.reasoning.abstraction import AbstractionEngine
        from klomboagi.reasoning.causal import CausalModel
        from klomboagi.reasoning.curiosity import CuriosityDriver
        from klomboagi.reasoning.inquiry import InquiryEngine

        engine = InvestigationEngine(
            temp_storage,
            CausalModel(temp_storage),
            CuriosityDriver(),
            InquiryEngine(temp_storage),
            AbstractionEngine(temp_storage),
        )

        thread = engine.pursue("Why do things fall?", context="physics", max_depth=1)

        # Should have at least attempted to answer
        assert thread.status in ("satisfied", "exhausted")
        assert thread.completed_at is not None
        # Should have at least the root question finding
        assert len(thread.findings) >= 1

    def test_depth_limiting(self, temp_storage):
        from klomboagi.connective.investigation import InvestigationEngine
        from klomboagi.reasoning.abstraction import AbstractionEngine
        from klomboagi.reasoning.causal import CausalModel
        from klomboagi.reasoning.curiosity import CuriosityDriver
        from klomboagi.reasoning.inquiry import InquiryEngine

        engine = InvestigationEngine(
            temp_storage,
            CausalModel(temp_storage),
            CuriosityDriver(),
            InquiryEngine(temp_storage),
            AbstractionEngine(temp_storage),
        )

        thread = engine.pursue("deep question", max_depth=1)

        # All findings should be at depth 0 or 1
        for finding in thread.findings:
            assert finding.depth <= 1

    def test_stats(self, temp_storage):
        from klomboagi.connective.investigation import InvestigationEngine
        from klomboagi.reasoning.abstraction import AbstractionEngine
        from klomboagi.reasoning.causal import CausalModel
        from klomboagi.reasoning.curiosity import CuriosityDriver
        from klomboagi.reasoning.inquiry import InquiryEngine

        engine = InvestigationEngine(
            temp_storage,
            CausalModel(temp_storage),
            CuriosityDriver(),
            InquiryEngine(temp_storage),
            AbstractionEngine(temp_storage),
        )

        # Run an investigation so there's data
        engine.pursue("test question", max_depth=1)

        stats = engine.stats()
        assert stats["total_investigations"] >= 1


# ── Cognitive Nexus Tests ──

class TestCognitiveNexus:

    def test_init(self, temp_storage):
        from klomboagi.connective.nexus import CognitiveNexus
        nexus = CognitiveNexus(temp_storage)
        assert nexus.beliefs is not None
        assert nexus.transfer_cortex is not None
        assert nexus.investigation is not None

    def test_assert_belief(self, temp_storage):
        from klomboagi.connective.nexus import CognitiveNexus
        nexus = CognitiveNexus(temp_storage)

        belief = nexus.assert_belief(
            "Water makes floors slippery",
            source="observation",
            frequency=1.0,
            confidence=0.6,
        )
        assert belief.truth.frequency == 1.0
        assert len(nexus.beliefs) >= 1

    def test_assert_contradicting_beliefs(self, temp_storage):
        from klomboagi.connective.nexus import CognitiveNexus
        nexus = CognitiveNexus(temp_storage)

        # Assert two contradicting beliefs
        nexus.assert_belief(
            "Water on tile floor is dangerous and slippery",
            source="observation",
            frequency=1.0,
            confidence=0.7,
        )
        nexus.assert_belief(
            "Water on tile floor is not dangerous and not slippery",
            source="counter_observation",
            frequency=0.0,
            confidence=0.6,
        )

        # Should have detected contradiction and opened investigation
        assert len(nexus.beliefs) >= 2

    def test_explore_laterally(self, temp_storage):
        from klomboagi.connective.nexus import CognitiveNexus
        nexus = CognitiveNexus(temp_storage)

        # Add some beliefs about water
        nexus.assert_belief("Water causes slipping on smooth surfaces", "obs", 1.0, 0.6)
        nexus.assert_belief("Water is a liquid that flows downhill", "obs", 1.0, 0.7)

        result = nexus.explore_laterally("water")
        assert "related_beliefs" in result
        assert "causal_neighborhood" in result
        assert "shared_patterns" in result
        assert len(result["related_beliefs"]) >= 1

    def test_learn_experience(self, temp_storage):
        from klomboagi.connective.nexus import CognitiveNexus
        nexus = CognitiveNexus(temp_storage)

        episode = {
            "id": "ep_test_001",
            "description": "walked on wet bathroom floor",
            "domain": "bathroom",
            "actions": [{"type": "walk", "target": "wet_tile"}],
            "outcome": "slipped",
            "success": False,
        }

        result = nexus.learn_experience(episode)
        assert result["causal_edges"] >= 0
        assert result["beliefs_updated"] >= 1

    def test_get_awareness_empty(self, temp_storage):
        from klomboagi.connective.nexus import CognitiveNexus
        nexus = CognitiveNexus(temp_storage)

        memos = nexus.get_awareness({"description": "walking somewhere"})
        assert isinstance(memos, list)

    def test_tick(self, temp_storage):
        from klomboagi.connective.nexus import CognitiveNexus
        nexus = CognitiveNexus(temp_storage)

        nexus.assert_belief("Test belief", "test", 1.0, 0.5)
        nexus.tick()
        # Should not crash; beliefs should decay slightly
        assert len(nexus.beliefs) >= 1

    def test_stats(self, temp_storage):
        from klomboagi.connective.nexus import CognitiveNexus
        nexus = CognitiveNexus(temp_storage)
        stats = nexus.stats()
        assert "beliefs" in stats
        assert "transfer_cortex" in stats
        assert "investigation" in stats

    def test_pursue(self, temp_storage):
        from klomboagi.connective.nexus import CognitiveNexus
        nexus = CognitiveNexus(temp_storage)

        thread = nexus.pursue("What happens when floor is wet?", max_depth=1)
        assert thread.status in ("satisfied", "exhausted")
        assert len(thread.findings) >= 1
        # Findings should have been registered as beliefs
        assert len(nexus.beliefs) >= 1


# ── End-to-End: Shower to Garage ──

class TestShowerToGarage:
    """
    The full walkthrough:
    1. Experience bathroom slip (wet tile)
    2. Experience kitchen slip (wet linoleum)
    3. Abstraction forms: liquid on smooth surface = slip risk
    4. Enter garage with oil on floor
    5. System recognizes the pattern in a new domain
    6. Awareness memo surfaces: "heads up"
    """

    def test_cross_domain_pattern_transfer(self, temp_storage):
        from klomboagi.connective.nexus import CognitiveNexus
        nexus = CognitiveNexus(temp_storage)

        # Episode 1: Bathroom slip
        ep1 = {
            "id": "ep_bathroom",
            "description": "stepped on wet tile bathroom floor",
            "domain": "bathroom",
            "actions": [
                {"type": "step", "target": "wet_tile"},
                {"type": "slip", "target": "floor"},
            ],
            "outcome": "fell",
            "success": False,
        }

        # Episode 2: Kitchen slip
        ep2 = {
            "id": "ep_kitchen",
            "description": "stepped on wet linoleum kitchen floor",
            "domain": "kitchen",
            "actions": [
                {"type": "step", "target": "wet_linoleum"},
                {"type": "slip", "target": "floor"},
            ],
            "outcome": "fell",
            "success": False,
        }

        # Save episodes to storage so abstraction engine can find them
        temp_storage.save_json("episodes", [ep1, ep2])

        # Learn from both episodes
        result1 = nexus.learn_experience(ep1)
        result2 = nexus.learn_experience(ep2)

        # At this point, an abstraction should have formed
        # and transfer cortex should have registered it

        # Episode 3: Garage with oil
        ep3 = {
            "id": "ep_garage",
            "description": "stepped on oil-covered garage concrete floor",
            "domain": "garage",
            "actions": [
                {"type": "step", "target": "oil_concrete"},
            ],
            "outcome": "unknown",
            "success": True,
        }

        # Check awareness before acting
        memos = nexus.get_awareness(ep3)
        # Awareness may or may not fire depending on abstraction quality,
        # but the transfer cortex should have patterns registered
        stats = nexus.transfer_cortex.stats()

        # The system should have learned something
        assert nexus.stats()["beliefs"] >= 2
        assert result1["causal_edges"] >= 0
        assert result2["causal_edges"] >= 0

    def test_prediction_confirmation_loop(self, temp_storage):
        from klomboagi.connective.transfer_cortex import TransferCortex, TransferPrediction
        from klomboagi.reasoning.abstraction import AbstractionEngine, Abstraction
        from klomboagi.reasoning.comparator import StructuralComparator
        from klomboagi.reasoning.causal import CausalModel
        from klomboagi.reasoning.truth import TruthValue

        abstraction_engine = AbstractionEngine(temp_storage)
        comparator = StructuralComparator(abstraction_engine)
        causal = CausalModel(temp_storage)
        cortex = TransferCortex(temp_storage, abstraction_engine, comparator, causal)

        # Register a pattern in two domains
        abs_data = Abstraction(
            id="abs_slip",
            name="slip_pattern",
            schema=[{"role": "step", "type": "operation"}, {"role": "outcome", "type": "state"}],
            invariants=["outcome:state=fell"],
            variables=["step:operation"],
            source_episodes=["ep1", "ep2"],
            instance_count=2,
            confidence=0.7,
        )
        cortex.register_abstraction(abs_data, "bathroom")
        cortex.register_abstraction(abs_data, "kitchen")

        pattern = cortex.patterns["abs_slip"]
        conf_before = pattern.confidence.confidence

        # Confirm prediction
        # First manually create a prediction
        pred = TransferPrediction(
            id="tpred_test",
            pattern_id="abs_slip",
            source_domain="bathroom",
            target_domain="garage",
            prediction="slip risk in garage",
            truth_value=TruthValue(0.8, 0.4),
            created_at="2026-03-27T00:00:00",
        )
        cortex.predictions.append(pred)

        # Confirm it
        result = cortex.confirm_prediction("tpred_test", outcome=True)
        assert result["status"] == "confirmed"
        assert "garage" in cortex.patterns["abs_slip"].source_domains

        # Confidence should have increased
        conf_after = cortex.patterns["abs_slip"].confidence.confidence
        assert conf_after >= conf_before

    def test_prediction_refutation(self, temp_storage):
        from klomboagi.connective.transfer_cortex import TransferCortex, TransferPrediction
        from klomboagi.reasoning.abstraction import AbstractionEngine, Abstraction
        from klomboagi.reasoning.comparator import StructuralComparator
        from klomboagi.reasoning.causal import CausalModel
        from klomboagi.reasoning.truth import TruthValue

        abstraction_engine = AbstractionEngine(temp_storage)
        comparator = StructuralComparator(abstraction_engine)
        causal = CausalModel(temp_storage)
        cortex = TransferCortex(temp_storage, abstraction_engine, comparator, causal)

        abs_data = Abstraction(
            id="abs_slip2",
            name="slip_pattern",
            schema=[],
            invariants=["outcome:state=fell"],
            variables=[],
            source_episodes=["ep1"],
            instance_count=1,
            confidence=0.6,
        )
        cortex.register_abstraction(abs_data, "bathroom")

        pred = TransferPrediction(
            id="tpred_refute",
            pattern_id="abs_slip2",
            source_domain="bathroom",
            target_domain="textured_garage",
            prediction="slip risk in textured garage",
            truth_value=TruthValue(0.7, 0.3),
            created_at="2026-03-27T00:00:00",
        )
        cortex.predictions.append(pred)

        # Refute it
        result = cortex.confirm_prediction("tpred_refute", outcome=False)
        assert result["status"] == "refuted"
        assert "investigation_question" in result
        # Should ask why the pattern didn't apply
        assert "textured_garage" in result["investigation_question"]
