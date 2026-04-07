"""
Runtime Bridge — wires the Cognitive Nexus into the actual runtime.

The RuntimeLoop runs the main execution cycle:
  plan → verify → critique → execute → reflect → consolidate → remember

The Nexus adds connective tissue after each cycle:
  → learn_experience (causal + abstraction + transfer + prediction + awareness)
  → get_awareness (before next action — "heads up" memos)
  → tick (decay attention, beliefs, predictions)

This bridge doesn't modify RuntimeLoop. It wraps it, intercepting the
cycle output and feeding it through the Nexus.

It also provides awareness injection: before each action, the system
checks if the current situation matches any known patterns from other
domains and surfaces that as contextual awareness in working memory.

No LLM. The bridge is pure wiring.
"""

from __future__ import annotations

from typing import Any

from klomboagi.core.loop import RuntimeLoop
from klomboagi.connective.nexus import CognitiveNexus
from klomboagi.storage.manager import StorageManager
from klomboagi.utils.time import utc_now


class ConnectedRuntime:
    """
    RuntimeLoop + CognitiveNexus = a brain that thinks AND connects.

    Usage:
        storage = StorageManager.bootstrap()
        runtime = ConnectedRuntime(storage)
        runtime.initialize()
        result = runtime.run_cycle()
        # result now includes awareness_memos, transfer_predictions,
        # investigation results, belief updates, and causal learning
    """

    def __init__(self, storage: StorageManager) -> None:
        self.storage = storage
        self.loop = RuntimeLoop(storage)
        self.nexus = CognitiveNexus(storage)

    def initialize(self) -> dict[str, object]:
        """Initialize the runtime and the nexus."""
        return self.loop.initialize()

    def status(self) -> dict[str, object]:
        """Get full status including nexus stats."""
        base_status = self.loop.status()
        base_status["nexus"] = self.nexus.stats()
        return base_status

    def run_cycle(self) -> dict[str, object]:
        """
        Run one full cycle with connective tissue active.

        1. Run the base RuntimeLoop cycle (plan → execute → reflect → learn)
        2. Feed the cycle result through the Nexus (cross-domain transfer,
           prediction, awareness, investigation, belief update)
        3. Inject awareness memos into the result
        4. Tick the Nexus (decay attention, beliefs, predictions)
        """
        # 1. Run the base cycle
        result = self.loop.run_cycle()

        if result.get("status") == "idle":
            # Nothing to learn from an idle cycle, but still tick
            self.nexus.tick()
            result["nexus"] = {"status": "idle"}
            return result

        # 2. Build an episode from the cycle result for the Nexus to learn from
        episode = self._cycle_to_episode(result)

        # 3. Feed through the Nexus unified learning pipeline
        nexus_result = self.nexus.learn_experience(episode)

        # 4. Get awareness memos for the current situation
        situation = {
            "description": result.get("mission", {}).get("description", ""),
            "domain": self._extract_domain(result),
            "outcome": result.get("action_outcome", {}).get("status", "unknown"),
        }
        awareness_memos = self.nexus.get_awareness(situation)

        # 5. If we have awareness memos, inject them into working memory notes
        if awareness_memos:
            mission_id = str(result.get("mission", {}).get("id", ""))
            if mission_id:
                memo_texts = [
                    f"[AWARENESS] {memo.message} — {memo.suggested_caution}"
                    for memo in awareness_memos[:3]  # Top 3 most relevant
                ]
                self._inject_awareness_to_working_memory(mission_id, memo_texts)

        # 6. Tick the Nexus
        self.nexus.tick()

        # 7. Attach nexus results to the cycle output
        result["nexus"] = {
            "causal_edges_learned": nexus_result.get("causal_edges", 0),
            "abstraction": nexus_result.get("abstraction"),
            "transfer_predictions": nexus_result.get("transfer_predictions", []),
            "prediction_confirmations": nexus_result.get("prediction_confirmations", []),
            "awareness_memos": [m.to_dict() for m in awareness_memos],
            "investigations_opened": nexus_result.get("investigations_opened", 0),
            "beliefs_updated": nexus_result.get("beliefs_updated", 0),
            "stats": self.nexus.stats(),
        }

        self.storage.event_log.append(
            "nexus.cycle.completed",
            {
                "mission_id": result.get("mission", {}).get("id"),
                "awareness_memos": len(awareness_memos),
                "transfer_predictions": len(nexus_result.get("transfer_predictions", [])),
                "beliefs": self.nexus.stats()["beliefs"],
            },
        )

        return result

    def think(self, problem: dict) -> Any:
        """
        Use the cognition loop for a standalone reasoning problem.
        The Nexus callbacks are wired in — deep investigation,
        awareness, and cross-domain transfer all active.
        """
        return self.nexus.think(problem)

    def pursue(self, question: str, context: str = "", max_depth: int = 3) -> Any:
        """Deep knowledge seeking — recursive investigation."""
        return self.nexus.pursue(question, context, max_depth)

    def explore(self, concept: str) -> dict:
        """Lateral exploration — look around a concept."""
        return self.nexus.explore_laterally(concept)

    def awareness(self, situation: dict) -> list:
        """Get awareness memos for a situation."""
        return self.nexus.get_awareness(situation)

    # ── Helpers ──

    def _cycle_to_episode(self, result: dict) -> dict:
        """Convert a RuntimeLoop cycle result into an episode for the Nexus."""
        mission = result.get("mission", {})
        action = result.get("next_action", {})
        outcome = result.get("action_outcome", {})
        cycle_trace = result.get("cycle_trace", {})

        # Build action list from cycle trace steps
        actions = []
        for step in cycle_trace.get("steps", []):
            next_action = step.get("next_action", {})
            action_outcome = step.get("action_outcome", {})
            actions.append({
                "type": next_action.get("type", "unknown"),
                "description": next_action.get("description", ""),
                "status": action_outcome.get("status", "unknown"),
            })

        return {
            "id": f"cycle_{mission.get('id', 'unknown')}_{utc_now()}",
            "description": str(mission.get("description", "")),
            "domain": self._extract_domain(result),
            "actions": actions,
            "outcome": str(outcome.get("status", "unknown")),
            "success": outcome.get("status") == "completed",
            "mission_id": str(mission.get("id", "")),
            "stop_reason": cycle_trace.get("stop_reason", "unknown"),
            "step_count": cycle_trace.get("step_count", 0),
        }

    def _extract_domain(self, result: dict) -> str:
        """Extract a domain from the cycle result for transfer tracking."""
        mission = result.get("mission", {})
        description = str(mission.get("description", ""))
        # Use first 3 meaningful words as domain
        words = [
            w.lower().strip(".,!?")
            for w in description.split()
            if len(w) > 3 and w.lower() not in (
                "the", "and", "for", "with", "from", "that", "this",
            )
        ]
        return "_".join(words[:3]) if words else "general"

    def _inject_awareness_to_working_memory(
        self, mission_id: str, memo_texts: list[str]
    ) -> None:
        """Inject awareness memos into working memory as notes."""
        try:
            self.loop.working_memory.update(
                mission_id,
                hypothesis=memo_texts[0] if memo_texts else None,
            )
        except (KeyError, TypeError):
            pass  # Working memory might not exist yet for this mission


class ConnectedAgent:
    """
    Wrapper for IntegratedAgent that adds Nexus awareness.

    Before task execution: checks for awareness memos
    After task execution: feeds the trajectory through Nexus learning

    Usage:
        from klomboagi.agent.integrated import IntegratedAgent
        from klomboagi.connective.runtime_bridge import ConnectedAgent

        storage = StorageManager.bootstrap()
        agent = IntegratedAgent()
        connected = ConnectedAgent(agent, storage)
        result = connected.execute(task)
        # result now includes awareness_before and nexus_learning
    """

    def __init__(self, agent: Any, storage: StorageManager) -> None:
        self.agent = agent
        self.nexus = CognitiveNexus(storage)

    def execute(self, task: dict) -> dict:
        """
        Execute a task with awareness and learning.

        1. Check awareness BEFORE execution
        2. Execute via the IntegratedAgent
        3. Learn from the outcome via the Nexus
        """
        # 1. Pre-execution awareness
        awareness = self.nexus.get_awareness({
            "description": task.get("description", ""),
            "domain": task.get("domain", "unknown"),
        })

        # 2. Execute the task
        result = self.agent.execute(task)

        # 3. Post-execution learning
        episode = {
            "id": task.get("id", "unknown"),
            "description": task.get("description", ""),
            "domain": task.get("domain", "unknown"),
            "actions": result.get("trace", []),
            "outcome": "success" if result.get("success") else "failure",
            "success": result.get("success", False),
        }
        nexus_learning = self.nexus.learn_experience(episode)

        # 4. Tick
        self.nexus.tick()

        # 5. Attach to result
        result["awareness_before"] = [m.to_dict() for m in awareness]
        result["nexus_learning"] = nexus_learning

        return result
