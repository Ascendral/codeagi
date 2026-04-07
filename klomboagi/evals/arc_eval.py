"""
ARC-AGI-1 Evaluation — get a real score.

Runs the SmartARCSolver (V18 + learned strategy ordering) against
the full ARC-AGI-1 dataset via arckit.

Reports:
  - Overall accuracy (correct / total)
  - Per-task results (pass/fail + which strategy worked)
  - Failure analysis (what types of tasks fail and why)
  - Timing

With --learn: records episodes, builds strategy profiles, and uses
learned state to influence strategy ordering on subsequent runs.

This is the single most important benchmark for KlomboAGI.
A real number tells us exactly where we are.
"""

from __future__ import annotations

import json
import os
import time
import sys
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ArcEvalResult:
    """Result of evaluating one ARC task."""
    task_id: str = ""
    correct: bool = False
    predicted: list[list[int]] | None = None
    expected: list[list[int]] | None = None
    strategy_used: str = ""
    time_ms: float = 0
    input_shape: tuple[int, int] = (0, 0)
    output_shape: tuple[int, int] = (0, 0)
    error: str = ""


@dataclass
class ArcEvalReport:
    """Full evaluation report."""
    results: list[ArcEvalResult] = field(default_factory=list)
    total: int = 0
    correct: int = 0
    failed: int = 0
    errors: int = 0
    total_time_s: float = 0
    learning_influence: int = 0  # Tasks where learned state changed ordering

    def accuracy(self) -> float:
        return self.correct / self.total if self.total > 0 else 0

    def summary(self) -> str:
        lines = [
            f"ARC-AGI-1 Evaluation: {self.correct}/{self.total} "
            f"({self.accuracy():.1%})",
            f"  Time: {self.total_time_s:.1f}s "
            f"({self.total_time_s / self.total * 1000:.0f}ms/task)" if self.total > 0 else "",
            f"  Correct: {self.correct}",
            f"  Failed: {self.failed}",
            f"  Errors: {self.errors}",
        ]

        if self.learning_influence > 0:
            lines.append(f"\n  Learning influence: {self.learning_influence}/{self.total} "
                         f"tasks had ordering changed by learned state")

        # Size analysis
        same_size_correct = sum(1 for r in self.results if r.correct and r.input_shape == r.output_shape)
        same_size_total = sum(1 for r in self.results if r.input_shape == r.output_shape)
        diff_size_correct = sum(1 for r in self.results if r.correct and r.input_shape != r.output_shape)
        diff_size_total = sum(1 for r in self.results if r.input_shape != r.output_shape)

        lines.append(f"\n  Same-size tasks: {same_size_correct}/{same_size_total} "
                     f"({same_size_correct/same_size_total:.1%})" if same_size_total > 0 else "")
        lines.append(f"  Diff-size tasks: {diff_size_correct}/{diff_size_total} "
                     f"({diff_size_correct/diff_size_total:.1%})" if diff_size_total > 0 else "")

        # Strategy phase breakdown
        phase_counts: dict[str, int] = {}
        for r in self.results:
            if r.correct and r.strategy_used:
                phase = r.strategy_used.split(":")[0]
                phase_counts[phase] = phase_counts.get(phase, 0) + 1
        if phase_counts:
            lines.append(f"\n  Wins by phase:")
            for phase, count in sorted(phase_counts.items(), key=lambda x: -x[1]):
                lines.append(f"    {phase}: {count}")

        # Show some failures
        failures = [r for r in self.results if not r.correct and not r.error]
        if failures:
            lines.append(f"\n  Sample failures:")
            for r in failures[:10]:
                lines.append(f"    {r.task_id}: {r.input_shape}->{r.output_shape} ({r.time_ms:.0f}ms)")

        return "\n".join(lines)


def run_arc_eval(max_tasks: int = 0, dataset: str = "training",
                 on_progress=None, use_learning: bool = False,
                 memory_path: str | None = None) -> ArcEvalReport:
    """
    Run ARC-AGI-1 evaluation.

    Args:
        max_tasks: 0 = all tasks, >0 = limit
        dataset: "training" (400 tasks) or "evaluation" (400 tasks)
        on_progress: callback(task_idx, total, task_id, correct)
        use_learning: if True, record episodes and use learned state
        memory_path: override default memory path for ARCLearner
    """
    import arckit
    import numpy as np
    from klomboagi.reasoning.arc_smart_solver import SmartARCSolverV2 as SmartARCSolver

    learner = None
    if use_learning:
        from klomboagi.reasoning.arc_learner import ARCLearner
        learner = ARCLearner(memory_path=memory_path) if memory_path else ARCLearner()
        prior_episodes = len(learner.episodes)
        prior_profiles = len(learner.strategy_profiles)
        print(f"  Learning: ON (loaded {prior_episodes} episodes, {prior_profiles} strategy profiles)")

    train_set, eval_set = arckit.load_data()
    tasks = train_set if dataset == "training" else eval_set

    if max_tasks > 0:
        tasks = list(tasks)[:max_tasks]

    solver = SmartARCSolver()
    report = ArcEvalReport()
    start = time.time()

    for idx, task in enumerate(tasks):
        task_start = time.time()
        result = ArcEvalResult(task_id=task.id)

        try:
            # Convert arckit format to solver format
            train_examples = []
            for ex in task.train:
                inp = np.array(ex[0]).tolist()
                out = np.array(ex[1]).tolist()
                train_examples.append({"input": inp, "output": out})

            # Get test input and expected output
            test_ex = task.test[0]
            test_input = np.array(test_ex[0]).tolist()
            test_expected = np.array(test_ex[1]).tolist()

            result.input_shape = (len(test_input), len(test_input[0]) if test_input else 0)
            result.output_shape = (len(test_expected), len(test_expected[0]) if test_expected else 0)
            result.expected = test_expected

            # Inject learned scores if learning is enabled
            if learner is not None:
                scores = learner.get_strategy_scores(train_examples, test_input)
                solver._learned_scores = scores
            else:
                solver._learned_scores = {}

            # Solve
            predicted = solver.solve(train_examples, test_input)
            result.predicted = predicted
            result.correct = (predicted == test_expected)
            result.strategy_used = getattr(solver, '_last_strategy', '') or ''
            result.time_ms = (time.time() - task_start) * 1000

            if getattr(solver, "_learning_reordered", False):
                report.learning_influence += 1

            # Record episode for learning
            if learner is not None:
                attempt_trace = list(getattr(solver, "_attempt_trace", []))
                strategies_tried = [
                    attempt["strategy"]
                    for attempt in attempt_trace
                    if attempt.get("strategy")
                ]
                learner.record_episode(
                    task_id=task.id,
                    solved=result.correct,
                    strategy_used=result.strategy_used or None,
                    train=train_examples,
                    test_input=test_input,
                    time_ms=result.time_ms,
                    strategies_tried=strategies_tried,
                    attempt_trace=attempt_trace,
                    save=False,
                )

        except Exception as e:
            result.error = str(e)[:200]
            result.correct = False

        if result.time_ms == 0:
            result.time_ms = (time.time() - task_start) * 1000
        report.results.append(result)
        report.total += 1
        if result.correct:
            report.correct += 1
        elif result.error:
            report.errors += 1
        else:
            report.failed += 1

        if on_progress:
            on_progress(idx + 1, len(tasks), task.id, result.correct)

    report.total_time_s = time.time() - start

    # Save learning state once at the end
    if learner is not None:
        learner._save_memory()
        print(f"\n  Learning: saved {len(learner.episodes)} episodes, "
              f"{len(learner.strategy_profiles)} strategy profiles "
              f"to {learner.memory_path}")

    return report


def save_eval_results(report: ArcEvalReport, use_learning: bool = False) -> str:
    """Save evaluation results to timestamped JSON file. Returns the file path."""
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    results_dir = os.path.join(project_root, "runtime", "state", "eval_results")
    os.makedirs(results_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(results_dir, f"{timestamp}.json")

    data = {
        "timestamp": timestamp,
        "total": report.total,
        "correct": report.correct,
        "accuracy": round(report.accuracy(), 4),
        "used_learning": use_learning,
        "learning_influence": report.learning_influence,
        "total_time_s": round(report.total_time_s, 1),
        "results": [
            {
                "task_id": r.task_id,
                "correct": r.correct,
                "strategy_used": r.strategy_used,
                "time_ms": round(r.time_ms, 1),
                "input_shape": list(r.input_shape),
                "output_shape": list(r.output_shape),
            }
            for r in report.results
        ],
    }

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)

    return filepath


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run ARC-AGI-1 evaluation")
    parser.add_argument("--max", type=int, default=0, help="Max tasks (0=all)")
    parser.add_argument("--dataset", default="training", choices=["training", "evaluation"])
    parser.add_argument("--learn", action="store_true", help="Enable learning (record episodes, use learned state)")
    parser.add_argument("--memory", type=str, default=None, help="Override memory file path")
    parser.add_argument("--save-results", action="store_true", help="Save results to JSON (auto-enabled with --learn)")
    args = parser.parse_args()

    def progress(idx, total, task_id, correct):
        mark = "\u2713" if correct else "\u2717"
        sys.stdout.write(f"\r  [{idx}/{total}] {task_id} {mark}   ")
        sys.stdout.flush()

    print(f"\n  ARC-AGI-1 Evaluation ({args.dataset} set)")
    print(f"  {'=' * 40}")

    report = run_arc_eval(
        max_tasks=args.max,
        dataset=args.dataset,
        on_progress=progress,
        use_learning=args.learn,
        memory_path=args.memory,
    )
    print(f"\n\n{report.summary()}")

    # Save results
    if args.save_results or args.learn:
        filepath = save_eval_results(report, use_learning=args.learn)
        print(f"\n  Results saved to: {filepath}")
