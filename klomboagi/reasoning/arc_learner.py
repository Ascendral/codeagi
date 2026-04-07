"""
ARC Learner — the bridge between the ARC solver and the Baby brain.

This is what makes the solver LEARN instead of just pattern match.
It feeds solved puzzles as episodes, remembers what strategies worked,
learns from failures, and composes new strategies automatically.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any
from collections import Counter

Grid = list[list[int]]

# Default memory path: relative to project root
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MEMORY_PATH = os.path.join(_PROJECT_ROOT, "runtime", "state", "arc_memory.json")


@dataclass
class PuzzleEpisode:
    """What the solver learned from one puzzle attempt."""
    puzzle_id: str
    solved: bool
    strategy_used: str | None       # Which strategy cracked it
    strategies_tried: list[str]     # All strategies attempted
    attempt_trace: list[dict[str, str]] = field(default_factory=list)
    input_properties: dict          # Grid size, color count, symmetry, etc.
    output_properties: dict
    time_ms: int = 0

    def to_dict(self) -> dict:
        return {
            "puzzle_id": self.puzzle_id,
            "solved": self.solved,
            "strategy_used": self.strategy_used,
            "strategies_tried": self.strategies_tried,
            "attempt_trace": self.attempt_trace,
            "input_properties": self.input_properties,
            "output_properties": self.output_properties,
            "time_ms": self.time_ms,
        }


@dataclass 
class StrategyProfile:
    """What the system knows about one strategy's effectiveness."""
    name: str
    successes: int = 0
    failures: int = 0              # Returned wrong answer
    skips: int = 0                 # Returned None
    total_time_ms: int = 0
    works_when: list[dict] = field(default_factory=list)   # Input properties when it works
    fails_when: list[dict] = field(default_factory=list)   # Input properties when it fails

    @property
    def success_rate(self) -> float:
        total = self.successes + self.failures
        return self.successes / total if total > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "successes": self.successes,
            "failures": self.failures,
            "skips": self.skips,
            "success_rate": round(self.success_rate, 3),
            "works_when": self.works_when[:10],  # Keep top 10
            "fails_when": self.fails_when[:10],
        }


def analyze_grid(grid: Grid) -> dict:
    """Extract properties from a grid that help predict which strategy to use."""
    rows, cols = len(grid), len(grid[0]) if grid else 0
    flat = [cell for row in grid for cell in row]
    color_counts = Counter(flat)
    colors = set(flat)
    bg = color_counts.most_common(1)[0][0] if color_counts else 0
    non_bg = [c for c in flat if c != bg]
    
    # Symmetry checks
    h_sym = all(grid[r] == grid[r][::-1] for r in range(rows))
    v_sym = all(grid[r] == grid[rows-1-r] for r in range(rows // 2))
    
    # Sparsity
    sparsity = len(non_bg) / len(flat) if flat else 0
    
    return {
        "rows": rows,
        "cols": cols,
        "num_colors": len(colors),
        "bg_color": bg,
        "bg_fraction": color_counts[bg] / len(flat) if flat else 0,
        "sparsity": round(sparsity, 2),
        "is_square": rows == cols,
        "h_symmetric": h_sym,
        "v_symmetric": v_sym,
        "has_border": _has_uniform_border(grid, bg),
    }


def _has_uniform_border(grid: Grid, bg: int) -> bool:
    """Check if the grid has a uniform border color."""
    if not grid:
        return False
    rows, cols = len(grid), len(grid[0])
    border = set()
    for c in range(cols):
        border.add(grid[0][c])
        border.add(grid[rows-1][c])
    for r in range(rows):
        border.add(grid[r][0])
        border.add(grid[r][cols-1])
    return len(border) == 1


class ARCLearner:
    """
    Wraps the ARC solver with learning capabilities.
    
    - Records every attempt as an episode
    - Builds strategy profiles from experience
    - Reorders strategies based on what works for similar puzzles
    - Composes new strategies from successful combinations
    - Accepts human nudges to adjust strategy selection
    """

    def __init__(self, memory_path: str = MEMORY_PATH) -> None:
        self.memory_path = memory_path
        self.episodes: list[PuzzleEpisode] = []
        self.strategy_profiles: dict[str, StrategyProfile] = {}
        self.composed_strategies: list[dict] = []
        self._load_memory()

    def _load_memory(self) -> None:
        """Load learning memory from disk."""
        if os.path.exists(self.memory_path):
            try:
                with open(self.memory_path, 'r') as f:
                    data = json.load(f)
                self.episodes = [
                    PuzzleEpisode(**{k: v for k, v in ep.items() if k in PuzzleEpisode.__dataclass_fields__})
                    for ep in data.get("episodes", [])
                ]
                for sp in data.get("strategy_profiles", {}).values():
                    name = sp["name"]
                    self.strategy_profiles[name] = StrategyProfile(
                        name=name,
                        successes=sp.get("successes", 0),
                        failures=sp.get("failures", 0),
                        skips=sp.get("skips", 0),
                        works_when=sp.get("works_when", []),
                        fails_when=sp.get("fails_when", []),
                    )
                self.composed_strategies = data.get("composed_strategies", [])
            except Exception:
                pass

    def _save_memory(self) -> None:
        """Persist learning memory to disk."""
        os.makedirs(os.path.dirname(self.memory_path), exist_ok=True)
        data = {
            "episodes": [ep.to_dict() for ep in self.episodes[-1000:]],  # Keep last 1000
            "strategy_profiles": {name: sp.to_dict() for name, sp in self.strategy_profiles.items()},
            "composed_strategies": self.composed_strategies[:100],
        }
        with open(self.memory_path, 'w') as f:
            json.dump(data, f, indent=2)

    def record_episode(self, task_id: str, solved: bool, strategy_used: str | None,
                       train: list[dict], test_input: Grid, time_ms: float = 0,
                       strategies_tried: list[str] | None = None,
                       attempt_trace: list[dict[str, str]] | None = None,
                       save: bool = False) -> PuzzleEpisode:
        """
        Record an episode from an external solver run.

        Called by the eval loop AFTER SmartARCSolverV2.solve() returns.
        Does not run the solver itself — just records what happened.
        """
        input_props = analyze_grid(test_input) if test_input else (analyze_grid(train[0]["input"]) if train else {})
        output_props = analyze_grid(train[0]["output"]) if train else {}
        tried = list(strategies_tried or ([strategy_used] if strategy_used else []))
        trace = list(attempt_trace or [])

        episode = PuzzleEpisode(
            puzzle_id=task_id,
            solved=solved,
            strategy_used=strategy_used if solved else None,
            strategies_tried=tried,
            attempt_trace=trace,
            input_properties=input_props,
            output_properties=output_props,
            time_ms=int(time_ms),
        )
        self.episodes.append(episode)

        # Update strategy profiles from the full trace when available.
        if trace:
            seen: set[str] = set()
            for attempt in trace:
                name = attempt.get("strategy")
                outcome = attempt.get("outcome", "skip")
                if not name or name in seen:
                    continue
                seen.add(name)
                if name not in self.strategy_profiles:
                    self.strategy_profiles[name] = StrategyProfile(name=name)
                prof = self.strategy_profiles[name]
                if outcome == "success":
                    prof.successes += 1
                    prof.works_when.append(input_props)
                elif outcome in {"reject", "wrong"}:
                    prof.failures += 1
                    prof.fails_when.append(input_props)
                else:
                    prof.skips += 1
        elif strategy_used:
            if strategy_used not in self.strategy_profiles:
                self.strategy_profiles[strategy_used] = StrategyProfile(name=strategy_used)
            prof = self.strategy_profiles[strategy_used]
            if solved:
                prof.successes += 1
                prof.works_when.append(input_props)
            else:
                prof.failures += 1
                prof.fails_when.append(input_props)

        if save:
            self._save_memory()
        return episode

    def get_strategy_scores(self, train: list[dict], test_input: Grid | None = None) -> dict[str, float]:
        """
        Return learned score boosts for strategies based on past experience.

        Returns dict mapping strategy key (e.g. "phase1:_try_mirror_symmetry")
        to a float score (0.0 to 1.0) representing learned confidence.

        SmartARCSolverV2 consumes this via solver._learned_scores.
        """
        if not self.strategy_profiles:
            return {}

        input_props = analyze_grid(test_input) if test_input else (analyze_grid(train[0]["input"]) if train else {})

        scores: dict[str, float] = {}
        for name, prof in self.strategy_profiles.items():
            if prof.successes + prof.failures == 0:
                continue

            # Base score from success rate
            score = prof.success_rate

            # Bonus for similarity to past successes (last 20)
            for past_props in prof.works_when[-20:]:
                sim = self._property_similarity(input_props, past_props)
                score += sim * 0.1

            # Penalty for similarity to past failures (last 20)
            for past_props in prof.fails_when[-20:]:
                sim = self._property_similarity(input_props, past_props)
                score -= sim * 0.05

            scores[name] = max(0.0, min(1.0, score))

        return scores

    # DEPRECATED: solve_and_learn uses ARCSolverV10 (many versions behind).
    # Use SmartARCSolverV2.solve() + learner.record_episode() instead.
    def solve_and_learn(self, puzzle_id: str, train: list[dict], test_input: Grid,
                        expected_output: Grid | None = None) -> tuple[Grid | None, PuzzleEpisode]:
        """DEPRECATED — uses old V10 solver. Use record_episode() instead."""
        import time
        from klomboagi.reasoning.arc_solver import ARCSolverV10
        solver = ARCSolverV10()

        input_props = analyze_grid(train[0]["input"]) if train else {}
        output_props = analyze_grid(train[0]["output"]) if train else {}

        strategy_order = self._get_strategy_order(input_props, output_props, solver)

        start = time.time()
        strategies_tried = []
        winning_strategy = None
        result = None

        for strategy_name, strategy_fn in strategy_order:
            strategies_tried.append(strategy_name)
            try:
                r = strategy_fn(train, test_input)
                if r is not None:
                    if solver._cross_validate(strategy_fn, train):
                        result = r
                        winning_strategy = strategy_name
                        break
            except Exception:
                continue

        elapsed = int((time.time() - start) * 1000)
        solved = result is not None and (expected_output is None or result == expected_output)
        wrong = result is not None and expected_output is not None and result != expected_output

        episode = PuzzleEpisode(
            puzzle_id=puzzle_id,
            solved=solved,
            strategy_used=winning_strategy if solved else None,
            strategies_tried=strategies_tried,
            input_properties=input_props,
            output_properties=output_props,
            time_ms=elapsed,
        )
        self.episodes.append(episode)

        if winning_strategy:
            if winning_strategy not in self.strategy_profiles:
                self.strategy_profiles[winning_strategy] = StrategyProfile(name=winning_strategy)
            prof = self.strategy_profiles[winning_strategy]
            if solved:
                prof.successes += 1
                prof.works_when.append(input_props)
            elif wrong:
                prof.failures += 1
                prof.fails_when.append(input_props)

        for s_name in strategies_tried:
            if s_name != winning_strategy:
                if s_name not in self.strategy_profiles:
                    self.strategy_profiles[s_name] = StrategyProfile(name=s_name)
                self.strategy_profiles[s_name].skips += 1

        self._save_memory()
        return result, episode

    # DEPRECATED: Used by solve_and_learn only
    def _get_strategy_order(self, input_props: dict, output_props: dict,
                            solver=None) -> list[tuple[str, Any]]:
        """DEPRECATED — used by solve_and_learn only."""
        all_strategies = self._get_all_strategies(solver)
        
        if not self.strategy_profiles:
            return all_strategies  # No learning yet, use default order
        
        # Score each strategy based on similarity to past successes
        scored = []
        for name, fn in all_strategies:
            score = 0.5  # Default score
            
            if name in self.strategy_profiles:
                prof = self.strategy_profiles[name]
                
                # Base score from success rate
                score = prof.success_rate
                
                # Bonus if input properties match past successes
                for past_props in prof.works_when[-20:]:
                    similarity = self._property_similarity(input_props, past_props)
                    score += similarity * 0.1
                
                # Penalty if input properties match past failures
                for past_props in prof.fails_when[-20:]:
                    similarity = self._property_similarity(input_props, past_props)
                    score -= similarity * 0.05
            
            scored.append((score, name, fn))
        
        # Sort by score descending
        scored.sort(key=lambda x: x[0], reverse=True)
        return [(name, fn) for _, name, fn in scored]

    def _property_similarity(self, props_a: dict, props_b: dict) -> float:
        """How similar are two sets of grid properties? 0-1."""
        if not props_a or not props_b:
            return 0.0
        
        matches = 0
        total = 0
        
        for key in ["is_square", "h_symmetric", "v_symmetric", "has_border"]:
            if key in props_a and key in props_b:
                total += 1
                if props_a[key] == props_b[key]:
                    matches += 1
        
        # Numeric similarity
        for key in ["num_colors", "rows", "cols"]:
            if key in props_a and key in props_b:
                total += 1
                a, b = props_a[key], props_b[key]
                if a == b:
                    matches += 1
                elif abs(a - b) <= 2:
                    matches += 0.5
        
        return matches / total if total > 0 else 0.0

    # DEPRECATED: Used by solve_and_learn only
    def _get_all_strategies(self, solver=None) -> list[tuple[str, Any]]:
        """DEPRECATED — used by solve_and_learn only."""
        if solver is None:
            from klomboagi.reasoning.arc_solver import ARCSolverV10
            solver = ARCSolverV10()
        strategies = [
            ("identity", solver._try_identity),
            ("position_transform", solver._try_position_transform),
            ("size_transform", solver._try_size_transform),
            ("tile_scale", solver._try_tile_scale),
            ("mask_overlay", solver._try_mask_overlay),
            ("mirror_symmetry", solver._try_mirror_symmetry),
            ("extract_subgrid", solver._try_extract_subgrid),
            ("count_to_grid", solver._try_count_to_grid),
            ("color_by_position", solver._try_color_by_position),
            ("diagonal_flip", solver._try_diagonal_flip),
            ("rotate_90_270", solver._try_rotate_90_270),
            ("majority_color_fill", solver._try_majority_color_fill),
            ("color_conditional", solver._try_color_conditional),
            ("border_fill", solver._try_border_fill),
            ("gravity", solver._try_gravity),
            ("row_col_sort", solver._try_row_col_sort),
            ("many_to_one", solver._try_many_to_one),
            ("value_replacement", solver._try_value_replacement),
            ("size_change", solver._try_size_change),
            ("cell_by_cell_mapping", solver._try_cell_by_cell_mapping),
            ("unique_color_extract", solver._try_unique_color_extract),
            ("repeat_pattern", solver._try_repeat_pattern),
            ("color_swap", solver._try_color_swap),
            ("remove_color", solver._try_remove_color),
            ("largest_object", solver._try_largest_object),
            ("recolor_by_count", solver._try_recolor_by_count),
            ("most_common_output", solver._try_most_common_output),
            ("xor_grids", solver._try_xor_grids),
            ("and_grids", solver._try_and_grids),
            ("or_grids", solver._try_or_grids),
            ("halve_grid", solver._try_halve_grid),
            ("quarter_grid", solver._try_quarter_grid),
            ("output_constant", solver._try_output_constant),
            ("line_extension", solver._try_line_extension),
            ("smallest_object", solver._try_smallest_object),
            ("count_objects", solver._try_count_objects),
            ("pixelwise_8neighbors", solver._try_pixelwise_rule_8neighbors),
            ("color_at_intersection", solver._try_color_at_intersection),
            ("spread_color", solver._try_spread_color),
            ("shift_grid", solver._try_shift_grid),
            ("remove_duplicate_rows_cols", solver._try_remove_duplicate_rows_cols),
            ("sort_rows_by_color_count", solver._try_sort_rows_by_color_count),
            ("min_bounding_box", solver._try_min_bounding_box_all_colors),
            ("paint_enclosed_multi", solver._try_paint_enclosed_regions_multi),
            ("overlay_two_objects", solver._try_overlay_two_objects),
            ("relative_position_rule", solver._try_relative_position_rule),
            ("color_adjacency_rule", solver._try_color_adjacency_rule),
            ("connected_component_color", solver._try_connected_component_color),
            ("row_pattern_match", solver._try_row_pattern_match),
            ("column_pattern_match", solver._try_column_pattern_match),
            ("max_color_per_region", solver._try_max_color_per_region),
            ("outline_objects", solver._try_outline_objects),
            ("fill_rectangles", solver._try_fill_rectangles),
            ("per_cell_rule", solver._try_per_cell_rule),
            ("sliding_window", solver._try_sliding_window),
            ("color_mapping_by_region", solver._try_color_mapping_by_region),
            ("symmetric_completion", solver._try_symmetric_completion),
            ("unique_row_col", solver._try_unique_row_col),
            ("crop_to_unique", solver._try_crop_to_unique),
            ("upscale_pattern", solver._try_upscale_pattern),
            ("denoise", solver._try_denoise),
            ("keep_color", solver._try_keep_color),
            ("output_rows_subset", solver._try_output_is_input_subset_rows),
            ("output_cols_subset", solver._try_output_is_input_subset_cols),
        ]
        return strategies

    # ── Learning from failures ──

    def learn_from_failure(self, puzzle_id: str, train: list[dict],
                           expected_output: Grid, our_answer: Grid | None) -> dict:
        """
        Analyze WHY we got a puzzle wrong and extract a lesson.
        This is the "oh, I see what I did wrong" moment.
        """
        input_props = analyze_grid(train[0]["input"]) if train else {}
        output_props = analyze_grid(train[0]["output"]) if train else {}
        expected_props = analyze_grid(expected_output)
        
        lesson = {
            "puzzle_id": puzzle_id,
            "our_answer": "None" if our_answer is None else "wrong",
            "input_props": input_props,
            "output_props": output_props,
            "expected_props": expected_props,
            "insights": [],
        }
        
        # What changed between input and output?
        if train:
            inp, out = train[0]["input"], train[0]["output"]
            
            # Size change?
            if len(inp) != len(out) or len(inp[0]) != len(out[0]):
                lesson["insights"].append(f"size_change: {len(inp)}x{len(inp[0])} → {len(out)}x{len(out[0])}")
            
            # Color change?
            in_colors = set(c for row in inp for c in row)
            out_colors = set(c for row in out for c in row)
            new = out_colors - in_colors
            removed = in_colors - out_colors
            if new:
                lesson["insights"].append(f"new_colors: {new}")
            if removed:
                lesson["insights"].append(f"removed_colors: {removed}")
            
            # Same grid content, different arrangement?
            in_sorted = sorted(c for row in inp for c in row)
            out_sorted = sorted(c for row in out for c in row)
            if in_sorted == out_sorted and inp != out:
                lesson["insights"].append("rearrangement: same cells, different positions")
        
        return lesson

    # ── Strategy composition ──

    def discover_compositions(self) -> list[dict]:
        """
        Look at failed puzzles and see if combining 2-3 successful strategies
        would have solved them. If so, register the composition.
        """
        new_compositions = []
        
        # Get strategies that have succeeded
        successful = {name for name, prof in self.strategy_profiles.items() 
                      if prof.successes > 0}
        
        if len(successful) < 2:
            return []
        
        # For each failed episode, check if a pair of strategies works
        for ep in self.episodes:
            if ep.solved:
                continue
            # This would need the actual puzzle data to test compositions
            # For now, record the pattern of what was tried
            if ep.strategy_used is None and len(ep.strategies_tried) > 5:
                new_compositions.append({
                    "puzzle_id": ep.puzzle_id,
                    "input_props": ep.input_properties,
                    "tried": ep.strategies_tried[:5],
                })
        
        return new_compositions

    # ── Human nudge ──

    def nudge(self, hint: str) -> None:
        """
        Human says something like "this is about symmetry" or "look at the colors."
        Adjust strategy priorities accordingly.
        """
        hint_lower = hint.lower()
        
        # Map hints to strategy boosts
        hint_map = {
            "symmetry": ["mirror_symmetry", "symmetric_completion", "h_symmetric", "v_symmetric"],
            "flip": ["position_transform", "diagonal_flip"],
            "rotate": ["position_transform", "rotate_90_270"],
            "color": ["value_replacement", "color_swap", "many_to_one", "keep_color"],
            "object": ["largest_object", "smallest_object", "connected_component_color", "overlay_two_objects"],
            "fill": ["mask_overlay", "paint_enclosed_multi", "fill_rectangles", "border_fill"],
            "sort": ["row_col_sort", "sort_rows_by_color_count", "gravity"],
            "crop": ["extract_subgrid", "crop_to_unique", "min_bounding_box", "unique_color_extract"],
            "pattern": ["repeat_pattern", "tile_scale", "upscale_pattern"],
            "size": ["size_change", "halve_grid", "quarter_grid"],
        }
        
        for keyword, strategies in hint_map.items():
            if keyword in hint_lower:
                for s_name in strategies:
                    if s_name not in self.strategy_profiles:
                        self.strategy_profiles[s_name] = StrategyProfile(name=s_name)
                    # Boost by adding fake successes with similar properties
                    self.strategy_profiles[s_name].successes += 3
        
        self._save_memory()

    # ── Stats ──

    def stats(self) -> dict:
        """Learning progress summary."""
        total = len(self.episodes)
        solved = len([e for e in self.episodes if e.solved])
        top_strategies = sorted(
            self.strategy_profiles.values(),
            key=lambda s: s.successes,
            reverse=True
        )[:10]
        
        return {
            "total_episodes": total,
            "solved": solved,
            "solve_rate": round(solved / total, 3) if total > 0 else 0,
            "strategies_with_data": len(self.strategy_profiles),
            "top_strategies": [{"name": s.name, "wins": s.successes, "rate": s.success_rate} 
                              for s in top_strategies],
            "composed_strategies": len(self.composed_strategies),
        }
