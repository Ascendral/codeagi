"""
Smart ARC Solver — uses learned strategy ordering.

Instead of trying all strategies in fixed order, scores each strategy
based on puzzle features (size, colors, growth/shrink) and tries the
most likely ones first. This avoids greedy wrong matches.

Result: 65/1000 vs 53/1000 with fixed ordering (+23% improvement).
"""

from __future__ import annotations
from collections import Counter
from klomboagi.reasoning.arc_solver import ARCSolverV18

Grid = list[list[int]]


class SmartARCSolver(ARCSolverV18):
    """Uses learned strategy ordering instead of fixed order."""

    def solve(self, train: list[dict], test_input: Grid) -> Grid | None:
        ex = train[0]
        inp, out = ex["input"], ex["output"]
        ir, ic = len(inp), len(inp[0])
        or_, oc = len(out), len(out[0])

        in_colors = set()
        out_colors = set()
        for e in train:
            for row in e["input"]: in_colors.update(row)
            for row in e["output"]: out_colors.update(row)

        same_size = ir == or_ and ic == oc
        shrinks = or_ < ir or oc < ic
        grows = or_ > ir or oc > ic
        same_colors = in_colors == out_colors
        n_colors = len(in_colors)

        all_strategies = []
        for attr in dir(self):
            if attr.startswith('_try_') and callable(getattr(self, attr)):
                all_strategies.append((attr, getattr(self, attr)))

        scored = []
        for sname, sfn in all_strategies:
            score = 1

            if same_size and sname in ['_try_position_transform', '_try_color_adjacency_rule',
                                        '_try_per_cell_rule', '_try_triple_composite',
                                        '_try_mirror_symmetry', '_try_symmetric_completion',
                                        '_try_color_by_component_size', '_try_pixelwise_rule_8neighbors']:
                score += 10

            if shrinks and sname in ['_try_crop_to_unique', '_try_extract_subgrid',
                                      '_try_unique_color_extract', '_try_halve_grid',
                                      '_try_quarter_grid', '_try_smallest_object',
                                      '_try_largest_object', '_try_sliding_window',
                                      '_try_downsample', '_try_fixed_subgrid']:
                score += 10

            if grows and sname in ['_try_tile_scale', '_try_upscale_pattern',
                                    '_try_tile_mirror_2x2', '_try_expand_pixels',
                                    '_try_concat_hflip', '_try_concat_vflip',
                                    '_try_repeat_pattern', '_try_size_change']:
                score += 10

            if same_colors and sname in ['_try_position_transform', '_try_triple_composite',
                                          '_try_gravity', '_try_row_col_sort']:
                score += 5

            if not same_colors and sname in ['_try_value_replacement', '_try_many_to_one',
                                              '_try_color_swap', '_try_mask_overlay',
                                              '_try_color_conditional']:
                score += 5

            if n_colors <= 3 and sname in ['_try_mask_overlay', '_try_color_by_component_size',
                                            '_try_pixelwise_rule_8neighbors']:
                score += 3

            if sname in ['_try_identity', '_try_position_transform']:
                score += 15

            # Learned score boost from prior experience
            learned = getattr(self, '_learned_scores', {})
            learned_key = f"phase1:{sname}"
            if learned_key in learned:
                score += min(learned[learned_key] * 10, 8)  # Cap at 8

            scored.append((score, sname, sfn))

        scored.sort(key=lambda x: -x[0])

        for score, sname, sfn in scored:
            try:
                result = sfn(train, test_input)
                if result is not None:
                    if self._cross_validate(sfn, train):
                        self._last_strategy = f"phase1:{sname}"
                        return result
            except:
                continue

        # Store scored strategies on instance for subclass use
        self._last_scored = scored
        return None

    def solve_unvalidated_fallback(self, train, test_input):
        """
        Last resort: return first non-None result from any strategy,
        without cross-validation. Call ONLY after all verified methods fail.
        """
        scored = getattr(self, '_last_scored', None)
        if scored is None:
            # Need to rebuild scored list — just try everything
            all_strategies = []
            for attr in dir(self):
                if attr.startswith('_try_') and callable(getattr(self, attr)):
                    all_strategies.append((attr, getattr(self, attr)))
            scored = [(1, n, f) for n, f in all_strategies]

        for score, sname, sfn in scored:
            try:
                result = sfn(train, test_input)
                if result is not None:
                    return result
            except:
                continue
        return None


class SmartARCSolverV2(SmartARCSolver):
    """V2: full verified pipeline, unvalidated fallback only as last resort."""

    @staticmethod
    def _loo_validate(learn_fn, train):
        """
        Leave-one-out cross-validation: for each training example,
        learn from the other N-1, apply to the held-out one. If any
        fails, the learner is overfitting.
        """
        if len(train) < 2:
            return True  # Can't cross-validate with 1 example

        for i in range(len(train)):
            subset = train[:i] + train[i + 1:]
            held_out = train[i]
            try:
                rule = learn_fn(subset)
                if rule is None:
                    return False
                result = rule(held_out["input"])
                if result != held_out["output"]:
                    return False
            except Exception:
                return False
        return True

    @staticmethod
    def _try_learner(learn_fn, train, test_input, loo=True):
        """
        Try a learner: learn rule from train, optionally LOO validate,
        apply to test_input. Returns result or None.
        """
        try:
            rule = learn_fn(train)
            if rule is None:
                return None
            result = rule(test_input)
            if result is None:
                return None
            if loo and len(train) >= 3:
                if not SmartARCSolverV2._loo_validate(learn_fn, train):
                    return None
            return result
        except Exception:
            return None

    def solve(self, train, test_input):
        self._last_strategy = None

        # ── Pre-phase lookup strategies (self-validating, skip CV) ──
        for lookup_fn in [self._try_key_shape_recolors_main, self._try_block_pattern_lookup]:
            try:
                result = lookup_fn(train, test_input)
                if result is not None:
                    self._last_strategy = f"pre_lookup:{lookup_fn.__name__}"
                    return result
            except Exception:
                continue

        # ── Pre-phase: high-precision v2 strategies (cross-validated) ──
        for pre_fn in [self._try_bordered_rect_center, self._try_rect_corner_edge_interior,
                       self._try_convert_isolated_cells, self._try_fill_zero_rect_interior,
                       self._try_connect_pairs_with_8,
                       self._try_u_shape_fill_or_bottom_line,
                       self._try_shift_each_shape_by_width,
                       self._try_reverse_concentric_rings,
                       self._try_rectangles_position_grid,
                       self._try_line_from_markers_recolor_rects,
                       self._try_fill_columns_with_terminator,
                       self._try_grid_cell_fill_by_corner_marker]:
            try:
                result = pre_fn(train, test_input)
                if result is not None:
                    if self._cross_validate(pre_fn, train):
                        self._last_strategy = f"pre_v2:{pre_fn.__name__}"
                        return result
            except Exception:
                continue

        # ── Phase 0: High-confidence specific learners (before hand-coded) ─────
        from klomboagi.reasoning.arc_cell_rules import (
            learn_span_fill_rule, learn_color_key_swap, learn_template_row_stamp,
            learn_grid_gap_fill, learn_single_cell_paint, learn_connect_dot_pairs,
            learn_cross_from_dots, learn_diamond_expand, learn_arrow_ray,
            learn_lshape_concavity, learn_conditional_span_fill,
            learn_ushape_gap_drop, learn_template_stamp_at_marker
        )
        for p0_fn in [learn_span_fill_rule, learn_conditional_span_fill,
                       learn_color_key_swap,
                       learn_template_row_stamp, learn_grid_gap_fill,
                       learn_single_cell_paint, learn_connect_dot_pairs,
                       learn_cross_from_dots, learn_diamond_expand,
                       learn_arrow_ray, learn_lshape_concavity,
                       learn_ushape_gap_drop, learn_template_stamp_at_marker]:
            result = SmartARCSolverV2._try_learner(p0_fn, train, test_input, loo=True)
            if result is not None:
                self._last_strategy = f"phase0:{p0_fn.__name__}"
                return result

        # ── Phase 1: 106 hand-coded strategies (cross-validated) ──────────────
        result = super().solve(train, test_input)
        if result is not None:
            # _last_strategy set by SmartARCSolver.solve() below
            return result

        # ── Phase 2: Learned rule families (LOO validated) ────────────────────
        from klomboagi.reasoning.arc_cell_rules import (
            learn_cell_rule, learn_span_fill_rule, learn_color_key_swap,
            learn_template_row_stamp, learn_grid_gap_fill, learn_single_cell_paint
        )
        from klomboagi.reasoning.arc_object_rules import learn_object_rule
        from klomboagi.reasoning.arc_pattern_match import learn_pattern_rule
        from klomboagi.reasoning.arc_extraction import learn_extraction_rule
        from klomboagi.reasoning.arc_grid_ops import learn_grid_rule
        from klomboagi.reasoning.arc_region import learn_region_rule
        from klomboagi.reasoning.arc_gravity import learn_gravity_rule
        from klomboagi.reasoning.arc_advanced import learn_advanced_rule
        from klomboagi.reasoning.arc_tiling import learn_tiling_rule
        from klomboagi.reasoning.arc_context_rules import learn_context_rule
        from klomboagi.reasoning.arc_ranking import learn_ranking_rule
        from klomboagi.reasoning.arc_legend import learn_legend_rule
        from klomboagi.reasoning.arc_compose import learn_compose_rule
        from klomboagi.reasoning.arc_multiobj import learn_multiobj_rule

        # Order: fast → slow, specific → general
        # LOO=True only for pattern_match which is prone to overfitting
        #
        # Family name mapping for classifier-guided reordering
        family_learner_map = {
            "cell_rule": (learn_cell_rule, False),
            "region": (learn_region_rule, False),
            "context": (learn_context_rule, False),
            "ranking": (learn_ranking_rule, False),
            "legend": (learn_legend_rule, False),
            "compose": (learn_compose_rule, False),
            "gravity": (learn_gravity_rule, False),
            "tiling": (learn_tiling_rule, False),
            "object_rule": (learn_object_rule, False),
            "multiobj": (learn_multiobj_rule, False),
            "extraction": (learn_extraction_rule, False),
            "grid_ops": (learn_grid_rule, False),
            "advanced": (learn_advanced_rule, False),
            "pattern": (learn_pattern_rule, True),
        }

        default_order = [
            "cell_rule", "region", "context", "ranking", "legend",
            "compose", "gravity", "tiling", "object_rule", "multiobj",
            "extraction", "grid_ops", "advanced", "pattern",
        ]

        # ── Classifier-guided reordering ──
        # If classifier predicts a specific family, try it first
        try:
            from klomboagi.reasoning.arc_classifier import predict_family_proba
            ranked = predict_family_proba(train)
            if ranked:
                # Get top predicted families (non-"none", probability > 0.1)
                priority = [fam for fam, prob in ranked
                            if fam != "none" and fam in family_learner_map
                            and prob > 0.1]
                if priority:
                    # Reorder: priority families first, then rest in default order
                    rest = [f for f in default_order if f not in priority]
                    ordered = priority + rest
                else:
                    ordered = default_order
            else:
                ordered = default_order
        except Exception:
            ordered = default_order

        # Also include span_fill variants in the loop
        extra_learners = [
            (learn_span_fill_rule, False),
            (learn_color_key_swap, False),
            (learn_template_row_stamp, False),
            (learn_grid_gap_fill, False),
            (learn_single_cell_paint, False),
        ]

        # Try extra learners first (fast, specific)
        for learn_fn, loo in extra_learners:
            result = self._try_learner(learn_fn, train, test_input, loo=loo)
            if result is not None:
                self._last_strategy = f"phase2_extra:{learn_fn.__name__}"
                return result

        # Try classifier-ordered families
        for family in ordered:
            learn_fn, loo = family_learner_map[family]
            result = self._try_learner(learn_fn, train, test_input, loo=loo)
            if result is not None:
                self._last_strategy = f"phase2:{family}"
                return result

        # DSL program synthesis (composable primitives, depth 1-3)
        from klomboagi.reasoning.arc_dsl_v2 import synthesize
        synth_result = synthesize(train, test_input, max_depth=3, timeout_ms=2000)
        if synth_result is not None:
            self._last_strategy = "dsl"
            return synth_result

        # ── Phase 2b: Object-level compositional solver ──────────────────────
        try:
            from klomboagi.reasoning.arc_object_solver import CompositionalObjectSolver
            obj_solver = CompositionalObjectSolver()
            obj_result = obj_solver.solve(train, test_input)
            if obj_result is not None:
                self._last_strategy = "object_solver"
                return obj_result
        except Exception:
            pass

        # ── Phase 2c: Reasoning-driven solver ────────────────────────────────
        try:
            from klomboagi.reasoning.arc_reasoner import ARCReasoner
            reasoner = ARCReasoner()
            reason_result = reasoner.solve(train, test_input)
            if reason_result is not None:
                self._last_strategy = "reasoner"
                return reason_result
        except Exception:
            pass

        # ── Phase 3: V2 hand-coded strategies (cross-validated) ───────────────
        v2 = [
            self._try_slide_object_to_anchor,
            self._try_tile_complement,
            self._try_fractal_mark_tile,
            self._try_paint_shape_with_color,
            self._try_separate_and_combine,
            self._try_count_unique_colors,
            self._try_fill_holes_in_objects,
            self._try_extend_pattern_to_edge,
            self._try_draw_crosshairs,
            self._try_denoise_repeating_block,
            self._try_flood_fill_interior,
            self._try_hollow_rectangles,
            self._try_connect_same_rowcol_pairs,
            self._try_outer_corners_of_rectangle,
            self._try_most_frequent_at_center_bottom,
            self._try_ring_at_crosshair_intersection,
            self._try_fill_to_nearest_corner,
            self._try_tile_from_marked_rows,
            self._try_nine_ball_toward_six,
            self._try_chebyshev_equidistant_center,
            self._try_connect_aligned_diamonds,
            self._try_connect_diagonal_crosses,
            self._try_uniform_rows_to_five,
            self._try_erase_outside_cross_quadrant,
            self._try_fill_square_rect_interiors,
            self._try_reflect_diagonal_segment,
            self._try_fill_innermost_row_gap,
            self._try_explode_center_to_corners,
            self._try_color_holes_by_column_rank,
            self._try_fill_shape_bounding_box,
            self._try_fill_max_zero_rect,
            self._try_attract_color_pairs,
            self._try_column_height_balance,
            self._try_perpendicular_8_blocks,
            self._try_grid_complement_separator,
            self._try_ones_in_8_bbox_to_3,
            self._try_swap_colors_in_components,
            self._try_remove_isolated_cells,
            self._try_quadrant_color_map,
            self._try_shift_parallelogram_top,
            self._try_ring_rotate_3x3,
            self._try_bordered_rect_center,
            self._try_rect_corner_edge_interior,
            self._try_convert_isolated_cells,
            self._try_row_permutation,
            self._try_broadcast_direction,
            self._try_fill_interior_through_gap,
            self._try_cross_quadrant_fill,
            self._try_crossing_bars_interrupted,
            self._try_dot_to_filled_ring,
            self._try_dot_to_corner_edge_ring,
            self._try_shift_cluster_color,
            self._try_two_row_checkerboard,
            self._try_keep_majority_replace_rest,
            self._try_ring_to_cross,
            self._try_tallest_shortest_column,
            self._try_alternating_row_shift,
            self._try_diamond_center_mark,
            self._try_extend_bars_to_cross,
            self._try_l_shape_diagonal_ray,
            self._try_2x2_block_corner_markers,
            self._try_grid_room_diagonal_colors,
            self._try_fg_color_swap_lookup,
            self._try_signal_to_quadrant_block,
            self._try_component_size_coloring,
            self._try_remove_low_neighbor_cells,
            self._try_remove_row_col_sandwiched_cells,
            self._try_diagonal_alternating_color,
            self._try_8conn_equal_size_coloring,
            self._try_top2_freq_stay,
            self._try_5block_nearest_noise_color,
            self._try_complete_rect_outline_to_3,
            self._try_marker_recolor_blob,
            self._try_small_components_to_3,
            self._try_path_value_reversal,
            self._try_connect_pairs_with_8,
            self._try_solid3row_middle_alternate,
            self._try_5border_swap_interior,
            self._try_ones_expand_3x3,
            self._try_stripe_mode_fill,
            self._try_largest_solid_rect_only,
            self._try_concentric_ring_rotation,
            self._try_fill_1frame_by_interior_parity,
            self._try_tile_pattern_extend_recolor,
            self._try_key_template_rotate_fill,
            self._try_odd_patch_out,
            self._try_middle_col_only,
            self._try_bar_bottom_half_to_8,
            self._try_color_stripe_list,
            self._try_mirror_horizontal_append,
            self._try_mirror_vertical_append,
            self._try_zoom_outer_double,
            self._try_lr_symmetric_color,
            self._try_grid_diagonal_indicator,
            self._try_count_ones_fill_order,
            self._try_nonzero_to_checkerblock,
            self._try_four_quadrant_rotations,
            self._try_row_rev_row_tile,
            self._try_border_pad_extend,
            self._try_diagonal_reveal,
            self._try_swap_ends_keep_mid,
            self._try_row_tile_mirror_tile,
            self._try_rev_row_block_mirror_tile,
            self._try_dominant_band_color,
            self._try_pair_bbox_fill,
            self._try_rect_interior_fill2,
            self._try_fill_bbox_holes_7,
            self._try_grid_split_unique_quadrant,
            self._try_swap_colors_per_shape,
            self._try_nor_pattern_match,
            self._try_rect_concentric_rings,
            self._try_gravity_sink_1_through_5,
            self._try_unique_cell_box,
            self._try_band_zero_extend,
            self._try_rot180,
            self._try_vflip_then_original,
            self._try_or_pattern_match,
            self._try_merge_3_2_to_8,
            self._try_find_hollow_shape_color,
            self._try_row_color_fill_5,
            self._try_reflect_bottom_to_top,
            self._try_rect_hollow_out,
            self._try_max_zero_rect_fill6,
            self._try_extend_pattern_rows_10,
            self._try_cross_to_4fold,
            self._try_cell_grid_count_nonbg,
            self._try_sep_col_both_zero_to_8,
            self._try_shape_cross_pattern,
            self._try_original_then_vflip,
            self._try_dedup_horiz_tile,
            self._try_fill_border_8,
            self._try_lower_to_upper_triangle,
            self._try_extract_symmetric_shape,
            self._try_corner_color_quadrant,
            self._try_mirror_top_rows_to_bottom,
            self._try_color_col_diagonal_and_bottom,
            self._try_dedup_consecutive_rows_cols,
            self._try_split_insert_9_sep,
            self._try_extend_row_adding_one,
            self._try_nor_top_bottom_half,
            self._try_or_left_right_half,
            self._try_extract_diamond_5x5,
            self._try_unique_quadrant,
            self._try_fill_endpoints_with_midpoint5,
            self._try_recolor_8_blobs_by_palette,
            self._try_cross_and_diagonals,
            self._try_flow_to_wall,
            self._try_row_bars_with_zone_borders,
            self._try_compact_anchor_corners,
            self._try_zone_recolor_by_identity,
            self._try_gravity_toward_separator,
            self._try_radiating_line_78,
            self._try_stamp_pattern_on_5_regions,
            self._try_key_grid_recolor,
            self._try_complete_symmetric_pattern,
            self._try_nested_rect_concentric_fill,
            self._try_project_color_to_border,
            self._try_flood_fill_between_pairs,
            self._try_border_extraction,
            self._try_majority_color_per_row_col,
            self._try_object_bbox_self_fill,
            self._try_reflect_objects_across_axis,
            self._try_sort_colors_to_regions,
            self._try_separator_grid_dimensions,
            self._try_quadrant_color_map,
            self._try_assemble_around_fives,
            self._try_blobs_sorted_by_size,
            self._try_extract_uniform_cells_from_sep_grid,
            self._try_pyramid_inner_diagonal_extend,
            self._try_diagonal_color_markers,
            self._try_move_toward_target,
            self._try_midpoint_plus,
            self._try_color_swap_mapping,
            self._try_shift_grid_down_one,
            self._try_and_halves_sep,
            self._try_zone_extract_single_value,
            self._try_compress_border_to_3x3,
            self._try_select_block_most_minority,
            self._try_checkerboard_from_rows,
            self._try_color_bands_in_order,
            self._try_rotate_line_at_pivot,
            self._try_tile_along_uniform_edge,
            self._try_rotate_180_around_pivot_color,
            self._try_xor_halves,
            self._try_nor_left_right_half,
            self._try_largest_rect_color_2x2,
            self._try_triangular_number_repeat,
            self._try_reflect_shape_by_pointer,
            self._try_extend_tips_diagonally,
            self._try_cascade_complement_shrink,
            self._try_extract_corner_2x2_by_parity,
            self._try_stamp_template_per_dot,
            self._try_scale_with_block_sub,
            self._try_gravity_drop_in_columns,
            self._try_extract_block_at_marker,
            self._try_border_extend_with_corners,
            self._try_block_dominant_color_ignoring_marker,
            self._try_count_blocks_to_checkerboard,
            self._try_max_count_colors_tiled,
            self._try_concentric_frames_by_size,
            self._try_shift_objects_right_by_width,
            self._try_drop_objects_into_ground_gaps,
            self._try_parity_fill_above_cell,
            self._try_separator_below_above_ratio,
            self._try_four_rotation_tiling,
            self._try_drop_shape_to_separator,
            self._try_recolor_by_nearest_border,
            self._try_crosshairs_fill_rect,
            self._try_draw_diagonal_on_zeros,
            self._try_fill_enclosed_zero_regions,
            self._try_or_halves_recolor,
            self._try_or_halves_with_separator_col,
            self._try_draw_x_from_dot,
            self._try_draw_border_from_dot,
            self._try_staircase_from_dot,
            self._try_bouncing_path_from_dot,
            self._try_L_path_between_dots,
            self._try_alternating_cols_from_2dots,
            self._try_count_8s_fill_matching_color,
            self._try_simple_tile_nxm,
            self._try_extract_5border_with_padding,
            self._try_extract_unique_or_largest_component,
            self._try_shift_each_shape_by_width,
            self._try_reverse_concentric_rings,
            self._try_key_shape_recolors_main,
            self._try_rectangles_position_grid,
            self._try_line_from_markers_recolor_rects,
            self._try_fill_columns_with_terminator,
            self._try_grid_cell_fill_by_corner_marker,
            self._try_u_shape_fill_or_bottom_line,
            self._try_enclosed_3x3_hollow,
            self._try_box_with_gap_fill_and_extend,
            self._try_majority_pixel_of_shapes,
            self._try_overlay_bordered_rect_interiors,
            self._try_xor_halves,
            self._try_shape_colored_by_key_columns,
            self._try_rect_count_staircase,
            self._try_patch_into_5rect_hole,
            self._try_stack_3_panels_priority,
            self._try_stack_4_panels_priority,
            self._try_tile_shape_by_marker_count,
            self._try_extract_shape_with_8_marker,
            self._try_merge_complementary_shapes,
            self._try_grid_cell_marker_count_threshold,
            self._try_most_common_cross_arm_color,
        ]
        for s in v2:
            try:
                r = s(train, test_input)
                if r is not None and self._cross_validate(s, train):
                    self._last_strategy = f"phase3:{s.__name__}"
                    return r
            except:
                continue

        # ── Phase 3.5: Lookup strategies (skip cross-validation) ─────────────
        for s in [self._try_shape_pattern_to_value]:
            try:
                r = s(train, test_input)
                if r is not None:
                    self._last_strategy = f"phase3_lookup:{s.__name__}"
                    return r
            except:
                pass

        # ── Phase 4: LLM refinement loop (verified) ───────────────────────────
        from klomboagi.reasoning.arc_llm_solver import solve_with_llm
        llm_result = solve_with_llm(train, test_input, max_attempts=5)
        if llm_result is not None:
            self._last_strategy = "llm"
            return llm_result

        # ── Phase 5: Disabled — unvalidated fallback produces too many wrong
        # results (pattern_rule alone fires 176 times incorrectly without LOO).
        # Return None to avoid polluting results.
        return None

    def _try_paint_shape_with_color(self, train, test_input):
        """A shape outline exists, fill interior with a specific color found elsewhere."""
        from collections import Counter
        
        for ex in train:
            if len(ex["input"]) != len(ex["output"]) or len(ex["input"][0]) != len(ex["output"][0]):
                return None
        
        all_v = []
        for ex in train:
            for row in ex["input"]: all_v.extend(row)
        bg = Counter(all_v).most_common(1)[0][0]
        
        # Find: which bg cells become non-bg in output?
        # And what determines their color?
        for ex in train:
            inp, out = ex["input"], ex["output"]
            rows, cols = len(inp), len(inp[0])
            
            # Find filled cells
            filled = {}
            for r in range(rows):
                for c in range(cols):
                    if inp[r][c] == bg and out[r][c] != bg:
                        filled[(r,c)] = out[r][c]
            
            if not filled:
                return None
            
            # Check: are filled cells enclosed by a single non-bg color?
            fill_colors = set(filled.values())
            if len(fill_colors) != 1:
                return None  # Multiple fill colors — too complex for this strategy
        
        # Simple case: fill enclosed bg regions with the enclosing color
        def fill_enclosed(grid, bg_val):
            rows, cols = len(grid), len(grid[0])
            result = [row[:] for row in grid]
            visited = [[False]*cols for _ in range(rows)]
            
            for r in range(rows):
                for c in range(cols):
                    if not visited[r][c] and grid[r][c] == bg_val:
                        region = []
                        queue = [(r,c)]
                        touches_border = False
                        adj_colors = Counter()
                        
                        while queue:
                            cr, cc = queue.pop(0)
                            if visited[cr][cc]: continue
                            if grid[cr][cc] != bg_val:
                                adj_colors[grid[cr][cc]] += 1
                                continue
                            visited[cr][cc] = True
                            region.append((cr,cc))
                            if cr==0 or cr==rows-1 or cc==0 or cc==cols-1:
                                touches_border = True
                            for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                                nr,nc = cr+dr, cc+dc
                                if 0<=nr<rows and 0<=nc<cols and not visited[nr][nc]:
                                    queue.append((nr,nc))
                        
                        if not touches_border and adj_colors:
                            fill_color = adj_colors.most_common(1)[0][0]
                            for rr,cc in region:
                                result[rr][cc] = fill_color
            return result
        
        if all(fill_enclosed(ex["input"], bg) == ex["output"] for ex in train):
            return fill_enclosed(test_input, bg)
        return None

    def _try_grid_diff_overlay(self, train, test_input):
        """Output = input with changes from a second 'layer' overlaid."""
        # Check if input has a divider line splitting it into two regions
        from collections import Counter
        
        for ex in train:
            inp = ex["input"]
            rows, cols = len(inp), len(inp[0])
            
            # Check for horizontal divider
            for r in range(1, rows-1):
                if len(set(inp[r])) == 1 and inp[r][0] != 0:
                    # Row r is a divider
                    top = [row[:] for row in inp[:r]]
                    bot = [row[:] for row in inp[r+1:]]
                    if len(top) == len(bot) and len(top[0]) == len(bot[0]):
                        pass  # Could be an overlay puzzle
            
            # Check for vertical divider
            for c in range(1, cols-1):
                col_vals = [inp[r][c] for r in range(rows)]
                if len(set(col_vals)) == 1 and col_vals[0] != 0:
                    pass
        
        return None  # Complex — skip detailed impl for now

    def _try_stamp_template(self, train, test_input):
        """A small template is stamped/copied to locations marked by a specific color."""
        from collections import Counter
        
        for ex in train:
            if len(ex["input"]) != len(ex["output"]) or len(ex["input"][0]) != len(ex["output"][0]):
                return None
        
        all_v = []
        for ex in train:
            for row in ex["input"]: all_v.extend(row)
        bg = Counter(all_v).most_common(1)[0][0]
        
        # Find non-bg colors
        colors = set(all_v) - {bg}
        if len(colors) < 2:
            return None
        
        return None  # Complex

    def _try_color_at_grid_intersections(self, train, test_input):
        """Grid lines exist, color cells at intersections."""
        return None  # Complex

    def _try_separate_and_combine(self, train, test_input):
        """Input has separator (line of one color), output combines the two halves."""
        from collections import Counter
        
        all_v = []
        for ex in train:
            for row in ex["input"]: all_v.extend(row)
        bg = Counter(all_v).most_common(1)[0][0]
        
        # Check for horizontal separator
        def find_h_separator(grid):
            rows = len(grid)
            for r in range(rows):
                vals = set(grid[r])
                if len(vals) == 1 and list(vals)[0] != bg:
                    return r, list(vals)[0]
            return None, None
        
        def find_v_separator(grid):
            rows, cols = len(grid), len(grid[0])
            for c in range(cols):
                vals = set(grid[r][c] for r in range(rows))
                if len(vals) == 1 and list(vals)[0] != bg:
                    return c, list(vals)[0]
            return None, None
        
        # Check horizontal separator + OR combine
        sep_r, sep_c_val = find_h_separator(train[0]["input"])
        if sep_r is not None:
            def combine_h_or(grid, bg_val):
                r, _ = find_h_separator(grid)
                if r is None: return None
                top = grid[:r]
                bot = grid[r+1:]
                if len(top) != len(bot): return None
                if len(top[0]) != len(bot[0]): return None
                result = []
                for i in range(len(top)):
                    row = []
                    for j in range(len(top[i])):
                        a, b = top[i][j], bot[i][j]
                        if a != bg_val: row.append(a)
                        elif b != bg_val: row.append(b)
                        else: row.append(bg_val)
                    result.append(row)
                return result
            
            def combine_h_xor(grid, bg_val):
                r, _ = find_h_separator(grid)
                if r is None: return None
                top = grid[:r]
                bot = grid[r+1:]
                if len(top) != len(bot) or len(top[0]) != len(bot[0]): return None
                result = []
                for i in range(len(top)):
                    row = []
                    for j in range(len(top[i])):
                        a, b = top[i][j], bot[i][j]
                        if a != bg_val and b == bg_val: row.append(a)
                        elif a == bg_val and b != bg_val: row.append(b)
                        else: row.append(bg_val)
                    result.append(row)
                return result
            
            def combine_h_and(grid, bg_val):
                r, _ = find_h_separator(grid)
                if r is None: return None
                top = grid[:r]
                bot = grid[r+1:]
                if len(top) != len(bot) or len(top[0]) != len(bot[0]): return None
                result = []
                for i in range(len(top)):
                    row = []
                    for j in range(len(top[i])):
                        a, b = top[i][j], bot[i][j]
                        if a != bg_val and b != bg_val: row.append(a)
                        else: row.append(bg_val)
                    result.append(row)
                return result
            
            for fn in [combine_h_or, combine_h_xor, combine_h_and]:
                try:
                    if all(fn(ex["input"], bg) == ex["output"] for ex in train):
                        return fn(test_input, bg)
                except:
                    continue
        
        # Check vertical separator
        sep_c, _ = find_v_separator(train[0]["input"])
        if sep_c is not None:
            def combine_v_or(grid, bg_val):
                c, _ = find_v_separator(grid)
                if c is None: return None
                rows = len(grid)
                left = [grid[r][:c] for r in range(rows)]
                right = [grid[r][c+1:] for r in range(rows)]
                if len(left[0]) != len(right[0]): return None
                result = []
                for r in range(rows):
                    row = []
                    for j in range(len(left[r])):
                        a, b = left[r][j], right[r][j]
                        if a != bg_val: row.append(a)
                        elif b != bg_val: row.append(b)
                        else: row.append(bg_val)
                    result.append(row)
                return result
            
            def combine_v_xor(grid, bg_val):
                c, _ = find_v_separator(grid)
                if c is None: return None
                rows = len(grid)
                left = [grid[r][:c] for r in range(rows)]
                right = [grid[r][c+1:] for r in range(rows)]
                if len(left[0]) != len(right[0]): return None
                result = []
                for r in range(rows):
                    row = []
                    for j in range(len(left[r])):
                        a, b = left[r][j], right[r][j]
                        if a != bg_val and b == bg_val: row.append(a)
                        elif a == bg_val and b != bg_val: row.append(b)
                        else: row.append(bg_val)
                    result.append(row)
                return result
            
            def combine_v_and(grid, bg_val):
                c, _ = find_v_separator(grid)
                if c is None: return None
                rows = len(grid)
                left = [grid[r][:c] for r in range(rows)]
                right = [grid[r][c+1:] for r in range(rows)]
                if len(left[0]) != len(right[0]): return None
                result = []
                for r in range(rows):
                    row = []
                    for j in range(len(left[r])):
                        a, b = left[r][j], right[r][j]
                        if a != bg_val and b != bg_val: row.append(a)
                        else: row.append(bg_val)
                    result.append(row)
                return result
            
            for fn in [combine_v_or, combine_v_xor, combine_v_and]:
                try:
                    if all(fn(ex["input"], bg) == ex["output"] for ex in train):
                        return fn(test_input, bg)
                except:
                    continue
        
        return None

    def _try_replicate_subgrid(self, train, test_input):
        """Find a small pattern in input and replicate it to fill the output."""
        return None  # Complex

    def _try_count_unique_colors(self, train, test_input):
        """Output 1x1 = number of unique non-bg colors."""
        from collections import Counter
        
        for ex in train:
            if len(ex["output"]) != 1 or len(ex["output"][0]) != 1:
                return None
        
        all_v = []
        for ex in train:
            for row in ex["input"]: all_v.extend(row)
        bg = Counter(all_v).most_common(1)[0][0]
        
        if all(ex["output"][0][0] == len(set(v for row in ex["input"] for v in row) - {bg}) for ex in train):
            count = len(set(v for row in test_input for v in row) - {bg})
            return [[count]]
        return None

    def _try_border_color_determines(self, train, test_input):
        """Border color of a region determines what fills it."""
        return None  # Complex

    def _try_fill_holes_in_objects(self, train, test_input):
        """Fill 'holes' (bg pixels surrounded by same color) in objects."""
        from collections import Counter
        
        for ex in train:
            if len(ex["input"]) != len(ex["output"]) or len(ex["input"][0]) != len(ex["output"][0]):
                return None
        
        all_v = []
        for ex in train:
            for row in ex["input"]: all_v.extend(row)
        bg = Counter(all_v).most_common(1)[0][0]
        
        def fill_holes(grid, bg_val):
            rows, cols = len(grid), len(grid[0])
            result = [row[:] for row in grid]
            
            for r in range(1, rows-1):
                for c in range(1, cols-1):
                    if grid[r][c] == bg_val:
                        neighbors = []
                        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                            nr, nc = r+dr, c+dc
                            if 0<=nr<rows and 0<=nc<cols and grid[nr][nc] != bg_val:
                                neighbors.append(grid[nr][nc])
                        if len(neighbors) >= 3:
                            mc = Counter(neighbors).most_common(1)[0][0]
                            result[r][c] = mc
            return result
        
        if all(fill_holes(ex["input"], bg) == ex["output"] for ex in train):
            return fill_holes(test_input, bg)
        
        # Try iterative fill
        def fill_holes_iter(grid, bg_val, max_iter=5):
            result = [row[:] for row in grid]
            for _ in range(max_iter):
                new_result = fill_holes(result, bg_val)
                if new_result == result:
                    break
                result = new_result
            return result
        
        if all(fill_holes_iter(ex["input"], bg) == ex["output"] for ex in train):
            return fill_holes_iter(test_input, bg)
        
        return None

    def _try_extend_pattern_to_edge(self, train, test_input):
        """Extend existing lines/patterns to the grid edges."""
        from collections import Counter
        
        for ex in train:
            if len(ex["input"]) != len(ex["output"]) or len(ex["input"][0]) != len(ex["output"][0]):
                return None
        
        all_v = []
        for ex in train:
            for row in ex["input"]: all_v.extend(row)
        bg = Counter(all_v).most_common(1)[0][0]
        
        def extend_all(grid, bg_val):
            rows, cols = len(grid), len(grid[0])
            result = [row[:] for row in grid]
            
            # Extend horizontal: if a row has non-bg cells, fill the whole row
            for r in range(rows):
                non_bg = [c for c in range(cols) if grid[r][c] != bg_val]
                if len(non_bg) >= 2:
                    color = grid[r][non_bg[0]]
                    if all(grid[r][c] == color for c in non_bg):
                        for c in range(cols):
                            result[r][c] = color
            
            # Extend vertical
            for c in range(cols):
                non_bg = [r for r in range(rows) if grid[r][c] != bg_val]
                if len(non_bg) >= 2:
                    color = grid[non_bg[0]][c]
                    if all(grid[r][c] == color for r in non_bg):
                        for r in range(rows):
                            result[r][c] = color
            return result
        
        if all(extend_all(ex["input"], bg) == ex["output"] for ex in train):
            return extend_all(test_input, bg)
        return None

    def _try_draw_crosshairs(self, train, test_input):
        """
        Two markers define a cross-hair: draw vertical line through one,
        horizontal through the paired marker's row, connecting them.
        Pattern: find pairs of same-color markers, draw lines through a shared anchor.
        """
        from collections import Counter, defaultdict

        for ex in train:
            inp, out = ex["input"], ex["output"]
            if len(inp) != len(out) or len(inp[0]) != len(out[0]):
                return None

        def find_markers(grid, bg):
            """Find isolated non-bg cells (markers)."""
            rows, cols = len(grid), len(grid[0])
            markers = defaultdict(list)
            for r in range(rows):
                for c in range(cols):
                    if grid[r][c] != bg:
                        markers[grid[r][c]].append((r, c))
            return markers

        all_v = []
        for ex in train:
            for row in ex["input"]: all_v.extend(row)
        bg = Counter(all_v).most_common(1)[0][0]

        def apply_crosshairs(grid, bg_val):
            rows, cols = len(grid), len(grid[0])
            result = [row[:] for row in grid]
            markers = find_markers(grid, bg_val)

            for color, positions in markers.items():
                if len(positions) == 2:
                    (r1, c1), (r2, c2) = positions
                    if r1 == r2:
                        # Same row — draw horizontal line between them
                        for c in range(min(c1, c2), max(c1, c2) + 1):
                            result[r1][c] = color
                    elif c1 == c2:
                        # Same column — draw vertical line between them
                        for r in range(min(r1, r2), max(r1, r2) + 1):
                            result[r][c1] = color
                    else:
                        # Different row and col — draw L-shape or cross
                        # Draw vertical through c1 from r1 toward r2
                        for r in range(min(r1, r2), max(r1, r2) + 1):
                            result[r][c1] = color
                        # Draw horizontal through r2 from c1 toward c2
                        for c in range(min(c1, c2), max(c1, c2) + 1):
                            result[r2][c] = color
            return result

        if all(apply_crosshairs(ex["input"], bg) == ex["output"] for ex in train):
            return apply_crosshairs(test_input, bg)
        return None

    def _try_denoise_repeating_block(self, train, test_input):
        """
        Grid has a repeating block pattern with some cells corrupted.
        Output = clean version where each block matches the majority pattern.
        """
        from collections import Counter

        for ex in train:
            if len(ex["input"]) != len(ex["output"]) or len(ex["input"][0]) != len(ex["output"][0]):
                return None

        # Try to detect block size from grid structure (look for separator lines)
        def find_block_size(grid):
            rows, cols = len(grid), len(grid[0])
            # Check for repeating row patterns
            for bh in range(2, rows // 2 + 1):
                if rows % bh == 0:
                    # Check if blocks repeat
                    blocks = []
                    for i in range(0, rows, bh):
                        blocks.append(tuple(tuple(row) for row in grid[i:i+bh]))
                    if len(set(blocks)) <= len(blocks) // 2:  # Some blocks match
                        return bh
            return None

        # Try to find the "clean" block pattern via majority voting
        def denoise(grid, block_h, block_w):
            rows, cols = len(grid), len(grid[0])
            result = [row[:] for row in grid]

            # Collect all blocks
            blocks = []
            for br in range(0, rows, block_h):
                for bc in range(0, cols, block_w):
                    block = []
                    for r in range(br, min(br + block_h, rows)):
                        row = []
                        for c in range(bc, min(bc + block_w, cols)):
                            row.append(grid[r][c])
                        block.append(tuple(row))
                    blocks.append((br, bc, tuple(block)))

            # Majority vote per cell position within the block
            clean_block = []
            for local_r in range(block_h):
                row = []
                for local_c in range(block_w):
                    votes = Counter()
                    for br, bc, block in blocks:
                        if local_r < len(block) and local_c < len(block[local_r]):
                            votes[block[local_r][local_c]] += 1
                    row.append(votes.most_common(1)[0][0])
                clean_block.append(row)

            # Apply clean block everywhere
            for br in range(0, rows, block_h):
                for bc in range(0, cols, block_w):
                    for lr in range(block_h):
                        for lc in range(block_w):
                            r, c = br + lr, bc + lc
                            if r < rows and c < cols:
                                result[r][c] = clean_block[lr][lc]
            return result

        # Try various block sizes
        for bh in range(2, min(len(test_input), 12)):
            for bw in range(2, min(len(test_input[0]), 12)):
                if len(test_input) % bh == 0 and len(test_input[0]) % bw == 0:
                    if all(denoise(ex["input"], bh, bw) == ex["output"] for ex in train):
                        return denoise(test_input, bh, bw)
        return None

    def _try_flood_fill_interior(self, train, test_input):
        """
        Shapes made of color 1 have a colored marker inside.
        Fill the interior of each shape with the marker's color.
        """
        from collections import Counter

        for ex in train:
            if len(ex["input"]) != len(ex["output"]) or len(ex["input"][0]) != len(ex["output"][0]):
                return None

        # Determine bg and wall color
        all_v = []
        for ex in train:
            for row in ex["input"]: all_v.extend(row)
        bg = Counter(all_v).most_common(1)[0][0]
        wall_candidates = Counter(all_v).most_common(3)
        wall = wall_candidates[1][0] if len(wall_candidates) > 1 else None
        if wall is None or wall == bg:
            return None

        def fill_shapes(grid, bg_val, wall_val):
            rows, cols = len(grid), len(grid[0])
            result = [row[:] for row in grid]

            # Find connected regions of bg enclosed by wall
            visited = [[False] * cols for _ in range(rows)]

            for r in range(rows):
                for c in range(cols):
                    if visited[r][c] or grid[r][c] == wall_val:
                        continue
                    # BFS to find this region
                    region = []
                    queue = [(r, c)]
                    touches_border = False
                    marker_color = None

                    while queue:
                        cr, cc = queue.pop(0)
                        if cr < 0 or cr >= rows or cc < 0 or cc >= cols:
                            touches_border = True
                            continue
                        if visited[cr][cc] or grid[cr][cc] == wall_val:
                            continue
                        visited[cr][cc] = True
                        region.append((cr, cc))
                        if grid[cr][cc] != bg_val and grid[cr][cc] != wall_val:
                            marker_color = grid[cr][cc]
                        if cr == 0 or cr == rows - 1 or cc == 0 or cc == cols - 1:
                            touches_border = True
                        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            queue.append((cr + dr, cc + dc))

                    if not touches_border and marker_color is not None:
                        for rr, cc in region:
                            if grid[rr][cc] == bg_val:
                                result[rr][cc] = marker_color

            return result

        if all(fill_shapes(ex["input"], bg, wall) == ex["output"] for ex in train):
            return fill_shapes(test_input, bg, wall)
        return None

    def _try_slide_object_to_anchor(self, train, test_input):
        """One object slides toward a fixed 'anchor' object until adjacent."""
        from collections import Counter

        def get_objects(grid, bg):
            """Return dict: color -> set of (r,c) cells."""
            rows, cols = len(grid), len(grid[0])
            objs = {}
            for r in range(rows):
                for c in range(cols):
                    v = grid[r][c]
                    if v != bg:
                        if v not in objs:
                            objs[v] = []
                        objs[v].append((r, c))
            return objs

        def bbox(cells):
            rs = [r for r,_ in cells]
            cs = [c for _,c in cells]
            return min(rs), min(cs), max(rs), max(cs)

        def slide_to_adjacent(grid, mover_color, anchor_color, bg):
            """Slide mover_color cells toward anchor_color until touching."""
            rows, cols = len(grid), len(grid[0])
            result = [row[:] for row in grid]

            mover = [(r,c) for r in range(rows) for c in range(cols) if grid[r][c] == mover_color]
            anchor = [(r,c) for r in range(rows) for c in range(cols) if grid[r][c] == anchor_color]
            if not mover or not anchor:
                return None

            mr0, mc0, mr1, mc1 = bbox(mover)
            ar0, ac0, ar1, ac1 = bbox(anchor)

            # Gap in each dimension (positive = mover before anchor, negative = after, 0 = overlap)
            v_dist = (ar0 - mr1) if ar0 > mr1 else (-(mr0 - ar1) if mr0 > ar1 else 0)
            h_dist = (ac0 - mc1) if ac0 > mc1 else (-(mc0 - ac1) if mc0 > ac1 else 0)

            # Determine direction: slide along the axis with larger gap
            if abs(v_dist) >= abs(h_dist) and v_dist != 0:
                if v_dist > 0:
                    # Anchor is below mover — slide down
                    shift_r, shift_c = ar0 - mr1 - 1, 0
                else:
                    # Anchor is above mover — slide up
                    shift_r, shift_c = ar1 + 1 - mr0, 0
            elif h_dist != 0:
                if h_dist > 0:
                    # Anchor is right of mover — slide right
                    shift_r, shift_c = 0, ac0 - mc1 - 1
                else:
                    # Anchor is left of mover — slide left
                    shift_r, shift_c = 0, ac1 + 1 - mc0
            else:
                return None  # Already adjacent or overlapping

            if shift_r == 0 and shift_c == 0:
                return None

            # Move mover cells
            for r, c in mover:
                result[r][c] = bg
            for r, c in mover:
                nr, nc = r + shift_r, c + shift_c
                if 0 <= nr < rows and 0 <= nc < cols:
                    result[nr][nc] = mover_color
                else:
                    return None
            return result

        # Find consistent anchor (doesn't move) and mover (slides) colors
        if not train:
            return None
        all_v = []
        for ex in train:
            for row in ex["input"]: all_v.extend(row)
        bg = Counter(all_v).most_common(1)[0][0]

        # Find colors that move vs stay
        colors = sorted(set(all_v) - {bg})
        if len(colors) != 2:
            return None  # Only handle 2-object case

        c1, c2 = colors[0], colors[1]

        # Check which stays (anchor) and which moves
        for anchor_c, mover_c in [(c1, c2), (c2, c1)]:
            def apply_slide(inp):
                return slide_to_adjacent(inp, mover_c, anchor_c, bg)

            valid = True
            for ex in train:
                result = apply_slide(ex["input"])
                if result != ex["output"]:
                    valid = False
                    break
            if valid:
                return apply_slide(test_input)

        return None

    def _try_tile_complement(self, train, test_input):
        """Output is the complement (bg<->fg) of the input, tiled k×k times."""
        from collections import Counter

        def find_bg(grid):
            vals = [v for row in grid for v in row]
            return Counter(vals).most_common(1)[0][0]

        factors = set()
        for ex in train:
            ir, ic = len(ex["input"]), len(ex["input"][0])
            or_, oc = len(ex["output"]), len(ex["output"][0])
            if or_ % ir != 0 or oc % ic != 0:
                return None
            fr, fc = or_ // ir, oc // ic
            if fr != fc:
                return None
            factors.add(fr)

        if len(factors) != 1:
            return None
        k = list(factors)[0]
        if k <= 1:
            return None

        for ex in train:
            inp = ex["input"]
            out = ex["output"]
            ir, ic = len(inp), len(inp[0])
            bg = find_bg(inp)
            # Find the only other color
            non_bg = sorted(set(v for row in inp for v in row) - {bg})
            if len(non_bg) != 1:
                return None
            fg = non_bg[0]
            # Build complement
            comp = [[fg if inp[r][c] == bg else bg for c in range(ic)] for r in range(ir)]
            # Check all tiles in output are comp
            for ti in range(k):
                for tj in range(k):
                    tile = [out[ti*ir+r][tj*ic:tj*ic+ic] for r in range(ir)]
                    if tile != comp:
                        return None

        # Apply to test
        ir, ic = len(test_input), len(test_input[0])
        bg = find_bg(test_input)
        non_bg = sorted(set(v for row in test_input for v in row) - {bg})
        if len(non_bg) != 1:
            return None
        fg = non_bg[0]
        comp = [[fg if test_input[r][c] == bg else bg for c in range(ic)] for r in range(ir)]
        result = []
        for ti in range(k):
            for r in range(ir):
                result.append(comp[r] * k)
        return result

    def _try_fractal_mark_tile(self, train, test_input):
        """Output tiles input at positions where input == mark_color; 0 elsewhere.

        mark_color is the most common non-background color in the input.
        """
        from collections import Counter

        def find_bg(grid):
            vals = [v for row in grid for v in row]
            c = Counter(vals)
            return c.most_common(1)[0][0]

        factors = set()
        for ex in train:
            ir, ic = len(ex["input"]), len(ex["input"][0])
            or_, oc = len(ex["output"]), len(ex["output"][0])
            if ir != ic:
                return None  # Only square inputs
            if or_ != ir * ir or oc != ic * ic:
                return None
            factors.add(ir)

        if len(factors) != 1:
            return None
        N = list(factors)[0]

        for ex in train:
            inp = ex["input"]
            out = ex["output"]
            bg = 0

            # Find mark color: most common non-bg value
            vals = [v for row in inp for v in row if v != bg]
            if not vals:
                return None
            mark = Counter(vals).most_common(1)[0][0]

            # Verify: tile at (i,j) is inp if inp[i,j]==mark, else all bg
            valid = True
            for i in range(N):
                for j in range(N):
                    tile = [out[i*N+r][j*N:j*N+N] for r in range(N)]
                    if inp[i][j] == mark:
                        if tile != inp:
                            valid = False; break
                    else:
                        if any(tile[r][c] != bg for r in range(N) for c in range(N)):
                            valid = False; break
                if not valid:
                    break
            if not valid:
                return None

        # Apply to test
        N = len(test_input)
        if N != len(test_input[0]):
            return None
        bg = 0
        vals = [v for row in test_input for v in row if v != bg]
        if not vals:
            return None
        mark = Counter(vals).most_common(1)[0][0]

        result = [[bg] * (N * N) for _ in range(N * N)]
        for i in range(N):
            for j in range(N):
                if test_input[i][j] == mark:
                    for r in range(N):
                        for c in range(N):
                            result[i*N+r][j*N+c] = test_input[r][c]
        return result

    def _try_remove_isolated_pixels(self, train, test_input):
        """Remove non-background cells that have no 4-connected same-color neighbor."""
        def _remove_isolated(grid):
            rows, cols = len(grid), len(grid[0])
            bg = 0
            result = [row[:] for row in grid]
            for r in range(rows):
                for c in range(cols):
                    v = grid[r][c]
                    if v == bg:
                        continue
                    neighbors = [
                        grid[r-1][c] if r > 0 else bg,
                        grid[r+1][c] if r < rows-1 else bg,
                        grid[r][c-1] if c > 0 else bg,
                        grid[r][c+1] if c < cols-1 else bg,
                    ]
                    if v not in neighbors:
                        result[r][c] = bg
            return result

        # Validate on training examples
        for ex in train:
            expected = ex["output"]
            predicted = _remove_isolated(ex["input"])
            if predicted != expected:
                return None
            # Must actually change something (avoid no-op pass-through)
            if predicted == list(list(row) for row in ex["input"]):
                return None

        return _remove_isolated(test_input)

    def _try_hollow_rectangles(self, train, test_input):
        """Hollow out solid-color rectangles: keep outer border, set interior to bg."""
        from collections import deque

        def _hollow(grid):
            rows, cols = len(grid), len(grid[0])
            result = [row[:] for row in grid]
            visited = [[False] * cols for _ in range(rows)]
            bg = 0
            for sr in range(rows):
                for sc in range(cols):
                    if grid[sr][sc] == bg or visited[sr][sc]:
                        continue
                    color = grid[sr][sc]
                    comp = []
                    q = deque([(sr, sc)])
                    visited[sr][sc] = True
                    while q:
                        r, c = q.popleft()
                        comp.append((r, c))
                        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and grid[nr][nc] == color:
                                visited[nr][nc] = True
                                q.append((nr, nc))
                    r0 = min(p[0] for p in comp)
                    r1 = max(p[0] for p in comp)
                    c0 = min(p[1] for p in comp)
                    c1 = max(p[1] for p in comp)
                    if len(comp) != (r1 - r0 + 1) * (c1 - c0 + 1):
                        continue  # not a solid rectangle
                    for r in range(r0 + 1, r1):
                        for c in range(c0 + 1, c1):
                            result[r][c] = bg
            return result

        for ex in train:
            predicted = _hollow(ex["input"])
            if predicted != ex["output"]:
                return None
            if predicted == [list(row) for row in ex["input"]]:
                return None

        return _hollow(test_input)

    def _try_connect_same_rowcol_pairs(self, train, test_input):
        """Connect pairs of same-color marker cells that share a row or column with a fill color."""
        from collections import Counter

        def _find_fill_color(grid_in, grid_out, bg):
            rows, cols = len(grid_in), len(grid_out[0])
            fills = set()
            for r in range(rows):
                for c in range(cols):
                    if grid_in[r][c] == bg and grid_out[r][c] != bg:
                        fills.add(grid_out[r][c])
            return list(fills)

        def _apply(grid, marker_color, fill_color, bg):
            rows, cols = len(grid), len(grid[0])
            result = [row[:] for row in grid]
            # Group marker cells by row and column
            by_row = {}
            by_col = {}
            for r in range(rows):
                for c in range(cols):
                    if grid[r][c] == marker_color:
                        by_row.setdefault(r, []).append(c)
                        by_col.setdefault(c, []).append(r)
            # Connect pairs in same row
            for r, cols_list in by_row.items():
                if len(cols_list) == 2:
                    c1, c2 = sorted(cols_list)
                    for c in range(c1 + 1, c2):
                        if result[r][c] == bg:
                            result[r][c] = fill_color
            # Connect pairs in same column
            for c, rows_list in by_col.items():
                if len(rows_list) == 2:
                    r1, r2 = sorted(rows_list)
                    for r in range(r1 + 1, r2):
                        if result[r][c] == bg:
                            result[r][c] = fill_color
            return result

        # Determine bg, marker color, fill color from training examples
        bg = 0
        marker_color = None
        fill_color = None

        for ex in train:
            inp, out = ex["input"], ex["output"]
            rows, cols = len(inp), len(inp[0])
            if len(inp) != len(out) or len(inp[0]) != len(out[0]):
                return None
            # Find bg as most common color in input
            all_in = [inp[r][c] for r in range(rows) for c in range(cols)]
            bg_candidate = Counter(all_in).most_common(1)[0][0]
            if marker_color is None:
                bg = bg_candidate
                # Marker = non-bg color in input
                m_colors = set(v for v in all_in if v != bg)
                if len(m_colors) != 1:
                    return None
                marker_color = list(m_colors)[0]
            elif bg_candidate != bg:
                return None

            fills = _find_fill_color(inp, out, bg)
            if not fills:
                # No fills in this example — OK if there are no pairs
                # Verify the output equals the input
                if inp != out:
                    return None
                continue
            if len(fills) > 1:
                return None
            f = fills[0]
            if f == marker_color or f == bg:
                return None
            if fill_color is None:
                fill_color = f
            elif fill_color != f:
                return None

        if marker_color is None or fill_color is None:
            return None

        # Validate on all training examples
        for ex in train:
            predicted = _apply(ex["input"], marker_color, fill_color, bg)
            if predicted != ex["output"]:
                return None

        return _apply(test_input, marker_color, fill_color, bg)

    def _try_outer_corners_of_rectangle(self, train, test_input):
        """Mark the 8 outer-corner positions of each hollow rectangular border with a new color.

        A hollow rectangular border: cells form only the perimeter of a bounding box
        (top row, bottom row, left col, right col fully filled, interior is background).

        For a frame with bounding box (r0,c0)-(r1,c1):
          outer corners = (r0-1,c0),(r0,c0-1),(r0-1,c1),(r0,c1+1),
                          (r1+1,c0),(r1,c0-1),(r1+1,c1),(r1,c1+1)
        """
        from collections import Counter, deque

        def _find_rect_shapes(grid):
            """Return list of (color, r0, c0, r1, c1) for hollow frames or solid rectangles."""
            rows, cols = len(grid), len(grid[0])
            bg = Counter(grid[r][c] for r in range(rows) for c in range(cols)).most_common(1)[0][0]
            visited = [[False] * cols for _ in range(rows)]
            rects = []
            for sr in range(rows):
                for sc in range(cols):
                    if grid[sr][sc] == bg or visited[sr][sc]:
                        continue
                    color = grid[sr][sc]
                    comp = []
                    q = deque([(sr, sc)])
                    visited[sr][sc] = True
                    while q:
                        r, c = q.popleft()
                        comp.append((r, c))
                        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                            nr, nc = r+dr, c+dc
                            if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and grid[nr][nc] == color:
                                visited[nr][nc] = True
                                q.append((nr, nc))
                    r0 = min(p[0] for p in comp)
                    r1 = max(p[0] for p in comp)
                    c0 = min(p[1] for p in comp)
                    c1 = max(p[1] for p in comp)
                    if r1 == r0 and c1 == c0:
                        continue  # single cell
                    # Only square shapes get outer corners
                    if (r1 - r0) != (c1 - c0):
                        continue
                    # Check if solid rectangle
                    solid_size = (r1-r0+1) * (c1-c0+1)
                    if len(comp) == solid_size and r1 > r0 and c1 > c0:
                        rects.append((color, r0, c0, r1, c1))
                        continue
                    # Check if hollow frame (perimeter-only, ≥3×3 bounding box)
                    if r1 - r0 < 2 or c1 - c0 < 2:
                        continue
                    perimeter = set()
                    for c in range(c0, c1+1):
                        perimeter.add((r0, c)); perimeter.add((r1, c))
                    for r in range(r0+1, r1):
                        perimeter.add((r, c0)); perimeter.add((r, c1))
                    if set(comp) == perimeter:
                        rects.append((color, r0, c0, r1, c1))
            return rects, bg

        def _outer_corner_cells(r0, c0, r1, c1):
            return [
                (r0-1, c0), (r0, c0-1),   # top-left outer corner
                (r0-1, c1), (r0, c1+1),   # top-right outer corner
                (r1+1, c0), (r1, c0-1),   # bottom-left outer corner
                (r1+1, c1), (r1, c1+1),   # bottom-right outer corner
            ]

        def _apply(grid, fill_color):
            rows, cols = len(grid), len(grid[0])
            rects, bg = _find_rect_shapes(grid)
            if not rects:
                return None
            result = [row[:] for row in grid]
            for _, r0, c0, r1, c1 in rects:
                for r, c in _outer_corner_cells(r0, c0, r1, c1):
                    if 0 <= r < rows and 0 <= c < cols and result[r][c] == bg:
                        result[r][c] = fill_color
            return result

        # Determine fill_color from training examples
        fill_color = None
        for ex in train:
            inp, out = ex["input"], ex["output"]
            rows, cols = len(inp), len(inp[0])
            rects, bg = _find_rect_shapes(inp)
            if not rects:
                return None
            # Find what color was added in the output
            added = {}
            for r in range(rows):
                for c in range(cols):
                    if inp[r][c] == bg and out[r][c] != bg:
                        added[out[r][c]] = added.get(out[r][c], 0) + 1
            if not added:
                return None
            if len(added) != 1:
                return None
            fc = list(added.keys())[0]
            if fill_color is None:
                fill_color = fc
            elif fill_color != fc:
                return None
            # Verify all added cells match outer corner positions
            expected_cells = set()
            for _, r0, c0, r1, c1 in rects:
                for r, c in _outer_corner_cells(r0, c0, r1, c1):
                    if 0 <= r < rows and 0 <= c < cols and inp[r][c] == bg:
                        expected_cells.add((r, c))
            actual_cells = {(r, c) for r in range(rows) for c in range(cols)
                            if inp[r][c] == bg and out[r][c] != bg}
            if actual_cells != expected_cells:
                return None

        if fill_color is None:
            return None

        result = _apply(test_input, fill_color)
        if result is None:
            return None
        if result == [list(row) for row in test_input]:
            return None
        return result

    def _try_most_frequent_at_center_bottom(self, train, test_input):
        """Grid divided by a row of 5s: top has colored cells, bottom is zeros.
        Output: place most-frequent color from top section at (last_row, center_col).
        """
        from collections import Counter

        def _find_divider(grid):
            rows, cols = len(grid), len(grid[0])
            for r in range(rows):
                if all(int(grid[r][c]) == 5 for c in range(cols)):
                    return r
            return None

        def _apply(grid, color):
            rows, cols = len(grid), len(grid[0])
            div = _find_divider(grid)
            if div is None:
                return None
            # Bottom section must be all zeros (below divider)
            for r in range(div + 1, rows):
                for c in range(cols):
                    if int(grid[r][c]) != 0:
                        return None
            result = [list(map(int, row)) for row in grid]
            last_row = rows - 1
            center_col = (cols - 1) // 2
            if result[last_row][center_col] != 0:
                return None
            result[last_row][center_col] = color
            return result

        # Validate on all training examples (color varies per example — computed from top section)
        for ex in train:
            inp, out = ex["input"], ex["output"]
            rows, cols = len(inp), len(inp[0])
            div = _find_divider(inp)
            if div is None or div == 0:
                return None
            # Bottom must be zeros
            bottom_ok = all(int(inp[r][c]) == 0 for r in range(div+1, rows) for c in range(cols))
            if not bottom_ok:
                return None
            # Find the single change
            changes = [(r, c, int(out[r][c])) for r in range(rows) for c in range(cols)
                       if int(inp[r][c]) != int(out[r][c])]
            if len(changes) != 1:
                return None
            cr, cc, cv = changes[0]
            if cr != rows - 1 or cc != (cols - 1) // 2:
                return None
            # Verify color matches most frequent in top section
            top_cells = [int(inp[r][c]) for r in range(div) for c in range(cols) if int(inp[r][c]) != 5]
            if not top_cells:
                return None
            most_common = Counter(top_cells).most_common(1)[0][0]
            if cv != most_common:
                return None

        # For test, compute most frequent color from top section
        test_grid = [list(map(int, row)) for row in test_input]
        div = _find_divider(test_grid)
        if div is None:
            return None
        top_cells = [test_grid[r][c] for r in range(div) for c in range(len(test_grid[0])) if test_grid[r][c] != 5]
        if not top_cells:
            return None
        test_color = Counter(top_cells).most_common(1)[0][0]
        return _apply(test_grid, test_color)

    def _try_ring_at_crosshair_intersection(self, train, test_input):
        """Find two perpendicular lines (one row, one col of same/different color).
        Draw a 3x3 ring of color 4 around their intersection (center unchanged).
        """
        def _find_lines(grid):
            """Return (row_idx, col_idx) for the pair of perpendicular full lines."""
            rows, cols = len(grid), len(grid[0])
            from collections import Counter
            bg = Counter(grid[r][c] for r in range(rows) for c in range(cols)).most_common(1)[0][0]
            # Find rows and cols that are fully non-bg (a complete line)
            full_rows = [r for r in range(rows) if all(int(grid[r][c]) != bg for c in range(cols))]
            full_cols = [c for c in range(cols) if all(int(grid[r][c]) != bg for r in range(rows))]
            if len(full_rows) == 1 and len(full_cols) == 1:
                return full_rows[0], full_cols[0], bg
            return None, None, bg

        def _apply(grid, ring_color):
            rows, cols = len(grid), len(grid[0])
            row_idx, col_idx, bg = _find_lines(grid)
            if row_idx is None:
                return None
            result = [list(map(int, r)) for r in grid]
            ir, ic = row_idx, col_idx
            # Draw 3x3 ring (8 cells) around intersection (ir, ic)
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue  # leave center unchanged
                    nr, nc = ir + dr, ic + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        result[nr][nc] = ring_color
            return result

        # Validate on training examples
        ring_color = None
        for ex in train:
            inp, out = ex["input"], ex["output"]
            rows, cols = len(inp), len(inp[0])
            row_idx, col_idx, bg = _find_lines(inp)
            if row_idx is None:
                return None
            # Find what color was added (ring color)
            added = set()
            for r in range(rows):
                for c in range(cols):
                    if int(inp[r][c]) != int(out[r][c]):
                        added.add(int(out[r][c]))
            if len(added) != 1:
                return None
            rc = list(added)[0]
            if ring_color is None:
                ring_color = rc
            elif ring_color != rc:
                return None
            # Verify: expected ring cells match actual changes
            expected = set()
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = row_idx + dr, col_idx + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        expected.add((nr, nc))
            actual = {(r, c) for r in range(rows) for c in range(cols)
                      if int(inp[r][c]) != int(out[r][c])}
            if actual != expected:
                return None

        if ring_color is None:
            return None

        result = _apply(test_input, ring_color)
        if result is None or result == [list(map(int, r)) for r in test_input]:
            return None
        return result

    def _try_fill_to_nearest_corner(self, train, test_input):
        """Single non-bg pixel in a uniform grid: fill rectangle from pixel to nearest corner."""
        from collections import Counter

        def _get_pixel(grid):
            rows, cols = len(grid), len(grid[0])
            bg = Counter(int(grid[r][c]) for r in range(rows) for c in range(cols)).most_common(1)[0][0]
            pixels = [(r, c, int(grid[r][c])) for r in range(rows) for c in range(cols) if int(grid[r][c]) != bg]
            if len(pixels) != 1:
                return None, None, None, bg
            r0, c0, color = pixels[0]
            return r0, c0, color, bg

        def _apply(grid):
            rows, cols = len(grid), len(grid[0])
            r0, c0, color, bg = _get_pixel(grid)
            if r0 is None:
                return None
            # Find nearest corner by Chebyshev distance
            corners = [(0, 0), (0, cols-1), (rows-1, 0), (rows-1, cols-1)]
            nearest = min(corners, key=lambda p: max(abs(p[0]-r0), abs(p[1]-c0)))
            cr, cc = nearest
            rmin, rmax = min(r0, cr), max(r0, cr)
            cmin, cmax = min(c0, cc), max(c0, cc)
            result = [list(map(int, row)) for row in grid]
            for r in range(rmin, rmax+1):
                for c in range(cmin, cmax+1):
                    result[r][c] = color
            return result

        # Validate on training examples
        for ex in train:
            inp, out = ex["input"], ex["output"]
            r0, c0, color, bg = _get_pixel(inp)
            if r0 is None:
                return None
            # All cells in input must be bg except the one pixel
            rows, cols = len(inp), len(inp[0])
            if any(int(inp[r][c]) != bg and not (r == r0 and c == c0)
                   for r in range(rows) for c in range(cols)):
                return None
            predicted = _apply(inp)
            if predicted is None or predicted != [list(map(int, row)) for row in out]:
                return None

        result = _apply(test_input)
        if result is None or result == [list(map(int, row)) for row in test_input]:
            return None
        return result

    def _try_nine_ball_toward_six(self, train, test_input):
        """Grid divided by 0-rows/cols. A 3x3 block of 5s contains a 9 at center.
        A lone 6 appears in another cell. Move 9 to the edge of the 5s facing the 6's cell,
        and replace the 6 with 9.
        """
        def _get_dividers(grid):
            rows, cols = len(grid), len(grid[0])
            div_rows = [r for r in range(rows) if all(int(grid[r][c]) == 0 for c in range(cols))]
            div_cols = [c for c in range(cols) if all(int(grid[r][c]) == 0 for r in range(rows))]
            return div_rows, div_cols

        def _cell_idx(pos, dividers, size):
            """Return which cell (0-indexed) pos belongs to."""
            idx = 0
            for d in dividers:
                if pos > d:
                    idx += 1
                else:
                    break
            return idx

        def _apply(grid):
            rows, cols = len(grid), len(grid[0])
            div_rows, div_cols = _get_dividers(grid)
            if not div_rows or not div_cols:
                return None
            # Find center of 5s block (which contains a 9)
            center_r = center_c = None
            for r in range(1, rows - 1):
                for c in range(1, cols - 1):
                    if int(grid[r][c]) == 9:
                        # Check if surrounded by 5s
                        neighbors = [(r+dr, c+dc) for dr in (-1,0,1) for dc in (-1,0,1) if (dr,dc) != (0,0)]
                        if all(0 <= nr < rows and 0 <= nc < cols and int(grid[nr][nc]) == 5
                               for nr, nc in neighbors):
                            center_r, center_c = r, c
            if center_r is None:
                return None
            # Find the 6
            six_r = six_c = None
            for r in range(rows):
                for c in range(cols):
                    if int(grid[r][c]) == 6:
                        six_r, six_c = r, c
            if six_r is None:
                return None
            # Determine cell indices
            cell_5s_r = _cell_idx(center_r, div_rows, rows)
            cell_5s_c = _cell_idx(center_c, div_cols, cols)
            cell_6_r = _cell_idx(six_r, div_rows, rows)
            cell_6_c = _cell_idx(six_c, div_cols, cols)
            dr = 0 if cell_5s_r == cell_6_r else (1 if cell_6_r > cell_5s_r else -1)
            dc = 0 if cell_5s_c == cell_6_c else (1 if cell_6_c > cell_5s_c else -1)
            if dr == 0 and dc == 0:
                return None
            exit_r, exit_c = center_r + dr, center_c + dc
            result = [list(map(int, row)) for row in grid]
            result[center_r][center_c] = 5
            result[exit_r][exit_c] = 9
            result[six_r][six_c] = 9
            return result

        for ex in train:
            predicted = _apply(ex["input"])
            if predicted is None or predicted != [list(map(int, row)) for row in ex["output"]]:
                return None
        result = _apply(test_input)
        if result is None or result == [list(map(int, row)) for row in test_input]:
            return None
        return result

    def _try_chebyshev_equidistant_center(self, train, test_input):
        """Exactly 3 colored (non-bg) cells. Find the point equidistant (Chebyshev)
        from all 3, place 5 there. Place each color one step toward its source cell.
        """
        def _apply(grid):
            rows, cols = len(grid), len(grid[0])
            bg = 0
            cells = [(r, c, int(grid[r][c])) for r in range(rows) for c in range(cols)
                     if int(grid[r][c]) != bg]
            if len(cells) != 3:
                return None
            (r1,c1,v1), (r2,c2,v2), (r3,c3,v3) = cells
            # Find equidistant point with minimum Chebyshev distance
            center = None
            best_d = float('inf')
            for r in range(rows):
                for c in range(cols):
                    if int(grid[r][c]) != bg:
                        continue
                    d1 = max(abs(r-r1), abs(c-c1))
                    d2 = max(abs(r-r2), abs(c-c2))
                    d3 = max(abs(r-r3), abs(c-c3))
                    if d1 == d2 == d3 and d1 > 0 and d1 < best_d:
                        best_d = d1
                        center = (r, c)
            if center is None:
                return None
            cr, cc = center
            result = [list(map(int, row)) for row in grid]
            result[cr][cc] = 5
            for (rs, cs, vs) in cells:
                dr = 0 if rs == cr else (1 if rs > cr else -1)
                dc = 0 if cs == cc else (1 if cs > cc else -1)
                result[cr + dr][cc + dc] = vs
            return result

        for ex in train:
            predicted = _apply(ex["input"])
            if predicted is None or predicted != [list(map(int, row)) for row in ex["output"]]:
                return None
        result = _apply(test_input)
        if result is None or result == [list(map(int, row)) for row in test_input]:
            return None
        return result

    def _try_tile_from_marked_rows(self, train, test_input):
        """Rows marked by a uniform marker color in col 0 define a repeating tile.
        After the last content row, fill remaining empty rows by repeating the tile.
        The tile = marked rows with the marker replaced by background (0).
        """
        def _apply(grid):
            rows, cols = len(grid), len(grid[0])
            bg = 0
            # Find marker column: a column where all non-bg values are the same color
            marker_col = None
            marker_color = None
            for c in range(cols):
                col_vals = [int(grid[r][c]) for r in range(rows)]
                non_bg = [v for v in col_vals if v != bg]
                if len(non_bg) >= 1 and len(set(non_bg)) == 1:
                    marker_col = c
                    marker_color = non_bg[0]
                    break
            if marker_col is None:
                return None
            # Tile = rows where marker_col == marker_color, with that cell zeroed out
            tile_rows = [r for r in range(rows) if int(grid[r][marker_col]) == marker_color]
            if not tile_rows:
                return None
            tile = [[int(grid[r][c]) if c != marker_col else bg for c in range(cols)] for r in tile_rows]
            # Last non-empty row
            last_content = max(
                (r for r in range(rows) if any(int(grid[r][c]) != bg for c in range(cols))),
                default=0,
            )
            # Fill rows after last_content by repeating tile cyclically
            result = [list(map(int, row)) for row in grid]
            for i, r in enumerate(range(last_content + 1, rows)):
                result[r] = list(tile[i % len(tile)])
            return result

        for ex in train:
            predicted = _apply(ex["input"])
            if predicted is None or predicted != [list(map(int, row)) for row in ex["output"]]:
                return None

        result = _apply(test_input)
        if result is None or result == [list(map(int, row)) for row in test_input]:
            return None
        return result

    def _try_connect_aligned_diamonds(self, train, test_input):
        """Full diamond shapes (center with N/S/E/W arms of color 2) that share
        the same row or column get connected with 1s filling the gap between arms.
        """
        def _find_diamonds(grid):
            rows, cols = len(grid), len(grid[0])
            centers = []
            for r in range(1, rows - 1):
                for c in range(1, cols - 1):
                    if (int(grid[r][c]) == 0 and
                            int(grid[r-1][c]) == 2 and int(grid[r+1][c]) == 2 and
                            int(grid[r][c-1]) == 2 and int(grid[r][c+1]) == 2):
                        centers.append((r, c))
            return centers

        def _apply(grid):
            rows, cols = len(grid), len(grid[0])
            centers = _find_diamonds(grid)
            if len(centers) < 2:
                return None
            result = [list(map(int, row)) for row in grid]
            changed = False
            for i in range(len(centers)):
                for j in range(i + 1, len(centers)):
                    r1, c1 = centers[i]
                    r2, c2 = centers[j]
                    if r1 == r2:
                        cl, cr = (c1, c2) if c1 < c2 else (c2, c1)
                        gap = [result[r1][c] for c in range(cl + 2, cr - 1)]
                        if all(v == 0 for v in gap):
                            for c in range(cl + 2, cr - 1):
                                result[r1][c] = 1
                                changed = True
                    elif c1 == c2:
                        rt, rb = (r1, r2) if r1 < r2 else (r2, r1)
                        gap = [result[r][c1] for r in range(rt + 2, rb - 1)]
                        if all(v == 0 for v in gap):
                            for r in range(rt + 2, rb - 1):
                                result[r][c1] = 1
                                changed = True
            return result if changed else None

        for ex in train:
            predicted = _apply(ex["input"])
            if predicted is None or predicted != [list(map(int, row)) for row in ex["output"]]:
                return None
        result = _apply(test_input)
        if result is None or result == [list(map(int, row)) for row in test_input]:
            return None
        return result

    def _try_connect_diagonal_crosses(self, train, test_input):
        """Plus/cross shapes (center + N/S/E/W arms of same color) that are
        diagonally aligned (|dr|==|dc|) get connected with 2s along the diagonal.
        """
        def _find_crosses(grid):
            rows, cols = len(grid), len(grid[0])
            centers = []
            for r in range(1, rows - 1):
                for c in range(1, cols - 1):
                    v = int(grid[r][c])
                    if (v != 0 and
                            int(grid[r-1][c]) == v and int(grid[r+1][c]) == v and
                            int(grid[r][c-1]) == v and int(grid[r][c+1]) == v):
                        centers.append((r, c, v))
            return centers

        def _apply(grid):
            rows, cols = len(grid), len(grid[0])
            crosses = _find_crosses(grid)
            if len(crosses) < 2:
                return None
            result = [list(map(int, row)) for row in grid]
            changed = False
            for i in range(len(crosses)):
                for j in range(i + 1, len(crosses)):
                    r1, c1, v1 = crosses[i]
                    r2, c2, v2 = crosses[j]
                    dr = r2 - r1
                    dc = c2 - c1
                    if abs(dr) != abs(dc) or dr == 0:
                        continue
                    # Diagonal connection — fill path excluding endpoints
                    sr = 1 if dr > 0 else -1
                    sc = 1 if dc > 0 else -1
                    path = []
                    r, c = r1 + sr, c1 + sc
                    while (r, c) != (r2, c2):
                        path.append((r, c))
                        r += sr
                        c += sc
                    if all(int(result[pr][pc]) == 0 for pr, pc in path):
                        for pr, pc in path:
                            result[pr][pc] = 2
                            changed = True
            return result if changed else None

        for ex in train:
            predicted = _apply(ex["input"])
            if predicted is None or predicted != [list(map(int, row)) for row in ex["output"]]:
                return None
        result = _apply(test_input)
        if result is None or result == [list(map(int, row)) for row in test_input]:
            return None
        return result

    def _try_uniform_rows_to_five(self, train, test_input):
        """In a 3x3 grid: rows where all cells are the same color become [5,5,5];
        rows with mixed colors become [0,0,0].
        """
        def _apply(grid):
            rows, cols = len(grid), len(grid[0])
            if rows != 3 or cols != 3:
                return None
            result = []
            for row in grid:
                vals = [int(v) for v in row]
                if len(set(vals)) == 1:
                    result.append([5, 5, 5])
                else:
                    result.append([0, 0, 0])
            return result

        for ex in train:
            predicted = _apply(ex["input"])
            if predicted is None or predicted != [list(map(int, row)) for row in ex["output"]]:
                return None
        result = _apply(test_input)
        if result is None or result == [list(map(int, row)) for row in test_input]:
            return None
        return result

    def _try_erase_outside_cross_quadrant(self, train, test_input):
        """A full row and full column of 5s form a cross, creating a top-left quadrant.
        Colors that appear inside the quadrant have all their occurrences outside
        the quadrant erased (set to bg=0).
        """
        def _apply(grid):
            rows, cols = len(grid), len(grid[0])
            bg = 0
            # Find cross: a row where 5s span cols 0..k and a col where 5s span rows 0..k
            # (corner L-shape cross, not necessarily full-width/height)
            cross_row = cross_col = None
            for r in range(1, rows):
                five_cols = [c for c in range(cols) if int(grid[r][c]) == 5]
                if five_cols and five_cols == list(range(len(five_cols))):
                    cross_row = r
                    cross_col = len(five_cols) - 1
                    break
            if cross_row is None:
                return None
            # Verify cross_col: all cells in col cross_col from row 0 to cross_row are 5
            if not all(int(grid[r][cross_col]) == 5 for r in range(cross_row + 1)):
                return None
            # Find colors inside the top-left quadrant
            inside_colors = set()
            for r in range(cross_row):
                for c in range(cross_col):
                    v = int(grid[r][c])
                    if v != bg and v != 5:
                        inside_colors.add(v)
            if not inside_colors:
                return None
            # Remove outside occurrences of inside colors
            result = [list(map(int, row)) for row in grid]
            for r in range(rows):
                for c in range(cols):
                    v = result[r][c]
                    if v in inside_colors and (r >= cross_row or c >= cross_col):
                        result[r][c] = bg
            return result

        for ex in train:
            predicted = _apply(ex["input"])
            if predicted is None or predicted != [list(map(int, row)) for row in ex["output"]]:
                return None
        result = _apply(test_input)
        if result is None or result == [list(map(int, row)) for row in test_input]:
            return None
        return result

    def _try_fill_square_rect_interiors(self, train, test_input):
        """Fill with 2 the interior of closed rectangles whose interior is square (rows==cols).

        Solves 44d8ac46: grids have multiple closed 5-bordered rectangles;
        only those with a square interior get filled with 2.
        """
        from collections import Counter

        def _background(grid):
            vals = [v for row in grid for v in row]
            return Counter(vals).most_common(1)[0][0]

        def _apply(grid):
            rows, cols = len(grid), len(grid[0])
            bg = _background(grid)
            out = [list(map(int, row)) for row in grid]

            # Find border color: most common non-bg color
            non_bg = [int(grid[r][c]) for r in range(rows) for c in range(cols)
                      if int(grid[r][c]) != bg]
            if not non_bg:
                return out
            border_color = Counter(non_bg).most_common(1)[0][0]
            fill_color = 2

            for r1 in range(rows):
                for c1 in range(cols):
                    if int(grid[r1][c1]) != border_color:
                        continue
                    for r2 in range(r1 + 2, rows):
                        for c2 in range(c1 + 2, cols):
                            if int(grid[r2][c2]) != border_color:
                                continue
                            if not all(int(grid[r1][c]) == border_color for c in range(c1, c2 + 1)):
                                continue
                            if not all(int(grid[r2][c]) == border_color for c in range(c1, c2 + 1)):
                                continue
                            if not all(int(grid[r][c1]) == border_color for r in range(r1, r2 + 1)):
                                continue
                            if not all(int(grid[r][c2]) == border_color for r in range(r1, r2 + 1)):
                                continue
                            int_rows = r2 - r1 - 1
                            int_cols = c2 - c1 - 1
                            if int_rows <= 0 or int_cols <= 0:
                                continue
                            if not all(int(grid[r][c]) == bg
                                       for r in range(r1 + 1, r2)
                                       for c in range(c1 + 1, c2)):
                                continue
                            if int_rows == int_cols:
                                for r in range(r1 + 1, r2):
                                    for c in range(c1 + 1, c2):
                                        out[r][c] = fill_color
            return out

        for ex in train:
            if _apply(ex["input"]) != [list(map(int, row)) for row in ex["output"]]:
                return None
        return _apply(test_input)

    def _try_reflect_diagonal_segment(self, train, test_input):
        """Extend a diagonal segment by 1, reflected in the opposite direction.

        Solves 50c07299: segment of N cells gets replaced by N+1 cells going
        in the opposite direction from the anchor end.
        """
        from collections import Counter

        def _background(grid):
            vals = [v for row in grid for v in row]
            return Counter(vals).most_common(1)[0][0]

        def _find_diagonal_segment(grid, bg):
            rows, cols = len(grid), len(grid[0])
            non_bg = [(r, c, int(grid[r][c])) for r in range(rows)
                      for c in range(cols) if int(grid[r][c]) != bg]
            if not non_bg:
                return None
            colors = set(v for _, _, v in non_bg)
            if len(colors) != 1:
                return None
            color = list(colors)[0]
            cells = sorted((r, c) for r, c, _ in non_bg)
            n = len(cells)
            if n == 1:
                return (cells, None, color)
            dr = cells[1][0] - cells[0][0]
            dc = cells[1][1] - cells[0][1]
            if abs(dr) != 1 or abs(dc) != 1:
                return None
            r0, c0 = cells[0]
            for i, (r, c) in enumerate(cells):
                if r != r0 + i * dr or c != c0 + i * dc:
                    return None
            return (cells, (dr, dc), color)

        def _apply(grid, seg):
            rows, cols = len(grid), len(grid[0])
            bg = _background(grid)
            cells, direction, color = seg
            n = len(cells)
            out = [list(map(int, row)) for row in grid]

            if direction is None:
                r, c = cells[0]
                for step_dr, step_dc in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
                    opp_dr, opp_dc = -step_dr, -step_dc
                    new_cells = [(r + i * opp_dr, c + i * opp_dc) for i in range(1, n + 2)]
                    if all(0 <= nr < rows and 0 <= nc < cols for nr, nc in new_cells):
                        out[r][c] = bg
                        for nr, nc in new_cells:
                            out[nr][nc] = color
                        return out
                return None

            step_dr, step_dc = direction
            opp_dr, opp_dc = -step_dr, -step_dc
            anchor = cells[0]
            new_cells = [(anchor[0] + i * opp_dr, anchor[1] + i * opp_dc) for i in range(1, n + 2)]
            if not all(0 <= nr < rows and 0 <= nc < cols for nr, nc in new_cells):
                return None
            for r, c in cells:
                out[r][c] = bg
            for nr, nc in new_cells:
                out[nr][nc] = color
            return out

        for ex in train:
            bg = _background(ex["input"])
            seg = _find_diagonal_segment(ex["input"], bg)
            if seg is None:
                return None
            if _apply(ex["input"], seg) != [list(map(int, row)) for row in ex["output"]]:
                return None

        bg = _background(test_input)
        seg = _find_diagonal_segment(test_input, bg)
        if seg is None:
            return None
        result = _apply(test_input, seg)
        if result is None or result == [list(map(int, row)) for row in test_input]:
            return None
        return result

    def _try_fill_innermost_row_gap(self, train, test_input):
        """Fill the gap in innermost rows (rows with the most-interior non-bg pair endpoints).

        Solves 5ad8a7c0: rows whose 2s are at the most interior column positions
        get their gap filled. 'Innermost' = highest c1 (or equivalently lowest c2).
        """
        from collections import Counter

        def _background(grid):
            vals = [v for row in grid for v in row]
            return Counter(vals).most_common(1)[0][0]

        def _get_row_endpoints(row, bg):
            """Return (c1, c2, color) for a row with exactly 2 non-bg values, or None."""
            non_bg = [(c, int(v)) for c, v in enumerate(row) if int(v) != bg]
            if len(non_bg) != 2:
                return None
            colors = set(v for _, v in non_bg)
            if len(colors) != 1:
                return None
            c1, c2 = non_bg[0][0], non_bg[1][0]
            return (c1, c2, non_bg[0][1])

        def _apply(grid):
            rows, cols = len(grid), len(grid[0])
            bg = _background(grid)
            out = [list(map(int, row)) for row in grid]
            # Get endpoints for all rows with exactly 2 non-bg cells
            endpoints = {}
            for r in range(rows):
                ep = _get_row_endpoints(grid[r], bg)
                if ep is not None:
                    endpoints[r] = ep
            if not endpoints:
                return out
            # Innermost = highest c1 (most interior left position)
            max_c1 = max(ep[0] for ep in endpoints.values())
            # Fill only innermost rows that actually have a gap
            for r, (c1, c2, color) in endpoints.items():
                if c1 == max_c1 and c2 > c1 + 1:
                    for c in range(c1 + 1, c2):
                        out[r][c] = color
            return out

        for ex in train:
            if _apply(ex["input"]) != [list(map(int, row)) for row in ex["output"]]:
                return None
        return _apply(test_input)

    def _try_explode_center_to_corners(self, train, test_input):
        """2x2 block at the center of a grid explodes to the 4 corners; center is cleared.

        Solves 66e6c45b: a 4x4 grid has a 2x2 non-zero block at rows 1-2 cols 1-2.
        Each cell maps to the corresponding corner of the grid.
        """
        from collections import Counter

        def _background(grid):
            vals = [v for row in grid for v in row]
            return Counter(vals).most_common(1)[0][0]

        def _apply(grid):
            rows, cols = len(grid), len(grid[0])
            bg = _background(grid)
            out = [list(map(int, row)) for row in grid]
            # Find the 2x2 non-bg block
            for r1 in range(rows - 1):
                for c1 in range(cols - 1):
                    block = [[int(grid[r1 + dr][c1 + dc]) for dc in range(2)] for dr in range(2)]
                    if all(block[dr][dc] != bg for dr in range(2) for dc in range(2)):
                        # Check the rest of the grid is all bg
                        non_bg_outside = sum(
                            1 for r in range(rows) for c in range(cols)
                            if int(grid[r][c]) != bg
                            and not (r1 <= r <= r1 + 1 and c1 <= c <= c1 + 1)
                        )
                        if non_bg_outside > 0:
                            continue
                        # Map to corners: top-left, top-right, bottom-left, bottom-right
                        corners = [(0, 0), (0, cols - 1), (rows - 1, 0), (rows - 1, cols - 1)]
                        block_corners = [(r1, c1), (r1, c1 + 1), (r1 + 1, c1), (r1 + 1, c1 + 1)]
                        for (gr, gc), (br, bc) in zip(corners, block_corners):
                            out[gr][gc] = block[br - r1][bc - c1]
                        # Clear the 2x2 block
                        for dr in range(2):
                            for dc in range(2):
                                out[r1 + dr][c1 + dc] = bg
                        return out
            return None

        for ex in train:
            result = _apply(ex["input"])
            if result is None or result != [list(map(int, row)) for row in ex["output"]]:
                return None
        return _apply(test_input)

    def _try_color_holes_by_column_rank(self, train, test_input):
        """Holes (0s) in a grid of 5s get colored by the sorted rank of their column.

        Solves 575b1a71: find all unique columns containing holes, sort them left-to-right,
        assign color 1 to leftmost column's holes, 2 to next, 3 to next, 4 to rightmost.
        """
        from collections import Counter

        def _find_hole_color(grid):
            """Return the hole value (most common non-dominant value), or None."""
            vals = [v for row in grid for v in row]
            c = Counter(vals)
            bg = c.most_common(1)[0][0]
            holes = [v for v in vals if v != bg]
            if holes:
                return None  # Already colored
            return None

        def _apply(grid, col_color_map=None):
            rows, cols = len(grid), len(grid[0])
            vals = [v for row in grid for v in row]
            c = Counter(vals)
            if len(c) > 2:
                return None  # Only 2 distinct values (bg + hole)
            bg = c.most_common(1)[0][0]
            hole = c.most_common(2)[1][0] if len(c) == 2 else None

            # Find hole positions
            holes = [(r, col, int(grid[r][col]))
                     for r in range(rows) for col in range(cols)
                     if int(grid[r][col]) != bg]
            if not holes:
                return None

            # The "hole" value must be consistent (all same non-bg value)
            hole_val = holes[0][2]
            if not all(v == hole_val for _, _, v in holes):
                return None

            if col_color_map is None:
                # Build map from this grid's holes
                unique_cols = sorted(set(col for _, col, _ in holes))
                if len(unique_cols) < 2 or len(unique_cols) > 9:
                    return None
                col_color_map = {col: i + 1 for i, col in enumerate(unique_cols)}

            out = [list(map(int, row)) for row in grid]
            for r, col, _ in holes:
                if col not in col_color_map:
                    return None
                out[r][col] = col_color_map[col]
            return out

        # Verify each training example and extract consistent column→color mapping
        col_color_map = None
        for ex in train:
            inp = ex["input"]
            exp = [list(map(int, row)) for row in ex["output"]]
            # Determine expected map from output
            vals = [v for row in inp for v in row]
            c = Counter(vals)
            if len(c) != 2:
                return None
            bg = c.most_common(1)[0][0]
            hole_val = c.most_common(2)[1][0]
            holes = [(r, col) for r in range(len(inp)) for col in range(len(inp[0]))
                     if int(inp[r][col]) == hole_val]
            unique_cols = sorted(set(col for _, col in holes))
            if len(unique_cols) < 2:
                return None
            expected_map = {}
            for r, col in holes:
                expected_map[col] = exp[r][col]
            # Check ranks are 1..N in column order
            for i, col in enumerate(unique_cols):
                if expected_map.get(col) != i + 1:
                    return None
            if col_color_map is None:
                col_color_map = {col: i + 1 for i, col in enumerate(unique_cols)}

        # Apply to test
        vals = [v for row in test_input for v in row]
        c = Counter(vals)
        if len(c) != 2:
            return None
        bg = c.most_common(1)[0][0]
        hole_val = c.most_common(2)[1][0]
        holes = [(r, col) for r in range(len(test_input)) for col in range(len(test_input[0]))
                 if int(test_input[r][col]) == hole_val]
        unique_cols = sorted(set(col for _, col in holes))
        if len(unique_cols) < 2:
            return None
        test_map = {col: i + 1 for i, col in enumerate(unique_cols)}
        out = [list(map(int, row)) for row in test_input]
        for r, col in holes:
            out[r][col] = test_map[col]
        if out == [list(map(int, row)) for row in test_input]:
            return None
        return out

    def _try_fill_shape_bounding_box(self, train, test_input):
        """Fill the bounding box of each non-background shape with a fill color where background.

        Solves 60b61512: each connected component of 4s gets its bounding box
        filled with 7s wherever the cell is background (0).
        """
        from collections import Counter

        def _background(grid):
            vals = [v for row in grid for v in row]
            return Counter(vals).most_common(1)[0][0]

        def _connected_components(grid, bg):
            rows, cols = len(grid), len(grid[0])
            visited = [[False] * cols for _ in range(rows)]
            components = []
            for r0 in range(rows):
                for c0 in range(cols):
                    if int(grid[r0][c0]) != bg and not visited[r0][c0]:
                        # BFS
                        comp = []
                        stack = [(r0, c0)]
                        while stack:
                            r, c = stack.pop()
                            if visited[r][c]:
                                continue
                            visited[r][c] = True
                            comp.append((r, c))
                            for dr, dc in [(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]:
                                nr, nc = r + dr, c + dc
                                if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and int(grid[nr][nc]) != bg:
                                    stack.append((nr, nc))
                        components.append(comp)
            return components

        def _apply(grid, fill_color):
            rows, cols = len(grid), len(grid[0])
            bg = _background(grid)
            out = [list(map(int, row)) for row in grid]
            components = _connected_components(grid, bg)
            for comp in components:
                r1 = min(r for r, c in comp)
                r2 = max(r for r, c in comp)
                c1 = min(c for r, c in comp)
                c2 = max(c for r, c in comp)
                for r in range(r1, r2 + 1):
                    for c in range(c1, c2 + 1):
                        if int(grid[r][c]) == bg:
                            out[r][c] = fill_color
            return out

        # Infer fill color from training
        fill_color = None
        for ex in train:
            bg = _background(ex["input"])
            inp = [list(map(int, row)) for row in ex["input"]]
            exp = [list(map(int, row)) for row in ex["output"]]
            rows, cols = len(inp), len(inp[0])
            for r in range(rows):
                for c in range(cols):
                    if inp[r][c] == bg and exp[r][c] != bg:
                        candidate = exp[r][c]
                        if fill_color is None:
                            fill_color = candidate
                        elif fill_color != candidate:
                            return None
            if fill_color is None:
                return None

        if fill_color is None:
            return None

        for ex in train:
            if _apply(ex["input"], fill_color) != [list(map(int, row)) for row in ex["output"]]:
                return None

        result = _apply(test_input, fill_color)
        if result == [list(map(int, row)) for row in test_input]:
            return None
        return result

    def _try_fill_max_zero_rect(self, train, test_input):
        """Fill the largest all-zero rectangle with 6.

        Solves 3eda0437: in a grid of 0s and 1s, find the maximum rectangle
        containing only 0s and fill it with 6.
        """
        from collections import Counter

        def _max_zero_rect(grid):
            """Return (r1,c1,r2,c2) of the max all-zero rectangle."""
            rows, cols = len(grid), len(grid[0])
            # Heights: number of consecutive 0s above (including current) for each cell
            heights = [0] * cols
            best = (0, 0, 0, 0, 0)  # area, r1, c1, r2, c2

            for r in range(rows):
                for c in range(cols):
                    heights[c] = heights[c] + 1 if int(grid[r][c]) == 0 else 0

                # Max rectangle in histogram using stack
                stack = []  # (col, height)
                for c in range(cols + 1):
                    h = heights[c] if c < cols else 0
                    start = c
                    while stack and stack[-1][1] > h:
                        sc, sh = stack.pop()
                        w = c - sc
                        area = sh * w
                        if area > best[0] and sh >= 2 and w >= 2:
                            best = (area, r - sh + 1, sc, r, sc + w - 1)
                        start = sc
                    stack.append((start, h))

            return best[1:]  # r1, c1, r2, c2

        def _apply(grid):
            rows, cols = len(grid), len(grid[0])
            out = [list(map(int, row)) for row in grid]
            r1, c1, r2, c2 = _max_zero_rect(grid)
            if r1 > r2 or c1 > c2:
                return None
            # Verify entire rectangle is 0
            if not all(int(grid[r][c]) == 0 for r in range(r1, r2 + 1) for c in range(c1, c2 + 1)):
                return None
            for r in range(r1, r2 + 1):
                for c in range(c1, c2 + 1):
                    out[r][c] = 6
            return out

        for ex in train:
            result = _apply(ex["input"])
            if result is None or result != [list(map(int, row)) for row in ex["output"]]:
                return None

        result = _apply(test_input)
        if result is None or result == [list(map(int, row)) for row in test_input]:
            return None
        return result

    def _try_attract_color_pairs(self, train, test_input):
        """Rows with attracting color pairs: right color moves adjacent to left.

        Attracting pairs are learned from training examples (rows that change).
        Solves 494ef9d7.
        """
        def _find_attracting_pairs(examples):
            pairs = set()
            for ex in examples:
                inp = [list(map(int, r)) for r in ex["input"]]
                out = [list(map(int, r)) for r in ex["output"]]
                for r in range(len(inp)):
                    if inp[r] != out[r]:
                        nz = [(c, inp[r][c]) for c in range(len(inp[r])) if inp[r][c] != 0]
                        if len(nz) == 2:
                            pairs.add(frozenset([nz[0][1], nz[1][1]]))
            return pairs

        def _apply(grid, attracting_pairs):
            rows, cols = len(grid), len(grid[0])
            g = [list(map(int, row)) for row in grid]
            out = [row[:] for row in g]
            for r in range(rows):
                nz = [(c, g[r][c]) for c in range(cols) if g[r][c] != 0]
                if len(nz) == 2:
                    c1, v1 = nz[0]
                    c2, v2 = nz[1]
                    if frozenset([v1, v2]) in attracting_pairs and c2 > c1 + 1:
                        out[r][c2] = 0
                        out[r][c1 + 1] = v2
            return out

        attracting_pairs = _find_attracting_pairs(train)
        if not attracting_pairs:
            return None

        for ex in train:
            result = _apply(ex["input"], attracting_pairs)
            if result != [list(map(int, r)) for r in ex["output"]]:
                return None

        result = _apply(test_input, attracting_pairs)
        if result == [list(map(int, r)) for r in test_input]:
            return None
        return result

    def _try_column_height_balance(self, train, test_input):
        """Bar chart balance: 5-bar height = sum(8-heights) - sum(2-heights).

        Bars are vertical runs of a single color (8 or 2) aligned to the bottom.
        5-bar placed at next evenly-spaced column, same bottom alignment.
        Solves 37ce87bb.
        """
        def _parse_bars(grid):
            rows, cols = len(grid), len(grid[0])
            bars = {}
            for c in range(cols):
                col_cells = [(r, int(grid[r][c])) for r in range(rows) if int(grid[r][c]) != 7]
                if not col_cells:
                    continue
                colors = set(v for _, v in col_cells)
                if len(colors) != 1:
                    return None
                color = next(iter(colors))
                if color not in (8, 2):
                    return None
                height = len(col_cells)
                expected_rows = list(range(rows - height, rows))
                actual_rows = sorted(r for r, _ in col_cells)
                if actual_rows != expected_rows:
                    return None
                bars[c] = (color, height)
            return bars

        def _apply(grid):
            rows, cols = len(grid), len(grid[0])
            bars = _parse_bars(grid)
            if bars is None or len(bars) == 0:
                return None
            bar_cols = sorted(bars.keys())
            if len(bar_cols) >= 2:
                step = bar_cols[1] - bar_cols[0]
                for i in range(1, len(bar_cols)):
                    if bar_cols[i] - bar_cols[i - 1] != step:
                        return None
            else:
                step = 2
            sum_8 = sum(h for c, (color, h) in bars.items() if color == 8)
            sum_2 = sum(h for c, (color, h) in bars.items() if color == 2)
            H = sum_8 - sum_2
            if H <= 0:
                return None
            new_col = max(bar_cols) + step
            if new_col >= cols or H > rows:
                return None
            out = [list(map(int, row)) for row in grid]
            for r in range(rows - H, rows):
                out[r][new_col] = 5
            return out

        for ex in train:
            result = _apply(ex["input"])
            if result is None or result != [list(map(int, r)) for r in ex["output"]]:
                return None

        result = _apply(test_input)
        if result is None or result == [list(map(int, r)) for r in test_input]:
            return None
        return result

    def _try_perpendicular_8_blocks(self, train, test_input):
        """Two groups of 3-cells get 8-blocks placed perpendicularly.

        Rule: Given two connected components A and B of 3-cells with centers A, B:
        - Midpoint M = (A+B)/2
        - CD_vector = 3 * rotate_CCW(AB) where AB = B - A
        - 8-blocks placed at C = M - CD/2 and D = M + CD/2
        - Each 8-block has same shape as component A (cell offsets from center)
        Solves 22233c11.
        """
        def _split_mst(cells):
            """Split cells into exactly 2 groups by removing the longest MST edge."""
            n = len(cells)
            if n < 2:
                return None
            if n == 2:
                return [cells[:1], cells[1:]]
            edges = sorted(
                [((cells[i][0] - cells[j][0]) ** 2 + (cells[i][1] - cells[j][1]) ** 2, i, j)
                 for i in range(n) for j in range(i + 1, n)]
            )
            parent = list(range(n))

            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            def union(x, y):
                parent[find(x)] = find(y)

            mst_edges = []
            for dist, i, j in edges:
                if find(i) != find(j):
                    mst_edges.append((dist, i, j))
                    union(i, j)
            if not mst_edges:
                return None
            max_edge = max(mst_edges)
            adj = [[] for _ in range(n)]
            for dist, i, j in mst_edges:
                if (dist, i, j) != max_edge:
                    adj[i].append(j)
                    adj[j].append(i)
            visited = [False] * n
            groups = []
            for start in range(n):
                if not visited[start]:
                    comp = []
                    q = [start]
                    visited[start] = True
                    while q:
                        node = q.pop(0)
                        comp.append(cells[node])
                        for nb in adj[node]:
                            if not visited[nb]:
                                visited[nb] = True
                                q.append(nb)
                    groups.append(comp)
            return groups if len(groups) == 2 else None

        def _center(comp):
            return (sum(r for r, c in comp) / len(comp),
                    sum(c for r, c in comp) / len(comp))

        def _place_8_pair(out, A_center, B_center, offsets, rows, cols):
            """Place 8-shaped blocks perpendicular to A→B at C and D."""
            ab_r = B_center[0] - A_center[0]
            ab_c = B_center[1] - A_center[1]
            M = ((A_center[0] + B_center[0]) / 2,
                 (A_center[1] + B_center[1]) / 2)
            cd_r = 3 * (-ab_c)
            cd_c = 3 * ab_r
            center_C = (M[0] - cd_r / 2, M[1] - cd_c / 2)
            center_D = (M[0] + cd_r / 2, M[1] + cd_c / 2)
            for cx, cy in [center_C, center_D]:
                for dr, dc in offsets:
                    nr_f = cx + dr
                    nc_f = cy + dc
                    nr = round(nr_f)
                    nc = round(nc_f)
                    if abs(nr_f - nr) > 1e-6 or abs(nc_f - nc) > 1e-6:
                        continue
                    if 0 <= nr < rows and 0 <= nc < cols:
                        out[nr][nc] = 8

        def _apply(grid):
            rows, cols = len(grid), len(grid[0])
            g = [list(map(int, row)) for row in grid]
            if any(g[r][c] != 0 and g[r][c] != 3
                   for r in range(rows) for c in range(cols)):
                return None
            all_3s = [(r, c) for r in range(rows) for c in range(cols) if g[r][c] == 3]
            if len(all_3s) < 2:
                return None
            out = [row[:] for row in g]
            if len(all_3s) == 2:
                # Single segment: A and B are the 2 cells, offsets = [(-0.5,-0.5),(0.5,0.5)]
                A_center = (float(all_3s[0][0]), float(all_3s[0][1]))
                B_center = (float(all_3s[1][0]), float(all_3s[1][1]))
                offsets = [(r - _center(all_3s)[0], c - _center(all_3s)[1])
                           for r, c in all_3s]
                _place_8_pair(out, A_center, B_center, [(0.0, 0.0)], rows, cols)
            else:
                comps = _split_mst(all_3s)
                if comps is None:
                    return None
                if all(len(c) == 2 for c in comps):
                    # Per-segment: each 2-cell group independently generates 2 8-cells
                    for comp in comps:
                        A_c = (float(comp[0][0]), float(comp[0][1]))
                        B_c = (float(comp[1][0]), float(comp[1][1]))
                        _place_8_pair(out, A_c, B_c, [(0.0, 0.0)], rows, cols)
                else:
                    # Center-to-center: group centers define the line, shape = group offsets
                    A_center = _center(comps[0])
                    B_center = _center(comps[1])
                    offsets = [(r - A_center[0], c - A_center[1]) for r, c in comps[0]]
                    _place_8_pair(out, A_center, B_center, offsets, rows, cols)
            return out

        for ex in train:
            result = _apply(ex["input"])
            if result is None or result != [list(map(int, r)) for r in ex["output"]]:
                return None

        result = _apply(test_input)
        if result is None or result == [list(map(int, r)) for r in test_input]:
            return None
        return result

    def _try_grid_complement_separator(self, train, test_input):
        """Separator rows/cols (all-zero) stay 0; non-sep cells: non-zero→0, zero→2."""
        def _apply(grid):
            g = [list(map(int, row)) for row in grid]
            rows, cols = len(g), len(g[0])
            sep_rows = {r for r in range(rows) if all(g[r][c] == 0 for c in range(cols))}
            sep_cols = {c for c in range(cols) if all(g[r][c] == 0 for r in range(rows))}
            out = []
            for r in range(rows):
                row = []
                for c in range(cols):
                    if r in sep_rows or c in sep_cols:
                        row.append(0)
                    elif g[r][c] != 0:
                        row.append(0)
                    else:
                        row.append(2)
                out.append(row)
            return out

        # Need at least one separator row and one separator col
        g0 = [list(map(int, row)) for row in train[0]["input"]]
        rows0, cols0 = len(g0), len(g0[0])
        sep_rows0 = [r for r in range(rows0) if all(g0[r][c] == 0 for c in range(cols0))]
        sep_cols0 = [c for c in range(cols0) if all(g0[r][c] == 0 for r in range(rows0))]
        if not sep_rows0 or not sep_cols0:
            return None

        for ex in train:
            result = _apply(ex["input"])
            if result != [list(map(int, r)) for r in ex["output"]]:
                return None

        result = _apply(test_input)
        if result == [list(map(int, r)) for r in test_input]:
            return None
        return result

    def _try_ones_in_8_bbox_to_3(self, train, test_input):
        """All 1-cells inside the bounding box of 8-cells become 3; others unchanged."""
        def _apply(grid):
            g = [list(map(int, row)) for row in grid]
            rows, cols = len(g), len(g[0])
            eights = [(r, c) for r in range(rows) for c in range(cols) if g[r][c] == 8]
            if not eights:
                return None
            r_min = min(r for r, c in eights)
            r_max = max(r for r, c in eights)
            c_min = min(c for r, c in eights)
            c_max = max(c for r, c in eights)
            out = [row[:] for row in g]
            for r in range(r_min, r_max + 1):
                for c in range(c_min, c_max + 1):
                    if g[r][c] == 1:
                        out[r][c] = 3
            return out

        # Verify at least one 1→3 change occurs in training
        any_change = False
        for ex in train:
            result = _apply(ex["input"])
            if result is None or result != [list(map(int, r)) for r in ex["output"]]:
                return None
            if result != [list(map(int, r)) for r in ex["input"]]:
                any_change = True
        if not any_change:
            return None

        result = _apply(test_input)
        if result is None or result == [list(map(int, r)) for r in test_input]:
            return None
        return result

    def _try_swap_colors_in_components(self, train, test_input):
        """Swap the two non-background colors within each connected component."""
        from collections import Counter

        def _find_bg(grid):
            rows, cols = len(grid), len(grid[0])
            return Counter(grid[r][c] for r in range(rows) for c in range(cols)).most_common(1)[0][0]

        def _components(grid, bg):
            rows, cols = len(grid), len(grid[0])
            visited = [[False] * cols for _ in range(rows)]
            comps = []
            for sr in range(rows):
                for sc in range(cols):
                    if grid[sr][sc] != bg and not visited[sr][sc]:
                        comp = []
                        queue = [(sr, sc)]
                        visited[sr][sc] = True
                        while queue:
                            r, c = queue.pop(0)
                            comp.append((r, c))
                            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                                nr, nc = r+dr, c+dc
                                if 0<=nr<rows and 0<=nc<cols and not visited[nr][nc] and grid[nr][nc] != bg:
                                    visited[nr][nc] = True
                                    queue.append((nr, nc))
                        comps.append(comp)
            return comps

        def _apply(grid):
            g = [list(map(int, row)) for row in grid]
            bg = _find_bg(g)
            comps = _components(g, bg)
            if not comps:
                return None
            out = [row[:] for row in g]
            any_swap = False
            for comp in comps:
                colors = list(set(g[r][c] for r, c in comp))
                if len(colors) == 2:
                    a, b = colors[0], colors[1]
                    for r, c in comp:
                        out[r][c] = b if g[r][c] == a else a
                    any_swap = True
            return out if any_swap else None

        for ex in train:
            result = _apply(ex["input"])
            if result is None or result != [list(map(int, r)) for r in ex["output"]]:
                return None
        result = _apply(test_input)
        if result is None or result == [list(map(int, r)) for r in test_input]:
            return None
        return result

    def _try_remove_isolated_cells(self, train, test_input):
        """Remove cells that have no 8-connected neighbor of the same color."""
        from collections import Counter

        def _apply(grid):
            g = [list(map(int, row)) for row in grid]
            rows, cols = len(g), len(g[0])
            bg = Counter(g[r][c] for r in range(rows) for c in range(cols)).most_common(1)[0][0]
            out = [row[:] for row in g]
            changed = False
            for r in range(rows):
                for c in range(cols):
                    if g[r][c] == bg:
                        continue
                    v = g[r][c]
                    has_neighbor = any(
                        0 <= r+dr < rows and 0 <= c+dc < cols and g[r+dr][c+dc] == v
                        for dr in [-1,0,1] for dc in [-1,0,1] if (dr,dc) != (0,0)
                    )
                    if not has_neighbor:
                        out[r][c] = bg
                        changed = True
            return out if changed else None

        for ex in train:
            result = _apply(ex["input"])
            if result is None or result != [list(map(int, r)) for r in ex["output"]]:
                return None
        result = _apply(test_input)
        if result is None or result == [list(map(int, r)) for r in test_input]:
            return None
        return result

    def _try_quadrant_color_map(self, train, test_input):
        """2x2 key in 8-bordered corner maps scattered single-color cells by quadrant."""
        def _find_key(grid):
            g = [list(map(int, row)) for row in grid]
            rows, cols = len(g), len(g[0])
            for r in range(rows - 1):
                for c in range(cols - 1):
                    block = [g[r][c], g[r][c+1], g[r+1][c], g[r+1][c+1]]
                    if all(v not in (0, 8) for v in block):
                        return (r, c, block[0], block[1], block[2], block[3])
            return None

        def _apply(grid):
            key = _find_key(grid)
            if key is None:
                return None
            key_r, key_c, tl, tr, bl, br = key
            g = [list(map(int, row)) for row in grid]
            rows, cols = len(g), len(g[0])
            # Collect main cells: non-0, non-8, outside key region
            main_cells = []
            for r in range(rows):
                for c in range(cols):
                    if g[r][c] not in (0, 8):
                        if key_r <= r <= key_r+1 and key_c <= c <= key_c+1:
                            continue
                        main_cells.append((r, c, g[r][c]))
            if not main_cells:
                return None
            colors_in_main = set(v for r, c, v in main_cells)
            if len(colors_in_main) != 1:
                return None
            # Main area bounds: rows/cols that are fully non-8 (not key rows/cols)
            main_rows = [r for r in range(rows)
                         if not (key_r <= r <= key_r+1) and any(g[r][c] != 8 for c in range(cols))]
            main_cols = [c for c in range(cols)
                         if not (key_c <= c <= key_c+1) and any(g[r][c] != 8 for r in range(rows))]
            if not main_rows or not main_cols:
                return None
            r_mid = (min(main_rows) + max(main_rows)) / 2.0
            c_mid = (min(main_cols) + max(main_cols)) / 2.0
            out = [row[:] for row in g]
            for r, c, v in main_cells:
                qr = 0 if r < r_mid else 1
                qc = 0 if c < c_mid else 1
                out[r][c] = [[tl, tr], [bl, br]][qr][qc]
            return out

        # Verify key consistent across training examples
        key0 = _find_key(train[0]["input"])
        if key0 is None:
            return None

        for ex in train:
            result = _apply(ex["input"])
            if result is None or result != [list(map(int, r)) for r in ex["output"]]:
                return None
        result = _apply(test_input)
        if result is None or result == [list(map(int, r)) for r in test_input]:
            return None
        return result

    def _try_shift_parallelogram_top(self, train, test_input):
        """Non-bottom-bar cells of parallelogram outlines shift right by 1 (clipped at bottom-bar-right)."""
        from collections import Counter

        def _bfs_component(g, sr, sc, bg, rows, cols):
            # 8-connected: parallelogram sides are diagonally adjacent
            visited = [[False]*cols for _ in range(rows)]
            comp = []
            queue = [(sr, sc)]
            visited[sr][sc] = True
            while queue:
                r, c = queue.pop(0)
                comp.append((r, c))
                for dr in [-1,0,1]:
                    for dc in [-1,0,1]:
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r+dr, c+dc
                        if 0<=nr<rows and 0<=nc<cols and not visited[nr][nc] and g[nr][nc] != bg:
                            visited[nr][nc] = True
                            queue.append((nr, nc))
            return comp

        def _apply(grid):
            g = [list(map(int, row)) for row in grid]
            rows, cols = len(g), len(g[0])
            bg = Counter(g[r][c] for r in range(rows) for c in range(cols)).most_common(1)[0][0]
            out = [row[:] for row in g]
            visited_global = set()
            changed = False
            for sr in range(rows):
                for sc in range(cols):
                    if g[sr][sc] == bg or (sr, sc) in visited_global:
                        continue
                    comp = _bfs_component(g, sr, sc, bg, rows, cols)
                    for r, c in comp:
                        visited_global.add((r, c))
                    # Find bottom bar row (max row index)
                    row_max = max(r for r, c in comp)
                    row_min = min(r for r, c in comp)
                    if row_max == row_min:
                        continue  # Single row, skip
                    # Bottom bar: all cells in row_max
                    bottom_bar = sorted(c for r, c in comp if r == row_max)
                    if len(bottom_bar) < 2:
                        continue  # Bottom bar must have at least 2 cells
                    bottom_bar_right = bottom_bar[-1]
                    # Verify top bar also has >= 2 cells
                    top_bar = sorted(c for r, c in comp if r == row_min)
                    if len(top_bar) < 2:
                        continue
                    # Check this is a parallelogram-like outline (each middle row has exactly 2 cells)
                    mid_rows_ok = True
                    for row in range(row_min + 1, row_max):
                        cells_in_row = [c for r, c in comp if r == row]
                        if len(cells_in_row) != 2:
                            mid_rows_ok = False
                            break
                    if not mid_rows_ok:
                        continue
                    # Shift all non-bottom-bar cells right by 1, clip rightmost to bottom_bar_right
                    v = g[sr][sc]
                    # Compute new positions first, then apply
                    new_positions = {}
                    for r, c in comp:
                        if r == row_max:
                            new_positions[(r, c)] = (r, c)
                        else:
                            cells_in_row = sorted(cc for rr, cc in comp if rr == r)
                            if c == cells_in_row[-1]:
                                new_c = min(c + 1, bottom_bar_right)
                            else:
                                new_c = c + 1
                            new_positions[(r, c)] = (r, new_c)
                    # Clear old positions, then place new ones
                    for r, c in comp:
                        out[r][c] = bg
                    for (r, c), (nr, nc) in new_positions.items():
                        if 0 <= nr < rows and 0 <= nc < cols:
                            out[nr][nc] = v
                    changed = True
            return out if changed else None

        for ex in train:
            result = _apply(ex["input"])
            if result is None or result != [list(map(int, r)) for r in ex["output"]]:
                return None
        result = _apply(test_input)
        if result is None or result == [list(map(int, r)) for r in test_input]:
            return None
        return result

    def _try_ring_rotate_3x3(self, train, test_input):
        """3x3 blocks: corners rotate CCW by 1, edge midpoints rotate CW by 1, center fixed."""
        from collections import Counter
        # Ring positions clockwise: even = corners, odd = edge midpoints
        RING = [(0,0),(0,1),(0,2),(1,2),(2,2),(2,1),(2,0),(1,0)]

        def _find_and_apply(grid):
            g = [list(map(int, row)) for row in grid]
            rows, cols = len(g), len(g[0])
            bg = Counter(g[r][c] for r in range(rows) for c in range(cols)).most_common(1)[0][0]
            out = [row[:] for row in g]
            changed = False
            for sr in range(rows - 2):
                for sc in range(cols - 2):
                    if g[sr+1][sc+1] != bg:
                        continue
                    border = [g[sr+dr][sc+dc] for dr, dc in RING]
                    if any(v == bg for v in border):
                        continue
                    new_border = [None] * 8
                    for i in range(8):
                        if i % 2 == 0:  # corner: CCW 1 step in corner ring = -2 in full ring
                            new_border[(i + 6) % 8] = border[i]
                        else:           # edge: CW 1 step in edge ring = +2 in full ring
                            new_border[(i + 2) % 8] = border[i]
                    for i, (dr, dc) in enumerate(RING):
                        out[sr+dr][sc+dc] = new_border[i]
                    changed = True
            return out if changed else None

        for ex in train:
            result = _find_and_apply(ex["input"])
            if result is None or result != [list(map(int, r)) for r in ex["output"]]:
                return None
        result = _find_and_apply(test_input)
        if result is None or result == [list(map(int, r)) for r in test_input]:
            return None
        return result

    def _try_bordered_rect_center(self, train, test_input):
        """Find the unique NxM rectangle whose entire border is one color and center differs.
        Output is a 1x1 grid with the center color.
        Handles any rectangle size >=3x3 where border is uniform and center is a single different value."""
        from collections import Counter

        def _find_center(grid):
            g = [list(map(int, row)) for row in grid]
            rows, cols = len(g), len(g[0])
            bg = Counter(g[r][c] for r in range(rows) for c in range(cols)).most_common(1)[0][0]
            # Try all possible rectangles >=3x3
            centers = []
            for r0 in range(rows - 2):
                for c0 in range(cols - 2):
                    for r1 in range(r0 + 2, rows):
                        for c1 in range(c0 + 2, cols):
                            # Collect border cells
                            border = set()
                            for c in range(c0, c1 + 1):
                                border.add((r0, c))
                                border.add((r1, c))
                            for r in range(r0, r1 + 1):
                                border.add((r, c0))
                                border.add((r, c1))
                            # All border cells same color?
                            border_colors = {g[r][c] for r, c in border}
                            if len(border_colors) != 1:
                                continue
                            border_color = next(iter(border_colors))
                            if border_color == bg:
                                continue
                            # Interior cells
                            interior = [(r, c) for r in range(r0 + 1, r1)
                                        for c in range(c0 + 1, c1)]
                            if not interior:
                                continue
                            # All interior same color, different from border?
                            int_colors = {g[r][c] for r, c in interior}
                            if len(int_colors) != 1:
                                continue
                            int_color = next(iter(int_colors))
                            if int_color == border_color:
                                continue
                            centers.append(int_color)
            return centers

        # All train examples must have exactly one bordered rect with consistent rule
        train_centers = []
        for ex in train:
            centers = _find_center(ex["input"])
            if len(centers) != 1:
                return None
            if centers[0] != ex["output"][0][0]:
                return None
            train_centers.append(centers[0])

        test_centers = _find_center(test_input)
        if len(test_centers) != 1:
            return None
        return [[test_centers[0]]]

    def _try_rect_corner_edge_interior(self, train, test_input):
        """Solid color rectangles: corners → color_c, perimeter edges → color_e, interior → color_i.
        All three colors are learned from training. Handles multiple rectangles per grid."""
        from collections import Counter

        def _find_rects(grid, sep_color):
            """Return list of (color, r0, c0, r1, c1) for all solid-color rectangles
            that are not sep_color."""
            g = [list(map(int, row)) for row in grid]
            rows, cols = len(g), len(g[0])
            visited = [[False] * cols for _ in range(rows)]
            rects = []
            for r in range(rows):
                for c in range(cols):
                    if visited[r][c] or g[r][c] == sep_color:
                        continue
                    color = g[r][c]
                    # BFS to get component
                    comp = set()
                    q = [(r, c)]
                    while q:
                        cr, cc = q.pop()
                        if (cr, cc) in comp or not (0 <= cr < rows and 0 <= cc < cols):
                            continue
                        if g[cr][cc] != color:
                            continue
                        comp.add((cr, cc))
                        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                            q.append((cr+dr, cc+dc))
                    for pr, pc in comp:
                        visited[pr][pc] = True
                    r0 = min(rr for rr, _ in comp)
                    r1 = max(rr for rr, _ in comp)
                    c0 = min(cc for _, cc in comp)
                    c1 = max(cc for _, cc in comp)
                    # Verify all cells in bounding box are this color
                    expected_count = (r1 - r0 + 1) * (c1 - c0 + 1)
                    if len(comp) != expected_count:
                        return None  # not a solid rectangle
                    if r0 == r1 or c0 == c1:
                        continue  # too thin
                    rects.append((color, r0, c0, r1, c1))
            return rects

        # Detect separator color: the color that separates rectangles (usually 0)
        # Try 0 first, then most common color
        all_in = [v for ex in train for row in ex["input"] for v in row]
        sep_candidates = [0]
        mc = Counter(all_in).most_common()
        # Also try the least common color as separator if 0 isn't present
        if 0 not in [v for v, _ in mc]:
            sep_candidates = [mc[-1][0]]
        else:
            sep_candidates = [0]

        def _classify_cell(r, c, r0, c0, r1, c1):
            on_top = r == r0
            on_bot = r == r1
            on_lft = c == c0
            on_rgt = c == c1
            is_corner = (on_top or on_bot) and (on_lft or on_rgt)
            is_edge = (on_top or on_bot or on_lft or on_rgt) and not is_corner
            return 'corner' if is_corner else ('edge' if is_edge else 'interior')

        def _apply(grid, color_map, sep_color):
            g = [list(map(int, row)) for row in grid]
            rects = _find_rects(grid, sep_color)
            if not rects:
                return None
            out = [row[:] for row in g]
            for src_color, r0, c0, r1, c1 in rects:
                for r in range(r0, r1 + 1):
                    for c in range(c0, c1 + 1):
                        kind = _classify_cell(r, c, r0, c0, r1, c1)
                        out[r][c] = color_map[kind]
            return out

        for sep_color in sep_candidates:
            # Only applies when output is same size as input
            if len(train[0]["input"]) != len(train[0]["output"]) or \
               len(train[0]["input"][0]) != len(train[0]["output"][0]):
                continue

            # Learn color_map from first training example
            ex0 = train[0]
            rects0 = _find_rects(ex0["input"], sep_color)
            if not rects0:
                continue
            out0 = [list(map(int, row)) for row in ex0["output"]]
            color_map = {}
            ok = True
            for src_color, r0, c0, r1, c1 in rects0:
                for r in range(r0, r1 + 1):
                    for c in range(c0, c1 + 1):
                        kind = _classify_cell(r, c, r0, c0, r1, c1)
                        mapped = out0[r][c]
                        if kind in color_map and color_map[kind] != mapped:
                            ok = False
                            break
                        color_map[kind] = mapped
                    if not ok:
                        break
                if not ok:
                    break
            if not ok:
                continue
            if set(color_map.keys()) != {'corner', 'edge', 'interior'}:
                continue
            if len(set(color_map.values())) < 2:
                continue  # all same — not a meaningful transform

            # Cross-validate
            valid = True
            for ex in train:
                result = _apply(ex["input"], color_map, sep_color)
                if result is None or result != [list(map(int, r)) for r in ex["output"]]:
                    valid = False
                    break
            if not valid:
                continue
            result = _apply(test_input, color_map, sep_color)
            if result is None or result == [list(map(int, r)) for r in test_input]:
                continue
            return result
        return None

    def _try_convert_isolated_cells(self, train, test_input):
        """Cells of color A with no 4-connected neighbor of same color → color B.
        A and B are learned from training. Only same-size transforms."""
        from collections import Counter

        def _isolated_cells(grid, color):
            g = [list(map(int, row)) for row in grid]
            rows, cols = len(g), len(g[0])
            result = []
            for r in range(rows):
                for c in range(cols):
                    if g[r][c] != color:
                        continue
                    has_neighbor = any(
                        0 <= r+dr < rows and 0 <= c+dc < cols and g[r+dr][c+dc] == color
                        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]
                    )
                    if not has_neighbor:
                        result.append((r, c))
            return result

        def _apply(grid, src_color, dst_color):
            g = [list(map(int, row)) for row in grid]
            rows, cols = len(g), len(g[0])
            out = [row[:] for row in g]
            changed = False
            for r, c in _isolated_cells(grid, src_color):
                out[r][c] = dst_color
                changed = True
            return out if changed else None

        # Only same-size transforms
        if len(train[0]["input"]) != len(train[0]["output"]) or \
           len(train[0]["input"][0]) != len(train[0]["output"][0]):
            return None

        # Learn (src_color, dst_color) from training: find cells that change
        src_color = dst_color = None
        for ex in train:
            gin = [list(map(int, r)) for r in ex["input"]]
            gout = [list(map(int, r)) for r in ex["output"]]
            rows, cols = len(gin), len(gin[0])
            for r in range(rows):
                for c in range(cols):
                    if gin[r][c] != gout[r][c]:
                        sc, dc = gin[r][c], gout[r][c]
                        if src_color is None:
                            src_color, dst_color = sc, dc
                        elif src_color != sc or dst_color != dc:
                            return None  # multiple different change types
        if src_color is None:
            return None

        # Cross-validate: all isolated src_color cells must become dst_color
        for ex in train:
            expected_out = [list(map(int, r)) for r in ex["output"]]
            isolated = _isolated_cells(ex["input"], src_color)
            gin = [list(map(int, r)) for r in ex["input"]]
            for r, c in isolated:
                if expected_out[r][c] != dst_color:
                    return None
            # Non-isolated src cells must stay src
            rows, cols = len(gin), len(gin[0])
            for r in range(rows):
                for c in range(cols):
                    if gin[r][c] == src_color and (r, c) not in set(isolated):
                        if expected_out[r][c] != src_color:
                            return None

        result = _apply(test_input, src_color, dst_color)
        if result is None or result == [list(map(int, r)) for r in test_input]:
            return None
        return result

    def _try_row_permutation(self, train, test_input):
        """Output rows are a fixed permutation of input rows (same size only)."""
        # Only same-size transforms
        for ex in train:
            if len(ex["input"]) != len(ex["output"]) or \
               len(ex["input"][0]) != len(ex["output"][0]):
                return None

        rows = len(train[0]["input"])

        def _learn_perm(ex):
            """Return row-permutation list perm where out[r] = in[perm[r]], or None."""
            gin = [list(map(int, r)) for r in ex["input"]]
            gout = [list(map(int, r)) for r in ex["output"]]
            perm = []
            for r in range(rows):
                src = None
                for r2 in range(rows):
                    if gin[r2] == gout[r]:
                        if src is None:
                            src = r2
                        # If multiple matches, ambiguous — but accept if consistent
                if src is None:
                    return None
                perm.append(src)
            return perm

        # Learn permutation from first example
        perm = _learn_perm(train[0])
        if perm is None:
            return None

        # Verify it's a true permutation (no duplicates) and not identity
        if sorted(perm) != list(range(rows)):
            return None
        if perm == list(range(rows)):
            return None  # identity

        # Cross-validate on remaining examples
        for ex in train[1:]:
            if _learn_perm(ex) != perm:
                return None

        # Apply to test input
        gin = [list(map(int, r)) for r in test_input]
        result = [gin[perm[r]] for r in range(rows)]
        if result == gin:
            return None
        return result

    def _try_broadcast_direction(self, train, test_input):
        """Non-zero values broadcast in a direction (down/up/right/left), filling bg cells.
        Unlike gravity, values stay in place and fill the cells beyond them."""
        from collections import Counter

        def _apply(grid, direction, bg):
            g = [list(map(int, row)) for row in grid]
            rows, cols = len(g), len(g[0])
            out = [row[:] for row in g]
            if direction == 'down':
                for c in range(cols):
                    cur = bg
                    for r in range(rows):
                        if g[r][c] != bg:
                            cur = g[r][c]
                        elif cur != bg:
                            out[r][c] = cur
            elif direction == 'up':
                for c in range(cols):
                    cur = bg
                    for r in range(rows - 1, -1, -1):
                        if g[r][c] != bg:
                            cur = g[r][c]
                        elif cur != bg:
                            out[r][c] = cur
            elif direction == 'right':
                for r in range(rows):
                    cur = bg
                    for c in range(cols):
                        if g[r][c] != bg:
                            cur = g[r][c]
                        elif cur != bg:
                            out[r][c] = cur
            elif direction == 'left':
                for r in range(rows):
                    cur = bg
                    for c in range(cols - 1, -1, -1):
                        if g[r][c] != bg:
                            cur = g[r][c]
                        elif cur != bg:
                            out[r][c] = cur
            return out

        # Only same-size
        for ex in train:
            if len(ex["input"]) != len(ex["output"]) or \
               len(ex["input"][0]) != len(ex["output"][0]):
                return None

        all_in = [v for ex in train for row in ex["input"] for v in row]
        bg = Counter(all_in).most_common(1)[0][0]

        for direction in ['down', 'up', 'right', 'left']:
            valid = True
            for ex in train:
                r = _apply(ex["input"], direction, bg)
                if r != [list(map(int, row)) for row in ex["output"]]:
                    valid = False
                    break
            if valid:
                result = _apply(test_input, direction, bg)
                if result != [list(map(int, r)) for r in test_input]:
                    return result
        return None

    def _try_fill_interior_through_gap(self, train, test_input):
        """A bordered rectangle (single color) has one gap in its wall.
        Fill interior with 8, then project 8 from the gap outward to the grid edge."""
        from collections import Counter

        def _analyze(grid):
            g = [list(map(int, row)) for row in grid]
            rows, cols = len(g), len(g[0])
            # Find border color: non-bg color that forms a near-complete rectangle border
            all_vals = Counter(g[r][c] for r in range(rows) for c in range(cols))
            bg = all_vals.most_common(1)[0][0]
            # Try each non-bg color as the border color
            for border_color, _ in all_vals.most_common():
                if border_color == bg:
                    continue
                cells = {(r, c) for r in range(rows) for c in range(cols) if g[r][c] == border_color}
                if not cells:
                    continue
                r0 = min(r for r, c in cells)
                r1 = max(r for r, c in cells)
                c0 = min(c for r, c in cells)
                c1 = max(c for r, c in cells)
                if r1 - r0 < 2 or c1 - c0 < 2:
                    continue
                # Expected border cells (perimeter of bounding box)
                expected = set()
                for c in range(c0, c1 + 1):
                    expected.add((r0, c)); expected.add((r1, c))
                for r in range(r0, r1 + 1):
                    expected.add((r, c0)); expected.add((r, c1))
                # Find gap: cells in expected but not in cells
                gap_cells = expected - cells
                # Must have exactly 1 gap cell, and it must be on one wall
                if len(gap_cells) != 1:
                    continue
                gr, gc = next(iter(gap_cells))
                # Determine which wall and outward direction
                if gr == r0:
                    direction = 'up'; gap_r, gap_c = gr, gc
                elif gr == r1:
                    direction = 'down'; gap_r, gap_c = gr, gc
                elif gc == c0:
                    direction = 'left'; gap_r, gap_c = gr, gc
                elif gc == c1:
                    direction = 'right'; gap_r, gap_c = gr, gc
                else:
                    continue
                # Interior cells
                interior = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]
                return border_color, r0, c0, r1, c1, direction, gap_r, gap_c, interior
            return None

        def _apply(grid, analysis, fill_color=8):
            if analysis is None:
                return None
            g = [list(map(int, row)) for row in grid]
            rows, cols = len(g), len(g[0])
            border_color, r0, c0, r1, c1, direction, gap_r, gap_c, interior = analysis
            out = [row[:] for row in g]
            # Fill interior
            for r, c in interior:
                out[r][c] = fill_color
            # Fill gap cell
            out[gap_r][gap_c] = fill_color
            # Project outward from gap
            if direction == 'up':
                for r in range(gap_r - 1, -1, -1):
                    out[r][gap_c] = fill_color
            elif direction == 'down':
                for r in range(gap_r + 1, rows):
                    out[r][gap_c] = fill_color
            elif direction == 'left':
                for c in range(gap_c - 1, -1, -1):
                    out[gap_r][c] = fill_color
            elif direction == 'right':
                for c in range(gap_c + 1, cols):
                    out[gap_r][c] = fill_color
            return out

        # Validate on training examples
        for ex in train:
            analysis = _analyze(ex["input"])
            if analysis is None:
                return None
            result = _apply(ex["input"], analysis)
            if result is None or result != [list(map(int, r)) for r in ex["output"]]:
                return None

        test_analysis = _analyze(test_input)
        if test_analysis is None:
            return None
        result = _apply(test_input, test_analysis)
        if result is None or result == [list(map(int, r)) for r in test_input]:
            return None
        return result

    def _try_cross_quadrant_fill(self, train, test_input):
        """5x5-like grid with cross (middle row+col all same color), each quadrant has one
        cross-colored cell. Fill each quadrant's cross cell with the color missing so all
        quadrants have the same multiset of colors."""
        from collections import Counter

        def _solve(grid):
            rows, cols = len(grid), len(grid[0])
            if rows < 3 or cols < 3:
                return None
            # Find single uniform row and col (the cross)
            def is_uniform_row(r):
                return len(set(grid[r])) == 1
            def is_uniform_col(c):
                return len(set(grid[r][c] for r in range(rows))) == 1
            cross_rows = [r for r in range(rows) if is_uniform_row(r)]
            cross_cols = [c for c in range(cols) if is_uniform_col(c)]
            if len(cross_rows) != 1 or len(cross_cols) != 1:
                return None
            mr, mc = cross_rows[0], cross_cols[0]
            cross_val = grid[mr][mc]
            top_rows = list(range(0, mr))
            bot_rows = list(range(mr + 1, rows))
            left_cols = list(range(0, mc))
            right_cols = list(range(mc + 1, cols))
            if not top_rows or not bot_rows or not left_cols or not right_cols:
                return None
            quadrants = [
                (top_rows, left_cols),
                (top_rows, right_cols),
                (bot_rows, left_cols),
                (bot_rows, right_cols),
            ]
            q_vals = []
            q_cross_pos = []
            for qr, qc in quadrants:
                vals = [grid[r][c] for r in qr for c in qc]
                cross_cells = [(r, c) for r in qr for c in qc if grid[r][c] == cross_val]
                if len(cross_cells) != 1:
                    return None
                q_vals.append([v for v in vals if v != cross_val])
                q_cross_pos.append(cross_cells[0])
            all_non_cross = [v for q in q_vals for v in q]
            if not all_non_cross:
                return None
            color_counts = Counter(all_non_cross)
            target = {v: -(-cnt // 4) for v, cnt in color_counts.items()}  # ceil div
            out = [list(row) for row in grid]
            for qi in range(4):
                nc_count = Counter(q_vals[qi])
                missing = []
                for v, needed in target.items():
                    for _ in range(needed - nc_count.get(v, 0)):
                        missing.append(v)
                if len(missing) != 1:
                    return None
                r, c = q_cross_pos[qi]
                out[r][c] = missing[0]
            return out

        for ex in train:
            result = _solve(ex["input"])
            if result is None or result != ex["output"]:
                return None
        return _solve(test_input)

    def _try_crossing_bars_interrupted(self, train, test_input):
        """Grid has vertical stripe(s) and horizontal stripe(s). In input, one bar
        interrupts the other at the crossing. Output restores the interrupted bar."""
        from collections import Counter

        def _solve(grid):
            rows, cols = len(grid), len(grid[0])
            all_vals = [v for row in grid for v in row]
            bg = Counter(all_vals).most_common(1)[0][0]
            # Find horizontal stripes: rows where a non-bg color dominates
            h_rows = []
            for r in range(rows):
                non_bg = [v for v in grid[r] if v != bg]
                if len(non_bg) >= cols // 2:
                    h_color = Counter(non_bg).most_common(1)[0][0]
                    h_rows.append((r, h_color))
            # Find vertical stripes: cols where a non-bg color dominates
            v_cols = []
            for c in range(cols):
                col = [grid[r][c] for r in range(rows)]
                non_bg = [v for v in col if v != bg]
                if len(non_bg) >= rows // 2:
                    v_color = Counter(non_bg).most_common(1)[0][0]
                    v_cols.append((c, v_color))
            if not h_rows or not v_cols:
                return None
            out = [list(row) for row in grid]
            for r, h_color in h_rows:
                for c, v_color in v_cols:
                    if h_color == v_color:
                        continue
                    cell = grid[r][c]
                    # Check which bar is interrupted at this crossing
                    v_col_vals = [grid[rr][c] for rr in range(rows) if rr != r]
                    h_row_vals = [grid[r][cc] for cc in range(cols) if cc != c]
                    v_dominant = Counter(v_col_vals).most_common(1)[0][0]
                    h_dominant = Counter(h_row_vals).most_common(1)[0][0]
                    if v_dominant == v_color and cell != v_color:
                        out[r][c] = v_color  # v bar interrupted → restore
                    elif h_dominant == h_color and cell != h_color:
                        out[r][c] = h_color  # h bar interrupted → restore
            return out

        for ex in train:
            result = _solve(ex["input"])
            if result is None or result != ex["output"]:
                return None
        result = _solve(test_input)
        if result is None or result == test_input:
            return None
        return result

    def _try_dot_to_filled_ring(self, train, test_input):
        """Each non-bg dot at (r,c) with color v stays, all 8 neighbors set to a learned
        ring color (mapped per dot color). Handles edge dots by clamping."""
        from collections import Counter

        def _learn(train_examples):
            all_v = [v for ex in train_examples for row in ex["input"] for v in row]
            bg = Counter(all_v).most_common(1)[0][0]
            color_map = {}
            for ex in train_examples:
                rows, cols = len(ex["input"]), len(ex["input"][0])
                for r in range(rows):
                    for c in range(cols):
                        v = ex["input"][r][c]
                        if v == bg:
                            continue
                        # Collect 8-neighbor colors in output
                        nb_colors = set()
                        for dr, dc in [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]:
                            nr, nc = r+dr, c+dc
                            if 0 <= nr < rows and 0 <= nc < cols:
                                ov = ex["output"][nr][nc]
                                if ov != bg and ov != v:
                                    nb_colors.add(ov)
                        if not nb_colors:
                            return None, None
                        if len(nb_colors) != 1:
                            return None, None
                        ring_c = next(iter(nb_colors))
                        if v in color_map and color_map[v] != ring_c:
                            return None, None
                        color_map[v] = ring_c
            return bg, color_map

        def _apply(grid, bg, color_map):
            rows, cols = len(grid), len(grid[0])
            out = [list(row) for row in grid]
            for r in range(rows):
                for c in range(cols):
                    v = grid[r][c]
                    if v == bg or v not in color_map:
                        continue
                    ring_c = color_map[v]
                    for dr, dc in [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]:
                        nr, nc = r+dr, c+dc
                        if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == bg:
                            out[nr][nc] = ring_c
            return out

        bg, color_map = _learn(train)
        if bg is None or not color_map:
            return None
        for ex in train:
            result = _apply(ex["input"], bg, color_map)
            if result != ex["output"]:
                return None
        return _apply(test_input, bg, color_map)

    def _try_dot_to_corner_edge_ring(self, train, test_input):
        """Each non-bg dot at (r,c): center becomes bg, 4 diagonal neighbors = dot_color,
        4 orthogonal neighbors = a learned edge_color."""
        from collections import Counter

        def _learn_edge_color(train_examples):
            all_v = [v for ex in train_examples for row in ex["input"] for v in row]
            bg = Counter(all_v).most_common(1)[0][0]
            edge_color = None
            for ex in train_examples:
                rows, cols = len(ex["input"]), len(ex["input"][0])
                for r in range(rows):
                    for c in range(cols):
                        dot_v = ex["input"][r][c]
                        if dot_v == bg:
                            continue
                        # Center should become bg in output
                        if ex["output"][r][c] != bg:
                            return None, None
                        # Orthogonal neighbors should be edge_color
                        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                            nr, nc = r+dr, c+dc
                            if 0 <= nr < rows and 0 <= nc < cols:
                                ov = ex["output"][nr][nc]
                                if ov == bg or ov == dot_v:
                                    return None, None
                                if edge_color is None:
                                    edge_color = ov
                                elif edge_color != ov:
                                    return None, None
                        # Diagonal neighbors should be dot_v
                        for dr, dc in [(-1,-1),(-1,1),(1,-1),(1,1)]:
                            nr, nc = r+dr, c+dc
                            if 0 <= nr < rows and 0 <= nc < cols:
                                if ex["output"][nr][nc] != dot_v:
                                    return None, None
            return bg, edge_color

        def _apply(grid, bg, edge_color):
            rows, cols = len(grid), len(grid[0])
            out = [list(row) for row in grid]
            dots = [(r, c, grid[r][c]) for r in range(rows) for c in range(cols) if grid[r][c] != bg]
            for r, c, dot_v in dots:
                out[r][c] = bg
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr, nc = r+dr, c+dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        out[nr][nc] = edge_color
                for dr, dc in [(-1,-1),(-1,1),(1,-1),(1,1)]:
                    nr, nc = r+dr, c+dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        out[nr][nc] = dot_v
            return out

        bg, edge_color = _learn_edge_color(train)
        if bg is None or edge_color is None:
            return None
        for ex in train:
            result = _apply(ex["input"], bg, edge_color)
            if result != ex["output"]:
                return None
        return _apply(test_input, bg, edge_color)

    def _try_shift_cluster_color(self, train, test_input):
        """A cluster of cells with color S shifts by a fixed offset (dr, dc) and becomes
        color D. Learn S, D, and (dr, dc) from training examples."""
        from collections import Counter

        def _learn(train_examples):
            all_in = [v for ex in train_examples for row in ex["input"] for v in row]
            all_out = [v for ex in train_examples for row in ex["output"] for v in row]
            bg = Counter(all_in).most_common(1)[0][0]
            in_colors = set(all_in) - {bg}
            out_colors = set(all_out) - {bg}
            src_colors = in_colors - out_colors  # colors only in input
            dst_colors = out_colors - in_colors  # colors only in output
            if len(src_colors) != 1 or len(dst_colors) != 1:
                return None, None, None, None
            S = next(iter(src_colors))
            D = next(iter(dst_colors))
            # Find shift from first training example
            ex = train_examples[0]
            src_cells = [(r, c) for r in range(len(ex["input"])) for c in range(len(ex["input"][0])) if ex["input"][r][c] == S]
            dst_cells = [(r, c) for r in range(len(ex["output"])) for c in range(len(ex["output"][0])) if ex["output"][r][c] == D]
            if len(src_cells) != len(dst_cells) or not src_cells:
                return None, None, None, None
            dr = dst_cells[0][0] - src_cells[0][0]
            dc = dst_cells[0][1] - src_cells[0][1]
            # Verify consistent shift
            for (sr, sc), (dr2, dc2) in zip(src_cells, [(r,c) for r,c in dst_cells]):
                if dr2 - sr != dr or dc2 - sc != dc:
                    return None, None, None, None
            return bg, S, D, (dr, dc)

        def _apply(grid, bg, S, D, offset):
            dr, dc = offset
            rows, cols = len(grid), len(grid[0])
            out = [list(row) for row in grid]
            for r in range(rows):
                for c in range(cols):
                    if grid[r][c] == S:
                        out[r][c] = bg  # erase source
            for r in range(rows):
                for c in range(cols):
                    if grid[r][c] == S:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols:
                            out[nr][nc] = D
            return out

        bg, S, D, offset = _learn(train)
        if bg is None:
            return None
        for ex in train:
            result = _apply(ex["input"], bg, S, D, offset)
            if result != ex["output"]:
                return None
        return _apply(test_input, bg, S, D, offset)

    def _try_two_row_checkerboard(self, train, test_input):
        """2-row grid where each row is a solid color A/B. Output: checkerboard
        interleaving: output[r][c] = A if (r+c)%2==0 else B."""
        from collections import Counter

        def _check(grid):
            if len(grid) != 2:
                return None
            r0 = set(grid[0])
            r1 = set(grid[1])
            if len(r0) != 1 or len(r1) != 1:
                return None
            A, B = grid[0][0], grid[1][0]
            if A == B:
                return None
            cols = len(grid[0])
            return [[A if (r+c)%2==0 else B for c in range(cols)] for r in range(2)]

        for ex in train:
            result = _check(ex["input"])
            if result is None or result != ex["output"]:
                return None
        return _check(test_input)

    def _try_keep_majority_replace_rest(self, train, test_input):
        """Keep the most-common color; replace all other colors with a learned target color."""
        from collections import Counter

        def _find_target(train_examples):
            all_in = set(v for ex in train_examples for row in ex["input"] for v in row)
            all_out = set(v for ex in train_examples for row in ex["output"] for v in row)
            new_colors = all_out - all_in
            if len(new_colors) != 1:
                return None
            return next(iter(new_colors))

        def _apply(grid, target):
            all_v = [v for row in grid for v in row]
            majority = Counter(all_v).most_common(1)[0][0]
            return [[v if v == majority else target for v in row] for row in grid]

        target = _find_target(train)
        if target is None:
            return None
        for ex in train:
            result = _apply(ex["input"], target)
            if result != ex["output"]:
                return None
        return _apply(test_input, target)

    def _try_ring_to_cross(self, train, test_input):
        """3x3 hollow rings (8-cell border, hollow center) → cross/plus pattern with a
        learned replacement color. Other shapes unchanged."""
        from collections import Counter

        def _find_rings(grid, fg, bg):
            rows, cols = len(grid), len(grid[0])
            rings = []
            for r in range(1, rows - 1):
                for c in range(1, cols - 1):
                    if grid[r][c] != bg:
                        continue
                    corners = [(r-1,c-1),(r-1,c+1),(r+1,c-1),(r+1,c+1)]
                    ortho = [(r-1,c),(r+1,c),(r,c-1),(r,c+1)]
                    if all(grid[nr][nc] == fg for nr,nc in corners+ortho):
                        rings.append((r, c))
            return rings

        def _apply(grid, fg, bg, ring_color):
            rows, cols = len(grid), len(grid[0])
            out = [list(row) for row in grid]
            rings = _find_rings(grid, fg, bg)
            for r, c in rings:
                # Erase the ring border
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        out[r+dr][c+dc] = bg
                # Draw cross: center + 4 orthogonal = ring_color
                out[r][c] = ring_color
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    out[r+dr][c+dc] = ring_color
            return out

        all_in = [v for ex in train for row in ex["input"] for v in row]
        all_out = [v for ex in train for row in ex["output"] for v in row]
        bg = Counter(all_in).most_common(1)[0][0]
        fg_colors = set(all_in) - {bg}
        new_colors = set(all_out) - set(all_in)
        if len(fg_colors) != 1 or len(new_colors) != 1:
            return None
        fg = next(iter(fg_colors))
        ring_color = next(iter(new_colors))

        for ex in train:
            rings = _find_rings(ex["input"], fg, bg)
            if not rings:
                return None
            result = _apply(ex["input"], fg, bg, ring_color)
            if result != ex["output"]:
                return None
        return _apply(test_input, fg, bg, ring_color)

    def _try_tallest_shortest_column(self, train, test_input):
        """Columns of a single color with varying heights: tallest→color1, shortest→color2,
        all others→bg. Colors 1 and 2 are learned from training."""
        from collections import Counter

        def _get_columns(grid, dot_color):
            rows, cols = len(grid), len(grid[0])
            result = {}
            for c in range(cols):
                col_cells = [(r, c) for r in range(rows) if grid[r][c] == dot_color]
                if col_cells:
                    result[c] = col_cells
            return result

        def _learn(train_examples):
            all_in = [v for ex in train_examples for row in ex["input"] for v in row]
            all_out = [v for ex in train_examples for row in ex["output"] for v in row]
            bg = Counter(all_in).most_common(1)[0][0]
            dot_colors = set(all_in) - {bg}
            if len(dot_colors) != 1:
                return None, None, None, None
            dot_color = next(iter(dot_colors))
            out_colors = set(all_out) - {bg}
            if len(out_colors) != 2:
                return None, None, None, None
            # Figure out which out_color = tallest, which = shortest
            ex = train_examples[0]
            cols_map = _get_columns(ex["input"], dot_color)
            if len(cols_map) < 2:
                return None, None, None, None
            sorted_cols = sorted(cols_map.items(), key=lambda kv: len(kv[1]))
            shortest_c = sorted_cols[0][0]
            tallest_c = sorted_cols[-1][0]
            short_cells = set(sorted_cols[0][1])
            tall_cells = set(sorted_cols[-1][1])
            color_for_shortest = None
            color_for_tallest = None
            for r in range(len(ex["output"])):
                for c in range(len(ex["output"][0])):
                    v = ex["output"][r][c]
                    if v == bg:
                        continue
                    if (r, c) in short_cells:
                        color_for_shortest = v
                    elif (r, c) in tall_cells:
                        color_for_tallest = v
            if color_for_shortest is None or color_for_tallest is None:
                return None, None, None, None
            return bg, dot_color, color_for_tallest, color_for_shortest

        def _apply(grid, bg, dot_color, tallest_color, shortest_color):
            rows, cols = len(grid), len(grid[0])
            cols_map = _get_columns(grid, dot_color)
            if len(cols_map) < 2:
                return None
            sorted_cols = sorted(cols_map.items(), key=lambda kv: len(kv[1]))
            shortest_cells = set(sorted_cols[0][1])
            tallest_cells = set(sorted_cols[-1][1])
            out = [[bg]*cols for _ in range(rows)]
            for c_idx, cells in cols_map.items():
                cell_set = set(cells)
                if cell_set == tallest_cells:
                    color = tallest_color
                elif cell_set == shortest_cells:
                    color = shortest_color
                else:
                    continue  # middle columns → bg
                for r, c in cells:
                    out[r][c] = color
            return out

        bg, dot_color, tallest_color, shortest_color = _learn(train)
        if bg is None:
            return None
        for ex in train:
            result = _apply(ex["input"], bg, dot_color, tallest_color, shortest_color)
            if result is None or result != ex["output"]:
                return None
        return _apply(test_input, bg, dot_color, tallest_color, shortest_color)

    def _try_alternating_row_shift(self, train, test_input):
        """Alternating pattern rows: rows where non-bg values are in descending order at
        their positions shift those values right by 1 (vacating position fills with bg).
        Ascending-order rows and all-bg rows stay unchanged."""
        from collections import Counter

        def _find_bg(train_examples):
            all_v = [v for ex in train_examples for row in ex["input"] for v in row]
            return Counter(all_v).most_common(1)[0][0]

        def _row_type(row, bg):
            """Returns (positions, values, 'asc'|'desc'|'bg') for a row."""
            non_bg = [(c, v) for c, v in enumerate(row) if v != bg]
            if not non_bg:
                return None, None, 'bg'
            if len(non_bg) != 2:
                return None, None, None
            (c1, v1), (c2, v2) = non_bg
            if v1 < v2:
                return (c1, c2), (v1, v2), 'asc'
            elif v1 > v2:
                return (c1, c2), (v1, v2), 'desc'
            return None, None, None

        def _transform(row, bg):
            _, _, rtype = _row_type(row, bg)
            if rtype != 'desc':
                return list(row)
            out = list(row)
            non_bg = [(c, v) for c, v in enumerate(row) if v != bg]
            # Erase first, then write shifted positions (avoid collision)
            for c, v in non_bg:
                out[c] = bg
            for c, v in non_bg:
                if c + 1 < len(row):
                    out[c + 1] = v
            return out

        bg = _find_bg(train)
        # Validate on training
        for ex in train:
            result = [_transform(row, bg) for row in ex["input"]]
            if result != ex["output"]:
                return None
            # Ensure at least one desc row was shifted (not identity)
            if result == ex["input"]:
                return None
        result = [_transform(row, bg) for row in test_input]
        if result == test_input:
            return None
        return result

    def _try_diamond_center_mark(self, train, test_input):
        """Find 'diamond' structures: two horizontal dominos + two vertical dominos forming
        a symmetric cross (equal distance k from center). Place a mark color at center."""
        from collections import Counter

        def _find_h_dominos(grid, fg, bg):
            rows, cols = len(grid), len(grid[0])
            result = []
            for r in range(rows):
                for c in range(cols - 1):
                    if grid[r][c] == fg and grid[r][c+1] == fg:
                        # Check not part of a triple
                        left_ok = c == 0 or grid[r][c-1] != fg
                        right_ok = c+2 >= cols or grid[r][c+2] != fg
                        if left_ok and right_ok:
                            result.append((r, c, c+1))  # row, left_col, right_col
            return result

        def _find_v_dominos(grid, fg, bg):
            rows, cols = len(grid), len(grid[0])
            result = []
            for c in range(cols):
                for r in range(rows - 1):
                    if grid[r][c] == fg and grid[r+1][c] == fg:
                        top_ok = r == 0 or grid[r-1][c] != fg
                        bot_ok = r+2 >= rows or grid[r+2][c] != fg
                        if top_ok and bot_ok:
                            result.append((r, r+1, c))  # top_row, bot_row, col
            return result

        def _apply(grid, fg, bg, mark_color):
            rows, cols = len(grid), len(grid[0])
            out = [list(row) for row in grid]
            h_doms = _find_h_dominos(grid, fg, bg)
            v_doms = _find_v_dominos(grid, fg, bg)
            # Index vertical dominos by column
            v_by_col = {}
            for tr, br, c in v_doms:
                v_by_col.setdefault(c, []).append((tr, br))
            # For each pair of horizontal dominos in same row
            h_by_row = {}
            for r, lc, rc in h_doms:
                h_by_row.setdefault(r, []).append((lc, rc))
            for hr, h_list in h_by_row.items():
                for i in range(len(h_list)):
                    for j in range(i+1, len(h_list)):
                        lc1, rc1 = h_list[i]  # left domino
                        lc2, rc2 = h_list[j]  # right domino
                        if rc1 >= lc2:
                            continue  # overlapping
                        # Midpoint col must be integer
                        if (rc1 + lc2) % 2 != 0:
                            continue
                        mid_c = (rc1 + lc2) // 2
                        k = mid_c - rc1  # = lc2 - mid_c
                        if k <= 0:
                            continue
                        # Check vertical dominos at mid_c with symmetric distance k
                        for tr, br in v_by_col.get(mid_c, []):
                            if (hr - br) == k:  # above: inner edge (br) is k above hr
                                # Look for below domino
                                for tr2, br2 in v_by_col.get(mid_c, []):
                                    if (tr2 - hr) == k:  # below: inner edge (tr2) is k below hr
                                        out[hr][mid_c] = mark_color
            return out

        all_in = [v for ex in train for row in ex["input"] for v in row]
        all_out = [v for ex in train for row in ex["output"] for v in row]
        bg = Counter(all_in).most_common(1)[0][0]
        fg_colors = set(all_in) - {bg}
        new_colors = set(all_out) - set(all_in)
        if len(fg_colors) != 1 or len(new_colors) != 1:
            return None
        fg = next(iter(fg_colors))
        mark_color = next(iter(new_colors))

        for ex in train:
            result = _apply(ex["input"], fg, bg, mark_color)
            if result != ex["output"]:
                return None
        result = _apply(test_input, fg, bg, mark_color)
        if result == test_input:
            return None
        return result

    def _try_extend_bars_to_cross(self, train, test_input):
        """Two partial bars (a vertical run + a horizontal run, different colors) each
        extend to span the full column/row. A 3rd color (or learned color) is placed at
        the intersection."""
        from collections import Counter

        def _find_bars(grid):
            rows, cols = len(grid), len(grid[0])
            all_vals = [grid[r][c] for r in range(rows) for c in range(cols)]
            bg = Counter(all_vals).most_common(1)[0][0]
            non_bg = [(r, c, grid[r][c]) for r in range(rows) for c in range(cols) if grid[r][c] != bg]
            if not non_bg:
                return None
            # Group by color
            by_color = {}
            for r, c, v in non_bg:
                by_color.setdefault(v, []).append((r, c))
            # Look for exactly 2 colors, one forming a vertical run, one a horizontal run
            v_bar = None  # (col, color, row_set)
            h_bar = None  # (row, color, col_set)
            for color, cells in by_color.items():
                col_set = set(c for r, c in cells)
                row_set = set(r for r, c in cells)
                if len(col_set) == 1:
                    # All same column → vertical run
                    col = next(iter(col_set))
                    if v_bar is None:
                        v_bar = (col, color, row_set)
                    else:
                        return None  # two vertical bars
                elif len(row_set) == 1:
                    # All same row → horizontal run
                    row = next(iter(row_set))
                    if h_bar is None:
                        h_bar = (row, color, col_set)
                    else:
                        return None  # two horizontal bars
                else:
                    return None  # not a clean bar
            if v_bar is None or h_bar is None:
                return None
            v_col, v_color, v_rows = v_bar
            h_row, h_color, h_cols = h_bar
            if v_color == h_color:
                return None
            return bg, v_col, v_color, h_row, h_color

        # Learn intersection color from training examples
        inter_color = None
        for ex in train:
            parsed = _find_bars(ex["input"])
            if parsed is None:
                return None
            bg, v_col, v_color, h_row, h_color = parsed
            # Build expected output
            rows, cols = len(ex["input"]), len(ex["input"][0])
            out = [list(row) for row in ex["input"]]
            for r in range(rows):
                out[r][v_col] = v_color
            for c in range(cols):
                out[h_row][c] = h_color
            # Find the intersection color in actual output
            actual_inter = ex["output"][h_row][v_col]
            if actual_inter == v_color or actual_inter == h_color:
                # intersection is one of the bar colors — use bg or a 3rd color
                # determine: is there a 3rd color in output not in input?
                in_colors = set(v for row in ex["input"] for v in row)
                out_colors = set(v for row in ex["output"] for v in row)
                new_cols = out_colors - in_colors
                if new_cols:
                    # Check if actual output matches out with intersection replaced
                    cand = next(iter(new_cols))
                    out[h_row][v_col] = cand
                    if out != ex["output"]:
                        return None
                    if inter_color is None:
                        inter_color = cand
                    elif inter_color != cand:
                        return None
                else:
                    out[h_row][v_col] = actual_inter
                    if out != ex["output"]:
                        return None
                    if inter_color is None:
                        inter_color = actual_inter
                    elif inter_color != actual_inter:
                        return None
            else:
                out[h_row][v_col] = actual_inter
                if out != ex["output"]:
                    return None
                if inter_color is None:
                    inter_color = actual_inter
                elif inter_color != actual_inter:
                    return None

        # Apply to test input
        parsed = _find_bars(test_input)
        if parsed is None:
            return None
        bg, v_col, v_color, h_row, h_color = parsed
        rows, cols = len(test_input), len(test_input[0])
        out = [list(row) for row in test_input]
        for r in range(rows):
            out[r][v_col] = v_color
        for c in range(cols):
            out[h_row][c] = h_color
        out[h_row][v_col] = inter_color
        return out

    def _try_l_shape_diagonal_ray(self, train, test_input):
        """Each L-shaped piece (3 of 4 cells in a 2x2) shoots a diagonal ray from
        its open (missing) corner in the direction away from the block's center,
        until the ray goes out of bounds."""
        from collections import Counter

        def _find_l_shapes(grid):
            rows, cols = len(grid), len(grid[0])
            all_vals = [grid[r][c] for r in range(rows) for c in range(cols)]
            bg = Counter(all_vals).most_common(1)[0][0]
            shapes = []
            seen = set()
            for r in range(rows - 1):
                for c in range(cols - 1):
                    # Check 2x2 block
                    cells = [(r, c), (r, c+1), (r+1, c), (r+1, c+1)]
                    fg_cells = [(rr, cc) for rr, cc in cells if grid[rr][cc] != bg]
                    bg_cells = [(rr, cc) for rr, cc in cells if grid[rr][cc] == bg]
                    if len(fg_cells) == 3 and len(bg_cells) == 1:
                        # Check all fg cells same color
                        colors = set(grid[rr][cc] for rr, cc in fg_cells)
                        if len(colors) != 1:
                            continue
                        fg_color = next(iter(colors))
                        # Check none of the fg cells appear in any other shape
                        key = frozenset(fg_cells)
                        if key in seen:
                            continue
                        seen.add(key)
                        open_r, open_c = bg_cells[0]
                        # Direction: from center of 2x2 to open corner
                        center_r = r + 0.5
                        center_c = c + 0.5
                        dr = 1 if open_r > center_r else -1
                        dc = 1 if open_c > center_c else -1
                        shapes.append((open_r, open_c, dr, dc, fg_color))
            return shapes, bg

        def _apply(grid, shapes, bg):
            rows, cols = len(grid), len(grid[0])
            out = [list(row) for row in grid]
            for open_r, open_c, dr, dc, color in shapes:
                r, c = open_r + dr, open_c + dc
                while 0 <= r < rows and 0 <= c < cols:
                    if out[r][c] != bg:
                        break  # stop at non-bg cell
                    out[r][c] = color
                    r += dr
                    c += dc
            return out

        # Validate on training examples
        for ex in train:
            shapes, bg = _find_l_shapes(ex["input"])
            if not shapes:
                return None
            result = _apply(ex["input"], shapes, bg)
            if result != ex["output"]:
                return None

        shapes, bg = _find_l_shapes(test_input)
        if not shapes:
            return None
        return _apply(test_input, shapes, bg)

    def _try_2x2_block_corner_markers(self, train, test_input):
        """Each 2x2 block of a foreground color gets 4 corner markers placed
        diagonally just outside the block corners: top-left=1, top-right=2,
        bottom-left=3, bottom-right=4 (or learned colors)."""
        from collections import Counter

        def _find_2x2_blocks(grid, fg):
            rows, cols = len(grid), len(grid[0])
            blocks = []
            for r in range(rows - 1):
                for c in range(cols - 1):
                    if (grid[r][c] == fg and grid[r][c+1] == fg and
                            grid[r+1][c] == fg and grid[r+1][c+1] == fg):
                        blocks.append((r, c))
            return blocks

        def _apply(grid, fg, markers):
            rows, cols = len(grid), len(grid[0])
            out = [list(row) for row in grid]
            blocks = _find_2x2_blocks(grid, fg)
            for r, c in blocks:
                # (r-1, c-1) → markers[0]
                # (r-1, c+2) → markers[1]
                # (r+2, c-1) → markers[2]
                # (r+2, c+2) → markers[3]
                positions = [(r-1, c-1), (r-1, c+2), (r+2, c-1), (r+2, c+2)]
                for (pr, pc), m in zip(positions, markers):
                    if 0 <= pr < rows and 0 <= pc < cols:
                        out[pr][pc] = m
            return out

        # Determine fg color and marker colors from training
        all_in = [v for ex in train for row in ex["input"] for v in row]
        all_out = [v for ex in train for row in ex["output"] for v in row]
        bg = Counter(all_in).most_common(1)[0][0]
        new_colors = sorted(set(all_out) - set(all_in))
        if len(new_colors) != 4:
            return None
        markers = new_colors  # [1,2,3,4] or similar sorted order

        # Find fg: color that forms 2x2 blocks
        non_bg = [v for v in all_in if v != bg]
        if not non_bg:
            return None
        fg_counts = Counter(non_bg)
        # fg is the color that forms 2x2 blocks
        fg = None
        for ex in train:
            blocks = None
            for candidate in fg_counts:
                b = _find_2x2_blocks(ex["input"], candidate)
                if b:
                    if fg is None:
                        fg = candidate
                    elif fg != candidate:
                        return None
                    blocks = b
                    break
            if blocks is None:
                return None

        if fg is None:
            return None

        # Learn marker positions from training: which corner → which color
        # Try to determine the mapping of positions to marker colors
        # Look at the first training example to determine the ordering
        marker_map = None
        for ex in train:
            blocks = _find_2x2_blocks(ex["input"], fg)
            if not blocks:
                continue
            rows, cols = len(ex["input"]), len(ex["input"][0])
            # For each block, find what colors appear at corner positions in output
            for r, c in blocks:
                positions = [(r-1, c-1), (r-1, c+2), (r+2, c-1), (r+2, c+2)]
                in_grid = [v for pr, pc in positions
                           if 0 <= pr < rows and 0 <= pc < cols
                           for v in [ex["input"][pr][pc]]]
                out_vals = []
                for pr, pc in positions:
                    if 0 <= pr < rows and 0 <= pc < cols:
                        v = ex["output"][pr][pc]
                        if v != ex["input"][pr][pc]:
                            out_vals.append(v)
                        else:
                            out_vals.append(None)
                if any(v is not None for v in out_vals):
                    m = [v if v is not None else -1 for v in out_vals]
                    assigned = [v for v in m if v != -1]
                    if len(assigned) == 4 and len(set(assigned)) == 4:
                        if marker_map is None:
                            marker_map = m
                        elif marker_map != m:
                            return None
                        break

        if marker_map is None or any(v == -1 for v in marker_map):
            # Fall back to sorted new colors in position order
            marker_map = new_colors

        for ex in train:
            result = _apply(ex["input"], fg, marker_map)
            if result != ex["output"]:
                return None
        return _apply(test_input, fg, marker_map)

    def _try_fill_zero_rect_interior(self, train, test_input):
        """Find a rectangular region of all-zeros embedded in the grid (bordered by
        non-zero cells). Fill the inner cells (excluding 1-cell border) with a
        marker color learned from training."""
        from collections import Counter

        def _find_embedded_zero_rect(grid):
            """Find all rectangles that are all-zeros and have non-zero neighbors on all 4 sides."""
            rows, cols = len(grid), len(grid[0])
            all_vals = [grid[r][c] for r in range(rows) for c in range(cols)]
            bg = Counter(all_vals).most_common(1)[0][0]
            zero_val = 0  # the "hole" color

            # Build prefix sums for zero cells
            is_zero = [[1 if grid[r][c] == zero_val else 0 for c in range(cols)] for r in range(rows)]
            psum = [[0]*(cols+1) for _ in range(rows+1)]
            for r in range(rows):
                for c in range(cols):
                    psum[r+1][c+1] = psum[r][c+1] + psum[r+1][c] - psum[r][c] + is_zero[r][c]

            def rect_all_zero(r1, c1, r2, c2):
                # r1,c1 inclusive top-left; r2,c2 inclusive bottom-right
                total = psum[r2+1][c2+1] - psum[r1][c2+1] - psum[r2+1][c1] + psum[r1][c1]
                expected = (r2 - r1 + 1) * (c2 - c1 + 1)
                return total == expected

            candidates = []
            # Try all rectangles (efficient: iterate over all possible row pairs, use column sweep)
            # For each pair of rows, find all-zero columns using the prefix sum
            # This is O(rows^2 * cols^2) in worst case but grids are small
            for r1 in range(rows):
                for r2 in range(r1+2, rows):  # min height = 3 (to have inner rows)
                    for c1 in range(cols):
                        for c2 in range(c1+2, cols):  # min width = 3
                            if not rect_all_zero(r1, c1, r2, c2):
                                continue
                            # Check bordered: cells just outside should not ALL be zero
                            # (at least some non-zero in adjacent row/col)
                            has_border_above = r1 > 0 and any(grid[r1-1][c] != zero_val for c in range(c1, c2+1))
                            has_border_below = r2 < rows-1 and any(grid[r2+1][c] != zero_val for c in range(c1, c2+1))
                            has_border_left = c1 > 0 and any(grid[r][c1-1] != zero_val for r in range(r1, r2+1))
                            has_border_right = c2 < cols-1 and any(grid[r][c2+1] != zero_val for r in range(r1, r2+1))
                            if has_border_above or has_border_below or has_border_left or has_border_right:
                                candidates.append((r1, c1, r2, c2))
            return candidates

        def _get_marker_color(ex, rect):
            r1, c1, r2, c2 = rect
            in_rows = len(ex["input"])
            out_rows = len(ex["output"])
            if in_rows != out_rows:
                return None
            changed = set()
            for r in range(min(in_rows, out_rows)):
                in_row = ex["input"][r]
                out_row = ex["output"][r]
                if len(in_row) != len(out_row):
                    return None
                for c in range(len(out_row)):
                    if out_row[c] != in_row[c]:
                        changed.add(out_row[c])
            if len(changed) == 1:
                return next(iter(changed))
            return None

        # Learn from training: find the unique all-zero rectangle and marker color
        marker_color = None
        for ex in train:
            candidates = _find_embedded_zero_rect(ex["input"])
            if not candidates:
                return None
            # Find which candidate matches the output
            matched = None
            for r1, c1, r2, c2 in candidates:
                inner_rows = range(r1+1, r2)
                inner_cols = range(c1+1, c2)
                if not inner_rows or not inner_cols:
                    continue
                mc = _get_marker_color(ex, (r1, c1, r2, c2))
                if mc is None:
                    continue
                # Verify: all inner cells should be mc in output
                out_rows = len(ex["output"])
                out_cols = len(ex["output"][0]) if out_rows else 0
                if out_rows != len(ex["input"]) or out_cols != len(ex["input"][0]):
                    valid = False
                else:
                    valid = True
                    for r in inner_rows:
                        for c in inner_cols:
                            if r >= out_rows or c >= out_cols or ex["output"][r][c] != mc:
                                valid = False
                                break
                    # And no other cells changed
                    for r in range(out_rows):
                        for c in range(out_cols):
                            in_inner = r in inner_rows and c in inner_cols
                            if ex["output"][r][c] != ex["input"][r][c] and not in_inner:
                                valid = False
                                break
                if valid:
                    if marker_color is None:
                        marker_color = mc
                    elif marker_color != mc:
                        return None
                    matched = (r1, c1, r2, c2)
                    break
            if matched is None:
                return None

        if marker_color is None:
            return None

        # Apply to test input
        candidates = _find_embedded_zero_rect(test_input)
        if not candidates:
            return None

        # Pick the candidate that looks most like an embedded zero rect
        # Prefer largest area
        candidates.sort(key=lambda x: (x[2]-x[0]+1)*(x[3]-x[1]+1), reverse=True)
        # But only if it has a proper inner region
        for r1, c1, r2, c2 in candidates:
            inner_rows = range(r1+1, r2)
            inner_cols = range(c1+1, c2)
            if not inner_rows or not inner_cols:
                continue
            out = [list(row) for row in test_input]
            for r in inner_rows:
                for c in inner_cols:
                    out[r][c] = marker_color
            return out
        return None

    def _try_grid_room_diagonal_colors(self, train, test_input):
        """Grid divided by full-row/full-col dividers into rooms. Colors the rooms
        at evenly-spaced diagonal positions (top-left to bottom-right) with colors
        1, 2, 3 (or learned new colors in order)."""
        from collections import Counter

        def _get_dividers(grid):
            rows, cols = len(grid), len(grid[0])
            # Find divider color: forms at least one full row AND one full col
            all_colors = set(grid[r][c] for r in range(rows) for c in range(cols))
            div_color = None
            for color in sorted(all_colors):
                full_rows = [r for r in range(rows) if all(grid[r][c] == color for c in range(cols))]
                full_cols = [c for c in range(cols) if all(grid[r][c] == color for r in range(rows))]
                if full_rows and full_cols:
                    div_color = color
                    break
            if div_color is None:
                return None, None, None, None, None
            full_rows = sorted(r for r in range(rows) if all(grid[r][c] == div_color for c in range(cols)))
            full_cols = sorted(c for c in range(cols) if all(grid[r][c] == div_color for r in range(rows)))
            return div_color, full_rows, full_cols, rows, cols

        def _get_room_groups(div_rows, div_cols, rows, cols):
            # Row-groups: runs of non-divider rows
            row_groups = []
            start = 0
            for r in div_rows + [rows]:
                if r > start:
                    row_groups.append(list(range(start, r)))
                start = r + 1
            # Col-groups: runs of non-divider cols
            col_groups = []
            start = 0
            for c in div_cols + [cols]:
                if c > start:
                    col_groups.append(list(range(start, c)))
                start = c + 1
            return row_groups, col_groups

        def _apply(grid, row_groups, col_groups, colors):
            rows_count = len(row_groups)
            cols_count = len(col_groups)
            n_colors = len(colors)
            if n_colors < 2:
                return None
            out = [list(row) for row in grid]
            for k in range(n_colors):
                rg_idx = round(k * (rows_count - 1) / (n_colors - 1)) if n_colors > 1 else 0
                cg_idx = round(k * (cols_count - 1) / (n_colors - 1)) if n_colors > 1 else 0
                if rg_idx >= rows_count or cg_idx >= cols_count:
                    return None
                rg = row_groups[rg_idx]
                cg = col_groups[cg_idx]
                for r in rg:
                    for c in cg:
                        out[r][c] = colors[k]
            return out

        # Learn from training
        learned_colors = None
        for ex in train:
            dc, full_rows, full_cols, rows, cols = _get_dividers(ex["input"])
            if dc is None:
                return None
            row_groups, col_groups = _get_room_groups(full_rows, full_cols, rows, cols)
            if len(row_groups) < 2 or len(col_groups) < 2:
                return None
            # Find new colors in output (not in input)
            in_colors = set(v for row in ex["input"] for v in row)
            out_colors = set(v for row in ex["output"] for v in row)
            new_cols = sorted(out_colors - in_colors)
            if len(new_cols) < 2:
                return None
            result = _apply(ex["input"], row_groups, col_groups, new_cols)
            if result is None or result != ex["output"]:
                return None
            if learned_colors is None:
                learned_colors = new_cols
            elif learned_colors != new_cols:
                return None

        if learned_colors is None:
            return None

        dc, full_rows, full_cols, rows, cols = _get_dividers(test_input)
        if dc is None:
            return None
        row_groups, col_groups = _get_room_groups(full_rows, full_cols, rows, cols)
        if len(row_groups) < 2 or len(col_groups) < 2:
            return None
        return _apply(test_input, row_groups, col_groups, learned_colors)

    def _try_fg_color_swap_lookup(self, train, test_input):
        """Each training example has exactly 2 colors (bg=0 and one fg color).
        Mapping: fg→0, 0→new_color. Training examples collectively define a lookup
        table fg→new_color. Test applies the matching rule."""
        from collections import Counter

        bg = 0  # Always 0 for this pattern

        # Build lookup table from training
        lookup = {}
        for ex in train:
            in_colors = set(v for row in ex["input"] for v in row)
            out_colors = set(v for row in ex["output"] for v in row)
            fg_colors = in_colors - {bg}
            if len(fg_colors) != 1:
                return None
            fg = next(iter(fg_colors))
            new_colors = out_colors - {bg, fg}
            if len(new_colors) > 1:
                return None
            if len(new_colors) == 1:
                new_col = next(iter(new_colors))
            else:
                # new_col might already be in in_colors or output is just 0s
                # Check the cell mapping
                mapping = {}
                valid = True
                for r in range(len(ex["input"])):
                    for c in range(len(ex["input"][r])):
                        iv, ov = ex["input"][r][c], ex["output"][r][c]
                        if iv in mapping and mapping[iv] != ov:
                            valid = False
                            break
                        mapping[iv] = ov
                if not valid:
                    return None
                if fg not in mapping or mapping[fg] != bg:
                    return None
                new_col = mapping.get(bg, bg)
            # Verify the mapping fg→0, 0→new_col
            expected_out = [[new_col if v == bg else (bg if v == fg else v) for v in row] for row in ex["input"]]
            if expected_out != ex["output"]:
                return None
            if fg in lookup and lookup[fg] != new_col:
                return None
            lookup[fg] = new_col

        if not lookup:
            return None

        # Apply to test input
        test_in_colors = set(v for row in test_input for v in row)
        test_fg = test_in_colors - {bg}
        if len(test_fg) != 1:
            return None
        fg = next(iter(test_fg))
        if fg not in lookup:
            return None
        new_col = lookup[fg]
        return [[new_col if v == bg else (bg if v == fg else v) for v in row] for row in test_input]

    def _try_signal_to_quadrant_block(self, train, test_input):
        """Background color and one signal color. One or more 'noise' colors.
        Output: find signal cell, place signal as an NxN block at the NxN-grid
        quadrant containing the signal, remove all noise (replace with bg).
        Block size = grid_size / 2 for square grids."""
        from collections import Counter

        def _find_bg_signal(ex_list):
            # Find consistent bg and signal colors
            all_in = [v for ex in ex_list for row in ex["input"] for v in row]
            all_out = [v for ex in ex_list for row in ex["output"] for v in row]
            # bg appears in output (not changed much)
            out_cols = set(all_out)
            in_cols = set(all_in)
            # Signal = color in output that's not bg and consistent
            # bg = most common output color
            out_counts = Counter(all_out)
            bg = out_counts.most_common(1)[0][0]
            signal_cols = out_cols - {bg}
            if len(signal_cols) != 1:
                return None, None
            signal = next(iter(signal_cols))
            return bg, signal

        bg, signal = _find_bg_signal(train)
        if bg is None:
            return None

        def _solve(grid, rows, cols, block_r, block_c, bg, signal):
            out = [[bg]*cols for _ in range(rows)]
            for r in range(rows):
                for c in range(cols):
                    # Place signal block at target quadrant
                    in_block_r = (r // block_r) == (signal_row // block_r)
                    in_block_c = (c // block_c) == (signal_col // block_c)
                    if in_block_r and in_block_c:
                        out[r][c] = signal
            return out

        # Learn block size from training
        block_r = block_c = None
        signal_positions = []
        for ex in train:
            rows = len(ex["input"])
            cols = len(ex["input"][0]) if rows else 0
            if rows != len(ex["output"]) or cols != len(ex["output"][0]):
                return None
            # Find signal in input
            sig_cells = [(r, c) for r in range(rows) for c in range(cols)
                         if ex["input"][r][c] == signal]
            if not sig_cells:
                return None
            # Block size: rows/2 x cols/2 (requires even dimensions)
            if rows % 2 != 0 or cols % 2 != 0:
                return None
            br, bc = rows // 2, cols // 2
            if block_r is None:
                block_r, block_c = br, bc
            elif (block_r, block_c) != (br, bc):
                return None
            # The signal should map to a specific block
            # signal quadrant = (sig_r // br, sig_c // bc) → block fills (qr*br..., qc*bc...)
            for signal_row, signal_col in sig_cells:
                pass  # just need any sig cell
            # Verify output
            expected = [[bg]*cols for _ in range(rows)]
            for sr, sc in sig_cells:
                qr = sr // br
                qc = sc // bc
                for r in range(qr*br, (qr+1)*br):
                    for c in range(qc*bc, (qc+1)*bc):
                        expected[r][c] = signal
            if expected != ex["output"]:
                return None

        if block_r is None:
            return None

        # Apply to test
        rows = len(test_input)
        cols = len(test_input[0]) if rows else 0
        if rows != block_r * 2 or cols != block_c * 2:
            return None
        sig_cells = [(r, c) for r in range(rows) for c in range(cols)
                     if test_input[r][c] == signal]
        if not sig_cells:
            return None
        out = [[bg]*cols for _ in range(rows)]
        for sr, sc in sig_cells:
            qr = sr // block_r
            qc = sc // block_c
            for r in range(qr*block_r, (qr+1)*block_r):
                for c in range(qc*block_c, (qc+1)*block_c):
                    out[r][c] = signal
        return out

    def _try_component_size_coloring(self, train, test_input):
        """Each connected component (4-connected, non-zero) gets a color based on its size.
        Exactly one 'special' size maps to one color, all others map to another color.
        E.g. size==6 → color 2, all others → color 1."""
        def connected_components(grid):
            rows, cols = len(grid), len(grid[0])
            visited = [[False]*cols for _ in range(rows)]
            components = []
            for r in range(rows):
                for c in range(cols):
                    if grid[r][c] != 0 and not visited[r][c]:
                        comp = []
                        stack = [(r, c)]
                        while stack:
                            cr, cc = stack.pop()
                            if cr < 0 or cr >= rows or cc < 0 or cc >= cols:
                                continue
                            if visited[cr][cc] or grid[cr][cc] == 0:
                                continue
                            visited[cr][cc] = True
                            comp.append((cr, cc))
                            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                                stack.append((cr+dr, cc+dc))
                        components.append(comp)
            return components

        size_to_color = {}
        for ex in train:
            rows_in, cols_in = len(ex['input']), len(ex['input'][0])
            rows_out, cols_out = len(ex['output']), len(ex['output'][0])
            if rows_in != rows_out or cols_in != cols_out:
                return None
            comps = connected_components(ex['input'])
            for comp in comps:
                color = ex['output'][comp[0][0]][comp[0][1]]
                sz = len(comp)
                if sz in size_to_color:
                    if size_to_color[sz] != color:
                        return None
                else:
                    size_to_color[sz] = color

        if not size_to_color:
            return None

        colors = set(size_to_color.values())
        if len(colors) != 2:
            return None

        color_to_sizes = {}
        for sz, col in size_to_color.items():
            color_to_sizes.setdefault(col, []).append(sz)

        special_color = None
        special_size = None
        for col, sizes in color_to_sizes.items():
            if len(sizes) == 1:
                if special_color is not None:
                    return None
                special_color = col
                special_size = sizes[0]
        if special_color is None:
            return None
        other_color = next(c for c in colors if c != special_color)

        comps = connected_components(test_input)
        rows = len(test_input)
        cols = len(test_input[0]) if rows else 0
        out = [[0]*cols for _ in range(rows)]
        for comp in comps:
            color = special_color if len(comp) == special_size else other_color
            for r, c in comp:
                out[r][c] = color
        return out

    def _try_remove_low_neighbor_cells(self, train, test_input):
        """Remove cells that have fewer than 2 filled (non-zero) 4-connected neighbors.
        Keeps solid groups, removes isolated cells and thin protrusions."""
        def neighbor_count(r, c, grid):
            rows, cols = len(grid), len(grid[0])
            return sum(1 for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]
                       if 0 <= r+dr < rows and 0 <= c+dc < cols and grid[r+dr][c+dc] != 0)

        def apply(grid):
            rows, cols = len(grid), len(grid[0])
            return [[grid[r][c] if grid[r][c] != 0 and neighbor_count(r, c, grid) >= 2 else 0
                     for c in range(cols)] for r in range(rows)]

        for ex in train:
            if len(ex['input']) != len(ex['output']) or len(ex['input'][0]) != len(ex['output'][0]):
                return None
            if apply(ex['input']) != ex['output']:
                return None
        return apply(test_input)

    def _try_remove_row_col_sandwiched_cells(self, train, test_input):
        """Remove cell D at (r,c) if there exists a dominant color C that sandwiches D:
        C appears above AND below in the same column, OR left AND right in same row.
        Bilateral dominance: only C->D if sandwich_count(C,D) > sandwich_count(D,C)."""
        from collections import defaultdict

        def get_sandwichers(grid, r, c):
            cell = grid[r][c]
            if cell == 0:
                return set()
            rows, cols = len(grid), len(grid[0])
            col_above = {grid[rr][c] for rr in range(r) if grid[rr][c] not in (0, cell)}
            col_below = {grid[rr][c] for rr in range(r+1, rows) if grid[rr][c] not in (0, cell)}
            row_left  = {grid[r][cc] for cc in range(c) if grid[r][cc] not in (0, cell)}
            row_right = {grid[r][cc] for cc in range(c+1, cols) if grid[r][cc] not in (0, cell)}
            return (col_above & col_below) | (row_left & row_right)

        def apply_rule(grid):
            rows, cols = len(grid), len(grid[0])
            sandwich_count = defaultdict(int)
            sandwiched_by = {}
            for r in range(rows):
                for c in range(cols):
                    cell = grid[r][c]
                    if cell == 0:
                        continue
                    sw = get_sandwichers(grid, r, c)
                    sandwiched_by[(r, c)] = sw
                    for C in sw:
                        sandwich_count[(C, cell)] += 1
            out = [row[:] for row in grid]
            for r in range(rows):
                for c in range(cols):
                    cell = grid[r][c]
                    if cell == 0:
                        continue
                    for C in sandwiched_by.get((r, c), set()):
                        if sandwich_count[(C, cell)] > sandwich_count[(cell, C)]:
                            out[r][c] = 0
                            break
            return out

        for ex in train:
            if len(ex['input']) != len(ex['output']) or len(ex['input'][0]) != len(ex['output'][0]):
                return None
            if apply_rule(ex['input']) != ex['output']:
                return None
        return apply_rule(test_input)

    def _try_diagonal_alternating_color(self, train, test_input):
        """All non-zero cells lie on diagonals (r-c = constant).
        For each diagonal, alternate: original_color, 4, original_color, 4, ...
        starting from lowest-row cell."""
        from collections import defaultdict

        def apply_rule(grid):
            rows, cols = len(grid), len(grid[0])
            diag_groups = defaultdict(list)
            for r in range(rows):
                for c in range(cols):
                    if grid[r][c] != 0:
                        diag_groups[r - c].append((r, c, grid[r][c]))
            out = [row[:] for row in grid]
            for cells in diag_groups.values():
                cells.sort()
                for idx, (r, c, color) in enumerate(cells):
                    out[r][c] = color if idx % 2 == 0 else 4
            return out

        for ex in train:
            if len(ex['input']) != len(ex['output']) or len(ex['input'][0]) != len(ex['output'][0]):
                return None
            if apply_rule(ex['input']) != ex['output']:
                return None
        return apply_rule(test_input)

    def _try_8conn_equal_size_coloring(self, train, test_input):
        """Find all 8-connected components. Components whose size appears more than
        once (the 'common' size) get color 1; unique-size component gets color 2."""
        from collections import Counter

        def get_components(grid):
            rows, cols = len(grid), len(grid[0])
            visited = [[False]*cols for _ in range(rows)]
            components = []
            for r in range(rows):
                for c in range(cols):
                    if grid[r][c] != 0 and not visited[r][c]:
                        comp = []
                        stack = [(r, c)]
                        while stack:
                            cr, cc = stack.pop()
                            if cr < 0 or cr >= rows or cc < 0 or cc >= cols:
                                continue
                            if visited[cr][cc] or grid[cr][cc] == 0:
                                continue
                            visited[cr][cc] = True
                            comp.append((cr, cc))
                            for dr in [-1, 0, 1]:
                                for dc in [-1, 0, 1]:
                                    if dr == 0 and dc == 0:
                                        continue
                                    stack.append((cr+dr, cc+dc))
                        components.append(comp)
            return components

        def apply_rule(grid):
            comps = get_components(grid)
            size_counts = Counter(len(c) for c in comps)
            rows, cols = len(grid), len(grid[0])
            out = [[0]*cols for _ in range(rows)]
            for comp in comps:
                color = 1 if size_counts[len(comp)] > 1 else 2
                for r, c in comp:
                    out[r][c] = color
            return out

        for ex in train:
            if len(ex['input']) != len(ex['output']) or len(ex['input'][0]) != len(ex['output'][0]):
                return None
            if apply_rule(ex['input']) != ex['output']:
                return None
        return apply_rule(test_input)

    def _try_top2_freq_stay(self, train, test_input):
        """Top 2 most frequent colors stay, all others become 7.
        Handles: 9caf5b84"""
        from collections import Counter

        def apply_rule(grid):
            flat = [v for row in grid for v in row]
            if not flat:
                return grid
            counts = Counter(flat)
            sorted_colors = sorted(counts.keys(), key=lambda c: -counts[c])
            if len(sorted_colors) < 2:
                return None
            # Check for tie at the 2nd position
            top1_count = counts[sorted_colors[0]]
            top2_count = counts[sorted_colors[1]]
            # If there's a 3-way tie at top, can't determine top 2
            if len(sorted_colors) > 2 and counts[sorted_colors[2]] == top2_count:
                return None
            keep = {sorted_colors[0], sorted_colors[1]}
            rows, cols = len(grid), len(grid[0])
            out = []
            for r in range(rows):
                row_out = []
                for c in range(cols):
                    v = grid[r][c]
                    if v in keep:
                        row_out.append(v)
                    else:
                        row_out.append(7)
                out.append(row_out)
            return out

        for ex in train:
            if len(ex['input']) != len(ex['output']) or len(ex['input'][0]) != len(ex['output'][0]):
                return None
            result = apply_rule(ex['input'])
            if result is None or result != ex['output']:
                return None
        return apply_rule(test_input)

    def _try_5block_nearest_noise_color(self, train, test_input):
        """5-block (cells of value 5) + scattered noise. Find noise cell nearest to 5-block
        by Chebyshev distance. Among nearest, pick most frequent color. Replace 5s, zero rest.
        Handles: 6df30ad6"""
        from collections import Counter

        def solve(grid):
            rows, cols = len(grid), len(grid[0])
            block5 = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == 5]
            noise = [(r, c, grid[r][c]) for r in range(rows) for c in range(cols)
                     if grid[r][c] != 0 and grid[r][c] != 5]
            if not block5 or not noise:
                return None
            # Chebyshev distance from noise cell to nearest 5-cell
            def min_cheb(nr, nc):
                return min(max(abs(nr - br), abs(nc - bc)) for br, bc in block5)
            dists = [(min_cheb(r, c), r, c, color) for r, c, color in noise]
            min_d = min(d for d, _, _, _ in dists)
            nearest_colors = [color for d, _, _, color in dists if d == min_d]
            # Most frequent among nearest
            color_counts = Counter(nearest_colors)
            # If tie in nearest AND multiple colors, pick most frequent
            winner = color_counts.most_common(1)[0][0]
            # Check if tie exists (ambiguous)
            top_count = color_counts.most_common(1)[0][1]
            if sum(1 for _, cnt in color_counts.items() if cnt == top_count) > 1:
                return None  # ambiguous
            out = [[0]*cols for _ in range(rows)]
            for br, bc in block5:
                out[br][bc] = winner
            return out

        for ex in train:
            if len(ex['input']) != len(ex['output']) or len(ex['input'][0]) != len(ex['output'][0]):
                return None
            result = solve(ex['input'])
            if result is None or result != ex['output']:
                return None
        return solve(test_input)

    def _try_complete_rect_outline_to_3(self, train, test_input):
        """Connected components of 1s that form a complete hollow rectangle
        (all perimeter cells of bounding box present, interior empty, R>=3 and C>=3) -> 3.
        Incomplete shapes stay 1.
        Handles: 810b9b61"""
        def get_4conn_components(grid):
            rows, cols = len(grid), len(grid[0])
            visited = [[False]*cols for _ in range(rows)]
            components = []
            for sr in range(rows):
                for sc in range(cols):
                    if grid[sr][sc] == 1 and not visited[sr][sc]:
                        comp = []
                        stack = [(sr, sc)]
                        visited[sr][sc] = True
                        while stack:
                            r, c = stack.pop()
                            comp.append((r, c))
                            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                                nr, nc = r+dr, c+dc
                                if 0<=nr<rows and 0<=nc<cols and not visited[nr][nc] and grid[nr][nc]==1:
                                    visited[nr][nc] = True
                                    stack.append((nr, nc))
                        components.append(comp)
            return components

        def is_complete_hollow_rect(comp):
            cells = set(comp)
            min_r = min(r for r, c in cells)
            max_r = max(r for r, c in cells)
            min_c = min(c for r, c in cells)
            max_c = max(c for r, c in cells)
            R = max_r - min_r + 1
            C = max_c - min_c + 1
            if R < 3 or C < 3:
                return False
            perimeter = set()
            for r in range(min_r, max_r + 1):
                for c in range(min_c, max_c + 1):
                    if r == min_r or r == max_r or c == min_c or c == max_c:
                        perimeter.add((r, c))
            return cells == perimeter

        def apply_rule(grid):
            comps = get_4conn_components(grid)
            rows, cols = len(grid), len(grid[0])
            out = [row[:] for row in grid]
            for comp in comps:
                if is_complete_hollow_rect(comp):
                    for r, c in comp:
                        out[r][c] = 3
            return out

        for ex in train:
            if len(ex['input']) != len(ex['output']) or len(ex['input'][0]) != len(ex['output'][0]):
                return None
            if apply_rule(ex['input']) != ex['output']:
                return None
        return apply_rule(test_input)

    def _try_marker_recolor_blob(self, train, test_input):
        """One isolated singleton cell acts as a color marker. The main blob (connected
        component of a different color) gets recolored to the marker color, marker disappears.
        Handles: aabf363d"""
        def apply_rule(grid):
            rows, cols = len(grid), len(grid[0])
            non_zero = [(r, c, grid[r][c]) for r in range(rows) for c in range(cols) if grid[r][c] != 0]
            colors = set(v for _, _, v in non_zero)
            if len(colors) != 2:
                return None
            c1, c2 = list(colors)
            c1_cells = [(r, c) for r, c, v in non_zero if v == c1]
            c2_cells = [(r, c) for r, c, v in non_zero if v == c2]
            # One must be the singleton marker, other is the blob
            if len(c1_cells) == 1:
                marker_color, marker_cell = c1, c1_cells[0]
                blob_color = c2
            elif len(c2_cells) == 1:
                marker_color, marker_cell = c2, c2_cells[0]
                blob_color = c1
            else:
                return None
            out = [row[:] for row in grid]
            mr, mc = marker_cell
            out[mr][mc] = 0
            for r in range(rows):
                for c in range(cols):
                    if grid[r][c] == blob_color:
                        out[r][c] = marker_color
            return out

        for ex in train:
            if len(ex['input']) != len(ex['output']) or len(ex['input'][0]) != len(ex['output'][0]):
                return None
            result = apply_rule(ex['input'])
            if result is None or result != ex['output']:
                return None
        return apply_rule(test_input)

    def _try_small_components_to_3(self, train, test_input):
        """Same-value connected components with size <= 2 get recolored to 3.
        Components with size >= 3 stay unchanged.
        Handles: 12eac192"""
        def get_same_value_components(grid):
            rows, cols = len(grid), len(grid[0])
            visited = [[False]*cols for _ in range(rows)]
            components = []
            for sr in range(rows):
                for sc in range(cols):
                    if grid[sr][sc] != 0 and not visited[sr][sc]:
                        v = grid[sr][sc]
                        comp = []
                        stack = [(sr, sc)]
                        visited[sr][sc] = True
                        while stack:
                            r, c = stack.pop()
                            comp.append((r, c))
                            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                                nr, nc = r+dr, c+dc
                                if 0<=nr<rows and 0<=nc<cols and not visited[nr][nc] and grid[nr][nc]==v:
                                    visited[nr][nc] = True
                                    stack.append((nr, nc))
                        components.append((v, comp))
            return components

        def apply_rule(grid):
            comps = get_same_value_components(grid)
            out = [row[:] for row in grid]
            for v, comp in comps:
                if len(comp) <= 2:
                    for r, c in comp:
                        out[r][c] = 3
            return out

        for ex in train:
            if len(ex['input']) != len(ex['output']) or len(ex['input'][0]) != len(ex['output'][0]):
                return None
            if apply_rule(ex['input']) != ex['output']:
                return None
        return apply_rule(test_input)

    def _try_path_value_reversal(self, train, test_input):
        """Non-background cells form a connected path (snake). Reverse the values along the path.
        Background is the most frequent color.
        Handles: 5792cb4d"""
        from collections import Counter

        def find_bg(grid):
            flat = [v for row in grid for v in row]
            return Counter(flat).most_common(1)[0][0]

        def get_path_cells(grid, bg):
            """Return path cells in order from one end to other. Returns None if not a valid path."""
            rows, cols = len(grid), len(grid[0])
            non_bg = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] != bg]
            if not non_bg:
                return None
            # Build adjacency
            cell_set = set(non_bg)
            def neighbors(r, c):
                return [(r+dr, c+dc) for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]
                        if (r+dr, c+dc) in cell_set]
            # Check all cells have <= 2 neighbors (path property)
            for r, c in non_bg:
                if len(neighbors(r, c)) > 2:
                    return None
            # Find endpoints (cells with exactly 1 neighbor)
            endpoints = [(r, c) for r, c in non_bg if len(neighbors(r, c)) == 1]
            if len(endpoints) != 2:
                return None
            # Traverse from one endpoint
            path = []
            visited = set()
            curr = endpoints[0]
            while curr is not None:
                path.append(curr)
                visited.add(curr)
                nxt = None
                for n in neighbors(*curr):
                    if n not in visited:
                        nxt = n
                        break
                curr = nxt
            if len(path) != len(non_bg):
                return None  # disconnected
            return path

        def apply_rule(grid, bg):
            path = get_path_cells(grid, bg)
            if path is None:
                return None
            values = [grid[r][c] for r, c in path]
            reversed_values = values[::-1]
            out = [row[:] for row in grid]
            for (r, c), v in zip(path, reversed_values):
                out[r][c] = v
            return out

        # Determine background from first training example
        bg = find_bg(train[0]['input'])
        for ex in train:
            if len(ex['input']) != len(ex['output']) or len(ex['input'][0]) != len(ex['output'][0]):
                return None
            result = apply_rule(ex['input'], bg)
            if result is None or result != ex['output']:
                return None
        return apply_rule(test_input, bg)

    def _try_connect_pairs_with_8(self, train, test_input):
        """For each row, consecutive 1s get connected with 8s between them.
        For each column, consecutive 1s get connected with 8s between them.
        Handles: dbc1a6ce"""
        def apply_rule(grid):
            rows, cols = len(grid), len(grid[0])
            out = [row[:] for row in grid]
            # Connect same-row pairs
            for r in range(rows):
                ones = sorted(c for c in range(cols) if grid[r][c] == 1)
                for i in range(len(ones) - 1):
                    c1, c2 = ones[i], ones[i+1]
                    for c in range(c1+1, c2):
                        if out[r][c] == 0:
                            out[r][c] = 8
            # Connect same-column pairs
            for c in range(cols):
                ones = sorted(r for r in range(rows) if grid[r][c] == 1)
                for i in range(len(ones) - 1):
                    r1, r2 = ones[i], ones[i+1]
                    for r in range(r1+1, r2):
                        if out[r][c] == 0:
                            out[r][c] = 8
            return out

        for ex in train:
            if len(ex['input']) != len(ex['output']) or len(ex['input'][0]) != len(ex['output'][0]):
                return None
            if apply_rule(ex['input']) != ex['output']:
                return None
        return apply_rule(test_input)

    def _try_solid3row_middle_alternate(self, train, test_input):
        """Each solid 3-row rectangular block has its middle row converted to alternating
        color/0 pattern (starting with color at leftmost column). Top/bottom rows unchanged.
        Handles: 3bdb4ada"""
        def find_solid_rects(grid):
            """Find all solid-color rectangles of exactly 3 rows."""
            rows, cols = len(grid), len(grid[0])
            visited = [[False]*cols for _ in range(rows)]
            rects = []  # (color, min_r, max_r, min_c, max_c)
            for sr in range(rows):
                for sc in range(cols):
                    if grid[sr][sc] != 0 and not visited[sr][sc]:
                        v = grid[sr][sc]
                        # BFS for connected region
                        comp = []
                        queue = [(sr, sc)]
                        visited[sr][sc] = True
                        while queue:
                            r, c = queue.pop(0)
                            comp.append((r, c))
                            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                                nr, nc = r+dr, c+dc
                                if 0<=nr<rows and 0<=nc<cols and not visited[nr][nc] and grid[nr][nc]==v:
                                    visited[nr][nc] = True
                                    queue.append((nr, nc))
                        min_r = min(r for r,c in comp)
                        max_r = max(r for r,c in comp)
                        min_c = min(c for r,c in comp)
                        max_c = max(c for r,c in comp)
                        # Check if exactly 3 rows
                        if max_r - min_r != 2:
                            return None
                        # Check solid rectangle
                        expected = set((r, c) for r in range(min_r, max_r+1) for c in range(min_c, max_c+1))
                        if set(comp) != expected:
                            return None
                        rects.append((v, min_r, max_r, min_c, max_c))
            return rects

        def apply_rule(grid):
            rects = find_solid_rects(grid)
            if rects is None or not rects:
                return None
            out = [row[:] for row in grid]
            for v, min_r, max_r, min_c, max_c in rects:
                mid_r = min_r + 1
                for c in range(min_c, max_c + 1):
                    offset = c - min_c
                    out[mid_r][c] = v if offset % 2 == 0 else 0
            return out

        for ex in train:
            if len(ex['input']) != len(ex['output']) or len(ex['input'][0]) != len(ex['output'][0]):
                return None
            result = apply_rule(ex['input'])
            if result is None or result != ex['output']:
                return None
        return apply_rule(test_input)

    def _try_5border_swap_interior(self, train, test_input):
        """5-bordered rectangle: swap the two non-zero/non-5 colors inside."""
        def find_5_border(grid):
            rows, cols = len(grid), len(grid[0])
            fives = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == 5]
            if not fives:
                return None
            min_r = min(r for r, c in fives); max_r = max(r for r, c in fives)
            min_c = min(c for r, c in fives); max_c = max(c for r, c in fives)
            if max_r - min_r < 2 or max_c - min_c < 2:
                return None
            perimeter = set()
            for r in range(min_r, max_r + 1):
                for c in range(min_c, max_c + 1):
                    if r == min_r or r == max_r or c == min_c or c == max_c:
                        perimeter.add((r, c))
            if set(fives) != perimeter:
                return None
            return min_r, max_r, min_c, max_c

        def apply_rule(grid, border):
            min_r, max_r, min_c, max_c = border
            interior_colors = set()
            for r in range(min_r + 1, max_r):
                for c in range(min_c + 1, max_c):
                    v = grid[r][c]
                    if v != 0 and v != 5:
                        interior_colors.add(v)
            if len(interior_colors) != 2:
                return None
            c1, c2 = list(interior_colors)
            swap = {c1: c2, c2: c1}
            out = [row[:] for row in grid]
            for r in range(min_r + 1, max_r):
                for c in range(min_c + 1, max_c):
                    v = grid[r][c]
                    if v in swap:
                        out[r][c] = swap[v]
            return out

        for ex in train:
            if len(ex['input']) != len(ex['output']) or len(ex['input'][0]) != len(ex['output'][0]):
                return None
            border = find_5_border(ex['input'])
            if border is None:
                return None
            result = apply_rule(ex['input'], border)
            if result is None or result != ex['output']:
                return None
        border = find_5_border(test_input)
        if border is None:
            return None
        return apply_rule(test_input, border)

    def _try_ones_expand_3x3(self, train, test_input):
        """Each 1 expands to 3x3 block of 1s in output (only filling 0s)."""
        def apply_rule(grid):
            rows, cols = len(grid), len(grid[0])
            ones = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == 1]
            if not ones:
                return None
            out = [row[:] for row in grid]
            for (br, bc) in ones:
                for dr in range(-1, 2):
                    for dc in range(-1, 2):
                        nr, nc = br + dr, bc + dc
                        if 0 <= nr < rows and 0 <= nc < cols and out[nr][nc] == 0:
                            out[nr][nc] = 1
            return out

        for ex in train:
            if len(ex['input']) != len(ex['output']) or len(ex['input'][0]) != len(ex['output'][0]):
                return None
            result = apply_rule(ex['input'])
            if result is None or result != ex['output']:
                return None
        return apply_rule(test_input)

    def _try_stripe_mode_fill(self, train, test_input):
        """Vertical or horizontal stripes: each col/row gets its most frequent value."""
        from collections import Counter

        def col_mode_grid(grid):
            rows, cols = len(grid), len(grid[0])
            out = [row[:] for row in grid]
            for c in range(cols):
                vals = [grid[r][c] for r in range(rows)]
                counts = Counter(vals)
                top = counts.most_common(2)
                if len(top) > 1 and top[0][1] == top[1][1]:
                    return None
                mode = top[0][0]
                for r in range(rows):
                    out[r][c] = mode
            return out

        def row_mode_grid(grid):
            rows, cols = len(grid), len(grid[0])
            out = []
            for r in range(rows):
                counts = Counter(grid[r])
                top = counts.most_common(2)
                if len(top) > 1 and top[0][1] == top[1][1]:
                    return None
                out.append([top[0][0]] * cols)
            return out

        for ex in train:
            if len(ex['input']) != len(ex['output']) or len(ex['input'][0]) != len(ex['output'][0]):
                return None
            col_r = col_mode_grid(ex['input'])
            row_r = row_mode_grid(ex['input'])
            if col_r != ex['output'] and row_r != ex['output']:
                return None
        # Apply to test: prefer column mode, fall back to row mode
        result = col_mode_grid(test_input)
        if result is not None:
            return result
        return row_mode_grid(test_input)

    def _try_largest_solid_rect_only(self, train, test_input):
        """Keep only the largest solid single-color rectangle; zero everything else."""
        def find_largest_solid_rect(grid):
            rows, cols = len(grid), len(grid[0])
            from collections import defaultdict
            # Collect cells per color
            color_cells = defaultdict(set)
            for r in range(rows):
                for c in range(cols):
                    if grid[r][c] != 0:
                        color_cells[grid[r][c]].add((r, c))
            best_area = 0
            best = None
            for color, cells in color_cells.items():
                rs = sorted(set(r for r, c in cells))
                cs = sorted(set(c for r, c in cells))
                for i, r1 in enumerate(rs):
                    for r2 in rs[i:]:
                        for j, c1 in enumerate(cs):
                            for c2 in cs[j:]:
                                area = (r2 - r1 + 1) * (c2 - c1 + 1)
                                if area <= best_area:
                                    continue
                                rect = set((r, c) for r in range(r1, r2+1) for c in range(c1, c2+1))
                                if rect.issubset(cells):
                                    # Check all rect cells are color
                                    if all(grid[r][c] == color for r, c in rect):
                                        best_area = area
                                        best = (color, r1, r2, c1, c2)
            return best

        def apply_rule(grid, rect):
            if rect is None:
                return None
            color, r1, r2, c1, c2 = rect
            rows, cols = len(grid), len(grid[0])
            out = [[0] * cols for _ in range(rows)]
            for r in range(r1, r2 + 1):
                for c in range(c1, c2 + 1):
                    out[r][c] = color
            return out

        for ex in train:
            if len(ex['input']) != len(ex['output']) or len(ex['input'][0]) != len(ex['output'][0]):
                return None
            rect = find_largest_solid_rect(ex['input'])
            result = apply_rule(ex['input'], rect)
            if result is None or result != ex['output']:
                return None
        rect = find_largest_solid_rect(test_input)
        return apply_rule(test_input, rect)

    def _try_concentric_ring_rotation(self, train, test_input):
        """Concentric rectangular rings: rotate ring colors right by 1 (innermost→outermost)."""
        def get_ring_colors(grid):
            rows, cols = len(grid), len(grid[0])
            n_rings = min(rows, cols) // 2 + (1 if min(rows, cols) % 2 else 0)
            ring_colors = []
            for ring_i in range(n_rings):
                # Collect all cells on this ring
                cells = []
                r1, r2, c1, c2 = ring_i, rows - 1 - ring_i, ring_i, cols - 1 - ring_i
                if r1 > r2 or c1 > c2:
                    break
                if r1 == r2:
                    cells = [(r1, c) for c in range(c1, c2 + 1)]
                elif c1 == c2:
                    cells = [(r, c1) for r in range(r1, r2 + 1)]
                else:
                    cells = ([(r1, c) for c in range(c1, c2 + 1)] +
                             [(r2, c) for c in range(c1, c2 + 1)] +
                             [(r, c1) for r in range(r1 + 1, r2)] +
                             [(r, c2) for r in range(r1 + 1, r2)])
                colors = set(grid[r][c] for r, c in cells)
                if len(colors) != 1:
                    return None
                ring_colors.append(colors.pop())
            return ring_colors

        def apply_rule(grid):
            ring_colors = get_ring_colors(grid)
            if ring_colors is None or len(ring_colors) < 2:
                return None
            # Find cycle length
            cycle_len = len(ring_colors)
            for i in range(1, len(ring_colors)):
                if ring_colors[i] == ring_colors[0]:
                    cycle_len = i
                    break
            # Rotate right by 1 within cycle
            new_colors = [ring_colors[(i - 1) % cycle_len] for i in range(len(ring_colors))]
            rows, cols = len(grid), len(grid[0])
            out = [row[:] for row in grid]
            n_rings = len(ring_colors)
            for ring_i in range(n_rings):
                r1, r2, c1, c2 = ring_i, rows - 1 - ring_i, ring_i, cols - 1 - ring_i
                if r1 > r2 or c1 > c2:
                    break
                if r1 == r2:
                    cells = [(r1, c) for c in range(c1, c2 + 1)]
                elif c1 == c2:
                    cells = [(r, c1) for r in range(r1, r2 + 1)]
                else:
                    cells = ([(r1, c) for c in range(c1, c2 + 1)] +
                             [(r2, c) for c in range(c1, c2 + 1)] +
                             [(r, c1) for r in range(r1 + 1, r2)] +
                             [(r, c2) for r in range(r1 + 1, r2)])
                for r, c in cells:
                    out[r][c] = new_colors[ring_i]
            return out

        for ex in train:
            if len(ex['input']) != len(ex['output']) or len(ex['input'][0]) != len(ex['output'][0]):
                return None
            result = apply_rule(ex['input'])
            if result is None or result != ex['output']:
                return None
        return apply_rule(test_input)

    def _try_fill_1frame_by_interior_parity(self, train, test_input):
        """Fill hollow 1-frames: odd interior side→7, even interior side→2."""
        def find_1_frames(grid):
            rows, cols = len(grid), len(grid[0])
            frames = []
            for r1 in range(rows - 2):
                for r2 in range(r1 + 2, rows):
                    for c1 in range(cols - 2):
                        for c2 in range(c1 + 2, cols):
                            # Check all border cells are 1
                            border = ([(r1, c) for c in range(c1, c2 + 1)] +
                                      [(r2, c) for c in range(c1, c2 + 1)] +
                                      [(r, c1) for r in range(r1 + 1, r2)] +
                                      [(r, c2) for r in range(r1 + 1, r2)])
                            if not all(grid[r][c] == 1 for r, c in border):
                                continue
                            # Check interior is all 0
                            if not all(grid[r][c] == 0 for r in range(r1+1, r2) for c in range(c1+1, c2)):
                                continue
                            interior_h = r2 - r1 - 1
                            interior_w = c2 - c1 - 1
                            if interior_h != interior_w:
                                continue
                            frames.append((r1, r2, c1, c2, interior_h))
            return frames

        def apply_rule(grid):
            frames = find_1_frames(grid)
            if not frames:
                return None
            rows, cols = len(grid), len(grid[0])
            out = [row[:] for row in grid]
            for r1, r2, c1, c2, side in frames:
                fill = 7 if side % 2 == 1 else 2
                for r in range(r1 + 1, r2):
                    for c in range(c1 + 1, c2):
                        out[r][c] = fill
            return out

        for ex in train:
            if len(ex['input']) != len(ex['output']) or len(ex['input'][0]) != len(ex['output'][0]):
                return None
            result = apply_rule(ex['input'])
            if result is None or result != ex['output']:
                return None
        return apply_rule(test_input)

    def _try_tile_pattern_extend_recolor(self, train, test_input):
        """Tile repeating row pattern to output_rows=input_rows+3, replace color 1 with 2."""
        def find_period(grid):
            rows = len(grid)
            for p in range(1, rows + 1):
                if all(grid[i] == grid[i % p] for i in range(rows)):
                    return p
            return rows

        def apply_rule(grid):
            rows, cols = len(grid), len(grid[0])
            period = find_period(grid)
            out_rows = rows + 3
            out = []
            for i in range(out_rows):
                src_row = grid[i % period]
                out.append([2 if v == 1 else v for v in src_row])
            return out

        for ex in train:
            if len(ex['output']) != len(ex['input']) + 3:
                return None
            if len(ex['output'][0]) != len(ex['input'][0]):
                return None
            inp_colors = set(v for row in ex['input'] for v in row if v != 0)
            out_colors = set(v for row in ex['output'] for v in row if v != 0)
            if inp_colors != {1} or out_colors != {2}:
                return None
            result = apply_rule(ex['input'])
            if result != ex['output']:
                return None
        return apply_rule(test_input)

    def _try_key_template_rotate_fill(self, train, test_input):
        """Key (non-8 colored cells) rotated 90°CCW fills 8-template blocks."""
        def apply_rule(grid):
            rows, cols = len(grid), len(grid[0])
            # Separate key cells (non-zero, non-8) from template (8-cells)
            key_cells = [(r, c, grid[r][c]) for r in range(rows) for c in range(cols)
                         if grid[r][c] != 0 and grid[r][c] != 8]
            template_cells = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == 8]
            if not key_cells or not template_cells:
                return None
            # Key bounding box
            kr_min = min(r for r, c, v in key_cells); kr_max = max(r for r, c, v in key_cells)
            kc_min = min(c for r, c, v in key_cells); kc_max = max(c for r, c, v in key_cells)
            N = max(kr_max - kr_min + 1, kc_max - kc_min + 1)
            if kr_max - kr_min + 1 != N or kc_max - kc_min + 1 != N:
                return None
            # Extract N×N key matrix
            key_matrix = [[grid[kr_min + r][kc_min + c] for c in range(N)] for r in range(N)]
            # Template bounding box
            tr_min = min(r for r, c in template_cells); tr_max = max(r for r, c in template_cells)
            tc_min = min(c for r, c in template_cells); tc_max = max(c for r, c in template_cells)
            if (tr_max - tr_min + 1) % N != 0 or (tc_max - tc_min + 1) % N != 0:
                return None
            S_r = (tr_max - tr_min + 1) // N
            S_c = (tc_max - tc_min + 1) // N
            if S_r != S_c:
                return None
            S = S_r
            # Verify template: each S×S sub-block is all-8 or all-0
            for br in range(N):
                for bc in range(N):
                    block_vals = set(grid[tr_min + br*S + dr][tc_min + bc*S + dc]
                                     for dr in range(S) for dc in range(S))
                    if len(block_vals) != 1 or (block_vals.pop() not in (0, 8)):
                        return None
            # Fill template blocks using 90°CCW rotation: block(r,c) ← key[N-1-c][r]
            out = [row[:] for row in grid]
            for br in range(N):
                for bc in range(N):
                    color = key_matrix[N - 1 - bc][br]
                    if color == 0:
                        continue
                    # Check this block should be 8 in input
                    if grid[tr_min + br*S][tc_min + bc*S] != 8:
                        return None
                    for dr in range(S):
                        for dc in range(S):
                            out[tr_min + br*S + dr][tc_min + bc*S + dc] = color
            return out

        for ex in train:
            if len(ex['input']) != len(ex['output']) or len(ex['input'][0]) != len(ex['output'][0]):
                return None
            result = apply_rule(ex['input'])
            if result is None or result != ex['output']:
                return None
        return apply_rule(test_input)

    def _try_odd_patch_out(self, train, test_input):
        """Grid contains k equal 3x3 patches; find the one with unique binary pattern."""
        def get_patches(grid):
            rows, cols = len(grid), len(grid[0])
            if rows == 3 and cols % 3 == 0 and cols > 3:
                k = cols // 3
                return [[[grid[r][bc*3 + c] for c in range(3)] for r in range(3)] for bc in range(k)]
            elif cols == 3 and rows % 3 == 0 and rows > 3:
                k = rows // 3
                return [[[grid[br*3 + r][c] for c in range(3)] for r in range(3)] for br in range(k)]
            return None

        def binary(patch):
            return tuple(tuple(1 if v != 0 else 0 for v in row) for row in patch)

        def apply_rule(grid):
            patches = get_patches(grid)
            if patches is None or len(patches) < 3:
                return None
            patterns = [binary(p) for p in patches]
            from collections import Counter
            cnt = Counter(patterns)
            if len(cnt) != 2:
                return None
            odd_pat = [p for p, c in cnt.items() if c == 1]
            if len(odd_pat) != 1:
                return None
            idx = patterns.index(odd_pat[0])
            return patches[idx]

        for ex in train:
            result = apply_rule(ex['input'])
            if result is None or result != ex['output']:
                return None
        return apply_rule(test_input)

    def _try_middle_col_only(self, train, test_input):
        """Keep only the middle column (cols//2), zero everything else."""
        def apply_rule(grid):
            rows, cols = len(grid), len(grid[0])
            mid = cols // 2
            return [[grid[r][mid] if c == mid else 0 for c in range(cols)] for r in range(rows)]

        for ex in train:
            if len(ex['input']) != len(ex['output']) or len(ex['input'][0]) != len(ex['output'][0]):
                return None
            result = apply_rule(ex['input'])
            if result != ex['output']:
                return None
        return apply_rule(test_input)

    def _try_bar_bottom_half_to_8(self, train, test_input):
        """Vertical bars of a color: bottom floor(height/2) cells become 8."""
        def apply_rule(grid):
            rows, cols = len(grid), len(grid[0])
            out = [row[:] for row in grid]
            # Find non-zero, non-8 cells; must be single color per col, contiguous from some start row
            bar_color = None
            for c in range(cols):
                col_vals = [(r, grid[r][c]) for r in range(rows) if grid[r][c] != 0]
                if not col_vals:
                    continue
                colors = set(v for r, v in col_vals)
                if len(colors) != 1:
                    return None
                color = colors.pop()
                if color == 8:
                    return None
                if bar_color is None:
                    bar_color = color
                elif color != bar_color:
                    return None
                # Check contiguous from some start row
                rs = [r for r, v in col_vals]
                if rs != list(range(rs[0], rs[0] + len(rs))):
                    return None
                height = len(rs)
                n_eights = height // 2
                start = rs[0]
                # bottom n_eights become 8
                for r in range(start + height - n_eights, start + height):
                    out[r][c] = 8
            return out

        for ex in train:
            if len(ex['input']) != len(ex['output']) or len(ex['input'][0]) != len(ex['output'][0]):
                return None
            result = apply_rule(ex['input'])
            if result is None or result != ex['output']:
                return None
        return apply_rule(test_input)

    def _try_color_stripe_list(self, train, test_input):
        """Diagonal or horizontal color stripes → output 1xN row or Nx1 column listing colors in order."""
        def apply_rule(grid):
            rows, cols = len(grid), len(grid[0])
            # Find distinct colors (all non-zero values)
            colors_seen = []
            seen_set = set()
            for r in range(rows):
                for c in range(cols):
                    v = grid[r][c]
                    if v != 0 and v not in seen_set:
                        colors_seen.append(v)
                        seen_set.add(v)
            if len(colors_seen) < 2:
                return None
            # Determine orientation: if first row has multiple colors → horizontal stripes → row output
            # if first row is all one color → diagonal/mixed → column output
            first_row_colors = set(grid[0])
            if len(first_row_colors) > 1:
                # Multiple colors in first row → left-to-right order (already in colors_seen order)
                # Output: 1 x N row
                return [list(colors_seen)]
            else:
                # Horizontal bands → top-to-bottom (colors_seen is already in top-to-bottom order)
                # Output: N x 1 column
                return [[c] for c in colors_seen]

        for ex in train:
            result = apply_rule(ex['input'])
            if result is None or result != ex['output']:
                return None
        return apply_rule(test_input)

    def _try_mirror_horizontal_append(self, train, test_input):
        """Append the LR-mirror of the grid to the right (double width)."""
        def apply_rule(grid):
            return [row + row[::-1] for row in grid]

        for ex in train:
            if len(ex['input']) != len(ex['output']):
                return None
            if len(ex['output'][0]) != len(ex['input'][0]) * 2:
                return None
            result = apply_rule(ex['input'])
            if result != ex['output']:
                return None
        return apply_rule(test_input)

    def _try_mirror_vertical_append(self, train, test_input):
        """Append the UD-mirror of the grid below (double height)."""
        def apply_rule(grid):
            return list(grid) + list(grid[::-1])

        for ex in train:
            if len(ex['input'][0]) != len(ex['output'][0]):
                return None
            if len(ex['output']) != len(ex['input']) * 2:
                return None
            result = apply_rule(ex['input'])
            if result != ex['output']:
                return None
        return apply_rule(test_input)

    def _try_zoom_outer_double(self, train, test_input):
        """3x3 → 5x5: outer rows/cols doubled, center stays 1x."""
        def apply_rule(grid):
            if len(grid) != 3 or len(grid[0]) != 3:
                return None
            row_map = [0, 0, 1, 2, 2]
            col_map = [0, 0, 1, 2, 2]
            return [[grid[row_map[r]][col_map[c]] for c in range(5)] for r in range(5)]

        for ex in train:
            if len(ex['input']) != 3 or len(ex['input'][0]) != 3:
                return None
            if len(ex['output']) != 5 or len(ex['output'][0]) != 5:
                return None
            result = apply_rule(ex['input'])
            if result != ex['output']:
                return None
        return apply_rule(test_input)

    def _try_lr_symmetric_color(self, train, test_input):
        """3x3 binary patterns: LR-symmetric → color 1, else → color 7."""
        def apply_rule(grid):
            if len(grid) != 3 or len(grid[0]) != 3:
                return None
            binary = [[1 if v != 0 else 0 for v in row] for row in grid]
            lr_sym = all(binary[r][0] == binary[r][2] for r in range(3))
            return [[1]] if lr_sym else [[7]]

        for ex in train:
            if len(ex['input']) != 3 or len(ex['input'][0]) != 3:
                return None
            if ex['output'] not in [[[1]], [[7]]]:
                return None
            result = apply_rule(ex['input'])
            if result != ex['output']:
                return None
        return apply_rule(test_input)

    def _try_grid_diagonal_indicator(self, train, test_input):
        """3x3 → 3x3: homogeneous → top-row 5s; bottom-row uniform → main diag; else anti-diag."""
        def apply_rule(grid):
            if len(grid) != 3 or len(grid[0]) != 3:
                return None
            out = [[0]*3 for _ in range(3)]
            flat = [grid[r][c] for r in range(3) for c in range(3)]
            if len(set(flat)) == 1:
                out[0] = [5, 5, 5]
            elif len(set(grid[2])) == 1:
                for i in range(3): out[i][i] = 5
            else:
                for i in range(3): out[i][2-i] = 5
            return out

        for ex in train:
            if len(ex['input']) != 3 or len(ex['input'][0]) != 3:
                return None
            result = apply_rule(ex['input'])
            if result is None or result != ex['output']:
                return None
        return apply_rule(test_input)

    def _try_count_ones_fill_order(self, train, test_input):
        """3x3 grid: count 1s, fill that many cells in fixed order (top row, then (1,1), ...) with 2."""
        FILL_ORDER = [(0,0),(0,1),(0,2),(1,1),(1,0),(1,2),(2,1),(2,0),(2,2)]

        def apply_rule(grid):
            if len(grid) != 3 or len(grid[0]) != 3:
                return None
            n = sum(grid[r][c] != 0 for r in range(3) for c in range(3))
            if n > 9: return None
            out = [[0]*3 for _ in range(3)]
            for r, c in FILL_ORDER[:n]:
                out[r][c] = 2
            return out

        for ex in train:
            if len(ex['input']) != 3 or len(ex['input'][0]) != 3:
                return None
            result = apply_rule(ex['input'])
            if result is None or result != ex['output']:
                return None
        return apply_rule(test_input)

    def _try_nonzero_to_checkerblock(self, train, test_input):
        """Each non-zero cell expands to 2x2 [[1,2],[2,1]] block; zeros stay 2x2 zeros."""
        def apply_rule(grid):
            rows, cols = len(grid), len(grid[0])
            # Find the non-zero indicator value
            nz_vals = set(grid[r][c] for r in range(rows) for c in range(cols) if grid[r][c] != 0)
            if len(nz_vals) != 1:
                return None
            out = [[0]*(cols*2) for _ in range(rows*2)]
            for r in range(rows):
                for c in range(cols):
                    if grid[r][c] != 0:
                        out[r*2][c*2]=1; out[r*2][c*2+1]=2
                        out[r*2+1][c*2]=2; out[r*2+1][c*2+1]=1
            return out

        for ex in train:
            if len(ex['output']) != len(ex['input'])*2 or len(ex['output'][0]) != len(ex['input'][0])*2:
                return None
            result = apply_rule(ex['input'])
            if result is None or result != ex['output']:
                return None
        return apply_rule(test_input)

    def _try_four_quadrant_rotations(self, train, test_input):
        """Output = 4 quadrants: input, rot90CW, rot90CCW, rot180."""
        def rot90cw(grid):
            rows, cols = len(grid), len(grid[0])
            return [[grid[rows-1-c][r] for c in range(cols)] for r in range(rows)]

        def rot90ccw(grid):
            rows, cols = len(grid), len(grid[0])
            return [[grid[c][cols-1-r] for c in range(cols)] for r in range(rows)]

        def rot180(grid):
            return [row[::-1] for row in grid[::-1]]

        def apply_rule(grid):
            rows, cols = len(grid), len(grid[0])
            tl = grid
            tr = rot90cw(grid)
            bl = rot90ccw(grid)
            br = rot180(grid)
            out = []
            for r in range(rows):
                out.append(tl[r] + tr[r])
            for r in range(rows):
                out.append(bl[r] + br[r])
            return out

        for ex in train:
            if len(ex['output']) != len(ex['input'])*2 or len(ex['output'][0]) != len(ex['input'][0])*2:
                return None
            result = apply_rule(ex['input'])
            if result != ex['output']:
                return None
        return apply_rule(test_input)

    def _try_row_rev_row_tile(self, train, test_input):
        """Each row → rev(row) + row + rev(row) + row (4x width)."""
        def apply_rule(grid):
            out = []
            for row in grid:
                rev = row[::-1]
                out.append(rev + list(row) + rev + list(row))
            return out

        for ex in train:
            if len(ex['output']) != len(ex['input']) or len(ex['output'][0]) != len(ex['input'][0])*4:
                return None
            result = apply_rule(ex['input'])
            if result != ex['output']:
                return None
        return apply_rule(test_input)

    def _try_border_pad_extend(self, train, test_input):
        """M×N → (M+2)×(N+2): top/bottom zero-pad, each row repeated with edge-value sides."""
        def apply_rule(grid):
            rows = len(grid)
            out = [[0] + list(grid[0]) + [0]]
            for r in range(rows):
                out.append([grid[r][0]] + list(grid[r]) + [grid[r][-1]])
            out.append([0] + list(grid[-1]) + [0])
            return out

        for ex in train:
            if len(ex['output']) != len(ex['input'])+2 or len(ex['output'][0]) != len(ex['input'][0])+2:
                return None
            result = apply_rule(ex['input'])
            if result != ex['output']:
                return None
        return apply_rule(test_input)

    def _try_diagonal_reveal(self, train, test_input):
        """1×N row: tile it diagonally from bottom-left to top-right. Output size = N×k where k=non-zero count."""
        def apply_rule(grid):
            if len(grid) != 1:
                return None
            row = grid[0]
            N = len(row)
            k = sum(1 for v in row if v != 0)
            if k == 0:
                return None
            S = N * k  # output size
            out = [[0]*S for _ in range(S)]
            for r in range(S):
                start = (S - 1 - r)
                for j in range(N):
                    c = start + j
                    if 0 <= c < S:
                        out[r][c] = row[j]
            return out

        for ex in train:
            result = apply_rule(ex['input'])
            if result is None or result != ex['output']:
                return None
        return apply_rule(test_input)

    def _try_swap_ends_keep_mid(self, train, test_input):
        """5×1 column: swap elements (0,1), keep (2), swap (3,4)."""
        def apply_rule(grid):
            if len(grid) != 5 or len(grid[0]) != 1:
                return None
            vals = [grid[i][0] for i in range(5)]
            out_vals = [vals[1], vals[0], vals[2], vals[4], vals[3]]
            return [[v] for v in out_vals]

        for ex in train:
            result = apply_rule(ex['input'])
            if result is None or result != ex['output']:
                return None
        return apply_rule(test_input)

    def _try_row_tile_mirror_tile(self, train, test_input):
        """R×C → 3R×3C: [original_rows, reversed_rows, original_rows], each row tiled ×3 in cols."""
        def apply_rule(grid):
            R, C = len(grid), len(grid[0])
            rev_rows = [row[::-1] for row in grid]
            all_rows = list(grid) + rev_rows + list(grid)
            return [row * 3 for row in all_rows]

        for ex in train:
            if len(ex['output']) != len(ex['input'])*3 or len(ex['output'][0]) != len(ex['input'][0])*3:
                return None
            result = apply_rule(ex['input'])
            if result != ex['output']:
                return None
        return apply_rule(test_input)

    def _try_rev_row_block_mirror_tile(self, train, test_input):
        """R×C → 3R×2C: nat_block=rev(row)+row, output = rev_block + nat_block + rev_block."""
        def apply_rule(grid):
            nat = [row[::-1] + list(row) for row in grid]
            rev = nat[::-1]
            return rev + nat + rev

        for ex in train:
            if len(ex['output']) != len(ex['input'])*3 or len(ex['output'][0]) != len(ex['input'][0])*2:
                return None
            result = apply_rule(ex['input'])
            if result != ex['output']:
                return None
        return apply_rule(test_input)

    def _try_dominant_band_color(self, train, test_input):
        """Grid has H-bands and V-bands. Output is the color that wins at ALL its intersections."""
        def get_band_color(vals):
            """Return dominant non-zero color if it appears in >50% of non-zero cells."""
            nz = [v for v in vals if v != 0]
            if not nz:
                return None
            from collections import Counter
            c, n = Counter(nz).most_common(1)[0]
            return c if n >= len(vals) * 0.4 else None

        def apply_rule(grid):
            rows, cols = len(grid), len(grid[0])
            h_bands = {}  # color → list of row indices
            v_bands = {}  # color → list of col indices
            for r in range(rows):
                c = get_band_color([grid[r][col] for col in range(cols)])
                if c is not None:
                    h_bands.setdefault(c, []).append(r)
            for col in range(cols):
                c = get_band_color([grid[r][col] for r in range(rows)])
                if c is not None:
                    v_bands.setdefault(c, []).append(col)
            if not h_bands or not v_bands:
                return None
            # Check which color always wins at intersections
            for h_color, h_rows in h_bands.items():
                if all(
                    grid[r][vc] == h_color
                    for v_color, v_cols in v_bands.items()
                    for r in h_rows
                    for vc in v_cols
                    if v_color != h_color
                ):
                    return [[h_color]]
            for v_color, v_cols in v_bands.items():
                if all(
                    grid[hr][c] == v_color
                    for h_color, h_rows in h_bands.items()
                    for hr in h_rows
                    for c in v_cols
                    if h_color != v_color
                ):
                    return [[v_color]]
            return None

        for ex in train:
            if ex['output'] not in [[[v]] for v in range(10)]:
                return None
            result = apply_rule(ex['input'])
            if result is None or result != ex['output']:
                return None
        return apply_rule(test_input)

    def _try_pair_bbox_fill(self, train, test_input):
        """Two cells of same non-zero color define a bounding rectangle; fill it with that color."""
        from collections import defaultdict
        def apply_rule(grid):
            rows, cols = len(grid), len(grid[0])
            cp = defaultdict(list)
            for r in range(rows):
                for c in range(cols):
                    if grid[r][c] != 0:
                        cp[grid[r][c]].append((r, c))
            if not cp or any(len(v) != 2 for v in cp.values()):
                return None
            out = [[0]*cols for _ in range(rows)]
            for color, [(r1,c1),(r2,c2)] in cp.items():
                for r in range(min(r1,r2), max(r1,r2)+1):
                    for c in range(min(c1,c2), max(c1,c2)+1):
                        out[r][c] = color
            return out
        for ex in train:
            if apply_rule(ex['input']) != ex['output']:
                return None
        return apply_rule(test_input)

    def _try_rect_interior_fill2(self, train, test_input):
        """Solid rectangles of non-zero cells: interior (non-border-row/col) cells become 2."""
        from collections import deque
        def apply_rule(grid):
            rows, cols = len(grid), len(grid[0])
            out = [list(row) for row in grid]
            visited = [[False]*cols for _ in range(rows)]
            for sr in range(rows):
                for sc in range(cols):
                    if grid[sr][sc] != 0 and not visited[sr][sc]:
                        color = grid[sr][sc]
                        q = deque([(sr, sc)])
                        comp = []
                        while q:
                            r, c = q.popleft()
                            if visited[r][c]: continue
                            visited[r][c] = True
                            comp.append((r, c))
                            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                                nr, nc = r+dr, c+dc
                                if 0<=nr<rows and 0<=nc<cols and not visited[nr][nc] and grid[nr][nc] == color:
                                    q.append((nr, nc))
                        min_r = min(r for r,c in comp)
                        max_r = max(r for r,c in comp)
                        min_c = min(c for r,c in comp)
                        max_c = max(c for r,c in comp)
                        for r, c in comp:
                            if r != min_r and r != max_r and c != min_c and c != max_c:
                                out[r][c] = 2
            return out
        for ex in train:
            if apply_rule(ex['input']) != ex['output']:
                return None
        return apply_rule(test_input)

    def _try_fill_bbox_holes_7(self, train, test_input):
        """Shapes of non-zero color (8-connected): 0-cells within bounding box become 7."""
        from collections import deque
        DIRS8 = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
        def apply_rule(grid):
            rows, cols = len(grid), len(grid[0])
            out = [list(row) for row in grid]
            visited = [[False]*cols for _ in range(rows)]
            for sr in range(rows):
                for sc in range(cols):
                    if grid[sr][sc] != 0 and not visited[sr][sc]:
                        color = grid[sr][sc]
                        q = deque([(sr, sc)])
                        comp = []
                        while q:
                            r, c = q.popleft()
                            if visited[r][c]: continue
                            visited[r][c] = True
                            comp.append((r, c))
                            for dr, dc in DIRS8:
                                nr, nc = r+dr, c+dc
                                if 0<=nr<rows and 0<=nc<cols and not visited[nr][nc] and grid[nr][nc] == color:
                                    q.append((nr, nc))
                        min_r = min(r for r,c in comp)
                        max_r = max(r for r,c in comp)
                        min_c = min(c for r,c in comp)
                        max_c = max(c for r,c in comp)
                        for r in range(min_r, max_r+1):
                            for c in range(min_c, max_c+1):
                                if grid[r][c] == 0:
                                    out[r][c] = 7
            return out
        for ex in train:
            if apply_rule(ex['input']) != ex['output']:
                return None
        return apply_rule(test_input)

    def _try_grid_split_unique_quadrant(self, train, test_input):
        """Grid divided by a row+col of separator color; output is the quadrant with unique non-bg cell."""
        from collections import Counter
        def apply_rule(grid):
            rows, cols = len(grid), len(grid[0])
            sep_rows = [r for r in range(rows) if len(set(grid[r])) == 1 and grid[r][0] != 0]
            sep_cols = [c for c in range(cols)
                        if len(set(grid[r][c] for r in range(rows))) == 1 and grid[0][c] != 0]
            if not sep_rows or not sep_cols:
                return None
            sep_r = sep_rows[0]
            sep_c = sep_cols[0]
            sep_color = grid[sep_r][sep_c]
            all_vals = [grid[r][c] for r in range(rows) for c in range(cols) if grid[r][c] != sep_color]
            if not all_vals: return None
            bg = Counter(all_vals).most_common(1)[0][0]
            q_rows = [list(range(0, sep_r)), list(range(sep_r+1, rows))]
            q_cols = [list(range(0, sep_c)), list(range(sep_c+1, cols))]
            for rrange in q_rows:
                for crange in q_cols:
                    if not rrange or not crange: continue
                    has_unique = any(grid[r][c] != bg and grid[r][c] != sep_color
                                     for r in rrange for c in crange)
                    if has_unique:
                        return [[grid[r][c] for c in crange] for r in rrange]
            return None
        for ex in train:
            if apply_rule(ex['input']) != ex['output']:
                return None
        return apply_rule(test_input)

    def _try_swap_colors_per_shape(self, train, test_input):
        """Each connected shape (4-conn non-zero) has exactly 2 colors; swap them."""
        from collections import deque
        def apply_rule(grid):
            rows, cols = len(grid), len(grid[0])
            out = [list(row) for row in grid]
            visited = [[False]*cols for _ in range(rows)]
            for sr in range(rows):
                for sc in range(cols):
                    if grid[sr][sc] != 0 and not visited[sr][sc]:
                        q = deque([(sr, sc)])
                        comp = []
                        while q:
                            r, c = q.popleft()
                            if visited[r][c]: continue
                            visited[r][c] = True
                            comp.append((r, c))
                            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                                nr, nc = r+dr, c+dc
                                if 0<=nr<rows and 0<=nc<cols and not visited[nr][nc] and grid[nr][nc] != 0:
                                    q.append((nr, nc))
                        colors = set(grid[r][c] for r, c in comp)
                        if len(colors) != 2:
                            return None
                        c1, c2 = tuple(colors)
                        for r, c in comp:
                            out[r][c] = c2 if grid[r][c] == c1 else c1
            return out
        for ex in train:
            if apply_rule(ex['input']) != ex['output']:
                return None
        return apply_rule(test_input)

    def _try_nor_pattern_match(self, train, test_input):
        """Two patterns stacked with separator row; output 3 where both are 0, else 0."""
        def apply_rule(grid):
            rows, cols = len(grid), len(grid[0])
            sep_rows = [r for r in range(rows) if len(set(grid[r])) == 1 and grid[r][0] != 0]
            if not sep_rows:
                return None
            sep_r = sep_rows[0]
            top = grid[:sep_r]
            bot = grid[sep_r+1:]
            if len(top) != len(bot) or len(top) == 0:
                return None
            out = [[0]*cols for _ in range(len(top))]
            for r in range(len(top)):
                for c in range(cols):
                    if top[r][c] == 0 and bot[r][c] == 0:
                        out[r][c] = 3
            return out
        for ex in train:
            if apply_rule(ex['input']) != ex['output']:
                return None
        return apply_rule(test_input)

    def _try_rect_concentric_rings(self, train, test_input):
        """Solid rectangles of non-zero cells: corners=1, edges=4, interior=2."""
        from collections import deque
        def apply_rule(grid):
            rows, cols = len(grid), len(grid[0])
            out = [list(row) for row in grid]
            visited = [[False]*cols for _ in range(rows)]
            for sr in range(rows):
                for sc in range(cols):
                    if grid[sr][sc] != 0 and not visited[sr][sc]:
                        color = grid[sr][sc]
                        q = deque([(sr, sc)])
                        comp = []
                        while q:
                            r, c = q.popleft()
                            if visited[r][c]: continue
                            visited[r][c] = True
                            comp.append((r, c))
                            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                                nr, nc = r+dr, c+dc
                                if 0<=nr<rows and 0<=nc<cols and not visited[nr][nc] and grid[nr][nc] == color:
                                    q.append((nr, nc))
                        min_r = min(r for r, c in comp)
                        max_r = max(r for r, c in comp)
                        min_c = min(c for r, c in comp)
                        max_c = max(c for r, c in comp)
                        for r, c in comp:
                            on_r = (r == min_r or r == max_r)
                            on_c = (c == min_c or c == max_c)
                            if on_r and on_c:
                                out[r][c] = 1
                            elif on_r or on_c:
                                out[r][c] = 4
                            else:
                                out[r][c] = 2
            return out
        for ex in train:
            if apply_rule(ex['input']) != ex['output']:
                return None
        return apply_rule(test_input)

    def _try_gravity_sink_1_through_5(self, train, test_input):
        """Column [1, 5, ..., 5]: 1 gravity-sinks to bottom of 5-stack, original 1 pos becomes 0."""
        def apply_rule(grid):
            rows, cols = len(grid), len(grid[0])
            out = [list(row) for row in grid]
            for c in range(cols):
                col = [grid[r][c] for r in range(rows)]
                for r in range(rows):
                    if col[r] == 1:
                        end = r
                        while end + 1 < rows and col[end + 1] == 5:
                            end += 1
                        if end > r:
                            out[r][c] = 0
                            for rr in range(r + 1, end):
                                out[rr][c] = 5
                            out[end][c] = 1
                        break
            return out
        for ex in train:
            if apply_rule(ex['input']) != ex['output']:
                return None
        return apply_rule(test_input)

    def _try_unique_cell_box(self, train, test_input):
        """Value appearing exactly once: surround with 3x3 box of 2s (center keeps value)."""
        from collections import Counter
        def apply_rule(grid):
            rows, cols = len(grid), len(grid[0])
            flat = [grid[r][c] for r in range(rows) for c in range(cols) if grid[r][c] != 0]
            if not flat:
                return None
            cnt = Counter(flat)
            unique_vals = [v for v, n in cnt.items() if n == 1]
            if len(unique_vals) != 1:
                return None
            uv = unique_vals[0]
            pos = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == uv]
            r0, c0 = pos[0]
            out = [[0]*cols for _ in range(rows)]
            for dr in range(-1, 2):
                for dc in range(-1, 2):
                    nr, nc = r0+dr, c0+dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        out[nr][nc] = 2
            out[r0][c0] = uv
            return out
        for ex in train:
            if apply_rule(ex['input']) != ex['output']:
                return None
        return apply_rule(test_input)

    def _try_band_zero_extend(self, train, test_input):
        """In each colored band, a 0 hole extends as a full line perpendicular to band orientation."""
        from collections import deque
        def apply_rule(grid):
            rows, cols = len(grid), len(grid[0])
            out = [list(row) for row in grid]
            visited = [[False]*cols for _ in range(rows)]
            for sr in range(rows):
                for sc in range(cols):
                    if grid[sr][sc] != 0 and not visited[sr][sc]:
                        color = grid[sr][sc]
                        q = deque([(sr, sc)])
                        comp = []
                        while q:
                            r, c = q.popleft()
                            if visited[r][c]: continue
                            visited[r][c] = True
                            comp.append((r, c))
                            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                                nr, nc = r+dr, c+dc
                                if 0<=nr<rows and 0<=nc<cols and not visited[nr][nc] and grid[nr][nc] == color:
                                    q.append((nr, nc))
                        min_r = min(r for r, c in comp)
                        max_r = max(r for r, c in comp)
                        min_c = min(c for r, c in comp)
                        max_c = max(c for r, c in comp)
                        h = max_r - min_r + 1
                        w = max_c - min_c + 1
                        for r in range(min_r, max_r+1):
                            for c in range(min_c, max_c+1):
                                if grid[r][c] == 0:
                                    if w >= h:
                                        for rr in range(min_r, max_r+1):
                                            out[rr][c] = 0
                                    else:
                                        for cc in range(min_c, max_c+1):
                                            out[r][cc] = 0
            return out
        for ex in train:
            if apply_rule(ex['input']) != ex['output']:
                return None
        return apply_rule(test_input)

    def _try_rot180(self, train, test_input):
        """Output is input rotated 180 degrees."""
        def apply_rule(grid):
            return [row[::-1] for row in grid[::-1]]
        for ex in train:
            if apply_rule(ex['input']) != ex['output']:
                return None
        return apply_rule(test_input)

    def _try_vflip_then_original(self, train, test_input):
        """Output is reversed-rows + original rows (vertical mirror on top, original on bottom)."""
        def apply_rule(grid):
            return list(grid[::-1]) + list(grid)
        for ex in train:
            if apply_rule(ex['input']) != ex['output']:
                return None
        return apply_rule(test_input)

    def _try_or_pattern_match(self, train, test_input):
        """Two patterns stacked with separator row; output 3 where either is non-zero, else 0."""
        def apply_rule(grid):
            rows, cols = len(grid), len(grid[0])
            sep_rows = [r for r in range(rows) if len(set(grid[r])) == 1 and grid[r][0] != 0]
            if not sep_rows:
                return None
            sep_r = sep_rows[0]
            top = grid[:sep_r]
            bot = grid[sep_r+1:]
            if len(top) != len(bot) or len(top) == 0:
                return None
            out = [[0]*cols for _ in range(len(top))]
            for r in range(len(top)):
                for c in range(cols):
                    if top[r][c] != 0 or bot[r][c] != 0:
                        out[r][c] = 3
            return out
        for ex in train:
            if apply_rule(ex['input']) != ex['output']:
                return None
        return apply_rule(test_input)

    def _try_merge_3_2_to_8(self, train, test_input):
        """Adjacent 3-2 pairs merge: 3 becomes 8, 2 disappears."""
        def apply_rule(grid):
            rows, cols = len(grid), len(grid[0])
            out = [list(row) for row in grid]
            for r in range(rows):
                for c in range(cols):
                    if grid[r][c] == 3:
                        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                            nr, nc = r+dr, c+dc
                            if 0<=nr<rows and 0<=nc<cols and grid[nr][nc] == 2:
                                out[r][c] = 8
                                out[nr][nc] = 0
                                break
            return out
        for ex in train:
            if apply_rule(ex['input']) != ex['output']:
                return None
        return apply_rule(test_input)

    def _try_find_hollow_shape_color(self, train, test_input):
        """Find shape with interior 0s (hollow rectangle); output [[color]]."""
        from collections import deque
        def apply_rule(grid):
            rows, cols = len(grid), len(grid[0])
            visited = [[False]*cols for _ in range(rows)]
            for sr in range(rows):
                for sc in range(cols):
                    if grid[sr][sc] != 0 and not visited[sr][sc]:
                        color = grid[sr][sc]
                        q = deque([(sr, sc)])
                        comp = []
                        while q:
                            r, c = q.popleft()
                            if visited[r][c]: continue
                            visited[r][c] = True
                            comp.append((r, c))
                            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                                nr, nc = r+dr, c+dc
                                if 0<=nr<rows and 0<=nc<cols and not visited[nr][nc] and grid[nr][nc] == color:
                                    q.append((nr, nc))
                        min_r = min(r for r, c in comp)
                        max_r = max(r for r, c in comp)
                        min_c = min(c for r, c in comp)
                        max_c = max(c for r, c in comp)
                        for r in range(min_r+1, max_r):
                            for c in range(min_c+1, max_c):
                                if grid[r][c] == 0:
                                    return [[color]]
            return None
        for ex in train:
            if apply_rule(ex['input']) != ex['output']:
                return None
        return apply_rule(test_input)

    def _try_row_color_fill_5(self, train, test_input):
        """Indicator column: row's non-zero non-5 value fills all 5s in that row."""
        def find_indicator(grid):
            rows, cols = len(grid), len(grid[0])
            for c in range(cols):
                col_vals = [grid[r][c] for r in range(rows)]
                if any(v == 5 for v in col_vals):
                    continue
                if any(v != 0 for v in col_vals):
                    return c
            return None
        def apply_rule(grid):
            rows, cols = len(grid), len(grid[0])
            ind = find_indicator(grid)
            if ind is None:
                return None
            out = [list(row) for row in grid]
            for r in range(rows):
                color = grid[r][ind]
                if color != 0:
                    for c in range(cols):
                        if grid[r][c] == 5:
                            out[r][c] = color
            return out
        for ex in train:
            if apply_rule(ex['input']) != ex['output']:
                return None
        return apply_rule(test_input)

    def _try_reflect_bottom_to_top(self, train, test_input):
        """Top half becomes mirror of bottom half; bottom half unchanged (vertical symmetry)."""
        def apply_rule(grid):
            rows = len(grid)
            out = [list(row) for row in grid]
            for r in range(rows // 2):
                out[r] = list(grid[rows - 1 - r])
            return out
        for ex in train:
            if apply_rule(ex['input']) != ex['output']:
                return None
        return apply_rule(test_input)

    def _try_rect_hollow_out(self, train, test_input):
        """Solid filled rectangles: interior cells become 0 (border only stays)."""
        from collections import deque
        def apply_rule(grid):
            rows, cols = len(grid), len(grid[0])
            out = [list(row) for row in grid]
            visited = [[False]*cols for _ in range(rows)]
            for sr in range(rows):
                for sc in range(cols):
                    if grid[sr][sc] != 0 and not visited[sr][sc]:
                        color = grid[sr][sc]
                        q = deque([(sr, sc)])
                        comp = []
                        while q:
                            r, c = q.popleft()
                            if visited[r][c]: continue
                            visited[r][c] = True
                            comp.append((r, c))
                            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                                nr, nc = r+dr, c+dc
                                if 0<=nr<rows and 0<=nc<cols and not visited[nr][nc] and grid[nr][nc] == color:
                                    q.append((nr, nc))
                        min_r = min(r for r, c in comp)
                        max_r = max(r for r, c in comp)
                        min_c = min(c for r, c in comp)
                        max_c = max(c for r, c in comp)
                        for r, c in comp:
                            if r != min_r and r != max_r and c != min_c and c != max_c:
                                out[r][c] = 0
            return out
        for ex in train:
            if apply_rule(ex['input']) != ex['output']:
                return None
        return apply_rule(test_input)

    def _try_max_zero_rect_fill6(self, train, test_input):
        """Find the largest all-zero rectangle in the grid and fill with 6."""
        def apply_rule(grid):
            rows, cols = len(grid), len(grid[0])
            h = [0] * cols
            best_area, best = 0, None
            for r in range(rows):
                for c in range(cols):
                    h[c] = h[c] + 1 if grid[r][c] == 0 else 0
                stack = []
                for c in range(cols + 1):
                    height = h[c] if c < cols else 0
                    start = c
                    while stack and stack[-1][1] > height:
                        sc, sh = stack.pop()
                        w = c - sc
                        area = sh * w
                        if area > best_area:
                            best_area = area
                            best = (r - sh + 1, sc, r, sc + w - 1)
                        start = sc
                    stack.append((start, height))
            if best is None:
                return None
            r1, c1, r2, c2 = best
            out = [list(row) for row in grid]
            for r in range(r1, r2 + 1):
                for c in range(c1, c2 + 1):
                    out[r][c] = 6
            return out
        for ex in train:
            if apply_rule(ex['input']) != ex['output']:
                return None
        return apply_rule(test_input)

    def _try_extend_pattern_rows_10(self, train, test_input):
        """Detect repeating row period and extend to 10 rows."""
        def find_period(grid):
            n = len(grid)
            for p in range(1, n + 1):
                valid = True
                for i in range(n):
                    if grid[i] != grid[i % p]:
                        valid = False
                        break
                if valid:
                    return p
            return n
        def apply_rule(grid):
            rows = len(grid)
            if rows >= 10:
                return None
            p = find_period(grid)
            cycle = grid[:p]
            return [list(cycle[i % p]) for i in range(10)]
        for ex in train:
            if apply_rule(ex['input']) != ex['output']:
                return None
        return apply_rule(test_input)

    def _try_cross_to_4fold(self, train, test_input):
        """Pattern in one quadrant of a cross-divided grid → reflected to all 4 quadrants."""
        def apply_rule(grid):
            rows, cols = len(grid), len(grid[0])
            sep_rows = [r for r in range(rows) if all(grid[r][c] != 0 and grid[r][c] == grid[r][0] for c in range(cols))]
            sep_cols = [c for c in range(cols) if all(grid[r][c] != 0 and grid[r][c] == grid[0][c] for r in range(rows))]
            if not sep_rows or not sep_cols:
                return None
            sr, sc = sep_rows[0], sep_cols[0]
            sep_color = grid[sr][sc]
            q_ranges = [
                (list(range(0, sr)), list(range(0, sc))),
                (list(range(0, sr)), list(range(sc+1, cols))),
                (list(range(sr+1, rows)), list(range(0, sc))),
                (list(range(sr+1, rows)), list(range(sc+1, cols))),
            ]
            pattern_q = None
            for qi, (rr, cc) in enumerate(q_ranges):
                if rr and cc and any(grid[r][c] != 0 for r in rr for c in cc):
                    pattern_q = qi
                    break
            if pattern_q is None:
                return None
            rr, cc = q_ranges[pattern_q]
            h, w = len(rr), len(cc)
            if h == 0 or w == 0:
                return None
            pat = [[sep_color if grid[r][c] != 0 else 0 for c in cc] for r in rr]

            def hflip(p): return [row[::-1] for row in p]
            def vflip(p): return p[::-1]
            def rot180(p): return [row[::-1] for row in p[::-1]]

            if pattern_q == 0:
                tl, tr, bl, br = pat, hflip(pat), vflip(pat), rot180(pat)
            elif pattern_q == 1:
                tl, tr, bl, br = hflip(pat), pat, rot180(pat), vflip(pat)
            elif pattern_q == 2:
                tl, tr, bl, br = vflip(pat), rot180(pat), pat, hflip(pat)
            else:
                tl, tr, bl, br = rot180(pat), vflip(pat), hflip(pat), pat

            out = [[0]*(w*2) for _ in range(h*2)]
            for i in range(h):
                for j in range(w):
                    out[i][j] = tl[i][j]
                    out[i][w+j] = tr[i][j]
                    out[h+i][j] = bl[i][j]
                    out[h+i][w+j] = br[i][j]
            return out
        for ex in train:
            if apply_rule(ex['input']) != ex['output']:
                return None
        return apply_rule(test_input)

    def _try_cell_grid_count_nonbg(self, train, test_input):
        """Grid of 2x2 color blocks separated by 0-rows/cols. Count non-background colors (bg=most
        common). Output sorted column vector [[c1],[c2],[c3]] descending by count."""
        from collections import Counter
        def extract_cells(grid):
            rows, cols = len(grid), len(grid[0])
            sep_rows = [r for r in range(rows) if all(grid[r][c] == 0 for c in range(cols))]
            sep_cols = [c for c in range(cols) if all(grid[r][c] == 0 for r in range(rows))]
            if not sep_rows or not sep_cols:
                return None
            # data rows and cols are between separators
            data_rows = [r for r in range(rows) if r not in sep_rows]
            data_cols = [c for c in range(cols) if c not in sep_cols]
            if not data_rows or not data_cols:
                return None
            # Extract representative cell color for each grid cell (take [0][0] of each block)
            # First determine row groups (contiguous data rows) and col groups
            row_groups = []
            grp = []
            for r in data_rows:
                if not grp or r == grp[-1]+1:
                    grp.append(r)
                else:
                    row_groups.append(grp)
                    grp = [r]
            if grp:
                row_groups.append(grp)
            col_groups = []
            grp = []
            for c in data_cols:
                if not grp or c == grp[-1]+1:
                    grp.append(c)
                else:
                    col_groups.append(grp)
                    grp = [c]
            if grp:
                col_groups.append(grp)
            cells = []
            for rg in row_groups:
                for cg in col_groups:
                    color = grid[rg[0]][cg[0]]
                    if not all(grid[r][c] == color for r in rg for c in cg):
                        return None
                    cells.append(color)
            return cells
        def apply_rule(grid):
            cells = extract_cells(grid)
            if cells is None:
                return None
            cnt = Counter(cells)
            bg = cnt.most_common(1)[0][0]
            non_bg = {c: n for c, n in cnt.items() if c != bg}
            if not non_bg:
                return None
            sorted_colors = sorted(non_bg.keys(), key=lambda c: -non_bg[c])
            return [[c] for c in sorted_colors]
        for ex in train:
            if apply_rule(ex['input']) != ex['output']:
                return None
        return apply_rule(test_input)

    def _try_sep_col_both_zero_to_8(self, train, test_input):
        """Grid divided by a separator column (all same non-zero value). Left and right halves
        same size. Output 8 where both sides are 0, else 0."""
        def apply_rule(grid):
            rows, cols = len(grid), len(grid[0])
            # Find separator column
            sep_c = None
            for c in range(cols):
                col_vals = [grid[r][c] for r in range(rows)]
                if len(set(col_vals)) == 1 and col_vals[0] != 0:
                    sep_c = c
                    break
            if sep_c is None:
                return None
            left_w = sep_c
            right_w = cols - sep_c - 1
            if left_w != right_w or left_w == 0:
                return None
            out = []
            for r in range(rows):
                row = []
                for c in range(left_w):
                    lv = grid[r][c]
                    rv = grid[r][sep_c + 1 + c]
                    row.append(8 if lv == 0 and rv == 0 else 0)
                out.append(row)
            return out
        for ex in train:
            if apply_rule(ex['input']) != ex['output']:
                return None
        return apply_rule(test_input)

    def _try_shape_cross_pattern(self, train, test_input):
        """Single non-zero shape → cross/pinwheel: original left, hflip right,
        transpose top, vflip(transpose) bottom. Output size (2W+H)x(2W+H)."""
        def apply_rule(grid):
            rows, cols = len(grid), len(grid[0])
            # Find bounding box
            min_r = min((r for r in range(rows) if any(grid[r][c] != 0 for c in range(cols))), default=None)
            max_r = max((r for r in range(rows) if any(grid[r][c] != 0 for c in range(cols))), default=None)
            min_c = min((c for c in range(cols) if any(grid[r][c] != 0 for r in range(rows))), default=None)
            max_c = max((c for c in range(cols) if any(grid[r][c] != 0 for r in range(rows))), default=None)
            if min_r is None:
                return None
            H = max_r - min_r + 1
            W = max_c - min_c + 1
            shape = [grid[r][min_c:max_c+1] for r in range(min_r, max_r+1)]
            sz = 2 * W + H
            out = [[0] * sz for _ in range(sz)]
            # Left: original at [W:W+H, 0:W]
            for i in range(H):
                for j in range(W):
                    out[W+i][j] = shape[i][j]
            # Right: hflip at [W:W+H, W+H:2W+H]
            for i in range(H):
                for j in range(W):
                    out[W+i][W+H+j] = shape[i][W-1-j]
            # Transpose T: cols of shape become rows, placed at [0:W, W:W+H]
            for j in range(W):
                for i in range(H):
                    out[j][W+i] = shape[i][j]
            # VFlip(T) at [W+H:2W+H, W:W+H]
            for j in range(W):
                for i in range(H):
                    out[W+H+(W-1-j)][W+i] = shape[i][j]
            return out
        for ex in train:
            if apply_rule(ex['input']) != ex['output']:
                return None
        return apply_rule(test_input)

    def _try_original_then_vflip(self, train, test_input):
        """Output = input rows followed by input rows in reverse (original + vflip appended below)."""
        def apply_rule(grid):
            return list(grid) + list(reversed(grid))
        for ex in train:
            if apply_rule(ex['input']) != ex['output']:
                return None
        return apply_rule(test_input)

    def _try_dedup_horiz_tile(self, train, test_input):
        """Input tiles with a period along cols OR rows; output is one period unit."""
        def find_col_period(grid):
            rows, cols = len(grid), len(grid[0])
            for p in range(1, cols):
                if cols % p != 0:
                    continue
                if all(grid[r][c] == grid[r][c % p] for r in range(rows) for c in range(cols)):
                    return p
            return None
        def find_row_period(grid):
            rows, cols = len(grid), len(grid[0])
            for p in range(1, rows):
                if rows % p != 0:
                    continue
                if all(grid[r][c] == grid[r % p][c] for r in range(rows) for c in range(cols)):
                    return p
            return None
        def apply_rule(grid):
            pc = find_col_period(grid)
            if pc is not None and pc < len(grid[0]):
                return [row[:pc] for row in grid]
            pr = find_row_period(grid)
            if pr is not None and pr < len(grid):
                return grid[:pr]
            return None
        for ex in train:
            if apply_rule(ex['input']) != ex['output']:
                return None
        return apply_rule(test_input)

    def _try_fill_border_8(self, train, test_input):
        """All-zero input → output with border cells = 8, interior = 0."""
        def apply_rule(grid):
            rows, cols = len(grid), len(grid[0])
            if any(grid[r][c] != 0 for r in range(rows) for c in range(cols)):
                return None
            out = [[0]*cols for _ in range(rows)]
            for r in range(rows):
                for c in range(cols):
                    if r == 0 or r == rows-1 or c == 0 or c == cols-1:
                        out[r][c] = 8
            return out
        for ex in train:
            if apply_rule(ex['input']) != ex['output']:
                return None
        return apply_rule(test_input)

    def _try_lower_to_upper_triangle(self, train, test_input):
        """Grid with diagonal of identical values; transpose lower triangle into upper, zero lower."""
        def apply_rule(grid):
            rows, cols = len(grid), len(grid[0])
            if rows != cols:
                return None
            n = rows
            # Detect diagonal value (must be same on all diagonal cells)
            diag_vals = [grid[i][i] for i in range(n)]
            if len(set(diag_vals)) != 1:
                return None
            # Upper triangle should be all 0 in input
            if any(grid[r][c] != 0 for r in range(n) for c in range(r+1, n)):
                return None
            out = [[0]*n for _ in range(n)]
            for i in range(n):
                out[i][i] = diag_vals[0]
            # Transpose lower to upper: out[r][c] = grid[c][r] for r < c
            for r in range(n):
                for c in range(r+1, n):
                    out[r][c] = grid[c][r]
            return out
        for ex in train:
            if apply_rule(ex['input']) != ex['output']:
                return None
        return apply_rule(test_input)

    def _try_extract_symmetric_shape(self, train, test_input):
        """Grid has multiple non-zero shapes; output is the bounding-box of the
        horizontally symmetric (left-right) shape."""
        def get_color_bboxes(grid):
            rows, cols = len(grid), len(grid[0])
            from collections import defaultdict
            color_cells = defaultdict(list)
            for r in range(rows):
                for c in range(cols):
                    if grid[r][c] != 0:
                        color_cells[grid[r][c]].append((r, c))
            bboxes = {}
            for color, cells in color_cells.items():
                min_r = min(r for r,c in cells)
                max_r = max(r for r,c in cells)
                min_c = min(c for r,c in cells)
                max_c = max(c for r,c in cells)
                sub = [grid[r][min_c:max_c+1] for r in range(min_r, max_r+1)]
                bboxes[color] = sub
            return bboxes
        def is_h_symmetric(sub):
            return all(row == row[::-1] for row in sub)
        def apply_rule(grid):
            bboxes = get_color_bboxes(grid)
            if len(bboxes) < 2:
                return None
            sym_shapes = [(color, sub) for color, sub in bboxes.items() if is_h_symmetric(sub)]
            if len(sym_shapes) != 1:
                return None
            return sym_shapes[0][1]
        for ex in train:
            if apply_rule(ex['input']) != ex['output']:
                return None
        return apply_rule(test_input)

    def _try_corner_color_quadrant(self, train, test_input):
        """Grid with 1-filled border rows/cols forming a +. 4 corner cells have colors.
        Interior cells are 8 (active) or 0. Output: active cells colored by which quadrant
        corner they belong to; inactive cells = 0."""
        def apply_rule(grid):
            rows, cols = len(grid), len(grid[0])
            # Find separator rows and cols (all 1s)
            sep_rows = [r for r in range(rows) if all(grid[r][c] == 1 for c in range(cols))]
            sep_cols = [c for c in range(cols) if all(grid[r][c] == 1 for r in range(rows))]
            if len(sep_rows) != 2 or len(sep_cols) != 2:
                return None
            r1, r2 = sep_rows[0], sep_rows[1]
            c1, c2 = sep_cols[0], sep_cols[1]
            # Corners must be exactly at grid edges outside separators
            if r1 != 1 or r2 != rows-2 or c1 != 1 or c2 != cols-2:
                return None
            tl = grid[0][0]
            tr = grid[0][cols-1]
            bl = grid[rows-1][0]
            br = grid[rows-1][cols-1]
            if any(v in (0, 1) for v in (tl, tr, bl, br)):
                return None
            # Extract interior
            interior = [grid[r][c1+1:c2] for r in range(r1+1, r2)]
            ih, iw = len(interior), len(interior[0]) if interior else 0
            if ih == 0 or iw == 0:
                return None
            out = [[0]*iw for _ in range(ih)]
            for r in range(ih):
                for c in range(iw):
                    v = interior[r][c]
                    if v != 0:
                        corner = tl if r < ih//2 and c < iw//2 else \
                                 tr if r < ih//2 and c >= iw//2 else \
                                 bl if r >= ih//2 and c < iw//2 else br
                        out[r][c] = corner
            return out
        for ex in train:
            if apply_rule(ex['input']) != ex['output']:
                return None
        return apply_rule(test_input)

    def _try_mirror_top_rows_to_bottom(self, train, test_input):
        """Non-zero rows at the top; output = same grid with those rows reversed at the bottom."""
        def apply_rule(grid):
            rows = len(grid)
            # Find P = number of consecutive non-zero rows at top
            p = 0
            for row in grid:
                if any(v != 0 for v in row):
                    p += 1
                else:
                    break
            if p == 0 or p >= rows:
                return None
            # Check remaining rows are all zero
            if any(v != 0 for r in range(p, rows) for v in grid[r]):
                return None
            # Output: same but bottom p rows = reversed top p rows
            out = [list(row) for row in grid]
            for i in range(p):
                out[rows - p + i] = list(grid[p - 1 - i])
            return out
        for ex in train:
            if apply_rule(ex['input']) != ex['output']:
                return None
        return apply_rule(test_input)

    def _try_color_col_diagonal_and_bottom(self, train, test_input):
        """Left column filled with one color, rest zeros. Output: preserve left col,
        add diagonal of 2s from top-right down-left, fill bottom row with 4s."""
        def apply_rule(grid):
            rows, cols = len(grid), len(grid[0])
            # Check left col all same non-zero color
            colors = set(grid[r][0] for r in range(rows))
            if len(colors) != 1 or 0 in colors:
                return None
            color = grid[0][0]
            # Check rest is all zeros
            if any(grid[r][c] != 0 for r in range(rows) for c in range(1, cols)):
                return None
            out = [list(row) for row in grid]
            # Diagonal of 2s: from (0, cols-1) going down-left to (rows-2, 1)
            for i in range(rows - 1):
                r, c = i, cols - 1 - i
                if c >= 1:
                    out[r][c] = 2
            # Bottom row: col 0 stays color, rest = 4
            for c in range(1, cols):
                out[rows-1][c] = 4
            return out
        for ex in train:
            if apply_rule(ex['input']) != ex['output']:
                return None
        return apply_rule(test_input)

    def _try_dedup_consecutive_rows_cols(self, train, test_input):
        """Collapse consecutive identical rows, then consecutive identical columns."""
        def dedup_rows(grid):
            if not grid:
                return grid
            out = [grid[0]]
            for row in grid[1:]:
                if row != out[-1]:
                    out.append(row)
            return out
        def dedup_cols(grid):
            if not grid:
                return grid
            cols = len(grid[0])
            keep = [0]
            for c in range(1, cols):
                if any(grid[r][c] != grid[r][keep[-1]] for r in range(len(grid))):
                    keep.append(c)
            return [[row[c] for c in keep] for row in grid]
        def apply_rule(grid):
            g = dedup_rows(grid)
            g = dedup_cols(g)
            if g == grid:
                return None
            return g
        for ex in train:
            if apply_rule(ex['input']) != ex['output']:
                return None
        return apply_rule(test_input)

    def _try_split_insert_9_sep(self, train, test_input):
        """Split input at midrow. Output: top rows + right-9-col, separator row of 9s,
        bottom rows + left-9-col. Output size (H+1)x(W+1)."""
        def apply_rule(grid):
            rows, cols = len(grid), len(grid[0])
            if rows % 2 != 0:
                return None
            mid = rows // 2
            top = [list(row) + [9] for row in grid[:mid]]
            sep = [9] * (cols + 1)
            bot = [[9] + list(row) for row in grid[mid:]]
            return top + [sep] + bot
        for ex in train:
            if apply_rule(ex['input']) != ex['output']:
                return None
        return apply_rule(test_input)

    def _try_extend_row_adding_one(self, train, test_input):
        """1-row input: N non-zeros then zeros. Output K = cols//2 rows, each adding 1 more non-zero."""
        def apply_rule(grid):
            if len(grid) != 1:
                return None
            row = grid[0]
            cols = len(row)
            # Count initial non-zeros
            n = 0
            val = row[0]
            while n < cols and row[n] != 0:
                n += 1
            if n == 0:
                return None
            val = row[0]
            k = cols // 2
            out = []
            for i in range(k):
                filled = n + i
                out.append([val] * min(filled, cols) + [0] * max(0, cols - filled))
            return out
        for ex in train:
            if apply_rule(ex['input']) != ex['output']:
                return None
        return apply_rule(test_input)

    def _try_nor_top_bottom_half(self, train, test_input):
        """Stack of 2 identical-height patterns (no separator). Output 2 at positions
        where BOTH halves are 0, else 0."""
        def apply_rule(grid):
            rows, cols = len(grid), len(grid[0])
            if rows % 2 != 0:
                return None
            half = rows // 2
            top = [grid[r] for r in range(half)]
            bot = [grid[r] for r in range(half, rows)]
            out = []
            for r in range(half):
                row = []
                for c in range(cols):
                    row.append(2 if top[r][c] == 0 and bot[r][c] == 0 else 0)
                out.append(row)
            return out
        for ex in train:
            if apply_rule(ex['input']) != ex['output']:
                return None
        return apply_rule(test_input)

    def _try_or_left_right_half(self, train, test_input):
        """Side-by-side two identical-width patterns. Output 6 at positions where
        at least one half is non-zero, else 0."""
        def apply_rule(grid):
            rows, cols = len(grid), len(grid[0])
            if cols % 2 != 0:
                return None
            half = cols // 2
            out = []
            for r in range(rows):
                row = []
                for c in range(half):
                    row.append(6 if grid[r][c] != 0 or grid[r][c + half] != 0 else 0)
                out.append(row)
            return out
        for ex in train:
            if apply_rule(ex['input']) != ex['output']:
                return None
        return apply_rule(test_input)

    def _try_extract_diamond_5x5(self, train, test_input):
        """5x5 grid with values on main diagonal and anti-diagonal. Extract to 3x3."""
        def apply_rule(grid):
            if len(grid) != 5 or len(grid[0]) != 5:
                return None
            return [
                [grid[0][0], grid[1][1], grid[0][4]],
                [grid[1][3], grid[2][2], grid[3][1]],
                [grid[4][0], grid[3][3], grid[4][4]]
            ]
        for ex in train:
            if apply_rule(ex['input']) != ex['output']:
                return None
        return apply_rule(test_input)

    def _try_unique_quadrant(self, train, test_input):
        """5x5 grid with zero separator row and col. Find the unique 2x2 quadrant
        (differs from the other three) and return it."""
        def apply_rule(grid):
            rows, cols = len(grid), len(grid[0])
            # Find zero row and zero col separators
            sep_rows = [r for r in range(rows) if all(grid[r][c] == 0 for c in range(cols))]
            sep_cols = [c for c in range(cols) if all(grid[r][c] == 0 for r in range(rows))]
            if len(sep_rows) != 1 or len(sep_cols) != 1:
                return None
            sr, sc = sep_rows[0], sep_cols[0]
            # Extract 4 quadrants
            def quad(r_range, c_range):
                return [grid[r][c_range.start:c_range.stop] for r in r_range]
            q = [
                quad(range(0, sr), range(0, sc)),
                quad(range(0, sr), range(sc+1, cols)),
                quad(range(sr+1, rows), range(0, sc)),
                quad(range(sr+1, rows), range(sc+1, cols))
            ]
            # Find the unique quadrant
            for i in range(4):
                others = [q[j] for j in range(4) if j != i]
                if all(others[0] == o for o in others) and q[i] != others[0]:
                    return q[i]
            return None
        for ex in train:
            if apply_rule(ex['input']) != ex['output']:
                return None
        return apply_rule(test_input)

    def _try_row_bars_with_zone_borders(self, train, test_input):
        """Grid of zeros with single non-zero cells defining 'bar' rows.
        Each bar row fills entirely. Non-bar rows get left/right borders colored
        by nearest bar (nearest = smaller distance, ties to upper).
        First and last rows always become full bars of first/last bar color."""
        def apply_rule(grid):
            rows, cols = len(grid), len(grid[0])
            # Find bar rows: rows with exactly one non-zero cell anywhere
            bar_rows = []
            for r in range(rows):
                nz = [(c, grid[r][c]) for c in range(cols) if grid[r][c] != 0]
                if len(nz) == 1:
                    bar_rows.append((r, nz[0][1]))
                elif len(nz) > 1:
                    return None  # multiple non-zero cells in a row
            if len(bar_rows) < 2:
                return None
            bar_rows.sort()
            out = [[0]*cols for _ in range(rows)]
            # Set bar rows
            for r, v in bar_rows:
                out[r] = [v] * cols
            # Set first and last rows as full bars
            out[0] = [bar_rows[0][1]] * cols
            out[rows-1] = [bar_rows[-1][1]] * cols
            # Zone assignment: for each non-bar row, find zone color
            bar_positions = [r for r, v in bar_rows]
            bar_colors = {r: v for r, v in bar_rows}
            for r in range(rows):
                # skip bar rows and first/last
                if r == 0 or r == rows-1 or r in bar_colors:
                    continue
                # Find zone: which bar interval contains r
                zone_color = None
                if r < bar_positions[0]:
                    zone_color = bar_rows[0][1]
                elif r > bar_positions[-1]:
                    zone_color = bar_rows[-1][1]
                else:
                    for i in range(len(bar_rows)-1):
                        ri, ci = bar_rows[i]
                        ri1, ci1 = bar_rows[i+1]
                        if ri < r < ri1:
                            mid = (ri + ri1) // 2
                            zone_color = ci if r <= mid else ci1
                            break
                if zone_color is None:
                    return None
                out[r][0] = zone_color
                out[r][cols-1] = zone_color
            return out
        for ex in train:
            if apply_rule(ex['input']) != ex['output']:
                return None
        return apply_rule(test_input)

    def _try_cross_and_diagonals(self, train, test_input):
        """Grid is all one background color with a few 1s. Each 1 at (r,c):
        fills row r and col c with 1s, places 2 at intersection,
        places 3 at the 4 diagonal neighbors (if still background)."""
        def apply_rule(grid):
            rows, cols = len(grid), len(grid[0])
            # Find background and cross value
            from collections import Counter
            flat = [v for row in grid for v in row]
            cnt = Counter(flat)
            if len(cnt) < 2:
                return None
            bg = cnt.most_common(1)[0][0]
            cross_vals = [v for v, c in cnt.items() if v != bg]
            if len(cross_vals) != 1:
                return None
            cv = cross_vals[0]
            crosses = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == cv]
            if not crosses:
                return None
            out = [list(row) for row in grid]
            # Draw bars
            for r, c in crosses:
                for cc in range(cols):
                    out[r][cc] = cv
                for rr in range(rows):
                    out[rr][c] = cv
            # Mark intersections with 2
            for r, c in crosses:
                out[r][c] = 2
            # Mark diagonals with 3 (only if background)
            for r, c in crosses:
                for dr, dc in [(-1,-1),(-1,1),(1,-1),(1,1)]:
                    nr, nc = r+dr, c+dc
                    if 0 <= nr < rows and 0 <= nc < cols and out[nr][nc] == bg:
                        out[nr][nc] = 3
            return out
        for ex in train:
            if apply_rule(ex['input']) != ex['output']:
                return None
        return apply_rule(test_input)

    def _try_flow_to_wall(self, train, test_input):
        """Grid has a wall (row or col, all one non-zero color). Non-zero cells
        flow toward the wall, filling the line from their position to next
        obstacle or the wall."""
        def find_wall(grid):
            rows, cols = len(grid), len(grid[0])
            # Check columns (right then left)
            if all(grid[r][cols-1] != 0 for r in range(rows)):
                wc = grid[0][cols-1]
                if all(grid[r][cols-1] == wc for r in range(rows)):
                    return ('right', cols-1, wc)
            if all(grid[r][0] != 0 for r in range(rows)):
                wc = grid[0][0]
                if all(grid[r][0] == wc for r in range(rows)):
                    return ('left', 0, wc)
            # Check rows (bottom then top)
            if all(grid[rows-1][c] != 0 for c in range(cols)):
                wc = grid[rows-1][0]
                if all(grid[rows-1][c] == wc for c in range(cols)):
                    return ('down', rows-1, wc)
            if all(grid[0][c] != 0 for c in range(cols)):
                wc = grid[0][0]
                if all(grid[0][c] == wc for c in range(cols)):
                    return ('up', 0, wc)
            return None

        def apply_rule(grid):
            rows, cols = len(grid), len(grid[0])
            w = find_wall(grid)
            if w is None:
                return None
            direction, wall_idx, wall_color = w
            out = [list(row) for row in grid]
            if direction == 'right':
                for r in range(rows):
                    cells = sorted([(c, grid[r][c]) for c in range(cols-1) if grid[r][c] != 0], key=lambda x: x[0])
                    for i, (c, v) in enumerate(cells):
                        end = cells[i+1][0] if i+1 < len(cells) else cols-1
                        for cc in range(c, end):
                            out[r][cc] = v
            elif direction == 'left':
                for r in range(rows):
                    cells = sorted([(c, grid[r][c]) for c in range(1, cols) if grid[r][c] != 0], key=lambda x: -x[0])
                    for i, (c, v) in enumerate(cells):
                        end = cells[i+1][0] if i+1 < len(cells) else 0
                        for cc in range(end+1, c+1):
                            out[r][cc] = v
            elif direction == 'down':
                for c in range(cols):
                    cells = sorted([(r, grid[r][c]) for r in range(rows-1) if grid[r][c] != 0], key=lambda x: x[0])
                    for i, (r, v) in enumerate(cells):
                        end = cells[i+1][0] if i+1 < len(cells) else rows-1
                        for rr in range(r, end):
                            out[rr][c] = v
            elif direction == 'up':
                for c in range(cols):
                    cells = sorted([(r, grid[r][c]) for r in range(1, rows) if grid[r][c] != 0], key=lambda x: -x[0])
                    for i, (r, v) in enumerate(cells):
                        end = cells[i+1][0] if i+1 < len(cells) else 0
                        for rr in range(end+1, r+1):
                            out[rr][c] = v
            return out
        for ex in train:
            if apply_rule(ex['input']) != ex['output']:
                return None
        return apply_rule(test_input)

    def _try_recolor_8_blobs_by_palette(self, train, test_input):
        """Grid has colored 'palette' blob and multiple 8-blobs with same shape.
        Replace each 8-blob with palette colors (by relative position); clear palette."""
        def get_8_blobs(grid):
            rows, cols = len(grid), len(grid[0])
            visited = set()
            blobs = []
            for r in range(rows):
                for c in range(cols):
                    if grid[r][c] == 8 and (r, c) not in visited:
                        blob = []
                        stack = [(r, c)]
                        while stack:
                            cr, cc = stack.pop()
                            if (cr, cc) in visited or not (0 <= cr < rows and 0 <= cc < cols):
                                continue
                            if grid[cr][cc] != 8:
                                continue
                            visited.add((cr, cc))
                            blob.append((cr, cc))
                            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                                stack.append((cr+dr, cc+dc))
                        blobs.append(blob)
            return blobs

        def apply_rule(grid):
            rows, cols = len(grid), len(grid[0])
            palette_cells = [(r, c, grid[r][c]) for r in range(rows) for c in range(cols)
                             if grid[r][c] != 0 and grid[r][c] != 8]
            if not palette_cells:
                return None
            eight_blobs = get_8_blobs(grid)
            if not eight_blobs:
                return None
            min_r = min(r for r, c, v in palette_cells)
            min_c = min(c for r, c, v in palette_cells)
            palette_shape = {(r - min_r, c - min_c): v for r, c, v in palette_cells}
            out = [list(row) for row in grid]
            for r, c, v in palette_cells:
                out[r][c] = 0
            for blob in eight_blobs:
                br = min(r for r, c in blob)
                bc = min(c for r, c in blob)
                blob_offsets = {(r - br, c - bc) for r, c in blob}
                if blob_offsets != set(palette_shape.keys()):
                    return None
                for r, c in blob:
                    out[r][c] = palette_shape[(r - br, c - bc)]
            return out

        for ex in train:
            if apply_rule(ex['input']) != ex['output']:
                return None
        return apply_rule(test_input)

    def _try_fill_endpoints_with_midpoint5(self, train, test_input):
        """Rows with non-zero values only at col 0 and col -1.
        Fill: left half with left val, middle col with 5, right half with right val."""
        def apply_rule(grid):
            rows, cols = len(grid), len(grid[0])
            if cols % 2 == 0:
                return None  # need odd cols for clear midpoint
            mid = cols // 2
            out = [list(row) for row in grid]
            changed = False
            for r in range(rows):
                row = grid[r]
                left_v = row[0]
                right_v = row[-1]
                if left_v == 0 or right_v == 0:
                    continue
                if any(row[c] != 0 for c in range(1, cols - 1)):
                    continue
                for c in range(cols):
                    if c < mid:
                        out[r][c] = left_v
                    elif c == mid:
                        out[r][c] = 5
                    else:
                        out[r][c] = right_v
                changed = True
            return out if changed else None
        for ex in train:
            if apply_rule(ex['input']) != ex['output']:
                return None
        return apply_rule(test_input)

    def _try_compact_anchor_corners(self, train, test_input):
        """Two non-bg colors, each with a 3x3 anchor block in opposite corners.
        Scattered cells compact toward their anchor:
        - Each anchor col: count all cells of that color in col -> output depth, justified toward corner
        - Non-anchor-col cells: map to extra col (adjacent to anchor) at same row if in anchor rows,
          else at the edge anchor row (last for TL, first for BR)."""
        def find_anchor(grid, color, rows, cols):
            corners = [
                ([0,1,2],[0,1,2],'top','left'),
                ([0,1,2],[cols-3,cols-2,cols-1],'top','right'),
                ([rows-3,rows-2,rows-1],[0,1,2],'bottom','left'),
                ([rows-3,rows-2,rows-1],[cols-3,cols-2,cols-1],'bottom','right'),
            ]
            for ar,ac,vpos,hpos in corners:
                if all(grid[r][c]==color for r in ar for c in ac):
                    return ar,ac,vpos,hpos
            return None,None,None,None

        def apply_rule(grid):
            from collections import Counter
            rows, cols = len(grid), len(grid[0])
            if rows < 3 or cols < 3:
                return None
            flat = [v for row in grid for v in row]
            cnt = Counter(flat)
            bg = cnt.most_common(1)[0][0]
            colors = [v for v,_ in cnt.most_common() if v != bg]
            if len(colors) != 2:
                return None
            out = [[bg]*cols for _ in range(rows)]
            for color in colors:
                ar,ac,vpos,hpos = find_anchor(grid, color, rows, cols)
                if ar is None:
                    return None
                ar_set = set(ar)
                ac_set = set(ac)
                is_top = vpos == 'top'
                is_left = hpos == 'left'
                for c in ac:
                    depth = sum(1 for r in range(rows) if grid[r][c] == color)
                    if is_top:
                        for r in range(depth): out[r][c] = color
                    else:
                        for r in range(rows-depth, rows): out[r][c] = color
                extra_col = (max(ac)+1) if is_left else (min(ac)-1)
                if not (0 <= extra_col < cols):
                    return None
                clamp_row = max(ar) if is_top else min(ar)
                non_anc = [(r,c2) for r in range(rows) for c2 in range(cols)
                           if c2 not in ac_set and grid[r][c2] == color]
                row_map = {}
                for r,c2 in non_anc:
                    tr = r if r in ar_set else clamp_row
                    row_map[tr] = row_map.get(tr, 0) + 1
                for tr in row_map:
                    if 0 <= tr < rows:
                        out[tr][extra_col] = color
            return out

        for ex in train:
            if apply_rule(ex['input']) != ex['output']:
                return None
        return apply_rule(test_input)

    def _try_zone_recolor_by_identity(self, train, test_input):
        """Grid divided by separator rows/cols (all-same non-0 non-1 value) into zones.
        Zones with a single non-0 non-1 color are 'identity' zones.
        Zones with only 1s are 'pattern' zones.
        For each pattern zone, find identity zone in same row-band or col-band -> replace 1s with that color."""
        def apply_rule(grid):
            rows, cols = len(grid), len(grid[0])
            sep_val = None
            for r in range(rows):
                vals = set(grid[r])
                if len(vals) == 1 and list(vals)[0] not in (0, 1):
                    sep_val = list(vals)[0]
                    break
            if sep_val is None:
                return None
            sep_rows = [r for r in range(rows) if all(grid[r][c] == sep_val for c in range(cols))]
            sep_cols = [c for c in range(cols) if all(grid[r][c] == sep_val for r in range(rows))]
            def make_bands(sep_list, total):
                bands = []
                prev = 0
                for s in sep_list + [total]:
                    if s > prev:
                        bands.append((prev, s))
                    prev = s + 1
                return bands
            row_bands = make_bands(sep_rows, rows)
            col_bands = make_bands(sep_cols, cols)
            identity = {}
            patterns = {}
            for ri, (r0, r1) in enumerate(row_bands):
                for ci, (c0, c1) in enumerate(col_bands):
                    cells = [(grid[r][c], r, c) for r in range(r0, r1) for c in range(c0, c1)]
                    non_zero = [(v, r, c) for v, r, c in cells if v != 0 and v != sep_val]
                    ones = [(r, c) for v, r, c in non_zero if v == 1]
                    non_ones = [(v, r, c) for v, r, c in non_zero if v != 1]
                    if non_ones and not ones and len(set(v for v, r, c in non_ones)) == 1:
                        identity[(ri, ci)] = non_ones[0][0]
                    elif ones and not non_ones:
                        patterns[(ri, ci)] = ones
                    elif ones and non_ones:
                        return None
            out = [list(row) for row in grid]
            for (pri, pci), patt_cells in patterns.items():
                matches = [(ri, ci) for (ri, ci) in identity if ri == pri or ci == pci]
                if len(matches) != 1:
                    return None
                color = identity[matches[0]]
                for r, c in patt_cells:
                    out[r][c] = color
            return out
        for ex in train:
            if apply_rule(ex['input']) != ex['output']:
                return None
        return apply_rule(test_input)

    def _try_gravity_toward_separator(self, train, test_input):
        """Single separator row (all same non-0 non-1 value). Two non-bg, non-sep values.
        Value A (e.g. 2) falls toward the separator (creating a trail from position to sep).
        Value B (e.g. 1) falls away from the separator (creating a trail from position to edge)."""
        def apply_rule(grid):
            rows, cols = len(grid), len(grid[0])
            sep_rows = [r for r in range(rows)
                        if len(set(grid[r])) == 1 and list(set(grid[r]))[0] not in (0, 1)]
            if len(sep_rows) != 1:
                return None
            sep = sep_rows[0]
            sep_val = grid[sep][0]
            # Find exactly 2 non-bg non-sep values
            from collections import Counter
            flat = [v for row in grid for v in row if v != sep_val]
            cnt = Counter(flat)
            bg = cnt.most_common(1)[0][0]
            others = [v for v, _ in cnt.most_common() if v != bg]
            if len(others) != 2:
                return None
            # Determine which falls toward sep vs away
            # Score: toward_sep means cells appear between position and sep in output
            # Try both assignments
            def simulate(toward_val, away_val):
                out = [[sep_val if r == sep else bg for c in range(cols)] for r in range(rows)]
                for r in range(rows):
                    if r == sep:
                        continue
                    in_top = r < sep
                    for c in range(cols):
                        v = grid[r][c]
                        if v == toward_val:
                            if in_top:
                                for rr in range(r, sep): out[rr][c] = toward_val
                            else:
                                for rr in range(sep + 1, r + 1): out[rr][c] = toward_val
                        elif v == away_val:
                            if in_top:
                                for rr in range(0, r + 1): out[rr][c] = away_val
                            else:
                                for rr in range(r, rows): out[rr][c] = away_val
                return out
            v1, v2 = others[0], others[1]
            r1 = simulate(v1, v2)
            r2 = simulate(v2, v1)
            # Pick whichever matches training (will be cross-validated)
            return r1 if r1 is not None else r2
        # Check which assignment works for all training examples
        def apply_with_assignment(grid, toward_val, away_val):
            rows, cols = len(grid), len(grid[0])
            sep_rows = [r for r in range(rows)
                        if len(set(grid[r])) == 1 and list(set(grid[r]))[0] not in (0, 1)]
            if len(sep_rows) != 1:
                return None
            sep = sep_rows[0]
            sep_val = grid[sep][0]
            from collections import Counter
            flat = [v for row in grid for v in row if v != sep_val]
            cnt = Counter(flat)
            bg = cnt.most_common(1)[0][0]
            out = [[sep_val if r == sep else bg for c in range(cols)] for r in range(rows)]
            for r in range(rows):
                if r == sep:
                    continue
                in_top = r < sep
                for c in range(cols):
                    v = grid[r][c]
                    if v == toward_val:
                        if in_top:
                            for rr in range(r, sep): out[rr][c] = toward_val
                        else:
                            for rr in range(sep + 1, r + 1): out[rr][c] = toward_val
                    elif v == away_val:
                        if in_top:
                            for rr in range(0, r + 1): out[rr][c] = away_val
                        else:
                            for rr in range(r, rows): out[rr][c] = away_val
            return out
        # Detect toward/away from first training example
        grid0 = train[0]['input']
        from collections import Counter
        sep_rows_0 = [r for r in range(len(grid0))
                      if len(set(grid0[r])) == 1 and list(set(grid0[r]))[0] not in (0, 1)]
        if len(sep_rows_0) != 1:
            return None
        sep_val_0 = grid0[sep_rows_0[0]][0]
        flat0 = [v for row in grid0 for v in row if v != sep_val_0]
        cnt0 = Counter(flat0)
        bg0 = cnt0.most_common(1)[0][0]
        others0 = [v for v, _ in cnt0.most_common() if v != bg0]
        if len(others0) != 2:
            return None
        v1, v2 = others0[0], others0[1]
        for toward_val, away_val in [(v1, v2), (v2, v1)]:
            ok = True
            for ex in train:
                if apply_with_assignment(ex['input'], toward_val, away_val) != ex['output']:
                    ok = False
                    break
            if ok:
                return apply_with_assignment(test_input, toward_val, away_val)
        return None

    def _try_radiating_line_78(self, train, test_input):
        """A contiguous line of 7s (vertical or horizontal). Output: radiating pattern
        where at each position along the line, a spread of alternating 7/8 fans out
        perpendicularly. Spread = (line_length - 1 - index_from_start)."""
        def apply_rule(grid):
            rows, cols = len(grid), len(grid[0])
            non_zero = [(grid[r][c], r, c) for r in range(rows) for c in range(cols) if grid[r][c] != 0]
            vals = set(v for v,r,c in non_zero)
            if vals != {7}:
                return None
            seven_cells = [(r,c) for v,r,c in non_zero]
            if not seven_cells:
                return None
            rs = sorted(set(r for r,c in seven_cells))
            cs = sorted(set(c for r,c in seven_cells))
            out = [[0]*cols for _ in range(rows)]
            # Try vertical line
            if len(cs) == 1:
                c = cs[0]
                if sorted(r for r,c2 in seven_cells) != list(range(min(rs), max(rs)+1)):
                    return None
                top_r, bot_r = min(rs), max(rs)
                length = bot_r - top_r + 1
                for idx, r in enumerate(range(top_r, bot_r+1)):
                    spread = length - 1 - idx
                    for c2 in range(cols):
                        offset = c2 - c
                        if abs(offset) <= spread:
                            out[r][c2] = 7 if offset % 2 == 0 else 8
                return out
            # Try horizontal line
            if len(rs) == 1:
                r = rs[0]
                if sorted(c for r2,c in seven_cells) != list(range(min(cs), max(cs)+1)):
                    return None
                left_c, right_c = min(cs), max(cs)
                length = right_c - left_c + 1
                for idx, c in enumerate(range(left_c, right_c+1)):
                    spread = length - 1 - idx
                    for r2 in range(rows):
                        offset = r2 - r
                        if abs(offset) <= spread:
                            out[r2][c] = 7 if offset % 2 == 0 else 8
                return out
            return None
        for ex in train:
            if apply_rule(ex['input']) != ex['output']:
                return None
        return apply_rule(test_input)

    def _try_stamp_pattern_on_5_regions(self, train, test_input):
        """Grid has a non-zero/non-5 pattern block and one or more rectangular 5-regions.
        Copy the pattern (aligned by bounding box) into each 5-region. 5-regions same size as pattern."""
        def get_bounding_box(cells):
            rs = [r for r,c in cells]
            cs = [c for r,c in cells]
            return min(rs), max(rs), min(cs), max(cs)

        def apply_rule(grid):
            rows, cols = len(grid), len(grid[0])
            # Find pattern cells (non-0, non-5)
            pattern_cells = [(r, c) for r in range(rows) for c in range(cols)
                             if grid[r][c] != 0 and grid[r][c] != 5]
            # Find 5-regions
            five_cells = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == 5]
            if not pattern_cells or not five_cells:
                return None
            # Get pattern bounding box
            pr0, pr1, pc0, pc1 = get_bounding_box(pattern_cells)
            ph, pw = pr1 - pr0 + 1, pc1 - pc0 + 1
            # Build pattern relative values (including zeros inside bounding box)
            pat = {}
            for dr in range(ph):
                for dc in range(pw):
                    pat[(dr, dc)] = grid[pr0 + dr][pc0 + dc]
            # Find 5-rectangles (connected rectangular groups of 5s with same size as pattern)
            visited = set()
            five_rects = []
            for r0, c0 in five_cells:
                if (r0, c0) in visited:
                    continue
                # Try to form rectangle starting here
                r1 = r0
                while r1 + 1 < rows and grid[r1 + 1][c0] == 5:
                    r1 += 1
                c1 = c0
                while c1 + 1 < cols and grid[r0][c1 + 1] == 5:
                    c1 += 1
                h, w = r1 - r0 + 1, c1 - c0 + 1
                if h != ph or w != pw:
                    return None
                # Verify all cells in rect are 5
                if not all(grid[r][c] == 5 for r in range(r0, r1+1) for c in range(c0, c1+1)):
                    return None
                for r in range(r0, r1+1):
                    for c in range(c0, c1+1):
                        visited.add((r, c))
                five_rects.append((r0, c0))
            if not five_rects:
                return None
            out = [list(row) for row in grid]
            for sr, sc in five_rects:
                for dr in range(ph):
                    for dc in range(pw):
                        out[sr + dr][sc + dc] = pat[(dr, dc)]
            return out
        for ex in train:
            if apply_rule(ex['input']) != ex['output']:
                return None
        return apply_rule(test_input)

    def _try_key_grid_recolor(self, train, test_input):
        """Grid has a K×L key (non-0 non-1 values) and 1s arranged in a K×L spatial grid.
        Each spatial cell (i,j) of the 1-grid gets recolored with key_mat[i][j].
        1s in same row-band i and col-band j form one instance."""
        def apply_rule(grid):
            rows, cols = len(grid), len(grid[0])
            key_cells = [(r, c, grid[r][c]) for r in range(rows) for c in range(cols)
                         if grid[r][c] not in (0, 1)]
            if not key_cells:
                return None
            kr0 = min(r for r, c, v in key_cells)
            kr1 = max(r for r, c, v in key_cells)
            kc0 = min(c for r, c, v in key_cells)
            kc1 = max(c for r, c, v in key_cells)
            kh, kw = kr1 - kr0 + 1, kc1 - kc0 + 1
            key_mat = [[grid[kr0 + dr][kc0 + dc] for dc in range(kw)] for dr in range(kh)]
            one_rows = sorted(set(r for r in range(rows) for c in range(cols) if grid[r][c] == 1))
            one_cols = sorted(set(c for r in range(rows) for c in range(cols) if grid[r][c] == 1))
            def group_consecutive(indices):
                if not indices:
                    return []
                groups = [[indices[0]]]
                for idx in indices[1:]:
                    if idx == groups[-1][-1] + 1:
                        groups[-1].append(idx)
                    else:
                        groups.append([idx])
                return groups
            row_groups = group_consecutive(one_rows)
            col_groups = group_consecutive(one_cols)
            if len(row_groups) != kh or len(col_groups) != kw:
                return None
            row_group_of = {}
            for i, rg in enumerate(row_groups):
                for r in rg:
                    row_group_of[r] = i
            col_group_of = {}
            for j, cg in enumerate(col_groups):
                for c in cg:
                    col_group_of[c] = j
            out = [list(row) for row in grid]
            for r in range(rows):
                for c in range(cols):
                    if grid[r][c] == 1:
                        i = row_group_of.get(r)
                        j = col_group_of.get(c)
                        if i is None or j is None:
                            return None
                        out[r][c] = key_mat[i][j]
            return out
        for ex in train:
            if apply_rule(ex['input']) != ex['output']:
                return None
        return apply_rule(test_input)

    def _try_complete_symmetric_pattern(self, train, test_input):
        """Grid has two non-bg colors: one forms a solid rectangular block,
        the other is a scattered pattern with left-right symmetry.
        The block covers part of the pattern. Remove the block and complete
        the symmetric pattern by reflecting visible cells across the detected axis."""
        def apply_rule(grid):
            from collections import Counter
            rows, cols = len(grid), len(grid[0])
            flat = [v for row in grid for v in row]
            cnt = Counter(flat)
            bg = cnt.most_common(1)[0][0]
            colors = [v for v, _ in cnt.most_common() if v != bg]
            if len(colors) != 2:
                return None
            block_val = pat_val = None
            br0 = br1 = bc0 = bc1 = None
            for cv in colors:
                cells = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == cv]
                r0 = min(r for r, c in cells)
                r1 = max(r for r, c in cells)
                c0 = min(c for r, c in cells)
                c1 = max(c for r, c in cells)
                if all(grid[r][c] == cv for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)):
                    block_val = cv
                    br0, br1, bc0, bc1 = r0, r1, c0, c1
                    pat_val = [v for v in colors if v != cv][0]
                    break
            if block_val is None:
                return None
            block_rows = set(range(br0, br1 + 1))
            # Find axis from non-block rows with pattern
            axis_vals = []
            for r in range(rows):
                if r in block_rows:
                    continue
                pcols = [c for c in range(cols) if grid[r][c] == pat_val]
                if not pcols:
                    continue
                axis_vals.append(sum(pcols) / len(pcols))
            if not axis_vals:
                return None
            axis = sum(axis_vals) / len(axis_vals)
            axis = round(axis * 2) / 2
            out = [list(row) for row in grid]
            for r in range(br0, br1 + 1):
                for c in range(bc0, bc1 + 1):
                    rc = round(2 * axis - c)
                    if 0 <= rc < cols and grid[r][rc] == pat_val:
                        out[r][c] = pat_val
                    else:
                        out[r][c] = bg
            return out
        for ex in train:
            if apply_rule(ex['input']) != ex['output']:
                return None
        return apply_rule(test_input)
    def _try_nested_rect_concentric_fill(self, train, test_input):
        """Concentric ring coloring for solid rectangles: learn ring-depth → color
        mapping from training. Each ring layer (border=0, next=1, ...) gets a
        specific color. Generalizes _try_rect_concentric_rings to learned colors.
        Handles tasks like 694f12f3."""
        from collections import deque, Counter

        def find_solid_rects(grid, bg):
            """Find all solid-color rectangular components != bg."""
            rows, cols = len(grid), len(grid[0])
            visited = [[False] * cols for _ in range(rows)]
            rects = []
            for sr in range(rows):
                for sc in range(cols):
                    if grid[sr][sc] == bg or visited[sr][sc]:
                        continue
                    color = grid[sr][sc]
                    q = deque([(sr, sc)])
                    comp = []
                    while q:
                        r, c = q.popleft()
                        if visited[r][c]:
                            continue
                        visited[r][c] = True
                        comp.append((r, c))
                        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and grid[nr][nc] == color:
                                q.append((nr, nc))
                    r0 = min(r for r, _ in comp)
                    r1 = max(r for r, _ in comp)
                    c0 = min(c for _, c in comp)
                    c1 = max(c for _, c in comp)
                    if len(comp) == (r1 - r0 + 1) * (c1 - c0 + 1):
                        rects.append((color, r0, r1, c0, c1))
            return rects

        def ring_depth(r, c, r0, r1, c0, c1):
            """Chebyshev distance from border of rectangle."""
            return min(r - r0, r1 - r, c - c0, c1 - c)

        def get_bg(grid):
            return Counter(v for row in grid for v in row).most_common(1)[0][0]

        # Learn ring-depth → color mapping from training
        ring_map = None  # {depth: color}
        for ex in train:
            inp, out = ex["input"], ex["output"]
            if len(inp) != len(out) or len(inp[0]) != len(out[0]):
                return None
            bg = get_bg(inp)
            rects = find_solid_rects(inp, bg)
            if not rects:
                return None
            local_map = {}
            for color, r0, r1, c0, c1 in rects:
                max_d = min(r1 - r0, c1 - c0) // 2
                for r in range(r0, r1 + 1):
                    for c in range(c0, c1 + 1):
                        d = ring_depth(r, c, r0, r1, c0, c1)
                        out_color = out[r][c]
                        if d in local_map:
                            if local_map[d] != out_color:
                                return None
                        else:
                            local_map[d] = out_color
            if ring_map is None:
                ring_map = local_map
            else:
                for d, col in local_map.items():
                    if d in ring_map and ring_map[d] != col:
                        return None
                    ring_map[d] = col

        if not ring_map or len(ring_map) < 2:
            return None

        def apply_rule(grid):
            bg = get_bg(grid)
            rows, cols = len(grid), len(grid[0])
            out = [list(row) for row in grid]
            rects = find_solid_rects(grid, bg)
            for color, r0, r1, c0, c1 in rects:
                for r in range(r0, r1 + 1):
                    for c in range(c0, c1 + 1):
                        d = ring_depth(r, c, r0, r1, c0, c1)
                        if d in ring_map:
                            out[r][c] = ring_map[d]
                        else:
                            # Extrapolate: use max known depth color
                            max_known = max(ring_map.keys())
                            out[r][c] = ring_map[max_known]
            return out

        for ex in train:
            if apply_rule(ex["input"]) != ex["output"]:
                return None
        result = apply_rule(test_input)
        if result == [list(row) for row in test_input]:
            return None
        return result


# ─── STRATEGY 2: Object Projection to Border ──────────────────────────────
# Pattern: Non-bg cells project their color to the grid border in all 4
# cardinal directions (ray-casting to edges).

    def _try_project_color_to_border(self, train, test_input):
        """Non-background colored cells project their color outward to the grid
        borders along rows and columns (like shadows/rays). The original cell
        stays. Handles tasks like 689c358e."""
        from collections import Counter

        def get_bg(grid):
            return Counter(v for row in grid for v in row).most_common(1)[0][0]

        def get_nonbg_cells(grid, bg):
            cells = []
            for r in range(len(grid)):
                for c in range(len(grid[0])):
                    if grid[r][c] != bg:
                        cells.append((r, c, grid[r][c]))
            return cells

        # Learn which directions to project: up, down, left, right
        # Try all 16 subsets of {up, down, left, right} to find which matches
        directions_all = [
            ("up", -1, 0), ("down", 1, 0), ("left", 0, -1), ("right", 0, 1)
        ]

        def try_projection(grid, bg, dir_set, overwrite_nonbg=False):
            rows, cols = len(grid), len(grid[0])
            out = [list(row) for row in grid]
            cells = get_nonbg_cells(grid, bg)
            for r, c, color in cells:
                for name, dr, dc in dir_set:
                    nr, nc = r + dr, c + dc
                    while 0 <= nr < rows and 0 <= nc < cols:
                        if out[nr][nc] == bg or overwrite_nonbg:
                            out[nr][nc] = color
                        elif out[nr][nc] != color:
                            break  # stop at other non-bg
                        nr += dr
                        nc += dc
            return out

        best_dirs = None
        best_overwrite = None
        for mask in range(1, 16):
            dirs = [directions_all[i] for i in range(4) if mask & (1 << i)]
            for overwrite in [False, True]:
                match = True
                for ex in train:
                    bg = get_bg(ex["input"])
                    if len(ex["input"]) != len(ex["output"]) or len(ex["input"][0]) != len(ex["output"][0]):
                        match = False
                        break
                    if try_projection(ex["input"], bg, dirs, overwrite) != ex["output"]:
                        match = False
                        break
                if match:
                    best_dirs = dirs
                    best_overwrite = overwrite
                    break
            if best_dirs is not None:
                break

        if best_dirs is None:
            return None

        bg = get_bg(test_input)
        result = try_projection(test_input, bg, best_dirs, best_overwrite)
        if result == [list(row) for row in test_input]:
            return None
        return result


# ─── STRATEGY 3: Flood Fill Between Same-Color Pairs ──────────────────────
# Pattern: Two cells of the same non-bg color in the same row or column.
# Fill the gap between them with a learned fill color.

    def _try_flood_fill_between_pairs(self, train, test_input):
        """Find pairs of same-color cells aligned on same row or column.
        Fill the gap between them with a fill color learned from training.
        Different from _try_connect_same_rowcol_pairs: learns per-color fill mapping
        and handles diagonal pairs and multiple colors."""
        from collections import Counter

        def get_bg(grid):
            return Counter(v for row in grid for v in row).most_common(1)[0][0]

        def find_aligned_pairs(grid, bg):
            """Find all pairs of same non-bg color on same row or col."""
            rows, cols = len(grid), len(grid[0])
            pairs = []
            # Row pairs
            for r in range(rows):
                by_color = {}
                for c in range(cols):
                    v = grid[r][c]
                    if v != bg:
                        by_color.setdefault(v, []).append(c)
                for color, positions in by_color.items():
                    positions.sort()
                    for i in range(len(positions) - 1):
                        c1, c2 = positions[i], positions[i + 1]
                        # Check gap is all bg
                        if all(grid[r][c] == bg for c in range(c1 + 1, c2)):
                            pairs.append(("row", r, c1, c2, color))
            # Col pairs
            for c in range(cols):
                by_color = {}
                for r in range(rows):
                    v = grid[r][c]
                    if v != bg:
                        by_color.setdefault(v, []).append(r)
                for color, positions in by_color.items():
                    positions.sort()
                    for i in range(len(positions) - 1):
                        r1, r2 = positions[i], positions[i + 1]
                        if all(grid[r][c] == bg for r in range(r1 + 1, r2)):
                            pairs.append(("col", c, r1, r2, color))
            return pairs

        # Learn fill_color from training
        fill_color = None
        for ex in train:
            inp, out = ex["input"], ex["output"]
            if len(inp) != len(out) or len(inp[0]) != len(out[0]):
                return None
            bg = get_bg(inp)
            rows, cols = len(inp), len(inp[0])
            for r in range(rows):
                for c in range(cols):
                    if inp[r][c] == bg and out[r][c] != bg:
                        fc = out[r][c]
                        if fill_color is None:
                            fill_color = fc
                        elif fill_color != fc:
                            # Maybe fill color = same as pair color
                            fill_color = "same"

        if fill_color is None:
            return None

        def apply_rule(grid, fill_c):
            bg = get_bg(grid)
            rows, cols = len(grid), len(grid[0])
            out = [list(row) for row in grid]
            pairs = find_aligned_pairs(grid, bg)
            for p in pairs:
                if p[0] == "row":
                    _, r, c1, c2, color = p
                    fc = color if fill_c == "same" else fill_c
                    for c in range(c1 + 1, c2):
                        out[r][c] = fc
                else:
                    _, c, r1, r2, color = p
                    fc = color if fill_c == "same" else fill_c
                    for r in range(r1 + 1, r2):
                        out[r][c] = fc
            return out

        for ex in train:
            if apply_rule(ex["input"], fill_color) != ex["output"]:
                return None

        result = apply_rule(test_input, fill_color)
        if result == [list(row) for row in test_input]:
            return None
        return result


# ─── STRATEGY 4: Border Extraction (keep only border, erase interior) ─────
# Pattern: Extract border pixels of each connected component, remove interiors.

    def _try_border_extraction(self, train, test_input):
        """Extract only the border pixels of each non-bg connected component.
        Interior cells (all 4-neighbors are same color) become background.
        Inverse of fill operations."""
        from collections import Counter, deque

        def get_bg(grid):
            return Counter(v for row in grid for v in row).most_common(1)[0][0]

        def extract_borders(grid, bg):
            rows, cols = len(grid), len(grid[0])
            out = [list(row) for row in grid]
            for r in range(rows):
                for c in range(cols):
                    if grid[r][c] == bg:
                        continue
                    color = grid[r][c]
                    # Check if all 4-neighbors are same color (interior)
                    is_interior = True
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = r + dr, c + dc
                        if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                            is_interior = False
                            break
                        if grid[nr][nc] != color:
                            is_interior = False
                            break
                    if is_interior:
                        out[r][c] = bg
            return out

        for ex in train:
            if len(ex["input"]) != len(ex["output"]) or len(ex["input"][0]) != len(ex["output"][0]):
                return None
            bg = get_bg(ex["input"])
            if extract_borders(ex["input"], bg) != ex["output"]:
                return None
            # Make sure it actually changes something
            if [list(row) for row in ex["input"]] == ex["output"]:
                return None

        bg = get_bg(test_input)
        result = extract_borders(test_input, bg)
        if result == [list(row) for row in test_input]:
            return None
        return result


# ─── STRATEGY 5: Majority Color Per Row/Column ────────────────────────────
# Pattern: Each row (or column) gets uniformly filled with its majority
# non-background color.

    def _try_majority_color_per_row_col(self, train, test_input):
        """Each row or column becomes uniformly filled with its majority
        non-background color. Learns whether it's row-wise or column-wise
        from training."""
        from collections import Counter

        def get_bg(grid):
            return Counter(v for row in grid for v in row).most_common(1)[0][0]

        def fill_by_majority_row(grid, bg):
            rows, cols = len(grid), len(grid[0])
            out = [list(row) for row in grid]
            for r in range(rows):
                non_bg = [grid[r][c] for c in range(cols) if grid[r][c] != bg]
                if not non_bg:
                    continue
                majority = Counter(non_bg).most_common(1)[0][0]
                for c in range(cols):
                    if grid[r][c] != bg:
                        out[r][c] = majority
            return out

        def fill_by_majority_col(grid, bg):
            rows, cols = len(grid), len(grid[0])
            out = [list(row) for row in grid]
            for c in range(cols):
                non_bg = [grid[r][c] for r in range(rows) if grid[r][c] != bg]
                if not non_bg:
                    continue
                majority = Counter(non_bg).most_common(1)[0][0]
                for r in range(rows):
                    if grid[r][c] != bg:
                        out[r][c] = majority
            return out

        def fill_all_row(grid, bg):
            """Fill entire row with majority non-bg color (including bg cells)."""
            rows, cols = len(grid), len(grid[0])
            out = [list(row) for row in grid]
            for r in range(rows):
                non_bg = [grid[r][c] for c in range(cols) if grid[r][c] != bg]
                if not non_bg:
                    continue
                majority = Counter(non_bg).most_common(1)[0][0]
                for c in range(cols):
                    out[r][c] = majority
            return out

        def fill_all_col(grid, bg):
            """Fill entire column with majority non-bg color (including bg cells)."""
            rows, cols = len(grid), len(grid[0])
            out = [list(row) for row in grid]
            for c in range(cols):
                non_bg = [grid[r][c] for r in range(rows) if grid[r][c] != bg]
                if not non_bg:
                    continue
                majority = Counter(non_bg).most_common(1)[0][0]
                for r in range(rows):
                    out[r][c] = majority
            return out

        # Try all 4 variants
        for apply_fn in [fill_by_majority_row, fill_by_majority_col,
                         fill_all_row, fill_all_col]:
            match = True
            for ex in train:
                if len(ex["input"]) != len(ex["output"]) or len(ex["input"][0]) != len(ex["output"][0]):
                    match = False
                    break
                bg = get_bg(ex["input"])
                if apply_fn(ex["input"], bg) != ex["output"]:
                    match = False
                    break
                if [list(row) for row in ex["input"]] == ex["output"]:
                    match = False
                    break
            if match:
                bg = get_bg(test_input)
                result = apply_fn(test_input, bg)
                if result != [list(row) for row in test_input]:
                    return result
        return None


# ─── STRATEGY 6: Object Bounding Box Solid Fill ──────────────────────────
# Pattern: Each connected non-bg object gets its bounding box completely
# filled with the object's own color (eliminating holes/irregular shapes).
# Different from _try_fill_shape_bounding_box which fills bg cells with a
# DIFFERENT fill color.

    def _try_object_bbox_self_fill(self, train, test_input):
        """Each connected non-bg component gets its bounding box filled entirely
        with the component's own color, converting irregular shapes into solid
        rectangles. Unlike _try_fill_shape_bounding_box, uses same color as object."""
        from collections import Counter, deque

        def get_bg(grid):
            return Counter(v for row in grid for v in row).most_common(1)[0][0]

        def apply_rule(grid, bg):
            rows, cols = len(grid), len(grid[0])
            out = [list(row) for row in grid]
            visited = [[False] * cols for _ in range(rows)]
            for sr in range(rows):
                for sc in range(cols):
                    if grid[sr][sc] == bg or visited[sr][sc]:
                        continue
                    color = grid[sr][sc]
                    q = deque([(sr, sc)])
                    comp = []
                    while q:
                        r, c = q.popleft()
                        if visited[r][c]:
                            continue
                        visited[r][c] = True
                        comp.append((r, c))
                        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and grid[nr][nc] == color:
                                q.append((nr, nc))
                    r0 = min(r for r, _ in comp)
                    r1 = max(r for r, _ in comp)
                    c0 = min(c for _, c in comp)
                    c1 = max(c for _, c in comp)
                    for r in range(r0, r1 + 1):
                        for c in range(c0, c1 + 1):
                            out[r][c] = color
            return out

        for ex in train:
            if len(ex["input"]) != len(ex["output"]) or len(ex["input"][0]) != len(ex["output"][0]):
                return None
            bg = get_bg(ex["input"])
            if apply_rule(ex["input"], bg) != ex["output"]:
                return None
            if [list(row) for row in ex["input"]] == ex["output"]:
                return None

        bg = get_bg(test_input)
        result = apply_rule(test_input, bg)
        if result == [list(row) for row in test_input]:
            return None
        return result


# ─── STRATEGY 7: Mirror/Reflect Objects Across Grid Axis ──────────────────
# Pattern: Non-bg objects are reflected across the horizontal or vertical
# midline of the grid. The original stays and the reflection is added.

    def _try_reflect_objects_across_axis(self, train, test_input):
        """Non-bg objects are reflected/mirrored across the grid's horizontal
        or vertical midline. Both original and reflection are present in output.
        Handles horizontal, vertical, and both-axis reflections."""
        from collections import Counter

        def get_bg(grid):
            return Counter(v for row in grid for v in row).most_common(1)[0][0]

        def reflect_h(grid, bg):
            """Reflect non-bg cells across horizontal midline (top-bottom)."""
            rows, cols = len(grid), len(grid[0])
            out = [list(row) for row in grid]
            for r in range(rows):
                for c in range(cols):
                    if grid[r][c] != bg:
                        mr = rows - 1 - r
                        if out[mr][c] == bg:
                            out[mr][c] = grid[r][c]
            return out

        def reflect_v(grid, bg):
            """Reflect non-bg cells across vertical midline (left-right)."""
            rows, cols = len(grid), len(grid[0])
            out = [list(row) for row in grid]
            for r in range(rows):
                for c in range(cols):
                    if grid[r][c] != bg:
                        mc = cols - 1 - c
                        if out[r][mc] == bg:
                            out[r][mc] = grid[r][c]
            return out

        def reflect_both(grid, bg):
            """Reflect across both axes."""
            rows, cols = len(grid), len(grid[0])
            out = [list(row) for row in grid]
            for r in range(rows):
                for c in range(cols):
                    if grid[r][c] != bg:
                        v = grid[r][c]
                        mr, mc = rows - 1 - r, cols - 1 - c
                        if out[mr][c] == bg:
                            out[mr][c] = v
                        if out[r][mc] == bg:
                            out[r][mc] = v
                        if out[mr][mc] == bg:
                            out[mr][mc] = v
            return out

        def reflect_h_overwrite(grid, bg):
            """Reflect non-bg cells across horizontal midline (overwrite)."""
            rows, cols = len(grid), len(grid[0])
            out = [list(row) for row in grid]
            for r in range(rows):
                for c in range(cols):
                    if grid[r][c] != bg:
                        mr = rows - 1 - r
                        out[mr][c] = grid[r][c]
            return out

        def reflect_v_overwrite(grid, bg):
            """Reflect non-bg cells across vertical midline (overwrite)."""
            rows, cols = len(grid), len(grid[0])
            out = [list(row) for row in grid]
            for r in range(rows):
                for c in range(cols):
                    if grid[r][c] != bg:
                        mc = cols - 1 - c
                        out[r][mc] = grid[r][c]
            return out

        for apply_fn in [reflect_h, reflect_v, reflect_both,
                         reflect_h_overwrite, reflect_v_overwrite]:
            match = True
            for ex in train:
                if len(ex["input"]) != len(ex["output"]) or len(ex["input"][0]) != len(ex["output"][0]):
                    match = False
                    break
                bg = get_bg(ex["input"])
                if apply_fn(ex["input"], bg) != ex["output"]:
                    match = False
                    break
                if [list(row) for row in ex["input"]] == ex["output"]:
                    match = False
                    break
            if match:
                bg = get_bg(test_input)
                result = apply_fn(test_input, bg)
                if result != [list(row) for row in test_input]:
                    return result
        return None


# ─── STRATEGY 8: Color Region Sorting (scatter → organized regions) ───────
# Pattern: Grid has scattered colored cells. Output organizes them by
# sorting colors into regions — e.g., by distance from center, by quadrant,
# or by row/column position.

    def _try_sort_colors_to_regions(self, train, test_input):
        """Scattered colored cells get sorted/organized: each color occupies a
        contiguous band (rows or columns) in the output. The bands are ordered
        by the color's average position in the input.
        Handles tasks like 5751f35e."""
        from collections import Counter

        def get_bg(grid):
            return Counter(v for row in grid for v in row).most_common(1)[0][0]

        def sort_by_row_band(grid, bg):
            """Each non-bg color occupies contiguous rows, ordered by avg row."""
            rows, cols = len(grid), len(grid[0])
            # Collect color → positions
            color_pos = {}
            for r in range(rows):
                for c in range(cols):
                    v = grid[r][c]
                    if v != bg:
                        color_pos.setdefault(v, []).append((r, c))
            if not color_pos:
                return None

            # Sort colors by average row position
            color_avg = [(sum(r for r, c in pos) / len(pos), color)
                         for color, pos in color_pos.items()]
            color_avg.sort()

            # Count cells per color
            total_cells = sum(len(pos) for pos in color_pos.values())
            if total_cells != rows * cols - sum(1 for r in range(rows) for c in range(cols) if grid[r][c] == bg):
                pass  # OK, just non-bg cells

            # Assign rows proportionally
            out = [[bg] * cols for _ in range(rows)]
            row_idx = 0
            for _, color in color_avg:
                count = len(color_pos[color])
                rows_needed = max(1, round(count / cols))
                for r in range(row_idx, min(row_idx + rows_needed, rows)):
                    for c in range(cols):
                        out[r][c] = color
                row_idx += rows_needed
            return out

        def sort_by_col_band(grid, bg):
            """Each non-bg color occupies contiguous columns, ordered by avg col."""
            rows, cols = len(grid), len(grid[0])
            color_pos = {}
            for r in range(rows):
                for c in range(cols):
                    v = grid[r][c]
                    if v != bg:
                        color_pos.setdefault(v, []).append((r, c))
            if not color_pos:
                return None

            color_avg = [(sum(c for r, c in pos) / len(pos), color)
                         for color, pos in color_pos.items()]
            color_avg.sort()

            out = [[bg] * cols for _ in range(rows)]
            col_idx = 0
            for _, color in color_avg:
                count = len(color_pos[color])
                cols_needed = max(1, round(count / rows))
                for c in range(col_idx, min(col_idx + cols_needed, cols)):
                    for r in range(rows):
                        out[r][c] = color
                col_idx += cols_needed
            return out

        def sort_concentric(grid, bg):
            """Colors arranged in concentric rectangles from outside in,
            ordered by average distance from center."""
            rows, cols = len(grid), len(grid[0])
            cr, cc = (rows - 1) / 2.0, (cols - 1) / 2.0

            color_pos = {}
            for r in range(rows):
                for c in range(cols):
                    v = grid[r][c]
                    if v != bg:
                        color_pos.setdefault(v, []).append((r, c))
            if not color_pos:
                return None

            # Sort by avg chebyshev distance from center (outermost first)
            def avg_dist(positions):
                return sum(max(abs(r - cr), abs(c - cc)) for r, c in positions) / len(positions)

            color_dist = [(avg_dist(pos), color) for color, pos in color_pos.items()]
            color_dist.sort(reverse=True)  # outermost first
            colors_ordered = [c for _, c in color_dist]

            out = [[bg] * cols for _ in range(rows)]
            max_rings = min(rows, cols) // 2 + 1
            ring = 0
            ci = 0
            for d in range(max_rings):
                if ci >= len(colors_ordered):
                    break
                color = colors_ordered[ci]
                for r in range(rows):
                    for c in range(cols):
                        depth = min(r, rows - 1 - r, c, cols - 1 - c)
                        if depth == d:
                            out[r][c] = color
                ci += 1
            # Fill remaining interior
            for r in range(rows):
                for c in range(cols):
                    if out[r][c] == bg and ci > 0:
                        out[r][c] = colors_ordered[-1]
            return out

        for apply_fn in [sort_by_row_band, sort_by_col_band, sort_concentric]:
            match = True
            for ex in train:
                if len(ex["input"]) != len(ex["output"]) or len(ex["input"][0]) != len(ex["output"][0]):
                    match = False
                    break
                bg = get_bg(ex["input"])
                result = apply_fn(ex["input"], bg)
                if result is None or result != ex["output"]:
                    match = False
                    break
                if [list(row) for row in ex["input"]] == ex["output"]:
                    match = False
                    break
            if match:
                bg = get_bg(test_input)
                result = apply_fn(test_input, bg)
                if result is not None and result != [list(row) for row in test_input]:
                    return result
        return None

    # --- _try_separator_grid_dimensions (1190e5a7) ---
    def _try_separator_grid_dimensions(self, train, test_input):
        """Grid divided by separator rows/cols. Output = cell-grid dims filled with bg."""
        def solve(grid):
            rows, cols = len(grid), len(grid[0])
            sep_color = None
            for color in set(v for row in grid for v in row):
                full_rows = [r for r in range(rows) if all(grid[r][c] == color for c in range(cols))]
                full_cols = [c for c in range(cols) if all(grid[r][c] == color for r in range(rows))]
                if full_rows and full_cols:
                    sep_color = color
                    break
            if sep_color is None:
                return None
            full_rows = [r for r in range(rows) if all(grid[r][c] == sep_color for c in range(cols))]
            full_cols = [c for c in range(cols) if all(grid[r][c] == sep_color for r in range(rows))]
            full_rows_set, full_cols_set = set(full_rows), set(full_cols)
            bg_color = None
            for r in range(rows):
                if r in full_rows_set: continue
                for c in range(cols):
                    if c in full_cols_set: continue
                    if grid[r][c] != sep_color:
                        bg_color = grid[r][c]; break
                if bg_color is not None: break
            if bg_color is None:
                return None
            # Check all non-separator cells are bg_color
            for r in range(rows):
                if r in full_rows_set: continue
                for c in range(cols):
                    if c in full_cols_set: continue
                    if grid[r][c] != bg_color:
                        return None
            row_bands = col_bands = 0
            in_b = False
            for r in range(rows):
                if r not in full_rows_set:
                    if not in_b: row_bands += 1; in_b = True
                else: in_b = False
            in_b = False
            for c in range(cols):
                if c not in full_cols_set:
                    if not in_b: col_bands += 1; in_b = True
                else: in_b = False
            if row_bands < 1 or col_bands < 1:
                return None
            return [[bg_color] * col_bands for _ in range(row_bands)]

        # Verify on all training examples
        for ex in train:
            r = solve(ex['input'])
            if r != ex['output']:
                return None
        return solve(test_input)

    # --- _try_quadrant_color_map (19bb5feb) ---
    def _try_quadrant_color_map(self, train, test_input):
        """8-rectangle with colored patches inside. Output = 2x2 grid of patch colors by quadrant."""
        def solve(grid):
            rows, cols = len(grid), len(grid[0])
            eight_cells = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == 8]
            if len(eight_cells) < 4:
                return None
            min_r = min(r for r, c in eight_cells)
            max_r = max(r for r, c in eight_cells)
            min_c = min(c for r, c in eight_cells)
            max_c = max(c for r, c in eight_cells)
            patches = {}
            for r in range(min_r, max_r + 1):
                for c in range(min_c, max_c + 1):
                    v = grid[r][c]
                    if v != 8 and v != 0:
                        patches.setdefault(v, []).append((r, c))
            if not patches:
                return None
            mid_r = (min_r + max_r) / 2.0
            mid_c = (min_c + max_c) / 2.0
            result = [[0, 0], [0, 0]]
            for color, cells in patches.items():
                avg_r = sum(r for r, c in cells) / len(cells)
                avg_c = sum(c for r, c in cells) / len(cells)
                qr = 0 if avg_r < mid_r else 1
                qc = 0 if avg_c < mid_c else 1
                result[qr][qc] = color
            return result

        for ex in train:
            r = solve(ex['input'])
            if r != ex['output']:
                return None
        return solve(test_input)

    # --- _try_assemble_around_fives (137eaa0f) ---
    def _try_assemble_around_fives(self, train, test_input):
        """Scattered colored groups each adjacent to a 5. Assemble into 3x3 around center 5."""
        def solve(grid):
            rows, cols = len(grid), len(grid[0])
            fives = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == 5]
            if not fives:
                return None
            result = [[0]*3 for _ in range(3)]
            result[1][1] = 5
            for fr, fc in fives:
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = fr + dr, fc + dc
                        if 0 <= nr < rows and 0 <= nc < cols:
                            v = grid[nr][nc]
                            if v != 0 and v != 5:
                                or_, oc = 1 + dr, 1 + dc
                                if 0 <= or_ < 3 and 0 <= oc < 3:
                                    result[or_][oc] = v
            return result

        for ex in train:
            r = solve(ex['input'])
            if r != ex['output']:
                return None
        return solve(test_input)

    # --- _try_blobs_sorted_by_size (f8ff0b80) ---
    def _try_blobs_sorted_by_size(self, train, test_input):
        """Count cells per non-0 color, sort by count descending, output as column."""
        def solve(grid):
            from collections import Counter
            counts = Counter()
            for row in grid:
                for v in row:
                    if v != 0:
                        counts[v] += 1
            if not counts:
                return None
            ranked = sorted(counts.items(), key=lambda x: -x[1])
            return [[c] for c, _ in ranked]

        for ex in train:
            r = solve(ex['input'])
            if r != ex['output']:
                return None
        return solve(test_input)

    # --- _try_extract_uniform_cells_from_sep_grid (458e3a53) ---
    def _try_extract_uniform_cells_from_sep_grid(self, train, test_input):
        """Sep grid; some cells are uniform single-color, rest are noisy. Extract uniform sub-grid."""
        def solve(grid):
            rows, cols = len(grid), len(grid[0])
            sep_color = None
            for color in set(v for row in grid for v in row):
                fr = [r for r in range(rows) if all(grid[r][c] == color for c in range(cols))]
                fc = [c for c in range(cols) if all(grid[r][c] == color for r in range(rows))]
                if fr and fc:
                    sep_color = color; break
            if sep_color is None:
                return None
            fr_set = set(r for r in range(rows) if all(grid[r][c] == sep_color for c in range(cols)))
            fc_set = set(c for c in range(cols) if all(grid[r][c] == sep_color for r in range(rows)))
            # Build row/col bands
            rbands, cbands = [], []
            start = None
            for r in range(rows):
                if r not in fr_set:
                    if start is None: start = r
                else:
                    if start is not None: rbands.append((start, r-1)); start = None
            if start is not None: rbands.append((start, rows-1))
            start = None
            for c in range(cols):
                if c not in fc_set:
                    if start is None: start = c
                else:
                    if start is not None: cbands.append((start, c-1)); start = None
            if start is not None: cbands.append((start, cols-1))
            if len(rbands) < 2 or len(cbands) < 2:
                return None
            # For each cell, check if it's uniform (single non-sep color)
            cell_grid = []
            for ri, (r0, r1) in enumerate(rbands):
                row = []
                for ci, (c0, c1) in enumerate(cbands):
                    colors = set()
                    for r in range(r0, r1+1):
                        for c in range(c0, c1+1):
                            if grid[r][c] != sep_color:
                                colors.add(grid[r][c])
                    if len(colors) == 1:
                        row.append((True, colors.pop()))
                    else:
                        row.append((False, None))
                cell_grid.append(row)
            # Find bounding box of uniform cells
            uniform_cells = [(ri, ci, v) for ri, row in enumerate(cell_grid)
                           for ci, (u, v) in enumerate(row) if u]
            if not uniform_cells:
                return None
            min_ri = min(ri for ri,ci,v in uniform_cells)
            max_ri = max(ri for ri,ci,v in uniform_cells)
            min_ci = min(ci for ri,ci,v in uniform_cells)
            max_ci = max(ci for ri,ci,v in uniform_cells)
            # Extract the sub-grid
            result = []
            for ri in range(min_ri, max_ri+1):
                row = []
                for ci in range(min_ci, max_ci+1):
                    u, v = cell_grid[ri][ci]
                    if u:
                        row.append(v)
                    else:
                        return None  # gap in uniform block
                result.append(row)
            return result

        for ex in train:
            r = solve(ex['input'])
            if r != ex['output']:
                return None
        return solve(test_input)

    # --- _try_pyramid_inner_diagonal_extend (b8cdaf2b) ---
    def _try_pyramid_inner_diagonal_extend(self, train, test_input):
        """Pyramid shape at bottom with inner/outer colors. Inner extends diagonally above."""
        def solve(grid):
            rows, cols = len(grid), len(grid[0])
            result = [list(row) for row in grid]
            # Find bottom non-zero row
            bot = None
            for r in range(rows-1, -1, -1):
                if any(grid[r][c] != 0 for c in range(cols)):
                    bot = r
                    break
            if bot is None:
                return None
            # Bottom row should have 2 colors
            bot_colors = set(grid[bot][c] for c in range(cols) if grid[bot][c] != 0)
            if len(bot_colors) != 2:
                return None
            # Find inner and outer: inner is surrounded by outer
            bot_vals = [grid[bot][c] for c in range(cols)]
            # Find the non-zero spans
            outer = None
            inner = None
            for c in range(cols):
                if bot_vals[c] != 0:
                    outer = bot_vals[c]
                    break
            for c in range(cols):
                if bot_vals[c] != 0 and bot_vals[c] != outer:
                    inner = bot_vals[c]
                    break
            if outer is None or inner is None:
                return None
            # Find inner region bounds in bottom row
            inner_cols = [c for c in range(cols) if bot_vals[c] == inner]
            if not inner_cols:
                return None
            left_inner = min(inner_cols)
            right_inner = max(inner_cols)
            # Find top of pyramid (consecutive non-zero rows going up from bot)
            top = bot
            for r in range(bot-1, -1, -1):
                if any(grid[r][c] != 0 for c in range(cols)):
                    top = r
                else:
                    break
            # Extend inner color diagonally above the pyramid
            for i in range(1, rows):
                r = top - i
                if r < 0:
                    break
                lc = left_inner - i
                rc = right_inner + i
                placed = False
                if 0 <= lc < cols:
                    result[r][lc] = inner
                    placed = True
                if 0 <= rc < cols:
                    result[r][rc] = inner
                    placed = True
                if not placed:
                    break
            return result

        for ex in train:
            r = solve(ex['input'])
            if r != ex['output']:
                return None
        return solve(test_input)

    # --- _try_diagonal_color_markers (a9f96cdd) ---
    def _try_diagonal_color_markers(self, train, test_input):
        """Single marker cell replaced by colored diagonal neighbors (TL=c1, TR=c2, BL=c3, BR=c4)."""
        def solve(grid, marker, colors_map):
            rows, cols = len(grid), len(grid[0])
            result = [[0]*cols for _ in range(rows)]
            for r in range(rows):
                for c in range(cols):
                    if grid[r][c] == marker:
                        for (dr, dc), color in colors_map.items():
                            nr, nc = r+dr, c+dc
                            if 0 <= nr < rows and 0 <= nc < cols:
                                result[nr][nc] = color
            return result

        # Learn the marker and color map from training
        # Find the single non-0 value in each input
        marker = None
        colors_map = {}
        for ex in train:
            inp, out = ex['input'], ex['output']
            rows, cols = len(inp), len(inp[0])
            # Find marker
            marker_cells = [(r,c,inp[r][c]) for r in range(rows) for c in range(cols) if inp[r][c] != 0]
            if len(marker_cells) != 1:
                return None
            mr, mc, mv = marker_cells[0]
            if marker is None:
                marker = mv
            elif marker != mv:
                return None
            # Find output colors at diagonals
            for dr, dc in [(-1,-1),(-1,1),(1,-1),(1,1)]:
                nr, nc = mr+dr, mc+dc
                if 0 <= nr < rows and 0 <= nc < cols and out[nr][nc] != 0:
                    key = (dr, dc)
                    if key in colors_map:
                        if colors_map[key] != out[nr][nc]:
                            return None
                    else:
                        colors_map[key] = out[nr][nc]

        if marker is None or not colors_map:
            return None

        # Verify on all training
        for ex in train:
            r = solve(ex['input'], marker, colors_map)
            if r != ex['output']:
                return None
        return solve(test_input, marker, colors_map)

    # --- _try_move_toward_target (dc433765) ---
    def _try_move_toward_target(self, train, test_input):
        """Two colored dots on bg. One (mobile) moves one step toward the other (stationary)."""
        def solve(grid, mobile_color, static_color):
            rows, cols = len(grid), len(grid[0])
            mobile = static = None
            for r in range(rows):
                for c in range(cols):
                    if grid[r][c] == mobile_color:
                        mobile = (r, c)
                    elif grid[r][c] == static_color:
                        static = (r, c)
            if mobile is None or static is None:
                return None
            mr, mc = mobile
            sr, sc = static
            # Move one step toward static
            dr = (1 if sr > mr else -1 if sr < mr else 0)
            dc = (1 if sc > mc else -1 if sc < mc else 0)
            nr, nc = mr + dr, mc + dc
            result = [[0]*cols for _ in range(rows)]
            result[nr][nc] = mobile_color
            result[sr][sc] = static_color
            return result

        # Learn which color is mobile and which is static
        # Try both orderings
        for ex in train:
            inp = ex['input']
            colors = set(v for row in inp for v in row) - {0}
            if len(colors) != 2:
                return None
        
        colors = list(set(v for row in train[0]['input'] for v in row) - {0})
        if len(colors) != 2:
            return None

        for mobile_color, static_color in [(colors[0], colors[1]), (colors[1], colors[0])]:
            ok = True
            for ex in train:
                r = solve(ex['input'], mobile_color, static_color)
                if r != ex['output']:
                    ok = False
                    break
            if ok:
                return solve(test_input, mobile_color, static_color)
        return None

    # --- _try_midpoint_plus (e9614598) ---
    def _try_midpoint_plus(self, train, test_input):
        """Two dots of same color. Draw a plus/cross of color 3 at their midpoint."""
        def solve(grid, dot_color, plus_color):
            rows, cols = len(grid), len(grid[0])
            dots = [(r,c) for r in range(rows) for c in range(cols) if grid[r][c] == dot_color]
            if len(dots) != 2:
                return None
            (r1,c1), (r2,c2) = dots
            # Midpoint must be integer
            if (r1+r2) % 2 != 0 or (c1+c2) % 2 != 0:
                return None
            mr, mc = (r1+r2)//2, (c1+c2)//2
            result = [list(row) for row in grid]
            # Draw plus at midpoint
            result[mr][mc] = plus_color
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = mr+dr, mc+dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    result[nr][nc] = plus_color
            return result

        # Learn colors from first training example
        inp0, out0 = train[0]['input'], train[0]['output']
        rows0, cols0 = len(inp0), len(inp0[0])
        # Find dot color (non-zero in input)
        dot_colors = set(inp0[r][c] for r in range(rows0) for c in range(cols0) if inp0[r][c] != 0)
        if len(dot_colors) != 1:
            return None
        dot_color = dot_colors.pop()
        # Find plus color (new color in output)
        plus_colors = set(out0[r][c] for r in range(rows0) for c in range(cols0) if out0[r][c] != 0) - {dot_color}
        if len(plus_colors) != 1:
            return None
        plus_color = plus_colors.pop()

        for ex in train:
            r = solve(ex['input'], dot_color, plus_color)
            if r != ex['output']:
                return None
        return solve(test_input, dot_color, plus_color)

    # --- _try_color_swap_mapping (0d3d703e) ---
    def _try_color_swap_mapping(self, train, test_input):
        """Each input color maps to a fixed output color (learned from training)."""
        # Learn mapping from training
        mapping = {}
        for ex in train:
            inp, out = ex['input'], ex['output']
            if len(inp) != len(out) or len(inp[0]) != len(out[0]):
                return None
            for r in range(len(inp)):
                for c in range(len(inp[0])):
                    iv, ov = inp[r][c], out[r][c]
                    if iv in mapping:
                        if mapping[iv] != ov:
                            return None
                    else:
                        mapping[iv] = ov
        if not mapping:
            return None
        # Must be a non-identity mapping
        if all(k == v for k, v in mapping.items()):
            return None
        # Apply to test
        rows, cols = len(test_input), len(test_input[0])
        result = []
        for r in range(rows):
            row = []
            for c in range(cols):
                v = test_input[r][c]
                if v not in mapping:
                    return None
                row.append(mapping[v])
            result.append(row)
        # Verify on training
        for ex in train:
            inp = ex['input']
            predicted = [[mapping[inp[r][c]] for c in range(len(inp[0]))] for r in range(len(inp))]
            if predicted != ex['output']:
                return None
        return result

    # --- _try_shift_grid_down_one (25ff71a9) ---
    def _try_shift_grid_down_one(self, train, test_input):
        """Shift entire grid down by 1 row, top row becomes all zeros."""
        def solve(grid):
            rows, cols = len(grid), len(grid[0])
            result = [[0]*cols] + [list(grid[r]) for r in range(rows-1)]
            return result
        for ex in train:
            if solve(ex['input']) != ex['output']:
                return None
        return solve(test_input)

    # --- _try_and_halves_sep (0520fde7) ---
    def _try_and_halves_sep(self, train, test_input):
        """Grid split by separator column. Output = AND of left and right halves (1&1 -> marker)."""
        def solve(grid, sep_col, marker):
            rows = len(grid)
            left = [grid[r][:sep_col] for r in range(rows)]
            right = [grid[r][sep_col+1:] for r in range(rows)]
            if len(left[0]) != len(right[0]):
                return None
            w = len(left[0])
            result = [[0]*w for _ in range(rows)]
            for r in range(rows):
                for c in range(w):
                    if left[r][c] != 0 and right[r][c] != 0:
                        result[r][c] = marker
            return result

        # Find separator column (all same non-zero value)
        inp0 = train[0]['input']
        rows, cols = len(inp0), len(inp0[0])
        sep_col = None
        for c in range(cols):
            vals = set(inp0[r][c] for r in range(rows))
            if len(vals) == 1 and 0 not in vals:
                sep_col = c
                break
        if sep_col is None:
            return None
        # Find marker color from output
        out0 = train[0]['output']
        marker = None
        for r in range(len(out0)):
            for c in range(len(out0[0])):
                if out0[r][c] != 0:
                    marker = out0[r][c]
                    break
            if marker: break
        if marker is None:
            marker = 2  # default
        for ex in train:
            if solve(ex['input'], sep_col, marker) != ex['output']:
                return None
        return solve(test_input, sep_col, marker)

    # --- _try_zone_extract_single_value (5783df64) ---
    def _try_zone_extract_single_value(self, train, test_input):
        """Grid divided into equal zones, each with exactly 1 non-zero value. Extract to compact grid."""
        def solve(grid, out_rows, out_cols):
            rows, cols = len(grid), len(grid[0])
            if rows % out_rows != 0 or cols % out_cols != 0:
                return None
            zr = rows // out_rows
            zc = cols // out_cols
            result = [[0]*out_cols for _ in range(out_rows)]
            for ri in range(out_rows):
                for ci in range(out_cols):
                    val = 0
                    for r in range(ri*zr, (ri+1)*zr):
                        for c in range(ci*zc, (ci+1)*zc):
                            if grid[r][c] != 0:
                                if val != 0:
                                    return None  # more than 1 non-zero
                                val = grid[r][c]
                    result[ri][ci] = val
            return result

        # Determine output dimensions from training
        out0 = train[0]['output']
        out_rows, out_cols = len(out0), len(out0[0])
        # Check all outputs have same dims
        for ex in train:
            o = ex['output']
            if len(o) != out_rows or len(o[0]) != out_cols:
                return None
        for ex in train:
            r = solve(ex['input'], out_rows, out_cols)
            if r != ex['output']:
                return None
        return solve(test_input, out_rows, out_cols)

    # --- _try_compress_border_to_3x3 (bc1d5164) ---
    def _try_compress_border_to_3x3(self, train, test_input):
        """Compress grid to 3x3 by grouping rows/cols into 3 zones. OR within each zone."""
        def solve(grid):
            rows, cols = len(grid), len(grid[0])
            if rows < 3 or cols < 3:
                return None
            # Find the non-zero value
            nz = set(v for row in grid for v in row if v != 0)
            if len(nz) != 1:
                return None
            val = nz.pop()
            # Zone mapping: row 0 -> 0, rows 1..rows-2 -> 1, row rows-1 -> 2
            # Col 0 -> 0, cols 1..cols-2 -> 1, col cols-1 -> 2
            result = [[0]*3 for _ in range(3)]
            for r in range(rows):
                zr = 0 if r == 0 else (2 if r == rows-1 else 1)
                for c in range(cols):
                    zc = 0 if c == 0 else (2 if c == cols-1 else 1)
                    if grid[r][c] == val:
                        result[zr][zc] = val
            return result

        for ex in train:
            r = solve(ex['input'])
            if r != ex['output']:
                return None
        return solve(test_input)

    # --- _try_select_block_most_minority (ae4f1146) ---
    def _try_select_block_most_minority(self, train, test_input):
        """Multiple NxM blocks of 2 colors on 0-bg. Select the block with the most of the rarer color."""
        def find_blocks(grid):
            rows, cols = len(grid), len(grid[0])
            visited = [[False]*cols for _ in range(rows)]
            blocks = []
            for r in range(rows):
                for c in range(cols):
                    if grid[r][c] != 0 and not visited[r][c]:
                        queue = [(r, c)]
                        visited[r][c] = True
                        cells = []
                        while queue:
                            cr, cc = queue.pop(0)
                            cells.append((cr, cc))
                            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                                nr, nc = cr+dr, cc+dc
                                if 0<=nr<rows and 0<=nc<cols and not visited[nr][nc] and grid[nr][nc]!=0:
                                    visited[nr][nc] = True
                                    queue.append((nr, nc))
                        min_r = min(r for r,c in cells)
                        min_c = min(c for r,c in cells)
                        max_r = max(r for r,c in cells)
                        max_c = max(c for r,c in cells)
                        h, w = max_r-min_r+1, max_c-min_c+1
                        if h*w == len(cells):  # solid rectangle
                            block = [[grid[min_r+dr][min_c+dc] for dc in range(w)] for dr in range(h)]
                            blocks.append(block)
            return blocks

        def solve(grid, minority_color):
            blocks = find_blocks(grid)
            if len(blocks) < 2:
                return None
            # All blocks same size
            h, w = len(blocks[0]), len(blocks[0][0])
            if not all(len(b)==h and len(b[0])==w for b in blocks):
                return None
            # Select block with most minority_color cells
            best = max(blocks, key=lambda b: sum(1 for row in b for v in row if v == minority_color))
            return best

        # Learn minority color: the one that appears less overall in non-zero cells
        from collections import Counter
        all_nonzero = Counter()
        for ex in train:
            for row in ex['input']:
                for v in row:
                    if v != 0:
                        all_nonzero[v] += 1
        if len(all_nonzero) != 2:
            return None
        minority_color = all_nonzero.most_common()[-1][0]

        for ex in train:
            r = solve(ex['input'], minority_color)
            if r != ex['output']:
                return None
        return solve(test_input, minority_color)

    # --- _try_checkerboard_from_rows (e9afcf9a) ---
    def _try_checkerboard_from_rows(self, train, test_input):
        """Input has uniform rows of different colors. Output = checkerboard of those colors."""
        def solve(grid):
            rows, cols = len(grid), len(grid[0])
            # Check each row is uniform
            row_colors = []
            for r in range(rows):
                vals = set(grid[r])
                if len(vals) != 1:
                    return None
                row_colors.append(grid[r][0])
            # Need at least 2 different colors
            if len(set(row_colors)) < 2:
                return None
            # Build checkerboard
            result = [[0]*cols for _ in range(rows)]
            for r in range(rows):
                for c in range(cols):
                    # Offset by column
                    idx = (r + c) % len(row_colors)
                    result[r][c] = row_colors[idx]
            return result

        for ex in train:
            r = solve(ex['input'])
            if r != ex['output']:
                return None
        return solve(test_input)

    # --- _try_color_bands_in_order (4be741c5) ---
    def _try_color_bands_in_order(self, train, test_input):
        """Grid has color bands (top-to-bottom or left-to-right). Output = colors in spatial order."""
        def solve_tb(grid):
            """Try top-to-bottom band detection."""
            rows, cols = len(grid), len(grid[0])
            # For each row, find the dominant color
            from collections import Counter
            colors_in_order = []
            for r in range(rows):
                freq = Counter(grid[r])
                dom = freq.most_common(1)[0][0]
                if not colors_in_order or colors_in_order[-1] != dom:
                    colors_in_order.append(dom)
            if len(colors_in_order) < 2:
                return None
            return [[c] for c in colors_in_order]

        def solve_lr(grid):
            """Try left-to-right band detection."""
            rows, cols = len(grid), len(grid[0])
            from collections import Counter
            colors_in_order = []
            for c in range(cols):
                col_vals = [grid[r][c] for r in range(rows)]
                freq = Counter(col_vals)
                dom = freq.most_common(1)[0][0]
                if not colors_in_order or colors_in_order[-1] != dom:
                    colors_in_order.append(dom)
            if len(colors_in_order) < 2:
                return None
            return [colors_in_order]

        # Try top-to-bottom first
        ok = True
        for ex in train:
            r = solve_tb(ex['input'])
            if r != ex['output']:
                ok = False; break
        if ok:
            return solve_tb(test_input)

        # Try left-to-right
        ok = True
        for ex in train:
            r = solve_lr(ex['input'])
            if r != ex['output']:
                ok = False; break
        if ok:
            return solve_lr(test_input)
        return None

    # --- _try_rotate_line_at_pivot (c074846d) ---
    def _try_rotate_line_at_pivot(self, train, test_input):
        """Line of color A extends from pivot 5. Recolor A→3, draw new A-line perpendicular (90° CW)."""
        def solve(grid):
            rows, cols = len(grid), len(grid[0])
            # Find pivot (5)
            pivot = None
            for r in range(rows):
                for c in range(cols):
                    if grid[r][c] == 5:
                        pivot = (r, c)
                        break
                if pivot: break
            if pivot is None:
                return None
            pr, pc = pivot
            # Find line of 2s extending from pivot
            line_dir = None
            line_len = 0
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                n = 0
                r, c = pr+dr, pc+dc
                while 0 <= r < rows and 0 <= c < cols and grid[r][c] == 2:
                    n += 1
                    r += dr
                    c += dc
                if n > 0:
                    line_dir = (dr, dc)
                    line_len = n
                    break
            if line_dir is None or line_len == 0:
                return None
            # Rotate 90° clockwise: (dr,dc) -> (dc,-dr)
            new_dr, new_dc = line_dir[1], -line_dir[0]
            result = [list(row) for row in grid]
            # Recolor original 2s to 3
            dr, dc = line_dir
            for i in range(1, line_len+1):
                result[pr+dr*i][pc+dc*i] = 3
            # Draw new line of 2s in perpendicular direction
            for i in range(1, line_len+1):
                nr, nc = pr+new_dr*i, pc+new_dc*i
                if 0 <= nr < rows and 0 <= nc < cols:
                    result[nr][nc] = 2
            return result

        for ex in train:
            r = solve(ex['input'])
            if r != ex['output']:
                return None
        return solve(test_input)

    # --- _try_tile_along_uniform_edge (15696249) ---
    def _try_tile_along_uniform_edge(self, train, test_input):
        """3x3 input with one uniform row/col. Tile 3x in that direction at the uniform position in a 9x9 grid."""
        def solve(grid):
            rows, cols = len(grid), len(grid[0])
            if rows != 3 or cols != 3:
                return None
            result = [[0]*9 for _ in range(9)]
            # Check rows for uniformity
            for r in range(3):
                if len(set(grid[r])) == 1:
                    # Tile horizontally at row band r*3
                    for dr in range(3):
                        for rep in range(3):
                            for dc in range(3):
                                result[r*3+dr][rep*3+dc] = grid[dr][dc]
                    return result
            # Check cols for uniformity
            for c in range(3):
                col = [grid[r][c] for r in range(3)]
                if len(set(col)) == 1:
                    # Tile vertically at col band c*3
                    for dc in range(3):
                        for rep in range(3):
                            for dr in range(3):
                                result[rep*3+dr][c*3+dc] = grid[dr][dc]
                    return result
            return None

        for ex in train:
            r = solve(ex['input'])
            if r != ex['output']:
                return None
        return solve(test_input)

    # --- _try_rotate_180_around_pivot_color (90347967) ---
    def _try_rotate_180_around_pivot_color(self, train, test_input):
        """Non-zero pattern on 0-bg. One cell is a pivot color. Rotate all other cells 180 around it."""
        def solve(grid, pivot_color):
            rows, cols = len(grid), len(grid[0])
            # Find pivot
            pivot = None
            for r in range(rows):
                for c in range(cols):
                    if grid[r][c] == pivot_color:
                        pivot = (r, c)
                        break
                if pivot: break
            if pivot is None:
                return None
            pr, pc = pivot
            result = [[0]*cols for _ in range(rows)]
            result[pr][pc] = pivot_color
            for r in range(rows):
                for c in range(cols):
                    v = grid[r][c]
                    if v != 0 and v != pivot_color:
                        nr, nc = 2*pr - r, 2*pc - c
                        if 0 <= nr < rows and 0 <= nc < cols:
                            result[nr][nc] = v
                        else:
                            return None
            return result

        # Learn pivot color: the color that appears exactly once in all training inputs
        from collections import Counter
        candidate_pivots = None
        for ex in train:
            freq = Counter(v for row in ex['input'] for v in row if v != 0)
            singles = {c for c, n in freq.items() if n == 1}
            if candidate_pivots is None:
                candidate_pivots = singles
            else:
                candidate_pivots &= singles
        if not candidate_pivots:
            return None

        for pivot_color in candidate_pivots:
            ok = True
            for ex in train:
                r = solve(ex['input'], pivot_color)
                if r != ex['output']:
                    ok = False; break
            if ok:
                return solve(test_input, pivot_color)
        return None

    # --- _try_xor_halves (31d5ba1a) ---
    def _try_xor_halves(self, train, test_input):
        """Grid split into top/bottom or left/right halves. Output = XOR: marker where exactly one is non-zero."""
        def solve_tb(grid, marker):
            rows, cols = len(grid), len(grid[0])
            if rows % 2 != 0:
                return None
            h = rows // 2
            result = [[0]*cols for _ in range(h)]
            for r in range(h):
                for c in range(cols):
                    top = grid[r][c] != 0
                    bot = grid[h+r][c] != 0
                    if top != bot:  # XOR
                        result[r][c] = marker
            return result

        def solve_lr(grid, marker):
            rows, cols = len(grid), len(grid[0])
            if cols % 2 != 0:
                return None
            w = cols // 2
            result = [[0]*w for _ in range(rows)]
            for r in range(rows):
                for c in range(w):
                    left = grid[r][c] != 0
                    right = grid[r][w+c] != 0
                    if left != right:
                        result[r][c] = marker
            return result

        # Learn marker from output
        out0 = train[0]['output']
        marker = None
        for row in out0:
            for v in row:
                if v != 0:
                    marker = v; break
            if marker: break
        if marker is None:
            return None

        # Try top/bottom
        ok = True
        for ex in train:
            r = solve_tb(ex['input'], marker)
            if r != ex['output']:
                ok = False; break
        if ok:
            return solve_tb(test_input, marker)

        # Try left/right
        ok = True
        for ex in train:
            r = solve_lr(ex['input'], marker)
            if r != ex['output']:
                ok = False; break
        if ok:
            return solve_lr(test_input, marker)
        return None

    # --- _try_nor_left_right_half (e345f17b) ---
    def _try_nor_left_right_half(self, train, test_input):
        """Grid split left/right. Output = NOR: marker where BOTH halves are 0."""
        def solve(grid, marker):
            rows, cols = len(grid), len(grid[0])
            if cols % 2 != 0:
                return None
            w = cols // 2
            result = [[0]*w for _ in range(rows)]
            for r in range(rows):
                for c in range(w):
                    if grid[r][c] == 0 and grid[r][w+c] == 0:
                        result[r][c] = marker
            return result

        out0 = train[0]['output']
        marker = None
        for row in out0:
            for v in row:
                if v != 0:
                    marker = v; break
            if marker: break
        if marker is None:
            return None

        for ex in train:
            r = solve(ex['input'], marker)
            if r != ex['output']:
                return None
        return solve(test_input, marker)

    # --- _try_largest_rect_color_2x2 (445eab21) ---
    def _try_largest_rect_color_2x2(self, train, test_input):
        """Multiple colored rectangles on 0-bg. Output = 2x2 filled with the largest rectangle's color."""
        def solve(grid):
            rows, cols = len(grid), len(grid[0])
            colors = set(v for row in grid for v in row if v != 0)
            if len(colors) < 2:
                return None
            best_color = None
            best_area = -1
            for color in colors:
                cells = [(r,c) for r in range(rows) for c in range(cols) if grid[r][c] == color]
                if not cells:
                    continue
                min_r = min(r for r,c in cells)
                max_r = max(r for r,c in cells)
                min_c = min(c for r,c in cells)
                max_c = max(c for r,c in cells)
                area = (max_r - min_r + 1) * (max_c - min_c + 1)
                if area > best_area:
                    best_area = area
                    best_color = color
            if best_color is None:
                return None
            return [[best_color, best_color], [best_color, best_color]]

        for ex in train:
            r = solve(ex['input'])
            if r != ex['output']:
                return None
        return solve(test_input)

    # --- _try_triangular_number_repeat (72207abc) ---
    def _try_triangular_number_repeat(self, train, test_input):
        """Colors in a row repeat cyclically at triangular number positions (0,1,3,6,10,15,21,28,...)."""
        def solve(grid):
            rows, cols = len(grid), len(grid[0])
            # Find the row with non-zero values
            active_row = None
            for r in range(rows):
                if any(grid[r][c] != 0 for c in range(cols)):
                    if active_row is not None:
                        return None  # multiple active rows
                    active_row = r
            if active_row is None:
                return None
            # Get initial colors at triangular positions until we hit a gap
            colors = []
            pos = 0
            step = 1
            while pos < cols:
                v = grid[active_row][pos]
                if v != 0:
                    colors.append(v)
                else:
                    break
                pos += step
                step += 1
            if not colors:
                return None
            # Generate triangular positions
            result = [list(row) for row in grid]
            pos = 0
            step = 1
            idx = 0
            while pos < cols:
                result[active_row][pos] = colors[idx % len(colors)]
                idx += 1
                pos += step
                step += 1
            return result

        for ex in train:
            r = solve(ex['input'])
            if r != ex['output']:
                return None
        return solve(test_input)

    # --- _try_reflect_shape_by_pointer (760b3cac) ---
    def _try_reflect_shape_by_pointer(self, train, test_input):
        """Two shapes (colors A and B). B has a 'tip' pointing left or right. A gets horizontally reflected in that direction."""
        def solve(grid):
            rows, cols = len(grid), len(grid[0])
            colors = set(v for row in grid for v in row if v != 0)
            if len(colors) != 2:
                return None
            result = [list(row) for row in grid]
            for reflect_color in colors:
                pointer_color = (colors - {reflect_color}).pop()
                # Get pointer cells
                ptr_cells = [(r,c) for r in range(rows) for c in range(cols) if grid[r][c] == pointer_color]
                if not ptr_cells:
                    continue
                # Find the sparse row (row with only 1 pointer cell)
                from collections import Counter
                row_counts = Counter(r for r,c in ptr_cells)
                sparse_rows = [r for r, cnt in row_counts.items() if cnt == 1]
                if not sparse_rows:
                    continue
                tip_r = sparse_rows[0]
                tip_c = [c for r,c in ptr_cells if r == tip_r][0]
                center_c = sum(c for r,c in ptr_cells) / len(ptr_cells)
                # Get reflect shape
                ref_cells = [(r,c) for r in range(rows) for c in range(cols) if grid[r][c] == reflect_color]
                if not ref_cells:
                    continue
                min_c = min(c for r,c in ref_cells)
                max_c = max(c for r,c in ref_cells)
                width = max_c - min_c + 1
                # Determine direction
                if tip_c < center_c:
                    # Reflect left
                    new_min_c = min_c - width
                    for r,c in ref_cells:
                        nc = min_c - 1 - (c - min_c)  # mirror within span
                        if 0 <= nc < cols:
                            result[r][nc] = reflect_color
                elif tip_c > center_c:
                    # Reflect right
                    for r,c in ref_cells:
                        nc = max_c + 1 + (max_c - c)
                        if 0 <= nc < cols:
                            result[r][nc] = reflect_color
                else:
                    continue
                # Verify this worked on training by returning and checking
                return result
            return None

        for ex in train:
            r = solve(ex['input'])
            if r != ex['output']:
                return None
        return solve(test_input)

    # --- _try_extend_tips_diagonally (7ddcd7ec) ---
    def _try_extend_tips_diagonally(self, train, test_input):
        """Shape on 0-bg. Tip cells (1 diagonal neighbor) extend diagonally outward to grid edge."""
        def solve(grid):
            rows, cols = len(grid), len(grid[0])
            color_cells = set()
            color = None
            for r in range(rows):
                for c in range(cols):
                    if grid[r][c] != 0:
                        color_cells.add((r, c))
                        color = grid[r][c]
            if not color_cells or color is None:
                return None
            # Check single color
            if any(grid[r][c] != color for r, c in color_cells):
                return None
            result = [list(row) for row in grid]
            # Find tips: cells with exactly 1 neighbor in shape (8-connected)
            for r, c in color_cells:
                neighbors = []
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        if (r+dr, c+dc) in color_cells:
                            neighbors.append((dr, dc))
                if len(neighbors) == 1:
                    # This is a tip. Direction = opposite of neighbor (outward)
                    ndr, ndc = neighbors[0]
                    ext_dr, ext_dc = -ndr, -ndc
                    # Extend diagonally (must be diagonal)
                    if ext_dr == 0 or ext_dc == 0:
                        continue  # not diagonal
                    nr, nc = r + ext_dr, c + ext_dc
                    while 0 <= nr < rows and 0 <= nc < cols:
                        result[nr][nc] = color
                        nr += ext_dr
                        nc += ext_dc
            return result

        for ex in train:
            r = solve(ex['input'])
            if r != ex['output']:
                return None
        return solve(test_input)

    # --- _try_cascade_complement_shrink (b5bb5719) ---
    def _try_cascade_complement_shrink(self, train, test_input):
        """Row 0 has colored pattern. Each subsequent row = complement of prev, shrunk by 1 each side."""
        def solve(grid):
            rows, cols = len(grid), len(grid[0])
            # Find bg color
            from collections import Counter
            freq = Counter(v for row in grid for v in row)
            bg = freq.most_common(1)[0][0]
            # Row 0 should have non-bg values
            non_bg_row0 = [(c, grid[0][c]) for c in range(cols) if grid[0][c] != bg]
            if not non_bg_row0:
                return None
            # Find the two non-bg colors
            colors = set(v for _, v in non_bg_row0)
            if len(colors) > 2 or len(colors) < 1:
                return None
            if len(colors) == 1:
                # Need to know the complement color from training
                return None
            color_list = sorted(colors)
            def complement(v):
                return color_list[1] if v == color_list[0] else color_list[0]
            # Build cascade
            result = [list(row) for row in grid]
            left = min(c for c, _ in non_bg_row0)
            right = max(c for c, _ in non_bg_row0)
            prev_row = list(grid[0])
            for r in range(1, rows):
                left += 1
                right -= 1
                if left > right:
                    break
                for c in range(left, right + 1):
                    result[r][c] = complement(prev_row[c - 1])
                prev_row = list(result[r])
            return result

        # For single-color inputs, learn complement from training
        def solve_with_learned_complement(grid, color_pair):
            rows, cols = len(grid), len(grid[0])
            from collections import Counter
            freq = Counter(v for row in grid for v in row)
            bg = freq.most_common(1)[0][0]
            non_bg_row0 = [(c, grid[0][c]) for c in range(cols) if grid[0][c] != bg]
            if not non_bg_row0:
                return None
            colors = set(v for _, v in non_bg_row0)
            if len(colors) <= 2:
                color_list = sorted(color_pair)
            elif len(colors) == 2:
                color_list = sorted(colors)
            else:
                return None
            def complement(v):
                return color_list[1] if v == color_list[0] else color_list[0]
            result = [list(row) for row in grid]
            left = min(c for c, _ in non_bg_row0)
            right = max(c for c, _ in non_bg_row0)
            prev_row = list(grid[0])
            for r in range(1, rows):
                left += 1
                right -= 1
                if left > right:
                    break
                for c in range(left, right + 1):
                    result[r][c] = complement(prev_row[c - 1])
                prev_row = list(result[r])
            return result

        # Try with 2-color input first
        ok = True
        for ex in train:
            r = solve(ex['input'])
            if r != ex['output']:
                ok = False; break
        if ok:
            return solve(test_input)

        # Learn color pair from training outputs
        all_nonbg = set()
        for ex in train:
            from collections import Counter
            bg = Counter(v for row in ex['input'] for v in row).most_common(1)[0][0]
            all_nonbg |= set(v for row in ex['output'] for v in row if v != bg)
        if len(all_nonbg) == 2:
            color_pair = all_nonbg
            ok2 = True
            for ex2 in train:
                r = solve_with_learned_complement(ex2['input'], color_pair)
                if r != ex2['output']:
                    ok2 = False; break
            if ok2:
                return solve_with_learned_complement(test_input, color_pair)
        return None

    # --- _try_shape_pattern_to_value (27a28665) ---
    def _try_shape_pattern_to_value(self, train, test_input):
        """Grid with 0/non-0 pattern -> 1x1 output. Map the 0/1 pattern to a specific value."""
        # All outputs must be 1x1
        for ex in train:
            if len(ex['output']) != 1 or len(ex['output'][0]) != 1:
                return None

        # Build mapping: canonical pattern -> output value
        pattern_map = {}
        for ex in train:
            inp = ex['input']
            # Convert to 0/1 pattern
            pattern = tuple(tuple(1 if v != 0 else 0 for v in row) for row in inp)
            val = ex['output'][0][0]
            if pattern in pattern_map:
                if pattern_map[pattern] != val:
                    return None
            pattern_map[pattern] = val

        # Look up test pattern
        test_pattern = tuple(tuple(1 if v != 0 else 0 for v in row) for row in test_input)
        if test_pattern in pattern_map:
            return [[pattern_map[test_pattern]]]
        return None

    # --- _try_extract_corner_2x2_by_parity (a6953f00) ---
    def _try_extract_corner_2x2_by_parity(self, train, test_input):
        """Extract top-left or top-right 2x2 based on grid column parity (even→TR, odd→TL)."""
        def solve(grid):
            rows, cols = len(grid), len(grid[0])
            if rows < 2 or cols < 2:
                return None
            if cols % 2 == 0:
                return [[grid[r][cols-2+c] for c in range(2)] for r in range(2)]
            else:
                return [[grid[r][c] for c in range(2)] for r in range(2)]

        for ex in train:
            r = solve(ex['input'])
            if r != ex['output']:
                return None
        return solve(test_input)

    # --- _try_stamp_template_per_dot (12997ef3) ---
    def _try_stamp_template_per_dot(self, train, test_input):
        """1-colored template shape + colored dots. Output = template stamped per dot color, stacked vertically."""
        def solve(grid):
            rows, cols = len(grid), len(grid[0])
            # Find 1-cells (template)
            ones = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == 1]
            if not ones:
                return None
            # Find colored dots (non-0, non-1)
            dots = [(r, c, grid[r][c]) for r in range(rows) for c in range(cols)
                    if grid[r][c] not in (0, 1)]
            if not dots:
                return None
            # Extract template bounding box
            min_r = min(r for r, c in ones)
            min_c = min(c for r, c in ones)
            max_r = max(r for r, c in ones)
            max_c = max(c for r, c in ones)
            th = max_r - min_r + 1
            tw = max_c - min_c + 1
            template = [[0]*tw for _ in range(th)]
            for r, c in ones:
                template[r - min_r][c - min_c] = 1
            # Sort dots by row then col
            dots.sort()
            # Determine stacking direction: same col → vertical, same row → horizontal
            dot_rows = set(r for r, _, _ in dots)
            dot_cols = set(c for _, c, _ in dots)
            if len(dot_cols) == 1:
                # Stack vertically
                result = []
                for _, _, color in dots:
                    stamp = [[color if template[r][c] == 1 else 0 for c in range(tw)] for r in range(th)]
                    result.extend(stamp)
            elif len(dot_rows) == 1:
                # Stack horizontally
                result = [[0] * (tw * len(dots)) for _ in range(th)]
                for di, (_, _, color) in enumerate(dots):
                    for r in range(th):
                        for c in range(tw):
                            if template[r][c] == 1:
                                result[r][di * tw + c] = color
            else:
                return None
            return result

        for ex in train:
            r = solve(ex['input'])
            if r != ex['output']:
                return None
        return solve(test_input)

    # --- _try_scale_with_block_sub (2072aba6) ---
    def _try_scale_with_block_sub(self, train, test_input):
        """Each cell maps to a KxK block. Learn the block mapping from training."""
        # Determine scale factor from output/input dimensions
        inp0 = train[0]['input']
        out0 = train[0]['output']
        ir, ic = len(inp0), len(inp0[0])
        or_, oc = len(out0), len(out0[0])
        if or_ % ir != 0 or oc % ic != 0:
            return None
        kr, kc = or_ // ir, oc // ic
        if kr != kc or kr < 2:
            return None
        k = kr

        # Learn block mapping from first training example
        block_map = {}
        for r in range(ir):
            for c in range(ic):
                val = inp0[r][c]
                block = tuple(tuple(out0[r*k+dr][c*k+dc] for dc in range(k)) for dr in range(k))
                if val in block_map:
                    if block_map[val] != block:
                        return None
                block_map[val] = block

        def solve(grid):
            rows, cols = len(grid), len(grid[0])
            result = [[0]*(cols*k) for _ in range(rows*k)]
            for r in range(rows):
                for c in range(cols):
                    val = grid[r][c]
                    if val not in block_map:
                        return None
                    block = block_map[val]
                    for dr in range(k):
                        for dc in range(k):
                            result[r*k+dr][c*k+dc] = block[dr][dc]
            return result

        for ex in train:
            r = solve(ex['input'])
            if r != ex['output']:
                return None
        return solve(test_input)

    # --- _try_gravity_drop_in_columns (825aa9e9) ---
    def _try_gravity_drop_in_columns(self, train, test_input):
        """Non-bg values in each column drop down (gravity). Stack from bottom."""
        from collections import Counter

        def solve(grid, bg):
            rows, cols = len(grid), len(grid[0])
            result = [[bg]*cols for _ in range(rows)]
            for c in range(cols):
                # Collect non-bg values in this column
                vals = [grid[r][c] for r in range(rows) if grid[r][c] != bg]
                # Place them at the bottom
                for i, v in enumerate(reversed(vals)):
                    result[rows - 1 - i][c] = v
            return result

        # Determine bg
        inp0 = train[0]['input']
        bg = Counter(v for row in inp0 for v in row).most_common(1)[0][0]

        for ex in train:
            r = solve(ex['input'], bg)
            if r != ex['output']:
                return None
        return solve(test_input, bg)

    # --- _try_extract_block_at_marker (48d8fb45) ---
    def _try_extract_block_at_marker(self, train, test_input):
        """Marker cell indicates where to extract a 3x3 block. Block at (marker_row+1, marker_col-1)."""
        from collections import Counter

        def solve(grid, marker_color, bh, bw, dr, dc):
            rows, cols = len(grid), len(grid[0])
            # Find marker
            for r in range(rows):
                for c in range(cols):
                    if grid[r][c] == marker_color:
                        sr, sc = r + dr, c + dc
                        if 0 <= sr and sr + bh <= rows and 0 <= sc and sc + bw <= cols:
                            return [[grid[sr+rr][sc+cc] for cc in range(bw)] for rr in range(bh)]
            return None

        # Learn marker color and offset from training
        for ex in train:
            inp, out = ex['input'], ex['output']
            freq = Counter(v for row in inp for v in row if v != 0)
            if len(freq) < 2:
                return None

        # Find marker: color appearing exactly once
        inp0 = train[0]['input']
        freq0 = Counter(v for row in inp0 for v in row if v != 0)
        candidates = [c for c, n in freq0.items() if n == 1]
        if not candidates:
            return None

        out0 = train[0]['output']
        bh, bw = len(out0), len(out0[0])

        for marker_color in candidates:
            # Find marker position
            mr = mc = None
            for r in range(len(inp0)):
                for c in range(len(inp0[0])):
                    if inp0[r][c] == marker_color:
                        mr, mc = r, c
            # Find output position
            for sr in range(len(inp0) - bh + 1):
                for sc in range(len(inp0[0]) - bw + 1):
                    sub = [[inp0[sr+r][sc+c] for c in range(bw)] for r in range(bh)]
                    if sub == out0:
                        dr, dc = sr - mr, sc - mc
                        # Verify on all training
                        ok = True
                        for ex2 in train:
                            r = solve(ex2['input'], marker_color, bh, bw, dr, dc)
                            if r != ex2['output']:
                                ok = False; break
                        if ok:
                            return solve(test_input, marker_color, bh, bw, dr, dc)
        return None

    # --- _try_border_extend_with_corners (49d1d64f) ---
    def _try_border_extend_with_corners(self, train, test_input):
        """Expand grid by 1 on each side. Edge cells duplicate outward, corners get 0."""
        def solve(grid):
            rows, cols = len(grid), len(grid[0])
            result = []
            # Top row: 0 + first row + 0
            result.append([0] + list(grid[0]) + [0])
            # Each input row: first_cell + row + last_cell
            for r in range(rows):
                result.append([grid[r][0]] + list(grid[r]) + [grid[r][-1]])
            # Bottom row: 0 + last row + 0
            result.append([0] + list(grid[-1]) + [0])
            return result

        for ex in train:
            r = solve(ex['input'])
            if r != ex['output']:
                return None
        return solve(test_input)

    # --- _try_block_dominant_color_ignoring_marker (5614dbcf) ---
    def _try_block_dominant_color_ignoring_marker(self, train, test_input):
        """Grid divided into NxN blocks. Each block's dominant non-0 non-marker color → output cell."""
        def solve(grid, out_rows, out_cols, marker):
            rows, cols = len(grid), len(grid[0])
            if rows % out_rows != 0 or cols % out_cols != 0:
                return None
            br = rows // out_rows
            bc = cols // out_cols
            result = [[0]*out_cols for _ in range(out_rows)]
            for ri in range(out_rows):
                for ci in range(out_cols):
                    from collections import Counter
                    freq = Counter()
                    for r in range(ri*br, (ri+1)*br):
                        for c in range(ci*bc, (ci+1)*bc):
                            v = grid[r][c]
                            if v != 0 and v != marker:
                                freq[v] += 1
                    if freq:
                        result[ri][ci] = freq.most_common(1)[0][0]
            return result

        out0 = train[0]['output']
        or_, oc = len(out0), len(out0[0])

        # Try each possible marker color (appears in input but not as dominant in output)
        inp0 = train[0]['input']
        in_colors = set(v for row in inp0 for v in row if v != 0)
        out_colors = set(v for row in out0 for v in row if v != 0)

        for marker in in_colors - out_colors:
            ok = True
            for ex in train:
                r = solve(ex['input'], or_, oc, marker)
                if r != ex['output']:
                    ok = False; break
            if ok:
                return solve(test_input, or_, oc, marker)

        # Also try marker=None (no marker to ignore)
        ok = True
        for ex in train:
            r = solve(ex['input'], or_, oc, -1)
            if r != ex['output']:
                ok = False; break
        if ok:
            return solve(test_input, or_, oc, -1)
        return None

    # --- _try_count_blocks_to_checkerboard (ff28f65a) ---
    def _try_count_blocks_to_checkerboard(self, train, test_input):
        """Count 2x2 non-zero blocks. Fill first N checkerboard positions in 3x3 output."""
        CHECKER = [(0,0),(0,2),(1,1),(2,0),(2,2)]

        def count_blocks(grid):
            rows, cols = len(grid), len(grid[0])
            visited = set()
            count = 0
            for r in range(rows - 1):
                for c in range(cols - 1):
                    if (r,c) not in visited and grid[r][c] != 0:
                        if grid[r][c+1] != 0 and grid[r+1][c] != 0 and grid[r+1][c+1] != 0:
                            count += 1
                            visited.update([(r,c),(r,c+1),(r+1,c),(r+1,c+1)])
            return count

        def solve(grid):
            n = count_blocks(grid)
            if n < 1 or n > 5:
                return None
            result = [[0]*3 for _ in range(3)]
            for i in range(n):
                r, c = CHECKER[i]
                result[r][c] = 1
            return result

        for ex in train:
            r = solve(ex['input'])
            if r != ex['output']:
                return None
        return solve(test_input)

    # --- _try_max_count_colors_tiled (a3325580) ---
    def _try_max_count_colors_tiled(self, train, test_input):
        """Find colors with max cell count. Output = max_count rows x N cols, each row = sorted colors."""
        from collections import Counter

        def solve(grid):
            freq = Counter(v for row in grid for v in row if v != 0)
            if not freq:
                return None
            max_count = max(freq.values())
            max_colors = [c for c, n in freq.items() if n == max_count]
            if not max_colors:
                return None
            # Sort by column of first appearance
            first_col = {}
            rows, cols = len(grid), len(grid[0])
            for color in max_colors:
                for r in range(rows):
                    for c in range(cols):
                        if grid[r][c] == color:
                            first_col[color] = c
                            break
                    else:
                        continue
                    break
            max_colors.sort(key=lambda x: first_col.get(x, 0))
            return [list(max_colors) for _ in range(max_count)]

        for ex in train:
            r = solve(ex['input'])
            if r != ex['output']:
                return None
        return solve(test_input)

    # --- _try_concentric_frames_by_size (5587a8d0) ---
    def _try_concentric_frames_by_size(self, train, test_input):
        """Multiple shapes on bg. Build concentric square frames, largest shape outermost."""
        from collections import Counter

        def solve(grid):
            rows, cols = len(grid), len(grid[0])
            bg = Counter(v for row in grid for v in row).most_common(1)[0][0]
            # Find colored shapes
            colors = {}
            for r in range(rows):
                for c in range(cols):
                    v = grid[r][c]
                    if v != bg:
                        colors.setdefault(v, []).append((r, c))
            if len(colors) < 2:
                return None
            # Sort by bounding box area descending, then cell count descending
            def sort_key(item):
                color, cells = item
                min_r = min(r for r, c in cells)
                max_r = max(r for r, c in cells)
                min_c = min(c for r, c in cells)
                max_c = max(c for r, c in cells)
                bbox = (max_r - min_r + 1) * (max_c - min_c + 1)
                return (-bbox, -len(cells))

            ordered = sorted(colors.items(), key=sort_key)
            n = len(ordered)
            size = 2 * n - 1
            result = [[0] * size for _ in range(size)]
            for i, (color, _) in enumerate(ordered):
                # Fill frame at depth i
                for r in range(i, size - i):
                    for c in range(i, size - i):
                        if r == i or r == size - i - 1 or c == i or c == size - i - 1:
                            result[r][c] = color
                        elif i == n - 1:  # innermost = fill center
                            result[r][c] = color
            return result

        for ex in train:
            r = solve(ex['input'])
            if r != ex['output']:
                return None
        return solve(test_input)

    # --- _try_shift_objects_right_by_width (64a7c07e) ---
    def _try_shift_objects_right_by_width(self, train, test_input):
        """Each connected object shifts right by its own width."""
        def solve(grid):
            rows, cols = len(grid), len(grid[0])
            visited = [[False]*cols for _ in range(rows)]
            result = [[0]*cols for _ in range(rows)]
            for r in range(rows):
                for c in range(cols):
                    if grid[r][c] != 0 and not visited[r][c]:
                        queue = [(r, c)]
                        visited[r][c] = True
                        cells = []
                        while queue:
                            cr, cc = queue.pop(0)
                            cells.append((cr, cc, grid[cr][cc]))
                            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                                nr, nc = cr+dr, cc+dc
                                if 0<=nr<rows and 0<=nc<cols and not visited[nr][nc] and grid[nr][nc]!=0:
                                    visited[nr][nc] = True
                                    queue.append((nr, nc))
                        width = max(c for _, c, _ in cells) - min(c for _, c, _ in cells) + 1
                        for cr, cc, v in cells:
                            nc = cc + width
                            if 0 <= nc < cols:
                                result[cr][nc] = v
                            else:
                                return None
            return result

        for ex in train:
            r = solve(ex['input'])
            if r != ex['output']:
                return None
        return solve(test_input)

    # --- _try_drop_objects_into_ground_gaps (67c52801) ---
    def _try_drop_objects_into_ground_gaps(self, train, test_input):
        """Objects drop into gaps in the ground layer, reshaping to fit gap width."""
        from collections import Counter

        def solve(grid):
            rows, cols = len(grid), len(grid[0])
            # Find ground: bottom row should be solid (one color)
            ground_color = None
            if len(set(grid[-1])) == 1 and grid[-1][0] != 0:
                ground_color = grid[-1][0]
            else:
                return None
            # Find gap row (row above ground that has ground_color with gaps)
            gap_row = rows - 2
            if gap_row < 0:
                return None
            gaps = []  # (start_col, width)
            c = 0
            while c < cols:
                if grid[gap_row][c] == 0:
                    start = c
                    while c < cols and grid[gap_row][c] == 0:
                        c += 1
                    gaps.append((start, c - start))
                else:
                    c += 1
            if not gaps:
                return None
            # Find floating objects (non-0 non-ground_color connected components)
            visited = [[False]*cols for _ in range(rows)]
            # Mark ground rows as visited
            for c in range(cols):
                visited[-1][c] = True
                visited[gap_row][c] = True
            objects = []
            for r in range(rows):
                for c in range(cols):
                    if grid[r][c] != 0 and grid[r][c] != ground_color and not visited[r][c]:
                        queue = [(r, c)]
                        visited[r][c] = True
                        cells = []
                        color = grid[r][c]
                        while queue:
                            cr, cc = queue.pop(0)
                            cells.append((cr, cc))
                            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                                nr, nc = cr+dr, cc+dc
                                if 0<=nr<rows and 0<=nc<cols and not visited[nr][nc] and grid[nr][nc] == color:
                                    visited[nr][nc] = True
                                    queue.append((nr, nc))
                        objects.append((len(cells), color, cells))
            if not objects or len(objects) != len(gaps):
                return None
            # Match objects to gaps: sort both by size, smallest to smallest
            objects.sort()
            gaps.sort(key=lambda x: x[1])
            # Build output: start with all 0, add ground, fill gaps
            result = [[0]*cols for _ in range(rows)]
            # Copy ground
            result[-1] = list(grid[-1])
            for c in range(cols):
                if grid[gap_row][c] == ground_color:
                    result[gap_row][c] = ground_color
            # Fill each gap with its matched object
            for (area, color, _), (gap_start, gap_width) in zip(objects, gaps):
                if area % gap_width != 0:
                    return None
                height = area // gap_width
                for dr in range(height):
                    for dc in range(gap_width):
                        r = gap_row - dr
                        c = gap_start + dc
                        if 0 <= r < rows:
                            result[r][c] = color
            return result

        for ex in train:
            r = solve(ex['input'])
            if r != ex['output']:
                return None
        return solve(test_input)

    # --- _try_parity_fill_above_cell (834ec97d) ---
    def _try_parity_fill_above_cell(self, train, test_input):
        """Single colored cell. Fill 4 at same-parity columns for all rows above. Cell shifts down 1."""
        def solve(grid):
            rows, cols = len(grid), len(grid[0])
            # Find single non-zero cell
            nz = [(r, c, grid[r][c]) for r in range(rows) for c in range(cols) if grid[r][c] != 0]
            if len(nz) != 1:
                return None
            cr, cc, color = nz[0]
            parity = cc % 2
            result = [[0]*cols for _ in range(rows)]
            # Fill rows 0 to cr with 4 at same-parity columns
            for r in range(cr + 1):
                for c in range(cols):
                    if c % 2 == parity:
                        result[r][c] = 4
            # Place original color at cr+1
            if cr + 1 < rows:
                result[cr + 1][cc] = color
            return result

        for ex in train:
            r = solve(ex['input'])
            if r != ex['output']:
                return None
        return solve(test_input)

    # --- _try_separator_below_above_ratio (8597cfd7) ---
    def _try_separator_below_above_ratio(self, train, test_input):
        """Two colored lines + separator row. Winner = color with highest below/above ratio. Output = 2x2 of winner."""
        from collections import Counter

        def solve(grid):
            rows, cols = len(grid), len(grid[0])
            freq = Counter(v for row in grid for v in row if v != 0)
            if len(freq) < 3:
                return None
            # Find separator: color that forms a full row
            sep_color = None
            sep_row = None
            for r in range(rows):
                row_vals = set(grid[r])
                if len(row_vals) == 1 and grid[r][0] != 0:
                    sep_color = grid[r][0]
                    sep_row = r
                    break
            if sep_color is None:
                return None
            # Find the two non-sep non-0 colors
            other_colors = [c for c in freq if c != sep_color]
            if len(other_colors) != 2:
                return None
            # Count above and below separator for each color
            best = None
            best_ratio = -1
            for color in other_colors:
                above = sum(1 for r in range(sep_row) for c in range(cols) if grid[r][c] == color)
                below = sum(1 for r in range(sep_row+1, rows) for c in range(cols) if grid[r][c] == color)
                ratio = below / above if above > 0 else float('inf')
                if ratio > best_ratio or (ratio == best_ratio and (best is None or color > best)):
                    best_ratio = ratio
                    best = color
            if best is None:
                return None
            return [[best, best], [best, best]]

        for ex in train:
            r = solve(ex['input'])
            if r != ex['output']:
                return None
        return solve(test_input)

    # --- _try_four_rotation_tiling (7953d61e) ---
    def _try_four_rotation_tiling(self, train, test_input):
        """Output = 2x2 tiling of [original, rot270; rot180, rot90]."""
        def rot90cw(grid):
            r, c = len(grid), len(grid[0])
            return [[grid[r-1-j][i] for j in range(r)] for i in range(c)]

        def solve(grid):
            rows, cols = len(grid), len(grid[0])
            r90 = rot90cw(grid)
            r180 = rot90cw(r90)
            r270 = rot90cw(r180)
            # Build 2x2 tiling: [orig, r270; r180, r90]
            result = []
            for r in range(rows):
                row = list(grid[r]) + list(r270[r])
                result.append(row)
            for r in range(rows):
                row = list(r180[r]) + list(r90[r])
                result.append(row)
            return result

        for ex in train:
            r = solve(ex['input'])
            if r != ex['output']:
                return None
        return solve(test_input)

    # --- _try_drop_shape_to_separator (56dc2b01) ---
    def _try_drop_shape_to_separator(self, train, test_input):
        """Shape moves toward a full-row separator. 8-row appears on opposite side of moved shape."""
        from collections import Counter

        def solve(grid):
            rows, cols = len(grid), len(grid[0])
            # Find colors
            freq = Counter(v for row in grid for v in row if v != 0)
            if len(freq) != 2:
                return None
            # Find separator: color that forms exactly one full row
            sep_color = shape_color = None
            sep_row = None
            for color in freq:
                full_rows = [r for r in range(rows) if all(grid[r][c] == color for c in range(cols))]
                if len(full_rows) == 1:
                    sep_color = color
                    sep_row = full_rows[0]
            if sep_color is None:
                return None
            shape_color = [c for c in freq if c != sep_color][0]
            # Find shape cells
            shape_cells = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == shape_color]
            if not shape_cells:
                return None
            shape_min_r = min(r for r, c in shape_cells)
            shape_max_r = max(r for r, c in shape_cells)
            shape_h = shape_max_r - shape_min_r + 1
            # Determine direction: shape moves toward separator
            result = [[0]*cols for _ in range(rows)]
            # Keep separator
            for c in range(cols):
                result[sep_row][c] = sep_color
            if shape_min_r > sep_row:
                # Shape is below sep → move up (shape goes just below sep)
                new_start = sep_row + 1
                eight_row = new_start + shape_h
            else:
                # Shape is above sep → move down (shape goes just above sep)
                new_start = sep_row - shape_h
                eight_row = new_start - 1
            # Place shape at new position
            for r, c in shape_cells:
                nr = new_start + (r - shape_min_r)
                if 0 <= nr < rows:
                    result[nr][c] = shape_color
            # Place 8-row
            if 0 <= eight_row < rows:
                for c in range(cols):
                    result[eight_row][c] = 8
            return result

        for ex in train:
            r = solve(ex['input'])
            if r != ex['output']:
                return None
        return solve(test_input)

    # --- _try_recolor_by_nearest_border (2204b7a8) ---
    def _try_recolor_by_nearest_border(self, train, test_input):
        """Two border rows (top/bottom) of different colors. Scattered marker cells recolor to nearest border."""
        def solve(grid):
            rows, cols = len(grid), len(grid[0])
            if rows < 3:
                return None
            # Check top and bottom rows are solid single colors
            top_vals = set(grid[0])
            bot_vals = set(grid[-1])
            if len(top_vals) != 1 or len(bot_vals) != 1:
                return None
            top_color = grid[0][0]
            bot_color = grid[-1][0]
            if top_color == 0 or bot_color == 0 or top_color == bot_color:
                return None
            # Find marker color (the one that gets replaced)
            marker_color = None
            for r in range(1, rows-1):
                for c in range(cols):
                    v = grid[r][c]
                    if v != 0 and v != top_color and v != bot_color:
                        marker_color = v
                        break
                if marker_color: break
            if marker_color is None:
                return None
            result = [list(row) for row in grid]
            for r in range(1, rows-1):
                for c in range(cols):
                    if grid[r][c] == marker_color:
                        dist_top = r
                        dist_bot = rows - 1 - r
                        result[r][c] = top_color if dist_top <= dist_bot else bot_color
            return result

        for ex in train:
            r = solve(ex['input'])
            if r != ex['output']:
                return None
        return solve(test_input)

    # --- _try_crosshairs_fill_rect (21f83797) ---
    def _try_crosshairs_fill_rect(self, train, test_input):
        """Two dots on 0-bg. Draw cross through each (full row + col). Fill enclosed rectangle with new color."""
        def solve(grid, dot_color, fill_color):
            rows, cols = len(grid), len(grid[0])
            dots = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == dot_color]
            if len(dots) != 2:
                return None
            (r1, c1), (r2, c2) = dots
            result = [[0]*cols for _ in range(rows)]
            # Draw crosshairs
            for c in range(cols):
                result[r1][c] = dot_color
                result[r2][c] = dot_color
            for r in range(rows):
                result[r][c1] = dot_color
                result[r][c2] = dot_color
            # Fill rectangle between the two crosshairs
            min_r, max_r = min(r1, r2) + 1, max(r1, r2)
            min_c, max_c = min(c1, c2) + 1, max(c1, c2)
            for r in range(min_r, max_r):
                for c in range(min_c, max_c):
                    result[r][c] = fill_color
            return result

        # Learn colors from training
        inp0, out0 = train[0]['input'], train[0]['output']
        dot_color = None
        for r in range(len(inp0)):
            for c in range(len(inp0[0])):
                if inp0[r][c] != 0:
                    dot_color = inp0[r][c]; break
            if dot_color: break
        if dot_color is None:
            return None
        # Find fill color from output (new color not in input)
        in_colors = set(v for row in inp0 for v in row)
        out_colors = set(v for row in out0 for v in row)
        new_colors = out_colors - in_colors
        fill_color = new_colors.pop() if new_colors else 1

        for ex in train:
            r = solve(ex['input'], dot_color, fill_color)
            if r != ex['output']:
                return None
        return solve(test_input, dot_color, fill_color)

    # --- _try_draw_diagonal_on_zeros (aa300dc3) ---
    def _try_draw_diagonal_on_zeros(self, train, test_input):
        """Draw main diagonal line with a new color on 0-cells only."""
        def solve(grid, new_color):
            rows, cols = len(grid), len(grid[0])
            result = [list(row) for row in grid]
            for i in range(min(rows, cols)):
                if grid[i][i] == 0:
                    result[i][i] = new_color
            return result

        # Learn new color
        inp0, out0 = train[0]['input'], train[0]['output']
        new_color = None
        in_colors = set(v for row in inp0 for v in row)
        for r in range(len(out0)):
            for c in range(len(out0[0])):
                if out0[r][c] not in in_colors:
                    new_color = out0[r][c]; break
            if new_color: break
        if new_color is None:
            return None

        for ex in train:
            r = solve(ex['input'], new_color)
            if r != ex['output']:
                return None
        return solve(test_input, new_color)

    # --- _try_fill_enclosed_zero_regions (6cf79266) ---
    def _try_fill_enclosed_zero_regions(self, train, test_input):
        """Fill 0-regions that are fully enclosed by non-zero cells with a new color."""
        def solve(grid, fill_color):
            rows, cols = len(grid), len(grid[0])
            # Find 0-regions not connected to the border
            visited = [[False]*cols for _ in range(rows)]
            # BFS from all border 0-cells to mark "exterior" 0s
            queue = []
            for r in range(rows):
                for c in range(cols):
                    if (r == 0 or r == rows-1 or c == 0 or c == cols-1) and grid[r][c] == 0:
                        if not visited[r][c]:
                            visited[r][c] = True
                            queue.append((r, c))
            while queue:
                cr, cc = queue.pop(0)
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr, nc = cr+dr, cc+dc
                    if 0<=nr<rows and 0<=nc<cols and not visited[nr][nc] and grid[nr][nc] == 0:
                        visited[nr][nc] = True
                        queue.append((nr, nc))
            # Any unvisited 0-cell is enclosed
            result = [list(row) for row in grid]
            filled = False
            for r in range(rows):
                for c in range(cols):
                    if grid[r][c] == 0 and not visited[r][c]:
                        result[r][c] = fill_color
                        filled = True
            return result if filled else None

        # Learn fill color
        inp0, out0 = train[0]['input'], train[0]['output']
        in_colors = set(v for row in inp0 for v in row)
        out_colors = set(v for row in out0 for v in row)
        new_colors = out_colors - in_colors
        if not new_colors:
            return None
        fill_color = new_colors.pop()

        for ex in train:
            r = solve(ex['input'], fill_color)
            if r != ex['output']:
                return None
        return solve(test_input, fill_color)

    # --- _try_or_halves_recolor (d19f7514) ---
    def _try_or_halves_recolor(self, train, test_input):
        """Two halves (top/bottom). OR them: any non-zero → output_color, else 0."""
        def solve(grid, output_color):
            rows, cols = len(grid), len(grid[0])
            if rows % 2 != 0:
                return None
            h = rows // 2
            result = [[0]*cols for _ in range(h)]
            for r in range(h):
                for c in range(cols):
                    if grid[r][c] != 0 or grid[h+r][c] != 0:
                        result[r][c] = output_color
            return result

        # Learn output color from first training example
        out0 = train[0]['output']
        output_color = None
        for row in out0:
            for v in row:
                if v != 0:
                    output_color = v; break
            if output_color: break
        if output_color is None:
            return None

        for ex in train:
            r = solve(ex['input'], output_color)
            if r != ex['output']:
                return None
        return solve(test_input, output_color)

    # --- _try_or_halves_with_separator_col (195ba7dc) ---
    def _try_or_halves_with_separator_col(self, train, test_input):
        """Grid split by separator col (uniform non-0 color). OR left/right halves → marker color."""
        def solve(grid, sep_col, marker):
            rows = len(grid)
            left = [grid[r][:sep_col] for r in range(rows)]
            right = [grid[r][sep_col+1:] for r in range(rows)]
            if len(left[0]) != len(right[0]):
                return None
            w = len(left[0])
            result = [[0]*w for _ in range(rows)]
            for r in range(rows):
                for c in range(w):
                    if left[r][c] != 0 or right[r][c] != 0:
                        result[r][c] = marker
            return result

        # Find separator col
        inp0 = train[0]['input']
        rows, cols = len(inp0), len(inp0[0])
        sep_col = None
        for c in range(cols):
            vals = set(inp0[r][c] for r in range(rows))
            if len(vals) == 1 and 0 not in vals:
                sep_col = c
                break
        if sep_col is None:
            return None
        # Find marker from output
        out0 = train[0]['output']
        marker = None
        for row in out0:
            for v in row:
                if v != 0:
                    marker = v; break
            if marker: break
        if marker is None:
            marker = 1

        for ex in train:
            r = solve(ex['input'], sep_col, marker)
            if r != ex['output']:
                return None
        return solve(test_input, sep_col, marker)

    # --- _try_draw_x_from_dot (623ea044) ---
    def _try_draw_x_from_dot(self, train, test_input):
        """Single dot on 0-bg. Draw X (4 diagonals) extending to grid edges."""
        def solve(grid):
            rows, cols = len(grid), len(grid[0])
            nz = [(r, c, grid[r][c]) for r in range(rows) for c in range(cols) if grid[r][c] != 0]
            if len(nz) != 1:
                return None
            r, c, color = nz[0]
            result = [[0]*cols for _ in range(rows)]
            result[r][c] = color
            for dr, dc in [(-1,-1),(-1,1),(1,-1),(1,1)]:
                nr, nc = r + dr, c + dc
                while 0 <= nr < rows and 0 <= nc < cols:
                    result[nr][nc] = color
                    nr += dr
                    nc += dc
            return result

        for ex in train:
            r = solve(ex['input'])
            if r != ex['output']:
                return None
        return solve(test_input)

    # --- _try_draw_border_from_dot (fc754716) ---
    def _try_draw_border_from_dot(self, train, test_input):
        """Single dot. Draw grid border using dot's color."""
        def solve(grid):
            rows, cols = len(grid), len(grid[0])
            nz = [(r, c, grid[r][c]) for r in range(rows) for c in range(cols) if grid[r][c] != 0]
            if len(nz) != 1:
                return None
            _, _, color = nz[0]
            result = [[0]*cols for _ in range(rows)]
            for c in range(cols):
                result[0][c] = color
                result[rows-1][c] = color
            for r in range(rows):
                result[r][0] = color
                result[r][cols-1] = color
            return result

        for ex in train:
            r = solve(ex['input'])
            if r != ex['output']:
                return None
        return solve(test_input)

    # --- _try_staircase_from_dot (d06dbe63) ---
    def _try_staircase_from_dot(self, train, test_input):
        """Single dot. Staircase of 5s: vertical step + 3-cell horizontal bar alternating."""
        def solve(grid, new_color):
            rows, cols = len(grid), len(grid[0])
            nz = [(r, c, grid[r][c]) for r in range(rows) for c in range(cols) if grid[r][c] != 0]
            if len(nz) != 1:
                return None
            r, c, _ = nz[0]
            result = [list(row) for row in grid]
            # Going up-right
            cur_r, cur_c = r-1, c
            step = 0
            while 0 <= cur_r < rows and 0 <= cur_c < cols:
                result[cur_r][cur_c] = new_color
                if step % 2 == 1:
                    # horizontal bar of 3
                    if cur_c+1 < cols: result[cur_r][cur_c+1] = new_color
                    if cur_c+2 < cols: result[cur_r][cur_c+2] = new_color
                    cur_c += 2
                cur_r -= 1
                step += 1
            # Going down-left
            cur_r, cur_c = r+1, c
            step = 0
            while 0 <= cur_r < rows and 0 <= cur_c < cols:
                result[cur_r][cur_c] = new_color
                if step % 2 == 1:
                    # horizontal bar of 3 going left
                    if cur_c-1 >= 0: result[cur_r][cur_c-1] = new_color
                    if cur_c-2 >= 0: result[cur_r][cur_c-2] = new_color
                    cur_c -= 2
                cur_r += 1
                step += 1
            return result

        # Learn new color from output
        out0 = train[0]['output']
        in_colors = set(v for row in train[0]['input'] for v in row)
        out_colors = set(v for row in out0 for v in row)
        new_colors = out_colors - in_colors
        if not new_colors:
            return None
        new_color = new_colors.pop()

        for ex in train:
            r = solve(ex['input'], new_color)
            if r != ex['output']:
                return None
        return solve(test_input, new_color)

    # --- _try_bouncing_path_from_dot (e179c5f4) ---
    def _try_bouncing_path_from_dot(self, train, test_input):
        """Single dot. Path bounces up diagonally between side walls. Rest filled with bg."""
        def solve(grid, fill_color):
            rows, cols = len(grid), len(grid[0])
            nz = [(r, c, grid[r][c]) for r in range(rows) for c in range(cols) if grid[r][c] != 0]
            if len(nz) != 1:
                return None
            sr, sc, color = nz[0]
            result = [[fill_color]*cols for _ in range(rows)]
            # Trace path from dot
            r, c = sr, sc
            dc = 1 if sc == 0 else -1
            result[r][c] = color
            while r > 0:
                r -= 1
                c += dc
                if c < 0:
                    c = 1
                    dc = 1
                elif c >= cols:
                    c = cols - 2
                    dc = -1
                result[r][c] = color
            return result

        # Learn fill color (the non-dot color in output)
        inp0, out0 = train[0]['input'], train[0]['output']
        nz_in = set(v for row in inp0 for v in row if v != 0)
        all_out = set(v for row in out0 for v in row)
        fill_candidates = all_out - nz_in
        if not fill_candidates:
            return None
        fill_color = fill_candidates.pop()

        for ex in train:
            r = solve(ex['input'], fill_color)
            if r != ex['output']:
                return None
        return solve(test_input, fill_color)

    # --- _try_L_path_between_dots (a2fd1cf0) ---
    def _try_L_path_between_dots(self, train, test_input):
        """Two colored dots. Draw L-path between them with a fill color. Corner is fixed relative to dot colors."""
        def solve(grid, dotA, dotB, fill_color):
            rows, cols = len(grid), len(grid[0])
            pa = pb = None
            for r in range(rows):
                for c in range(cols):
                    if grid[r][c] == dotA: pa = (r, c)
                    elif grid[r][c] == dotB: pb = (r, c)
            if pa is None or pb is None:
                return None
            ar, ac = pa
            br, bc = pb
            # Corner at (A's row, B's col)
            cr, cc = ar, bc
            result = [list(row) for row in grid]
            # Path from A to corner (horizontal at row ar) INCLUDING corner
            if ac != bc:
                step = 1 if bc > ac else -1
                c = ac + step
                while c != bc + step:
                    result[ar][c] = fill_color
                    c += step
            # Path from corner to B (vertical at col bc) EXCLUDING corner (already painted)
            if ar != br:
                step = 1 if br > ar else -1
                r = ar + step
                while r != br:
                    result[r][bc] = fill_color
                    r += step
            return result

        # Learn dot colors (which is A, which is B) and fill color
        inp0, out0 = train[0]['input'], train[0]['output']
        nz_in = [(r,c,inp0[r][c]) for r in range(len(inp0)) for c in range(len(inp0[0])) if inp0[r][c] != 0]
        if len(nz_in) != 2:
            return None
        in_colors = {v for _,_,v in nz_in}
        out_colors = set(v for row in out0 for v in row) - {0}
        fill_cands = out_colors - in_colors
        if not fill_cands:
            return None
        fill_color = fill_cands.pop()
        # Try both orderings
        for dotA, dotB in [(nz_in[0][2], nz_in[1][2]), (nz_in[1][2], nz_in[0][2])]:
            ok = True
            for ex in train:
                r = solve(ex['input'], dotA, dotB, fill_color)
                if r != ex['output']:
                    ok = False; break
            if ok:
                return solve(test_input, dotA, dotB, fill_color)
        return None

    # --- _try_alternating_cols_from_2dots (0a938d79) ---
    def _try_alternating_cols_from_2dots(self, train, test_input):
        """Two dots at different cols. Fill vertical columns spaced by col-distance, alternating colors."""
        def solve(grid):
            rows, cols = len(grid), len(grid[0])
            nz = [(r, c, grid[r][c]) for r in range(rows) for c in range(cols) if grid[r][c] != 0]
            if len(nz) != 2:
                return None
            (r1, c1, col1), (r2, c2, col2) = sorted(nz, key=lambda x: x[1])
            if col1 == col2:
                return None
            step = c2 - c1
            result = [[0]*cols for _ in range(rows)]
            # Going right
            c = c1
            color_idx = 0
            colors = [col1, col2]
            while c < cols:
                for r in range(rows):
                    result[r][c] = colors[color_idx % 2]
                c += step
                color_idx += 1
            # Also rows case — but for this task, assume col-direction works
            return result

        # Also need to handle row-direction case
        def solve_rows(grid):
            rows, cols = len(grid), len(grid[0])
            nz = [(r, c, grid[r][c]) for r in range(rows) for c in range(cols) if grid[r][c] != 0]
            if len(nz) != 2:
                return None
            (r1, c1, col1), (r2, c2, col2) = sorted(nz, key=lambda x: x[0])
            if r1 == r2:
                return None
            step = r2 - r1
            result = [[0]*cols for _ in range(rows)]
            r = r1
            color_idx = 0
            colors = [col1, col2]
            while r < rows:
                for c in range(cols):
                    result[r][c] = colors[color_idx % 2]
                r += step
                color_idx += 1
            return result

        # Use the axis with SMALLER distance as spacing
        def solve_auto(grid):
            rows, cols = len(grid), len(grid[0])
            nz = [(r, c, grid[r][c]) for r in range(rows) for c in range(cols) if grid[r][c] != 0]
            if len(nz) != 2:
                return None
            (r1, c1, col1), (r2, c2, col2) = nz
            dr = abs(r2 - r1)
            dc = abs(c2 - c1)
            if dr == 0 and dc == 0:
                return None
            if dc == 0:
                dr_use = True
            elif dr == 0:
                dr_use = False
            else:
                dr_use = dr <= dc
            if dr_use:
                # Spacing along rows, fill horizontal lines
                (r1, c1, col1), (r2, c2, col2) = sorted(nz, key=lambda x: x[0])
                step = r2 - r1
                result = [[0]*cols for _ in range(rows)]
                r = r1
                color_idx = 0
                colors = [col1, col2]
                while r < rows:
                    for c in range(cols):
                        result[r][c] = colors[color_idx % 2]
                    r += step
                    color_idx += 1
                return result
            else:
                (r1, c1, col1), (r2, c2, col2) = sorted(nz, key=lambda x: x[1])
                step = c2 - c1
                result = [[0]*cols for _ in range(rows)]
                c = c1
                color_idx = 0
                colors = [col1, col2]
                while c < cols:
                    for r in range(rows):
                        result[r][c] = colors[color_idx % 2]
                    c += step
                    color_idx += 1
                return result

        for ex in train:
            r = solve_auto(ex['input'])
            if r != ex['output']:
                return None
        return solve_auto(test_input)

    # --- _try_count_8s_fill_matching_color (2685904e) ---
    def _try_count_8s_fill_matching_color(self, train, test_input):
        """Row 0 has N 8s. Find bottom row with color counts. Fill bars above separator at cols of the color with count N, height N."""
        from collections import Counter

        def solve(grid):
            rows, cols = len(grid), len(grid[0])
            # Find separator row (all 5s)
            sep_row = None
            for r in range(rows):
                if len(set(grid[r])) == 1 and grid[r][0] == 5:
                    sep_row = r
                    break
            if sep_row is None:
                return None
            # Count 8s in row 0
            n8 = sum(1 for c in range(cols) if grid[0][c] == 8)
            if n8 == 0:
                return None
            # Find bottom colored row (below separator, first non-empty row)
            bot_row = None
            for r in range(sep_row+1, rows):
                if any(v != 0 for v in grid[r]):
                    bot_row = r
                    break
            if bot_row is None:
                return None
            # Count colors in bottom row
            freq = Counter(grid[bot_row][c] for c in range(cols) if grid[bot_row][c] != 0)
            # Find ALL colors with count == n8
            targets = [color for color, count in freq.items() if count == n8]
            if not targets:
                return None
            result = [list(row) for row in grid]
            # Fill bars above separator with height n8 for each target color
            for target in targets:
                target_cols = [c for c in range(cols) if grid[bot_row][c] == target]
                for c in target_cols:
                    for r in range(sep_row - n8, sep_row):
                        if r >= 0:
                            result[r][c] = target
            return result

        for ex in train:
            r = solve(ex['input'])
            if r != ex['output']:
                return None
        return solve(test_input)

    # --- _try_simple_tile_nxm (ccd554ac) ---
    def _try_simple_tile_nxm(self, train, test_input):
        """Output = input tiled in a k×k pattern (output dims = k*input dims)."""
        def solve(grid):
            ir, ic = len(grid), len(grid[0])
            # Output size learned
            return None

        # Determine the scaling rule: output_dim = f(input_dim)
        # Check: output dim = input_dim * input_dim (fractal-size)
        inp0 = train[0]['input']
        out0 = train[0]['output']
        ir, ic = len(inp0), len(inp0[0])
        or_, oc = len(out0), len(out0[0])
        if or_ == ir * ir and oc == ic * ic:
            # Scale by input size (per example)
            def do_tile(grid):
                i_rows, i_cols = len(grid), len(grid[0])
                return [[grid[r % i_rows][c % i_cols] for c in range(i_cols * i_cols)] for r in range(i_rows * i_rows)]
        else:
            # Fixed scale
            if ir == 0 or ic == 0 or or_ % ir != 0 or oc % ic != 0:
                return None
            kr, kc = or_ // ir, oc // ic
            def do_tile(grid):
                i_rows, i_cols = len(grid), len(grid[0])
                return [[grid[r % i_rows][c % i_cols] for c in range(i_cols * kc)] for r in range(i_rows * kr)]

        for ex in train:
            r = do_tile(ex['input'])
            if r != ex['output']:
                return None
        return do_tile(test_input)

    # --- _try_extract_5border_with_padding (3f7978a0) ---
    def _try_extract_5border_with_padding(self, train, test_input):
        """Find bounding box of 5-cells (or any non-bg border color), extract with padding."""
        def solve(grid, marker, pad_r, pad_c):
            rows, cols = len(grid), len(grid[0])
            cells = [(r,c) for r in range(rows) for c in range(cols) if grid[r][c] == marker]
            if not cells:
                return None
            min_r = min(r for r,c in cells)
            max_r = max(r for r,c in cells)
            min_c = min(c for r,c in cells)
            max_c = max(c for r,c in cells)
            r0 = max(0, min_r - pad_r)
            r1 = min(rows, max_r + 1 + pad_r)
            c0 = max(0, min_c - pad_c)
            c1 = min(cols, max_c + 1 + pad_c)
            return [[grid[r][c] for c in range(c0, c1)] for r in range(r0, r1)]

        inp0 = train[0]['input']
        from collections import Counter
        colors = set(v for row in inp0 for v in row if v != 0)
        for marker in colors:
            for pr in range(4):
                for pc in range(4):
                    ok = True
                    for ex in train:
                        r = solve(ex['input'], marker, pr, pc)
                        if r != ex['output']:
                            ok = False; break
                    if ok:
                        return solve(test_input, marker, pr, pc)
        return None

    # --- _try_extract_unique_or_largest_component (7bb29440) ---
    def _try_extract_unique_or_largest_component(self, train, test_input):
        """Find multiple connected components (same color). Extract the one with unique size or largest bounding box."""
        from collections import Counter

        def get_components(grid, target_color):
            rows, cols = len(grid), len(grid[0])
            visited = [[False]*cols for _ in range(rows)]
            comps = []
            for r in range(rows):
                for c in range(cols):
                    if grid[r][c] == target_color and not visited[r][c]:
                        queue = [(r, c)]
                        visited[r][c] = True
                        cells = [(r, c)]
                        while queue:
                            cr, cc = queue.pop(0)
                            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                                nr, nc = cr+dr, cc+dc
                                if 0<=nr<rows and 0<=nc<cols and not visited[nr][nc] and grid[nr][nc] == target_color:
                                    visited[nr][nc] = True
                                    queue.append((nr, nc))
                                    cells.append((nr, nc))
                        comps.append(cells)
            return comps

        def extract_bbox(grid, cells):
            rows, cols = len(grid), len(grid[0])
            min_r = min(r for r,c in cells); max_r = max(r for r,c in cells)
            min_c = min(c for r,c in cells); max_c = max(c for r,c in cells)
            return [[grid[r][c] for c in range(min_c, max_c+1)] for r in range(min_r, max_r+1)]

        def solve(grid, target_color, selector):
            comps = get_components(grid, target_color)
            if not comps:
                return None
            if selector == 'largest':
                bbox_sizes = []
                for cells in comps:
                    min_r = min(r for r,c in cells); max_r = max(r for r,c in cells)
                    min_c = min(c for r,c in cells); max_c = max(c for r,c in cells)
                    bbox_sizes.append((max_r-min_r+1) * (max_c-min_c+1))
                idx = bbox_sizes.index(max(bbox_sizes))
                return extract_bbox(grid, comps[idx])
            elif selector == 'smallest':
                bbox_sizes = []
                for cells in comps:
                    min_r = min(r for r,c in cells); max_r = max(r for r,c in cells)
                    min_c = min(c for r,c in cells); max_c = max(c for r,c in cells)
                    bbox_sizes.append((max_r-min_r+1) * (max_c-min_c+1))
                idx = bbox_sizes.index(min(bbox_sizes))
                return extract_bbox(grid, comps[idx])
            return None

        # Learn target color (appears multiple times in distinct components)
        inp0 = train[0]['input']
        colors = set(v for row in inp0 for v in row if v != 0)
        for target_color in colors:
            for selector in ['largest', 'smallest']:
                ok = True
                for ex in train:
                    r = solve(ex['input'], target_color, selector)
                    if r != ex['output']:
                        ok = False; break
                if ok:
                    return solve(test_input, target_color, selector)
        return None

    # --- _try_shift_each_shape_by_width (64a7c07e) ---
    def _try_shift_each_shape_by_width(self, train, test_input):
        """Each connected component shifts horizontally by its own bbox width."""
        def get_components(grid):
            rows, cols = len(grid), len(grid[0])
            visited = [[False]*cols for _ in range(rows)]
            comps = []
            for r in range(rows):
                for c in range(cols):
                    if grid[r][c] != 0 and not visited[r][c]:
                        color = grid[r][c]
                        queue = [(r, c)]
                        visited[r][c] = True
                        cells = [(r, c)]
                        while queue:
                            cr, cc = queue.pop(0)
                            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
                                nr, nc = cr+dr, cc+dc
                                if 0<=nr<rows and 0<=nc<cols and not visited[nr][nc] and grid[nr][nc] != 0:
                                    visited[nr][nc] = True
                                    queue.append((nr, nc))
                                    cells.append((nr, nc))
                        comps.append((color, cells))
            return comps

        def solve(grid, direction):
            rows, cols = len(grid), len(grid[0])
            out = [[0]*cols for _ in range(rows)]
            comps = get_components(grid)
            for color, cells in comps:
                min_r = min(r for r,c in cells); max_r = max(r for r,c in cells)
                min_c = min(c for r,c in cells); max_c = max(c for r,c in cells)
                w = max_c - min_c + 1
                h = max_r - min_r + 1
                if direction == 'right':
                    dr, dc = 0, w
                elif direction == 'left':
                    dr, dc = 0, -w
                elif direction == 'down':
                    dr, dc = h, 0
                elif direction == 'up':
                    dr, dc = -h, 0
                else:
                    return None
                for r, c in cells:
                    nr, nc = r+dr, c+dc
                    if not (0 <= nr < rows and 0 <= nc < cols):
                        return None
                    # Need to preserve interior 0s (shape cells stored by pos, but real color)
                    out[nr][nc] = grid[r][c]
                # Also preserve original 0-cells inside bbox that are part of shape region
                # Actually cells only contains non-0 cells. For shapes with 0 interior (like hollow box),
                # we need to copy the whole bbox rectangle's non-0 cells only. That's what we do.
            return out

        for direction in ['right', 'left', 'down', 'up']:
            ok = True
            for ex in train:
                if solve(ex['input'], direction) != ex['output']:
                    ok = False
                    break
            if ok:
                return solve(test_input, direction)
        return None

    # --- _try_reverse_concentric_rings (85c4e7cd) ---
    def _try_reverse_concentric_rings(self, train, test_input):
        """Reverse the color order of concentric square rings."""
        def get_rings(grid):
            rows, cols = len(grid), len(grid[0])
            rings = []
            max_depth = min(rows, cols) // 2 + 1
            for d in range(max_depth):
                # Cells at depth d
                ring_cells = set()
                for r in range(d, rows - d):
                    for c in range(d, cols - d):
                        if r == d or r == rows - d - 1 or c == d or c == cols - d - 1:
                            ring_cells.add((r, c))
                if not ring_cells:
                    break
                colors = set(grid[r][c] for r, c in ring_cells)
                if len(colors) != 1:
                    return None
                rings.append((ring_cells, next(iter(colors))))
            return rings

        def solve(grid):
            rings = get_rings(grid)
            if rings is None or len(rings) < 2:
                return None
            rows, cols = len(grid), len(grid[0])
            out = [[0]*cols for _ in range(rows)]
            colors = [c for _, c in rings]
            reversed_colors = list(reversed(colors))
            for i, (cells, _) in enumerate(rings):
                new_color = reversed_colors[i]
                for r, c in cells:
                    out[r][c] = new_color
            return out

        for ex in train:
            if solve(ex['input']) != ex['output']:
                return None
        return solve(test_input)

    # --- _try_key_shape_recolors_main (009d5c81) ---
    def _try_key_shape_recolors_main(self, train, test_input):
        """Small colored 'key' shape looks up new color for the 8-shape; key then erased."""
        def get_key_and_main(grid):
            # Find all non-0 cells, group into components by color.
            rows, cols = len(grid), len(grid[0])
            colors = {}
            for r in range(rows):
                for c in range(cols):
                    v = grid[r][c]
                    if v != 0:
                        colors.setdefault(v, []).append((r, c))
            if len(colors) != 2:
                return None, None, None, None
            # main is the larger one
            items = sorted(colors.items(), key=lambda x: -len(x[1]))
            main_color, main_cells = items[0]
            key_color, key_cells = items[1]
            # Normalize key shape
            if not key_cells:
                return None, None, None, None
            mr = min(r for r,c in key_cells); mxr = max(r for r,c in key_cells)
            mc = min(c for r,c in key_cells); mxc = max(c for r,c in key_cells)
            shape = tuple(tuple(1 if (r+mr, c+mc) in set(key_cells) else 0 for c in range(mxc-mc+1)) for r in range(mxr-mr+1))
            # fix: above uses wrong indexing
            shape_set = set(key_cells)
            shape = tuple(tuple(1 if (mr+i, mc+j) in shape_set else 0 for j in range(mxc-mc+1)) for i in range(mxr-mr+1))
            return main_color, main_cells, key_cells, shape

        # Learn mapping shape -> target_color
        mapping = {}
        for ex in train:
            inp = ex['input']; out = ex['output']
            mc, mcells, kcells, shape = get_key_and_main(inp)
            if shape is None:
                return None
            # Determine target color: in output, the main_cells should have a new color
            new_color = out[mcells[0][0]][mcells[0][1]]
            # Verify: all main cells have new_color in output, and all key cells have 0
            for r, c in mcells:
                if out[r][c] != new_color:
                    return None
            for r, c in kcells:
                if out[r][c] != 0:
                    return None
            if shape in mapping and mapping[shape] != new_color:
                return None
            mapping[shape] = new_color
            # Stricter: verify ENTIRE output matches expected transformation
            # build predicted output: copy input, set main to new_color, key to 0
            pred = [r[:] for r in inp]
            for r, c in mcells:
                pred[r][c] = new_color
            for r, c in kcells:
                pred[r][c] = 0
            if pred != out:
                return None

        # Apply to test
        mc, mcells, kcells, shape = get_key_and_main(test_input)
        if shape is None or shape not in mapping:
            return None
        target = mapping[shape]
        rows, cols = len(test_input), len(test_input[0])
        out = [row[:] for row in test_input]
        for r, c in mcells:
            out[r][c] = target
        for r, c in kcells:
            out[r][c] = 0
        return out

    # --- _try_rectangles_position_grid (0a1d4ef5) ---
    def _try_rectangles_position_grid(self, train, test_input):
        """Find solid rectangles, sort by (row-cluster, col-cluster), return grid of colors."""
        def get_solid_rects(grid):
            rows, cols = len(grid), len(grid[0])
            visited = [[False]*cols for _ in range(rows)]
            rects = []
            for r in range(rows):
                for c in range(cols):
                    if grid[r][c] != 0 and not visited[r][c]:
                        color = grid[r][c]
                        # BFS 4-connected same color
                        queue = [(r, c)]
                        visited[r][c] = True
                        cells = [(r, c)]
                        while queue:
                            cr, cc = queue.pop(0)
                            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                                nr, nc = cr+dr, cc+dc
                                if 0<=nr<rows and 0<=nc<cols and not visited[nr][nc] and grid[nr][nc] == color:
                                    visited[nr][nc] = True
                                    queue.append((nr, nc))
                                    cells.append((nr, nc))
                        mr = min(rr for rr,cc in cells); mxr = max(rr for rr,cc in cells)
                        mc = min(cc for rr,cc in cells); mxc = max(cc for rr,cc in cells)
                        area = (mxr-mr+1)*(mxc-mc+1)
                        # Only solid rects of size >= 6
                        if area >= 6 and len(cells) == area:
                            rects.append((color, mr, mc, mxr, mxc))
            return rects

        def solve(grid):
            rects = get_solid_rects(grid)
            if not rects:
                return None
            # Cluster by row: sort by row center
            rect_info = [(c, (mr+mxr)/2, (mc+mxc)/2) for c,mr,mc,mxr,mxc in rects]
            # Cluster into rows
            rect_info.sort(key=lambda x: x[1])
            row_groups = []
            cur = [rect_info[0]]
            for r in rect_info[1:]:
                if abs(r[1] - cur[-1][1]) <= 3:
                    cur.append(r)
                else:
                    row_groups.append(cur)
                    cur = [r]
            row_groups.append(cur)
            # Sort each row by col
            for g in row_groups:
                g.sort(key=lambda x: x[2])
            # Check all rows have same length
            n = len(row_groups[0])
            for g in row_groups:
                if len(g) != n:
                    return None
            out = [[g[i][0] for i in range(n)] for g in row_groups]
            return out

        for ex in train:
            if solve(ex['input']) != ex['output']:
                return None
        return solve(test_input)

    # --- _try_line_from_markers_recolor_rects (0d87d2a6) ---
    def _try_line_from_markers_recolor_rects(self, train, test_input):
        """Pairs of 1-markers define lines; draw line; rectangles crossed by line get recolored to marker color."""
        def solve(grid):
            rows, cols = len(grid), len(grid[0])
            # Find marker color (not 0, not bg rect color)
            from collections import Counter
            counts = Counter()
            for row in grid:
                for v in row:
                    if v != 0:
                        counts[v] += 1
            if len(counts) < 2:
                return None
            # Marker is the less frequent isolated-cell color; rect color is the more frequent
            # Find cells by color
            by_color = {}
            for r in range(rows):
                for c in range(cols):
                    if grid[r][c] != 0:
                        by_color.setdefault(grid[r][c], []).append((r, c))
            # Marker = color whose cells are all singletons (no 4-adjacent same-color)
            marker = None
            rect_color = None
            for color, cells in by_color.items():
                cs = set(cells)
                is_singleton_only = all(
                    (r-1,c) not in cs and (r+1,c) not in cs and (r,c-1) not in cs and (r,c+1) not in cs
                    for r,c in cells
                )
                if is_singleton_only:
                    marker = color
                else:
                    rect_color = color
            if marker is None or rect_color is None:
                return None
            markers = by_color[marker]
            # Find pairs aligned on same row or col
            out = [row[:] for row in grid]
            lines_rows = set()  # rows with horizontal lines
            lines_cols = set()  # cols with vertical lines
            horz_ranges = []  # (r, c1, c2)
            vert_ranges = []  # (c, r1, r2)
            used = set()
            for i in range(len(markers)):
                for j in range(i+1, len(markers)):
                    r1, c1 = markers[i]; r2, c2 = markers[j]
                    # Only pair markers on opposite edges
                    if r1 == r2 and ((c1 == 0 and c2 == cols-1) or (c2 == 0 and c1 == cols-1)):
                        horz_ranges.append((r1, 0, cols-1))
                    elif c1 == c2 and ((r1 == 0 and r2 == rows-1) or (r2 == 0 and r1 == rows-1)):
                        vert_ranges.append((c1, 0, rows-1))
            if not horz_ranges and not vert_ranges:
                return None
            # Draw lines
            for r, ca, cb in horz_ranges:
                for c in range(ca, cb+1):
                    out[r][c] = marker
            for c, ra, rb in vert_ranges:
                for r in range(ra, rb+1):
                    out[r][c] = marker
            # Flood-fill: any rect_color cell 4-adjacent to a marker cell becomes marker, recursively.
            from collections import deque
            queue = deque()
            for r in range(rows):
                for c in range(cols):
                    if out[r][c] == marker:
                        queue.append((r, c))
            while queue:
                cr, cc = queue.popleft()
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr, nc = cr+dr, cc+dc
                    if 0<=nr<rows and 0<=nc<cols and out[nr][nc] == rect_color:
                        out[nr][nc] = marker
                        queue.append((nr, nc))
            return out

        for ex in train:
            if solve(ex['input']) != ex['output']:
                return None
        return solve(test_input)

    # --- _try_block_pattern_lookup (17cae0c1) ---
    def _try_block_pattern_lookup(self, train, test_input):
        """Split input into kxk blocks; map each block's marker-pattern to a color; output solid blocks."""
        def get_blocks(grid, bh, bw, nh, nw):
            blocks = []
            for br in range(nh):
                row_blocks = []
                for bc in range(nw):
                    patt = []
                    for r in range(bh):
                        row = []
                        for c in range(bw):
                            v = grid[br*bh+r][bc*bw+c]
                            row.append(1 if v != 0 else 0)
                        patt.append(tuple(row))
                    row_blocks.append(tuple(patt))
                blocks.append(row_blocks)
            return blocks

        def learn_mapping(train):
            # Determine block size from training
            # Try dividing input dims into blocks matching output dims
            ex0 = train[0]
            inp = ex0['input']; out = ex0['output']
            in_h, in_w = len(inp), len(inp[0])
            out_h, out_w = len(out), len(out[0])
            if in_h != out_h or in_w != out_w:
                return None, None, None, None
            # Find block size by finding output blocks of solid color
            # Try block sizes
            for bh in range(1, in_h+1):
                if in_h % bh != 0:
                    continue
                for bw in range(1, in_w+1):
                    if in_w % bw != 0:
                        continue
                    nh = in_h // bh
                    nw = in_w // bw
                    if nh * nw < 2:
                        continue
                    # Check that in all examples, each output block is uniform color
                    ok = True
                    mapping = {}
                    for ex in train:
                        inp = ex['input']; out = ex['output']
                        if len(inp) != in_h or len(inp[0]) != in_w:
                            ok = False; break
                        for br in range(nh):
                            for bc in range(nw):
                                # Check solid
                                c0 = out[br*bh][bc*bw]
                                for r in range(bh):
                                    for c in range(bw):
                                        if out[br*bh+r][bc*bw+c] != c0:
                                            ok = False; break
                                    if not ok: break
                                if not ok: break
                                # Input pattern
                                patt = tuple(tuple(1 if inp[br*bh+r][bc*bw+c] != 0 else 0 for c in range(bw)) for r in range(bh))
                                if patt in mapping and mapping[patt] != c0:
                                    ok = False; break
                                mapping[patt] = c0
                            if not ok: break
                        if not ok: break
                    if ok and mapping:
                        return bh, bw, nh, nw, mapping
            return None, None, None, None, None

        res = learn_mapping(train)
        if res[0] is None:
            return None
        bh, bw, nh, nw, mapping = res
        in_h, in_w = len(test_input), len(test_input[0])
        if in_h % bh != 0 or in_w % bw != 0:
            return None
        out = [[0]*in_w for _ in range(in_h)]
        for br in range(in_h // bh):
            for bc in range(in_w // bw):
                patt = tuple(tuple(1 if test_input[br*bh+r][bc*bw+c] != 0 else 0 for c in range(bw)) for r in range(bh))
                if patt not in mapping:
                    return None
                color = mapping[patt]
                for r in range(bh):
                    for c in range(bw):
                        out[br*bh+r][bc*bw+c] = color
        return out

    # --- _try_fill_columns_with_terminator (17b80ad2) ---
    def _try_fill_columns_with_terminator(self, train, test_input):
        """For each column ending with terminator (e.g., 5), sort markers by row, fill segments [prev+1..cur] with cur's color."""
        def detect_terminator(grid):
            # Find candidate terminator: value that appears in bottom rows of multiple columns
            rows, cols = len(grid), len(grid[0])
            from collections import Counter
            counts = Counter()
            for r in range(rows):
                for c in range(cols):
                    if grid[r][c] != 0:
                        counts[grid[r][c]] += 1
            if not counts:
                return None
            # Terminator: most frequent non-zero, or specifically the one that appears at bottom
            return None  # We'll try all candidates

        def solve(grid, term):
            rows, cols = len(grid), len(grid[0])
            out = [row[:] for row in grid]
            for c in range(cols):
                # Collect markers in this col
                markers = [(r, grid[r][c]) for r in range(rows) if grid[r][c] != 0]
                if not markers:
                    continue
                # Does col have terminator?
                has_term = any(v == term for r,v in markers)
                if not has_term:
                    continue
                # Sort by row
                markers.sort()
                prev = -1
                for r, v in markers:
                    for rr in range(prev+1, r+1):
                        out[rr][c] = v
                    prev = r
            # Also rows
            for r in range(rows):
                markers = [(c, grid[r][c]) for c in range(cols) if grid[r][c] != 0]
                if not markers:
                    continue
                has_term = any(v == term for c,v in markers)
                if not has_term:
                    continue
                markers.sort()
                # Check: don't overwrite col-filled cells unless consistent
                # Simpler: only do row fill if col fill didn't happen for this row
                # Actually the rule: do col fill only, not row. But some tasks may need both.
                # Skip row fills for now
            return out

        # Learn terminator from training
        candidates = set()
        for ex in train:
            for row in ex['input']:
                for v in row:
                    if v != 0:
                        candidates.add(v)
        for term in candidates:
            ok = True
            for ex in train:
                r = solve(ex['input'], term)
                if r != ex['output']:
                    ok = False; break
            if ok:
                return solve(test_input, term)
        return None

    # --- _try_shapes_float_to_edges (17829a00) ---
    def _try_shapes_float_to_edges(self, train, test_input):
        """Top-edge-color shapes float UP to row 1. Bottom-edge-color shapes sink DOWN to row last-1."""
        def solve(grid):
            rows, cols = len(grid), len(grid[0])
            # Top edge color (row 0 must be uniform)
            top_color = grid[0][0]
            if any(v != top_color for v in grid[0]):
                return None
            bot_color = grid[rows-1][0]
            if any(v != bot_color for v in grid[rows-1]):
                return None
            bg = None
            # bg = most common value in interior
            from collections import Counter
            c = Counter()
            for r in range(1, rows-1):
                for v in grid[r]:
                    c[v] += 1
            bg = c.most_common(1)[0][0]
            # Initialize output: keep edges, fill interior with bg
            out = [[bg]*cols for _ in range(rows)]
            out[0] = list(grid[0])
            out[rows-1] = list(grid[rows-1])
            # Find components of top_color and bot_color in interior
            def find_components(color):
                visited = [[False]*cols for _ in range(rows)]
                comps = []
                for r in range(1, rows-1):
                    for cc in range(cols):
                        if grid[r][cc] == color and not visited[r][cc]:
                            queue = [(r, cc)]
                            visited[r][cc] = True
                            cells = [(r, cc)]
                            while queue:
                                cr, cccc = queue.pop(0)
                                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
                                    nr, nc = cr+dr, cccc+dc
                                    if 1<=nr<rows-1 and 0<=nc<cols and not visited[nr][nc] and grid[nr][nc] == color:
                                        visited[nr][nc] = True
                                        queue.append((nr, nc))
                                        cells.append((nr, nc))
                            comps.append(cells)
                return comps

            for cells in find_components(top_color):
                mr = min(r for r,cc in cells)
                dr = 1 - mr  # move so top is at row 1
                for r, cc in cells:
                    nr = r + dr
                    if 0 < nr < rows-1:
                        out[nr][cc] = top_color
            for cells in find_components(bot_color):
                mxr = max(r for r,cc in cells)
                dr = (rows-2) - mxr
                for r, cc in cells:
                    nr = r + dr
                    if 0 < nr < rows-1:
                        out[nr][cc] = bot_color
            return out

        for ex in train:
            r = solve(ex['input'])
            if r != ex['output']:
                return None
        return solve(test_input)

    # --- _try_grid_cell_fill_by_corner_marker (17b866bd) ---
    def _try_grid_cell_fill_by_corner_marker(self, train, test_input):
        """Repeating grid cells separated by 8-lines. Non-zero marker at corner fills adjacent cell."""
        def solve(grid):
            rows, cols = len(grid), len(grid[0])
            # Detect step: find row with all 0 or 8 corner pattern, locate grid line rows
            # Assume step = some s such that every s-th row and col is a 'corner' row/col.
            # Find by looking for 0-majority pattern
            # Find all corner positions: where input typically has 0
            # Method: find the step by looking for first row after row 0 that matches row 0.
            for step in range(3, min(rows, cols)):
                if rows % step == 1 and cols % step == 1:
                    # Row 0 ~= row step, ignoring marker positions (corners)
                    def similar(r1, r2, st):
                        for i in range(len(r1)):
                            if r1[i] == r2[i]:
                                continue
                            if i % st == 0:
                                continue
                            return False
                        return True
                    if similar(grid[0], grid[step], step):
                        # Use this step
                        out = [row[:] for row in grid]
                        # Find markers
                        for kr in range(0, rows, step):
                            for kc in range(0, cols, step):
                                v = grid[kr][kc]
                                if v != 0:
                                    # Clear marker
                                    out[kr][kc] = 0
                                    # Fill cell to bottom-right
                                    br = kr + 1
                                    bc = kc + 1
                                    er = kr + step
                                    ec = kc + step
                                    if er > rows or ec > cols:
                                        continue
                                    for r in range(br, er):
                                        for c in range(bc, ec):
                                            if grid[r][c] == 0:
                                                out[r][c] = v
                        return out
            return None

        for ex in train:
            r = solve(ex['input'])
            if r != ex['output']:
                return None
        return solve(test_input)

    # --- _try_u_shape_fill_or_bottom_line (db7260a4) ---
    def _try_u_shape_fill_or_bottom_line(self, train, test_input):
        """1-marker above U-shape(s). If column falls inside closed-bottom U, fill interior. Else output 1s on bottom row."""
        def solve(grid):
            rows, cols = len(grid), len(grid[0])
            # Find 1-marker
            marker_col = None
            for r in range(rows):
                for c in range(cols):
                    if grid[r][c] == 1:
                        if marker_col is not None:
                            return None  # Multiple markers
                        marker_col = c
            if marker_col is None:
                return None
            # Find vertical walls: maximal contiguous spans of 2s in each column, length >= 3
            walls = []  # list of (col, r_top, r_bot)
            for c in range(cols):
                r = 0
                while r < rows:
                    if grid[r][c] == 2:
                        r_top = r
                        while r < rows and grid[r][c] == 2:
                            r += 1
                        r_bot = r - 1
                        if r_bot - r_top >= 2:
                            walls.append((c, r_top, r_bot))
                    else:
                        r += 1
            out = [row[:] for row in grid]
            # Clear the 1-marker
            for r in range(rows):
                for c in range(cols):
                    if out[r][c] == 1:
                        out[r][c] = 0
            filled = False
            # For each pair of walls with same span, check closure and fill
            for i in range(len(walls)):
                for j in range(i+1, len(walls)):
                    c1, t1, b1 = walls[i]
                    c2, t2, b2 = walls[j]
                    if t1 != t2 or b1 != b2 or c1 >= c2:
                        continue
                    # Check bottom row closed between cols
                    closed = all(grid[b1][c] == 2 for c in range(c1, c2+1))
                    if closed and c1 <= marker_col <= c2:
                        for r in range(t1, b1):
                            for c in range(c1+1, c2):
                                if out[r][c] == 0:
                                    out[r][c] = 1
                        filled = True
            if not filled:
                # Fill bottom row with 1s
                out[rows-1] = [1]*cols
            return out

        for ex in train:
            r = solve(ex['input'])
            if r != ex['output']:
                return None
        return solve(test_input)

    # --- _try_enclosed_3x3_hollow (bf699163) ---
    def _try_enclosed_3x3_hollow(self, train, test_input):
        """Many 3x3 hollow squares; one big enclosing shape. Return the 3x3 hollow whose center is in the big shape's bbox."""
        def solve(grid):
            rows, cols = len(grid), len(grid[0])
            from collections import Counter
            cnt = Counter(v for row in grid for v in row)
            bg = cnt.most_common(1)[0][0]
            hollows = []
            for r in range(rows-2):
                for c in range(cols-2):
                    cells = [grid[r+i][c+j] for i in range(3) for j in range(3)]
                    center = grid[r+1][c+1]
                    border = [cells[0],cells[1],cells[2],cells[3],cells[5],cells[6],cells[7],cells[8]]
                    if center == bg and len(set(border)) == 1 and border[0] != bg:
                        hollows.append((border[0], r+1, c+1))
            if not hollows:
                return None
            hollow_cells = set()
            for color, cr, cc in hollows:
                for i in range(-1,2):
                    for j in range(-1,2):
                        hollow_cells.add((cr+i, cc+j))
            big_cells = [(r,c) for r in range(rows) for c in range(cols)
                         if grid[r][c] != bg and (r,c) not in hollow_cells]
            if not big_cells:
                return None
            mr = min(r for r,c in big_cells); mxr = max(r for r,c in big_cells)
            mc = min(c for r,c in big_cells); mxc = max(c for r,c in big_cells)
            for color, cr, cc in hollows:
                if mr <= cr <= mxr and mc <= cc <= mxc:
                    return [[color,color,color],[color,bg,color],[color,color,color]]
            return None

        for ex in train:
            if solve(ex['input']) != ex['output']:
                return None
        return solve(test_input)

    # --- _try_box_with_gap_fill_and_extend (292dd178) ---
    def _try_box_with_gap_fill_and_extend(self, train, test_input):
        """1-box with gap in wall: fill interior with 2, extend 2s through gap to grid edge."""
        def solve(grid):
            rows, cols = len(grid), len(grid[0])
            from collections import Counter
            cnt = Counter(v for row in grid for v in row)
            bg = cnt.most_common(1)[0][0]
            # Find non-bg color (box color)
            box_colors = set(v for row in grid for v in row if v != bg)
            if len(box_colors) != 1:
                return None
            box_color = next(iter(box_colors))
            # Find 1-boxes (connected components)
            visited = [[False]*cols for _ in range(rows)]
            comps = []
            for r in range(rows):
                for c in range(cols):
                    if grid[r][c] == box_color and not visited[r][c]:
                        queue = [(r, c)]
                        visited[r][c] = True
                        cells = [(r, c)]
                        while queue:
                            cr, cc = queue.pop(0)
                            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                                nr, nc = cr+dr, cc+dc
                                if 0<=nr<rows and 0<=nc<cols and not visited[nr][nc] and grid[nr][nc] == box_color:
                                    visited[nr][nc] = True
                                    queue.append((nr, nc))
                                    cells.append((nr, nc))
                        comps.append(cells)
            out = [row[:] for row in grid]
            for cells in comps:
                mr = min(r for r,c in cells); mxr = max(r for r,c in cells)
                mc = min(c for r,c in cells); mxc = max(c for r,c in cells)
                if mxr - mr < 2 or mxc - mc < 2:
                    continue
                cs = set(cells)
                # Find gaps in walls
                gaps = []  # (r, c, direction_outward)
                for c in range(mc, mxc+1):
                    if (mr, c) not in cs:
                        gaps.append((mr, c, 'up'))
                    if (mxr, c) not in cs:
                        gaps.append((mxr, c, 'down'))
                for r in range(mr, mxr+1):
                    if (r, mc) not in cs:
                        gaps.append((r, mc, 'left'))
                    if (r, mxc) not in cs:
                        gaps.append((r, mxc, 'right'))
                if not gaps:
                    continue
                # Fill interior with 2
                for r in range(mr+1, mxr):
                    for c in range(mc+1, mxc):
                        if grid[r][c] == bg:
                            out[r][c] = 2
                # Extend through each gap to edge
                for gr, gc, dirn in gaps:
                    if dirn == 'up':
                        for rr in range(gr, -1, -1):
                            out[rr][gc] = 2
                    elif dirn == 'down':
                        for rr in range(gr, rows):
                            out[rr][gc] = 2
                    elif dirn == 'left':
                        for cc in range(gc, -1, -1):
                            out[gr][cc] = 2
                    elif dirn == 'right':
                        for cc in range(gc, cols):
                            out[gr][cc] = 2
            return out

        for ex in train:
            if solve(ex['input']) != ex['output']:
                return None
        return solve(test_input)

    # --- _try_majority_pixel_of_shapes (358ba94e) ---
    def _try_majority_pixel_of_shapes(self, train, test_input):
        """Find all NxM shapes. Output is the pixel-wise majority."""
        def solve(grid):
            rows, cols = len(grid), len(grid[0])
            from collections import Counter
            cnt = Counter(v for row in grid for v in row)
            bg = cnt.most_common(1)[0][0]
            fg_colors = set(v for row in grid for v in row if v != bg)
            if len(fg_colors) != 1:
                return None
            fg = next(iter(fg_colors))
            visited = [[False]*cols for _ in range(rows)]
            shapes = []
            for r in range(rows):
                for c in range(cols):
                    if grid[r][c] == fg and not visited[r][c]:
                        queue = [(r, c)]
                        visited[r][c] = True
                        cells = [(r, c)]
                        while queue:
                            cr, cc = queue.pop(0)
                            for dr in [-1,0,1]:
                                for dc in [-1,0,1]:
                                    if dr==0 and dc==0: continue
                                    nr, nc = cr+dr, cc+dc
                                    if 0<=nr<rows and 0<=nc<cols and not visited[nr][nc] and grid[nr][nc] == fg:
                                        visited[nr][nc] = True
                                        queue.append((nr, nc))
                                        cells.append((nr, nc))
                        shapes.append(cells)
            shape_grids = []
            for cells in shapes:
                cs = set(cells)
                mr = min(r for r,c in cells); mxr = max(r for r,c in cells)
                mc = min(c for r,c in cells); mxc = max(c for r,c in cells)
                sg = tuple(tuple(1 if (mr+i, mc+j) in cs else 0 for j in range(mxc-mc+1)) for i in range(mxr-mr+1))
                shape_grids.append((mxr-mr+1, mxc-mc+1, sg))
            if len(shape_grids) < 2:
                return None
            dim_counts = Counter((h,w) for h,w,_ in shape_grids)
            common_dim = dim_counts.most_common(1)[0][0]
            if dim_counts[common_dim] < 2:
                return None
            h, w = common_dim
            same_shapes = [sg for ih,iw,sg in shape_grids if (ih,iw) == common_dim]
            # Find shape with UNIQUE hole count
            hole_counts = [sum(1 for row in sg for v in row if v == 0) for sg in same_shapes]
            cc = Counter(hole_counts)
            unique_counts = [c for c,n in cc.items() if n == 1]
            if len(unique_counts) != 1:
                return None
            target_count = unique_counts[0]
            for sg, hc in zip(same_shapes, hole_counts):
                if hc == target_count:
                    return [[fg if v else bg for v in row] for row in sg]
            return None

        for ex in train:
            if solve(ex['input']) != ex['output']:
                return None
        return solve(test_input)

    # --- _try_overlay_bordered_rect_interiors (25e02866) ---
    def _try_overlay_bordered_rect_interiors(self, train, test_input):
        """Multiple bordered rectangles. Overlay their interior patterns into one combined rect."""
        def solve(grid):
            rows, cols = len(grid), len(grid[0])
            from collections import Counter
            cnt = Counter(v for row in grid for v in row)
            bg = cnt.most_common(1)[0][0]
            # Find bordered rectangles
            visited = [[False]*cols for _ in range(rows)]
            rects = []
            for r in range(rows):
                for c in range(cols):
                    if grid[r][c] != bg and not visited[r][c]:
                        color = grid[r][c]
                        queue = [(r, c)]
                        visited[r][c] = True
                        cells = [(r, c)]
                        while queue:
                            cr, cc = queue.pop(0)
                            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                                nr, nc = cr+dr, cc+dc
                                if 0<=nr<rows and 0<=nc<cols and not visited[nr][nc] and grid[nr][nc] != bg:
                                    visited[nr][nc] = True
                                    queue.append((nr, nc))
                                    cells.append((nr, nc))
                        mr = min(rr for rr,cc in cells); mxr = max(rr for rr,cc in cells)
                        mc = min(cc for rr,cc in cells); mxc = max(cc for rr,cc in cells)
                        rects.append((mr, mc, mxr, mxc))
            if len(rects) < 2:
                return None
            # All rects should have same dimensions
            dims = set((mxr-mr+1, mxc-mc+1) for mr,mc,mxr,mxc in rects)
            if len(dims) != 1:
                return None
            h, w = next(iter(dims))
            if h < 3 or w < 3:
                return None
            # Border color should be consistent
            border_color = grid[rects[0][0]][rects[0][1]]
            # Build output: start with border_color filled rect
            out = [[border_color]*w for _ in range(h)]
            # Overlay each rect's interior
            for mr, mc, mxr, mxc in rects:
                for i in range(1, h-1):
                    for j in range(1, w-1):
                        v = grid[mr+i][mc+j]
                        if v != border_color:
                            out[i][j] = v
            return out

        for ex in train:
            if solve(ex['input']) != ex['output']:
                return None
        return solve(test_input)

    # --- _try_xor_halves (31d5ba1a) ---
    def _try_xor_halves(self, train, test_input):
        """Split grid into top/bottom halves. Output = XOR: 6 where exactly one half is non-zero, else 0."""
        def solve(grid, fill_color):
            rows, cols = len(grid), len(grid[0])
            if rows % 2 != 0:
                return None
            half = rows // 2
            out = [[0]*cols for _ in range(half)]
            for r in range(half):
                for c in range(cols):
                    a = grid[r][c] != 0
                    b = grid[half+r][c] != 0
                    if a != b:
                        out[r][c] = fill_color
            return out

        # Learn fill_color from first example
        fill_colors = set()
        for ex in train:
            for row in ex['output']:
                for v in row:
                    if v != 0:
                        fill_colors.add(v)
        if len(fill_colors) != 1:
            return None
        fill_color = next(iter(fill_colors))
        for ex in train:
            if solve(ex['input'], fill_color) != ex['output']:
                return None
        return solve(test_input, fill_color)

    # --- _try_largest_solid_block_color (3194b014) ---
    def _try_largest_solid_block_color(self, train, test_input):
        """Find largest solid rectangular block. Output 3x3 filled with its color."""
        def solve(grid):
            rows, cols = len(grid), len(grid[0])
            from collections import Counter
            # Exclude top-2 most common colors (background noise)
            cnt = Counter(v for row in grid for v in row)
            bg_colors = set(c for c, _ in cnt.most_common(2))
            # For each remaining color, find the largest axis-aligned solid rectangle
            best_color = None
            best_area = 0
            for color in set(v for row in grid for v in row):
                if color in bg_colors:
                    continue
                # Find maximal solid rectangle of this color using brute force
                for r1 in range(rows):
                    for c1 in range(cols):
                        if grid[r1][c1] != color:
                            continue
                        # Expand down and right
                        max_c2 = cols - 1
                        for r2 in range(r1, rows):
                            # Find rightmost extent at this row
                            c2 = c1
                            while c2 <= max_c2 and grid[r2][c2] == color:
                                c2 += 1
                            c2 -= 1
                            if c2 < c1:
                                break
                            max_c2 = c2
                            area = (r2 - r1 + 1) * (c2 - c1 + 1)
                            if area > best_area:
                                best_area = area
                                best_color = color
            if best_color is None or best_area < 6:
                return None
            return [[best_color]*3 for _ in range(3)]

        for ex in train:
            if solve(ex['input']) != ex['output']:
                return None
        return solve(test_input)

    # --- _try_shape_colored_by_key_columns (278e5215) ---
    def _try_shape_colored_by_key_columns(self, train, test_input):
        """5-shape + color key. Output: each column gets its key color where shape=5, separator color where shape=0."""
        def solve(grid):
            rows, cols = len(grid), len(grid[0])
            # Find 5-cells (shape)
            shape_cells = [(r,c) for r in range(rows) for c in range(cols) if grid[r][c] == 5]
            if not shape_cells:
                return None
            mr = min(r for r,c in shape_cells); mxr = max(r for r,c in shape_cells)
            mc = min(c for r,c in shape_cells); mxc = max(c for r,c in shape_cells)
            sh = mxr - mr + 1; sw = mxc - mc + 1
            shape = [[1 if grid[mr+i][mc+j] == 5 else 0 for j in range(sw)] for i in range(sh)]
            # Find key block: non-0, non-5 rectangle
            key_cells = [(r,c) for r in range(rows) for c in range(cols) if grid[r][c] not in (0, 5)]
            if not key_cells:
                return None
            kr = min(r for r,c in key_cells); kxr = max(r for r,c in key_cells)
            kc = min(c for r,c in key_cells); kxc = max(c for r,c in key_cells)
            kw = kxc - kc + 1
            if kw != sw:
                return None
            # Extract column colors and separator
            key_rows = [[grid[kr+i][kc+j] for j in range(kw)] for i in range(kxr-kr+1)]
            # Find separator row (uniform color)
            sep_color = None
            col_colors = [None] * kw
            for row in key_rows:
                if len(set(row)) == 1:
                    sep_color = row[0]
                else:
                    for j in range(kw):
                        col_colors[j] = row[j]
            if sep_color is None or any(c is None for c in col_colors):
                return None
            # Build output
            out = [[0]*sw for _ in range(sh)]
            for i in range(sh):
                for j in range(sw):
                    out[i][j] = col_colors[j] if shape[i][j] else sep_color
            return out

        for ex in train:
            if solve(ex['input']) != ex['output']:
                return None
        return solve(test_input)

    # --- _try_rect_count_staircase (2753e76c) ---
    def _try_rect_count_staircase(self, train, test_input):
        """Count solid rectangles per color. Output staircase sorted by count descending, right-justified."""
        def solve(grid):
            rows, cols = len(grid), len(grid[0])
            from collections import Counter
            visited = [[False]*cols for _ in range(rows)]
            rect_counts = Counter()
            for r in range(rows):
                for c in range(cols):
                    if grid[r][c] != 0 and not visited[r][c]:
                        color = grid[r][c]
                        queue = [(r, c)]
                        visited[r][c] = True
                        cells = [(r, c)]
                        while queue:
                            cr, cc = queue.pop(0)
                            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                                nr, nc = cr+dr, cc+dc
                                if 0<=nr<rows and 0<=nc<cols and not visited[nr][nc] and grid[nr][nc] == color:
                                    visited[nr][nc] = True
                                    queue.append((nr, nc))
                                    cells.append((nr, nc))
                        # Check if this component is a solid rectangle
                        mr = min(rr for rr,cc in cells); mxr = max(rr for rr,cc in cells)
                        mc = min(cc for rr,cc in cells); mxc = max(cc for rr,cc in cells)
                        if len(cells) == (mxr-mr+1)*(mxc-mc+1):
                            rect_counts[color] += 1
            if not rect_counts:
                return None
            sorted_colors = sorted(rect_counts.items(), key=lambda x: -x[1])
            n = len(sorted_colors)
            max_count = sorted_colors[0][1]
            out = [[0]*max_count for _ in range(n)]
            for i, (color, count) in enumerate(sorted_colors):
                for j in range(max_count - count, max_count):
                    out[i][j] = color
            return out

        for ex in train:
            if solve(ex['input']) != ex['output']:
                return None
        return solve(test_input)

    # --- _try_patch_into_5rect_hole (228f6490) ---
    def _try_patch_into_5rect_hole(self, train, test_input):
        """Small colored patches move into the 0-holes of 5-bordered rectangles, matched by shape."""
        def solve(grid):
            rows, cols = len(grid), len(grid[0])
            # Find 5-bordered rectangles
            visited = [[False]*cols for _ in range(rows)]
            rects = []
            for r in range(rows):
                for c in range(cols):
                    if grid[r][c] == 5 and not visited[r][c]:
                        queue = [(r, c)]; visited[r][c] = True; cells = [(r, c)]
                        while queue:
                            cr, cc = queue.pop(0)
                            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                                nr, nc = cr+dr, cc+dc
                                if 0<=nr<rows and 0<=nc<cols and not visited[nr][nc] and grid[nr][nc] == 5:
                                    visited[nr][nc] = True; queue.append((nr, nc)); cells.append((nr, nc))
                        rects.append(cells)
            # Find holes in each rect (interior 0-cells)
            rect_holes = []
            for cells in rects:
                cs = set(cells)
                mr = min(r for r,c in cells); mxr = max(r for r,c in cells)
                mc = min(c for r,c in cells); mxc = max(c for r,c in cells)
                hole = []
                for r in range(mr, mxr+1):
                    for c in range(mc, mxc+1):
                        if (r, c) not in cs and grid[r][c] == 0:
                            hole.append((r, c))
                if hole:
                    rect_holes.append(hole)
            # Find colored patches (non-5 non-0)
            visited2 = [[False]*cols for _ in range(rows)]
            patches = []
            for r in range(rows):
                for c in range(cols):
                    v = grid[r][c]
                    if v != 0 and v != 5 and not visited2[r][c]:
                        queue = [(r, c)]; visited2[r][c] = True; cells = [(r, c)]; color = v
                        while queue:
                            cr, cc = queue.pop(0)
                            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                                nr, nc = cr+dr, cc+dc
                                if 0<=nr<rows and 0<=nc<cols and not visited2[nr][nc] and grid[nr][nc] == color:
                                    visited2[nr][nc] = True; queue.append((nr, nc)); cells.append((nr, nc))
                        patches.append((color, cells))
            # Normalize shape: relative to bbox
            def normalize(cells):
                mr = min(r for r,c in cells); mc = min(c for r,c in cells)
                return frozenset((r-mr, c-mc) for r,c in cells)
            # Match patches to holes by shape
            out = [row[:] for row in grid]
            # Match by shape AND proximity (closest pair first)
            def center(cells):
                return (sum(r for r,c in cells)/len(cells), sum(c for r,c in cells)/len(cells))
            matches = []
            for pi, (color, pcells) in enumerate(patches):
                pshape = normalize(pcells)
                pc = center(pcells)
                for hi, hole in enumerate(rect_holes):
                    hshape = normalize(hole)
                    if pshape == hshape:
                        hc = center(hole)
                        dist = abs(pc[0]-hc[0]) + abs(pc[1]-hc[1])
                        matches.append((dist, pi, hi))
            matches.sort()
            used_patches = set(); used_holes = set()
            for _, pi, hi in matches:
                if pi in used_patches or hi in used_holes:
                    continue
                color, pcells = patches[pi]
                hole = rect_holes[hi]
                for r, c in hole:
                    out[r][c] = color
                for r, c in pcells:
                    out[r][c] = 0
                used_patches.add(pi); used_holes.add(hi)
            return out

        for ex in train:
            if solve(ex['input']) != ex['output']:
                return None
        return solve(test_input)

    # --- _try_complete_symmetry_with_2 (1b60fb0c) ---
    def _try_complete_symmetry_with_2(self, train, test_input):
        """1-shape has partial symmetry. Reflect to complete it; new cells become color 2."""
        def solve(grid):
            rows, cols = len(grid), len(grid[0])
            ones = [(r,c) for r in range(rows) for c in range(cols) if grid[r][c] == 1]
            if not ones:
                return None
            mr = min(r for r,c in ones); mxr = max(r for r,c in ones)
            mc = min(c for r,c in ones); mxc = max(c for r,c in ones)
            one_set = set(ones)
            out = [row[:] for row in grid]
            # Try vertical axis (reflect left<->right)
            for axis_c2 in [mc + mxc]:  # axis at (mc+mxc)/2
                new_cells = []
                for r, c in ones:
                    rc = mc + mxc - c  # reflect c around axis
                    if (r, rc) not in one_set and 0 <= rc < cols and grid[r][rc] == 0:
                        new_cells.append((r, rc))
                if new_cells:
                    test_out = [row[:] for row in grid]
                    for r, c in new_cells:
                        test_out[r][c] = 2
                    return test_out
            # Try horizontal axis
            for axis_r2 in [mr + mxr]:
                new_cells = []
                for r, c in ones:
                    rr = mr + mxr - r
                    if (rr, c) not in one_set and 0 <= rr < rows and grid[rr][c] == 0:
                        new_cells.append((rr, c))
                if new_cells:
                    test_out = [row[:] for row in grid]
                    for r, c in new_cells:
                        test_out[r][c] = 2
                    return test_out
            return None

        for ex in train:
            if solve(ex['input']) != ex['output']:
                return None
        return solve(test_input)

    # --- _try_stack_3_panels_priority (6a11f6da) ---
    def _try_stack_3_panels_priority(self, train, test_input):
        """3 equal panels stacked vertically. Output = pixel from highest-priority non-zero panel."""
        from itertools import permutations
        def solve(grid, perm):
            rows, cols = len(grid), len(grid[0])
            if rows % 3 != 0:
                return None
            ph = rows // 3
            panels = [grid[i*ph:(i+1)*ph] for i in range(3)]
            out = [[0]*cols for _ in range(ph)]
            for r in range(ph):
                for c in range(cols):
                    for pi in perm:
                        if panels[pi][r][c] != 0:
                            out[r][c] = panels[pi][r][c]
                            break
            return out

        # Try all 6 permutations of priority order
        for perm in permutations([0, 1, 2]):
            ok = True
            for ex in train:
                if solve(ex['input'], perm) != ex['output']:
                    ok = False; break
            if ok:
                return solve(test_input, perm)
        return None

    # --- _try_stack_4_panels_priority (3d31c5b3) ---
    def _try_stack_4_panels_priority(self, train, test_input):
        """4 equal panels stacked vertically. Output = pixel from highest-priority non-zero panel."""
        from itertools import permutations
        def solve(grid, perm):
            rows, cols = len(grid), len(grid[0])
            if rows % 4 != 0:
                return None
            ph = rows // 4
            panels = [grid[i*ph:(i+1)*ph] for i in range(4)]
            out = [[0]*cols for _ in range(ph)]
            for r in range(ph):
                for c in range(cols):
                    for pi in perm:
                        if panels[pi][r][c] != 0:
                            out[r][c] = panels[pi][r][c]
                            break
            return out

        for perm in permutations([0, 1, 2, 3]):
            ok = True
            for ex in train:
                if solve(ex['input'], perm) != ex['output']:
                    ok = False; break
            if ok:
                return solve(test_input, perm)
        return None

    # --- _try_tile_shape_by_marker_count (4852f2fa) ---
    def _try_tile_shape_by_marker_count(self, train, test_input):
        """Find a shape of one color and count markers of another color. Tile shape N times horizontally."""
        def solve(grid):
            rows, cols = len(grid), len(grid[0])
            from collections import Counter
            cnt = Counter(v for row in grid for v in row if v != 0)
            if len(cnt) != 2:
                return None
            colors = cnt.most_common()
            shape_color = colors[0][0]
            marker_color = colors[1][0]
            n_markers = colors[1][1]
            # Extract shape bbox
            cells = [(r,c) for r in range(rows) for c in range(cols) if grid[r][c] == shape_color]
            if not cells:
                return None
            mr = min(r for r,c in cells); mxr = max(r for r,c in cells)
            mc = min(c for r,c in cells); mxc = max(c for r,c in cells)
            sh = mxr - mr + 1; sw = mxc - mc + 1
            shape = [[grid[mr+i][mc+j] for j in range(sw)] for i in range(sh)]
            # Tile horizontally
            out = [[0]*(sw*n_markers) for _ in range(sh)]
            for t in range(n_markers):
                for i in range(sh):
                    for j in range(sw):
                        out[i][t*sw+j] = shape[i][j]
            return out

        for ex in train:
            if solve(ex['input']) != ex['output']:
                return None
        return solve(test_input)

    # --- _try_extract_shape_with_8_marker (5117e062) ---
    def _try_extract_shape_with_8_marker(self, train, test_input):
        """Find the shape containing an 8. Extract bbox, replace 8 with shape's color."""
        def solve(grid):
            rows, cols = len(grid), len(grid[0])
            # Find 8 positions
            eights = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == 8]
            if len(eights) != 1:
                return None
            er, ec = eights[0]
            # Find the shape surrounding the 8: look at 4-neighbors for non-0 non-8 color
            shape_color = None
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = er+dr, ec+dc
                if 0<=nr<rows and 0<=nc<cols and grid[nr][nc] not in (0, 8):
                    shape_color = grid[nr][nc]
                    break
            if shape_color is None:
                return None
            # BFS to find shape cells (including 8 position)
            visited = [[False]*cols for _ in range(rows)]
            queue = [(er, ec)]
            visited[er][ec] = True
            cells = [(er, ec)]
            while queue:
                cr, cc = queue.pop(0)
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr, nc = cr+dr, cc+dc
                    if 0<=nr<rows and 0<=nc<cols and not visited[nr][nc] and grid[nr][nc] in (shape_color, 8):
                        visited[nr][nc] = True
                        queue.append((nr, nc))
                        cells.append((nr, nc))
            mr = min(r for r,c in cells); mxr = max(r for r,c in cells)
            mc = min(c for r,c in cells); mxc = max(c for r,c in cells)
            cs = set(cells)
            out = [[0]*(mxc-mc+1) for _ in range(mxr-mr+1)]
            for r, c in cells:
                out[r-mr][c-mc] = shape_color
            return out

        for ex in train:
            if solve(ex['input']) != ex['output']:
                return None
        return solve(test_input)

    # --- _try_merge_complementary_shapes (681b3aeb) ---
    def _try_merge_complementary_shapes(self, train, test_input):
        """Two shapes that together tile a rectangle. Output = merged rectangle with each cell colored."""
        def solve(grid):
            rows, cols = len(grid), len(grid[0])
            # Find non-zero connected components
            visited = [[False]*cols for _ in range(rows)]
            shapes = []
            for r in range(rows):
                for c in range(cols):
                    if grid[r][c] != 0 and not visited[r][c]:
                        color = grid[r][c]
                        queue = [(r, c)]; visited[r][c] = True; cells = [(r, c)]
                        while queue:
                            cr, cc = queue.pop(0)
                            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                                nr, nc = cr+dr, cc+dc
                                if 0<=nr<rows and 0<=nc<cols and not visited[nr][nc] and grid[nr][nc] == color:
                                    visited[nr][nc] = True; queue.append((nr, nc)); cells.append((nr, nc))
                        shapes.append((color, cells))
            if len(shapes) != 2:
                return None
            c1, cells1 = shapes[0]; c2, cells2 = shapes[1]
            # Normalize each shape to (0,0) origin
            def normalize(cells):
                mr = min(r for r,c in cells); mc = min(c for r,c in cells)
                return frozenset((r-mr, c-mc) for r,c in cells)
            n1 = normalize(cells1); n2 = normalize(cells2)
            h1 = max(r for r,c in n1)+1; w1 = max(c for r,c in n1)+1
            h2 = max(r for r,c in n2)+1; w2 = max(c for r,c in n2)+1
            # Try all offsets of shape2 relative to shape1 so they tile a rectangle
            for dr in range(-h2, h1+1):
                for dc in range(-w2, w1+1):
                    shifted2 = frozenset((r+dr, c+dc) for r,c in n2)
                    combined = n1 | shifted2
                    if n1 & shifted2:
                        continue  # overlap
                    mr = min(r for r,c in combined); mc = min(c for r,c in combined)
                    mxr = max(r for r,c in combined); mxc = max(c for r,c in combined)
                    h = mxr-mr+1; w = mxc-mc+1
                    if len(combined) == h * w:
                        # Full rectangle
                        out = [[0]*w for _ in range(h)]
                        for r, c in n1:
                            out[r-mr][c-mc] = c1
                        for r, c in shifted2:
                            out[r-mr][c-mc] = c2
                        return out
            return None

        for ex in train:
            if solve(ex['input']) != ex['output']:
                return None
        return solve(test_input)

    # --- _try_grid_cell_marker_count_threshold (6773b310) ---
    def _try_grid_cell_marker_count_threshold(self, train, test_input):
        """Grid divided by separator lines into NxN cells. Count markers per cell. Output 1 if count >= threshold."""
        def solve(grid, sep_color, marker_color, threshold):
            rows, cols = len(grid), len(grid[0])
            # Find separator rows and cols
            sep_rows = [r for r in range(rows) if all(grid[r][c] == sep_color for c in range(cols))]
            sep_cols = [c for c in range(cols) if all(grid[r][c] == sep_color for r in range(rows))]
            if not sep_rows or not sep_cols:
                return None
            # Build cell boundaries
            row_bounds = [-1] + sep_rows + [rows]
            col_bounds = [-1] + sep_cols + [cols]
            nr = len(row_bounds) - 1
            nc = len(col_bounds) - 1
            out = [[0]*nc for _ in range(nr)]
            for ri in range(nr):
                for ci in range(nc):
                    r1, r2 = row_bounds[ri]+1, row_bounds[ri+1]
                    c1, c2 = col_bounds[ci]+1, col_bounds[ci+1]
                    count = sum(1 for r in range(r1, r2) for c in range(c1, c2) if grid[r][c] == marker_color)
                    out[ri][ci] = 1 if count >= threshold else 0
            return out

        # Learn parameters from training
        for sep_color in range(10):
            for marker_color in range(10):
                if sep_color == marker_color:
                    continue
                for threshold in [2, 3, 1]:
                    ok = True
                    for ex in train:
                        r = solve(ex['input'], sep_color, marker_color, threshold)
                        if r != ex['output']:
                            ok = False; break
                    if ok:
                        return solve(test_input, sep_color, marker_color, threshold)
        return None

    # --- _try_most_common_cross_arm_color (642d658d) ---
    def _try_most_common_cross_arm_color(self, train, test_input):
        """Find plus-crosses with center=4. Count arm colors. Output = 1x1 of most common arm color."""
        def solve(grid):
            rows, cols = len(grid), len(grid[0])
            from collections import Counter
            arm_colors = Counter()
            for r in range(1, rows-1):
                for c in range(1, cols-1):
                    if grid[r][c] == 4:
                        arms = [grid[r-1][c], grid[r+1][c], grid[r][c-1], grid[r][c+1]]
                        arm_set = set(arms)
                        arm_set.discard(0)
                        # Find the non-bg arm colors
                        bg_colors = set()
                        for row in grid:
                            for v in row:
                                bg_colors.add(v)
                        # All 4 arms same non-4 color?
                        non4 = [a for a in arms if a != 4 and a != 0]
                        if len(non4) == 4 and len(set(non4)) == 1:
                            arm_colors[non4[0]] += 1
            if not arm_colors:
                return None
            return [[arm_colors.most_common(1)[0][0]]]

        for ex in train:
            if solve(ex['input']) != ex['output']:
                return None
        return solve(test_input)

    # --- _try_nested_rect_colors_by_size (68bc2e87) ---
    def _try_nested_rect_colors_by_size(self, train, test_input):
        """Nested rectangles. Output = colors peeled from outside inward."""
        def solve(grid):
            rows, cols = len(grid), len(grid[0])
            from collections import Counter
            cnt = Counter(v for row in grid for v in row)
            bg = cnt.most_common(1)[0][0]
            # Peel layers: find outermost non-bg, record, treat as bg, repeat
            working = [row[:] for row in grid]
            result = []
            seen = {bg}
            for _ in range(10):
                # Find the color closest to the edges
                best_color = None
                best_dist = float('inf')
                for r in range(rows):
                    for c in range(cols):
                        v = working[r][c]
                        if v in seen:
                            continue
                        dist = min(r, rows-1-r, c, cols-1-c)
                        if dist < best_dist or (dist == best_dist and best_color is None):
                            best_dist = dist
                            best_color = v
                if best_color is None:
                    break
                result.append([best_color])
                seen.add(best_color)
            return result if result else None

        for ex in train:
            if solve(ex['input']) != ex['output']:
                return None
        return solve(test_input)

    # --- _try_fill_frame_holes_with_patch_color (42f14c03) ---
    def _try_fill_frame_holes_with_patch_color(self, train, test_input):
        """Find a frame shape with holes. Find scattered patches of another color. Fill holes with patch color."""
        def solve(grid):
            rows, cols = len(grid), len(grid[0])
            from collections import Counter
            cnt = Counter(v for row in grid for v in row)
            bg = cnt.most_common(1)[0][0]
            # Find non-bg components
            visited = [[False]*cols for _ in range(rows)]
            comps = []
            for r in range(rows):
                for c in range(cols):
                    if grid[r][c] != bg and not visited[r][c]:
                        color = grid[r][c]
                        queue = [(r, c)]; visited[r][c] = True; cells = [(r, c)]
                        while queue:
                            cr, cc = queue.pop(0)
                            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                                nr, nc = cr+dr, cc+dc
                                if 0<=nr<rows and 0<=nc<cols and not visited[nr][nc] and grid[nr][nc] == color:
                                    visited[nr][nc] = True; queue.append((nr, nc)); cells.append((nr, nc))
                        comps.append((color, cells))
            # Find the "frame" = largest component with interior holes
            frame = None
            fill_color = None
            for color, cells in comps:
                cs = set(cells)
                mr = min(r for r,c in cells); mxr = max(r for r,c in cells)
                mc = min(c for r,c in cells); mxc = max(c for r,c in cells)
                h, w = mxr-mr+1, mxc-mc+1
                if h < 3 or w < 3:
                    continue
                holes = [(r,c) for r in range(mr, mxr+1) for c in range(mc, mxc+1) if (r,c) not in cs and grid[r][c] == bg]
                if holes and len(holes) < h*w//2:
                    frame = (color, cells, mr, mc, mxr, mxc, holes)
                    break
            if frame is None:
                return None
            frame_color, frame_cells, mr, mc, mxr, mxc, holes = frame
            # Find fill color: non-bg, non-frame color with fewest total cells
            from collections import defaultdict
            color_total = defaultdict(int)
            for color, cells in comps:
                if color != frame_color:
                    color_total[color] += len(cells)
            if not color_total:
                return None
            fill_color = min(color_total, key=lambda c: color_total[c])
            if fill_color is None:
                return None
            # Build output: frame bbox with holes filled
            h, w = mxr-mr+1, mxc-mc+1
            out = [[bg]*w for _ in range(h)]
            for r, c in frame_cells:
                out[r-mr][c-mc] = frame_color
            for r, c in holes:
                out[r-mr][c-mc] = fill_color
            return out

        for ex in train:
            if solve(ex['input']) != ex['output']:
                return None
        return solve(test_input)
