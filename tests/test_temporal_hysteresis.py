import ast
import bisect
import json
import math
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "06-inference-main.ipynb"


def notebook_source():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
        and not any(
            line.lstrip().startswith(("!", "%"))
            for line in "".join(cell.get("source", [])).splitlines()
        )
    )


SOURCE = notebook_source()
TREE = ast.parse(SOURCE)
NODES = {
    node.name: ast.get_source_segment(SOURCE, node)
    for node in TREE.body
    if isinstance(node, ast.FunctionDef)
}


def load_namespace(*extra_names):
    namespace = {
        "math": math,
        "bisect": bisect,
        "YOLO_ONLY_THRESHOLD": 0.2,
        "TEMPORAL_MAX_GAP": 5,
        "TEMPORAL_MIN_SEG": 3,
        "PRODUCTION_TEMPORAL_CONFIG": {
            "max_gap": 7,
            "min_seg_len": 24,
            "max_center_speed": 0.4,
            "max_log_scale_speed": None,
        },
        "HYSTERESIS_STRONG_THRESHOLD": 0.54,
        "HYSTERESIS_WEAK_THRESHOLD_GRID": [0.46, 0.48, 0.50, 0.52, 0.54],
        "HYSTERESIS_ANCHOR_SPAN_GRID": [8, 16, 32],
        "HYSTERESIS_MAX_CENTER_SPEED": 0.4,
    }
    order = [
        "iou_box",
        "compute_st_iou_video",
        "gap_motion_metrics",
        "gap_is_plausible",
        "temporal_smooth_detections",
        "temporal_smooth_detections_gated",
        "bracketed_hysteresis_accept",
        "hysteresis_config_grid",
        "render_ranked_predictions",
        "render_hysteresis_predictions",
    ]
    for name in order + list(extra_names):
        if name not in namespace:
            exec(NODES[name], namespace)
    return namespace


def item(x1, score):
    return {"bbox": [x1, 0, x1 + 10, 10], "score": score}


class TemporalHysteresisTests(unittest.TestCase):
    def test_bracketed_motion_consistent_weak_candidate_is_accepted(self):
        ns = load_namespace()
        ranked = {0: item(0, 0.8), 2: item(2, 0.5), 4: item(4, 0.8)}
        accepted = ns["bracketed_hysteresis_accept"](
            ranked, 0.46, 0.54, max_anchor_span=8, max_center_speed=0.4
        )
        self.assertEqual(set(accepted), {0, 2, 4})

    def test_weak_candidate_without_two_anchors_is_rejected(self):
        ns = load_namespace()
        ranked = {0: item(0, 0.8), 2: item(2, 0.5)}
        accepted = ns["bracketed_hysteresis_accept"](
            ranked, 0.46, 0.54, max_anchor_span=8, max_center_speed=0.4
        )
        self.assertEqual(set(accepted), {0})

    def test_large_motion_jump_is_rejected(self):
        ns = load_namespace()
        ranked = {0: item(0, 0.8), 2: item(1000, 0.5), 4: item(4, 0.8)}
        accepted = ns["bracketed_hysteresis_accept"](
            ranked, 0.46, 0.54, max_anchor_span=8, max_center_speed=0.4
        )
        self.assertEqual(set(accepted), {0, 4})

    def test_weak_equals_strong_is_exact_threshold_control(self):
        ns = load_namespace()
        ranked = {
            0: item(0, 0.8),
            1: item(1, 0.53),
            2: item(2, 0.54),
            3: item(3, 0.2),
            4: item(4, 0.9),
        }
        accepted = ns["bracketed_hysteresis_accept"](
            ranked, 0.54, 0.54, max_anchor_span=32, max_center_speed=0.4
        )
        self.assertEqual(accepted, {0: ranked[0], 2: ranked[2], 4: ranked[4]})

    def test_render_control_matches_production_temporal_output(self):
        ns = load_namespace()
        video_id = "Object_0"
        ranked = {
            frame: item(frame, 0.8 if frame < 30 else 0.53)
            for frame in range(35)
        }
        cache = {
            "signature": {"video_folders": [video_id]},
            "videos": {
                video_id: {
                    "has_siamese": True,
                    "has_color": False,
                    "max_frame_idx": 34,
                }
            },
        }
        gt = {video_id: {frame: [frame, 0, frame + 10, 10] for frame in range(35)}}
        production = ns["render_ranked_predictions"](
            cache,
            {video_id: ranked},
            gt,
            0.54,
            temporal_config=ns["PRODUCTION_TEMPORAL_CONFIG"],
        )
        control = ns["render_hysteresis_predictions"](
            cache,
            {video_id: ranked},
            gt,
            {
                "weak_threshold": 0.54,
                "max_anchor_span": 8,
                "strong_threshold": 0.54,
                "max_center_speed": 0.4,
            },
        )
        self.assertEqual(control["predictions"], production["predictions"])
        self.assertEqual(control["per_video"], production["per_video"])
        self.assertEqual(control["rescued_frames"], 0)

    def test_grid_is_fixed_small_and_contains_controls(self):
        ns = load_namespace()
        grid = ns["hysteresis_config_grid"]()
        self.assertEqual(len(grid), 15)
        self.assertEqual(len({tuple(config.items()) for config in grid}), 15)
        controls = [config for config in grid if config["weak_threshold"] == 0.54]
        self.assertEqual(len(controls), 3)

    def test_inference_rule_does_not_read_video_or_identity_names(self):
        source = NODES["bracketed_hysteresis_accept"]
        self.assertNotIn("video_id", source)
        self.assertNotIn("identity", source)

    def test_hysteresis_is_only_active_experiment(self):
        tracked = {
            "RUN_CALIBRATION",
            "RUN_TEMPORAL_CALIBRATION",
            "RUN_RERANKING_EXPERIMENT",
            "RUN_ORACLE_DIAGNOSTICS",
            "RUN_HYSTERESIS_EXPERIMENT",
            "USE_TEMPORAL_HYSTERESIS",
        }
        assignments = {}
        for node in TREE.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                if isinstance(target, ast.Name) and target.id in tracked:
                    assignments[target.id] = ast.literal_eval(node.value)
        self.assertFalse(assignments["RUN_CALIBRATION"])
        self.assertFalse(assignments["RUN_TEMPORAL_CALIBRATION"])
        self.assertFalse(assignments["RUN_RERANKING_EXPERIMENT"])
        self.assertFalse(assignments["RUN_ORACLE_DIAGNOSTICS"])
        self.assertFalse(assignments["USE_TEMPORAL_HYSTERESIS"])
        self.assertFalse(assignments["RUN_HYSTERESIS_EXPERIMENT"])

    def test_identity_cv_runner_uses_heldout_pairs_and_control_guard(self):
        video_ids = [
            "BlackBox_0",
            "BlackBox_1",
            "CardboardBox_0",
            "CardboardBox_1",
            "LifeJacket_0",
            "LifeJacket_1",
        ]
        baseline = {
            "mean_st_iou": 0.5,
            "detected_frames": 6,
            "per_video": {video_id: 0.5 for video_id in video_ids},
            "predictions": [{"control": True}],
        }

        def fake_render(cache, ranked_by_video, gt_map, config):
            if config["weak_threshold"] == 0.54:
                return {
                    **baseline,
                    "rescued_frames": 0,
                    "rescued_per_video": {video_id: 0 for video_id in video_ids},
                }
            return {
                "mean_st_iou": 0.51,
                "detected_frames": 12,
                "per_video": {video_id: 0.51 for video_id in video_ids},
                "predictions": [{"control": False}],
                "rescued_frames": 6,
                "rescued_per_video": {video_id: 1 for video_id in video_ids},
            }

        namespace = {
            "HYSTERESIS_STRONG_THRESHOLD": 0.54,
            "HYSTERESIS_WEAK_THRESHOLD_GRID": [0.50, 0.54],
            "HYSTERESIS_ANCHOR_SPAN_GRID": [8],
            "HYSTERESIS_MAX_CENTER_SPEED": 0.4,
            "HYSTERESIS_MIN_MEAN_CV_DELTA": 0.003,
            "HYSTERESIS_MIN_IMPROVED_IDENTITIES": 2,
            "HYSTERESIS_MAX_WORST_IDENTITY_DROP": 0.010,
            "tqdm": lambda iterable, desc=None: iterable,
            "rank_video_frames": lambda video_cache, weights, include_debug=False: ({}, None),
            "render_hysteresis_predictions": fake_render,
        }
        function_order = [
            "identity_id_from_video_id",
            "build_identity_folds",
            "mean_row_score",
            "hysteresis_config_grid",
            "hysteresis_result_row",
            "hysteresis_config_from_row",
            "select_hysteresis_row",
            "select_voted_hysteresis_config",
            "run_hysteresis_identity_cv",
        ]
        for name in function_order:
            exec(NODES[name], namespace)
        cache = {
            "signature": {"video_folders": video_ids},
            "videos": {video_id: {} for video_id in video_ids},
        }
        _, folds, diagnostics, recommended = namespace["run_hysteresis_identity_cv"](
            cache, {}, {}, baseline
        )
        self.assertEqual(len(folds), 3)
        self.assertTrue(diagnostics["weak_equals_strong_exact_prediction_parity"])
        self.assertTrue(diagnostics["promotion_passed"])
        self.assertEqual(diagnostics["recommended_config"]["weak_threshold"], 0.50)
        self.assertEqual(diagnostics["recommendation_votes"], 3)
        self.assertEqual(recommended["mean_st_iou"], 0.51)
        for fold in folds:
            self.assertEqual(len(fold["heldout_videos"].split("|")), 2)


if __name__ == "__main__":
    unittest.main()
