import ast
import json
import math
import time
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
    for node in TREE.body if isinstance(node, ast.FunctionDef)
}


def config(weak=0.50, anchor_gap=12, speed=0.6):
    return {
        "top_k": 5,
        "weak_threshold": weak,
        "max_anchor_gap": anchor_gap,
        "max_center_speed": speed,
        "max_candidate_gap": 3,
        "min_recovered_frames": 3,
        "min_gap_coverage": 0.5,
    }


def item(x, score=0.52, candidate_idx=0):
    return {
        "bbox": [x, 0, x + 10, 10],
        "score": score,
        "candidate_idx": candidate_idx,
    }


def namespace():
    ns = {
        "json": json,
        "math": math,
        "time": time,
        "YOLO_ONLY_THRESHOLD": 0.2,
        "PRODUCTION_TEMPORAL_CONFIG": {"max_gap": 7},
        "RESIDUAL_RECOVERY_TOP_K": 5,
        "RESIDUAL_RECOVERY_WEAK_THRESHOLD_GRID": [0.50, 0.52, 0.54],
        "RESIDUAL_RECOVERY_MAX_ANCHOR_GAP_GRID": [12, 20],
        "RESIDUAL_RECOVERY_MAX_CENTER_SPEED_GRID": [0.6, 1.0],
        "RESIDUAL_RECOVERY_MAX_CANDIDATE_GAP": 3,
        "RESIDUAL_RECOVERY_MIN_RECOVERED_FRAMES": 3,
        "RESIDUAL_RECOVERY_MIN_GAP_COVERAGE": 0.5,
    }
    for name in [
        "iou_box",
        "compute_st_iou_video",
        "prediction_frame_maps",
        "gap_motion_metrics",
        "gap_is_plausible",
        "residual_recovery_config_grid",
        "residual_recovery_config_key",
        "recover_candidate_chain_between_anchors",
        "render_anchor_preserving_residual",
    ]:
        exec(NODES[name], ns)
    return ns


class AnchorPreservingResidualV2Tests(unittest.TestCase):
    def test_grid_is_small_and_contains_exact_noop_controls(self):
        ns = namespace()
        grid = ns["residual_recovery_config_grid"]()
        self.assertEqual(len(grid), 12)
        self.assertEqual(
            len({ns["residual_recovery_config_key"](row) for row in grid}), 12
        )
        self.assertEqual(sum(row["weak_threshold"] == 0.54 for row in grid), 4)

    def test_dense_motion_consistent_chain_is_recovered(self):
        ns = namespace()
        candidates = {frame: [item(frame)] for frame in range(1, 10)}
        recovered, diagnostics = ns["recover_candidate_chain_between_anchors"](
            candidates, 0, item(0, 0.8), 10, item(10, 0.8), config()
        )
        self.assertEqual(set(recovered), set(range(1, 10)))
        self.assertEqual(diagnostics["reason"], "recovered")
        self.assertEqual(diagnostics["coverage"], 1.0)

    def test_camera_discontinuity_resets_instead_of_bridging(self):
        ns = namespace()
        candidates = {frame: [item(frame)] for frame in range(1, 10)}
        recovered, diagnostics = ns["recover_candidate_chain_between_anchors"](
            candidates, 0, item(0, 0.8), 10, item(1000, 0.8), config()
        )
        self.assertEqual(recovered, {})
        self.assertEqual(diagnostics["reason"], "scene_discontinuity_reset")

    def test_sparse_chain_is_not_added(self):
        ns = namespace()
        recovered, diagnostics = ns["recover_candidate_chain_between_anchors"](
            {4: [item(4)]}, 0, item(0, 0.8), 10, item(10, 0.8), config()
        )
        self.assertEqual(recovered, {})
        self.assertIn(
            diagnostics["reason"],
            {"no_anchor_to_anchor_path", "insufficient_path_coverage"},
        )

    def test_render_never_overwrites_or_deletes_production_bbox(self):
        ns = namespace()
        production = {
            "mean_st_iou": 1.0,
            "per_video": {"Object_0": 1.0},
            "detected_frames": 1,
            "predictions": [{
                "video_id": "Object_0",
                "detections": [{"bboxes": [
                    {"frame": 0, "x1": 0, "y1": 0, "x2": 10, "y2": 10}
                ]}],
            }],
        }
        gt = {"Object_0": {
            0: [0, 0, 10, 10], 1: [1, 0, 11, 10]
        }}
        result = ns["render_anchor_preserving_residual"](
            production,
            {"Object_0": {0: item(100), 1: item(1)}},
            gt,
        )
        boxes = result["predictions"][0]["detections"][0]["bboxes"]
        self.assertEqual(boxes[0], production["predictions"][0]["detections"][0]["bboxes"][0])
        self.assertEqual([box["frame"] for box in boxes], [0, 1])
        self.assertEqual(production["detected_frames"], 1)

    def test_empty_recovery_is_exact_prediction_control(self):
        ns = namespace()
        production = {
            "mean_st_iou": 1.0,
            "per_video": {"Object_0": 1.0},
            "detected_frames": 1,
            "predictions": [{
                "video_id": "Object_0",
                "detections": [{"bboxes": [
                    {"frame": 0, "x1": 0, "y1": 0, "x2": 10, "y2": 10}
                ]}],
            }],
        }
        result = ns["render_anchor_preserving_residual"](
            production, {"Object_0": {}}, {"Object_0": {0: [0, 0, 10, 10]}}
        )
        self.assertEqual(result["predictions"], production["predictions"])

    def test_weak_equals_strong_short_circuits_to_noop(self):
        ns = namespace()
        exec(NODES["build_residual_recovery_for_video"], ns)
        recovered, diagnostics = ns["build_residual_recovery_for_video"](
            {"has_siamese": True}, {}, {}, 0.54, config(weak=0.54)
        )
        self.assertEqual(recovered, {})
        self.assertEqual(diagnostics["reason_counts"], {"exact_noop_control": 1})

    def test_recovery_logic_does_not_read_gt_identity_or_object_name(self):
        source = "\n".join(NODES[name] for name in [
            "recover_candidate_chain_between_anchors",
            "build_residual_recovery_for_video",
        ])
        self.assertNotIn("gt_map", source)
        self.assertNotIn("identity", source)
        self.assertNotIn("Cardboard", source)

    def test_residual_v2_is_archived(self):
        assignments = {}
        for node in TREE.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                if isinstance(target, ast.Name) and target.id.startswith("RUN_"):
                    try:
                        assignments[target.id] = ast.literal_eval(node.value)
                    except (ValueError, TypeError):
                        pass
        self.assertFalse(assignments["RUN_RESIDUAL_RECOVERY_EXPERIMENT"])

    def test_identity_cv_runner_keeps_pairs_and_requires_enabled_consensus(self):
        video_ids = [
            "BlackBox_0", "BlackBox_1", "CardboardBox_0",
            "CardboardBox_1", "LifeJacket_0", "LifeJacket_1",
        ]
        production = {
            "mean_st_iou": 0.50,
            "detected_frames": 6,
            "per_video": {video_id: 0.50 for video_id in video_ids},
            "predictions": [{"control": True}],
        }
        enabled = config(weak=0.50)
        control = config(weak=0.54)
        ns = {
            "json": json,
            "tqdm": lambda values, desc=None: values,
            "RESIDUAL_RECOVERY_MIN_MEAN_CV_DELTA": 0.003,
            "RESIDUAL_RECOVERY_MIN_IMPROVED_IDENTITIES": 2,
            "RESIDUAL_RECOVERY_MAX_WORST_IDENTITY_DROP": 0.010,
            "RESIDUAL_RECOVERY_MIN_CONFIG_VOTES": 2,
            "RESIDUAL_RECOVERY_MIN_REPLAY_FPS": 25.0,
            "residual_recovery_config_grid": lambda: [enabled, control],
        }
        for name in [
            "identity_id_from_video_id",
            "build_identity_folds",
            "mean_row_score",
            "residual_recovery_config_key",
            "residual_recovery_result_row",
            "residual_recovery_config_from_row",
            "select_residual_recovery_row",
            "select_voted_residual_recovery_config",
            "run_residual_recovery_identity_cv",
        ]:
            exec(NODES[name], ns)

        def fake_evaluate(cache, gt, weights, threshold, baseline, selected):
            is_enabled = selected["weak_threshold"] < threshold
            score = 0.51 if is_enabled else 0.50
            per_video_diagnostics = {
                video_id: {
                    "strong_anchor_count": 2,
                    "candidate_gap_count": 1,
                    "recovered_gap_count": int(is_enabled),
                    "recovered_frame_count": int(is_enabled),
                    "scene_discontinuity_reset_count": 0,
                    "reason_counts": {},
                }
                for video_id in video_ids
            }
            return {
                "mean_st_iou": score,
                "detected_frames": 6 + int(is_enabled) * 6,
                "per_video": {video_id: score for video_id in video_ids},
                "predictions": ([{"enabled": True}] if is_enabled else baseline["predictions"]),
                "residual_diagnostics": {
                    "replay_fps": 100.0,
                    "recovered_frame_count": int(is_enabled) * 6,
                    "recovered_gap_count": int(is_enabled) * 6,
                    "scene_discontinuity_reset_count": 0,
                    "per_video": per_video_diagnostics,
                },
            }

        ns["evaluate_residual_recovery_config"] = fake_evaluate
        cache = {
            "signature": {"video_folders": video_ids},
            "videos": {video_id: {} for video_id in video_ids},
        }
        _, folds, _, diagnostics, recommended = ns[
            "run_residual_recovery_identity_cv"
        ](cache, {}, {}, 0.54, production)
        self.assertEqual(len(folds), 3)
        self.assertTrue(diagnostics["exact_noop_control_prediction_parity"])
        self.assertEqual(diagnostics["recommendation_votes"], 3)
        self.assertTrue(diagnostics["promotion_passed"])
        self.assertEqual(recommended["mean_st_iou"], 0.51)
        for fold in folds:
            self.assertEqual(len(fold["heldout_videos"].split("|")), 2)


if __name__ == "__main__":
    unittest.main()
