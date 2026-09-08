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


def graph_namespace():
    namespace = {
        "math": math,
        "time": time,
        "YOLO_ONLY_THRESHOLD": 0.2,
        "TRACKLET_GRAPH_TOP_K_GRID": [3, 5],
        "TRACKLET_GRAPH_MOTION_WEIGHT_GRID": [0.02, 0.05, 0.10],
        "TRACKLET_GRAPH_STATE_CHANGE_PENALTY_GRID": [0.02, 0.05],
        "TRACKLET_GRAPH_MAX_CENTER_SPEED_GRID": [0.4, 0.6],
        "TRACKLET_GRAPH_CAMERA_COMPENSATION_GRID": [False, True],
        "TRACKLET_GRAPH_EMISSION_SCALE": 1.0,
        "TRACKLET_GRAPH_MAX_GLOBAL_SHIFT": 256.0,
        "TRACKLET_GRAPH_MIN_GLOBAL_SHIFT_MATCHES": 3,
    }
    order = [
        "fuse_candidate_score",
        "rank_top_k_video_frames",
        "gap_motion_metrics",
        "tracklet_graph_config_grid",
        "tracklet_graph_config_key",
        "median_value",
        "bbox_center",
        "estimate_candidate_global_shift",
        "shifted_bbox",
        "tracklet_transition_score",
        "decode_candidate_tracklet",
    ]
    for name in order:
        exec(NODES[name], namespace)
    return namespace


def raw_candidate(x, score):
    return {
        "bbox": [x, 0, x + 10, 10],
        "scores": [score, score, score],
    }


def graph_config(camera=False, state_penalty=0.05, max_speed=0.4):
    return {
        "top_k": 3,
        "motion_weight": 0.05,
        "state_change_penalty": state_penalty,
        "max_center_speed": max_speed,
        "camera_compensation": camera,
        "emission_scale": 1.0,
    }


class CandidateTrackletGraphTests(unittest.TestCase):
    def test_fixed_grid_has_48_unique_global_configs(self):
        ns = graph_namespace()
        grid = ns["tracklet_graph_config_grid"]()
        self.assertEqual(len(grid), 48)
        self.assertEqual(
            len({ns["tracklet_graph_config_key"](config) for config in grid}), 48
        )

    def test_viterbi_rescues_coherent_below_threshold_candidate(self):
        ns = graph_namespace()
        video = {
            "max_frame_idx": 2,
            "has_siamese": True,
            "has_color": False,
            "frames": {
                "0": [raw_candidate(0, 0.80)],
                "1": [raw_candidate(1, 0.50)],
                "2": [raw_candidate(2, 0.80)],
            },
        }
        selected, diagnostics = ns["decode_candidate_tracklet"](
            video, {"yolo": 1.0, "siamese": 0.0, "color": 0.0},
            0.54, graph_config(),
        )
        self.assertEqual(set(selected), {0, 1, 2})
        self.assertEqual(diagnostics["rescued_below_threshold"], 1)

    def test_isolated_below_threshold_candidate_prefers_absent(self):
        ns = graph_namespace()
        video = {
            "max_frame_idx": 0,
            "has_siamese": True,
            "has_color": False,
            "frames": {"0": [raw_candidate(0, 0.50)]},
        }
        selected, diagnostics = ns["decode_candidate_tracklet"](
            video, {"yolo": 1.0, "siamese": 0.0, "color": 0.0},
            0.54, graph_config(),
        )
        self.assertEqual(selected, {})
        self.assertEqual(diagnostics["absent_frames"], 1)

    def test_implausible_jump_cannot_form_one_candidate_path(self):
        ns = graph_namespace()
        video = {
            "max_frame_idx": 1,
            "has_siamese": True,
            "has_color": False,
            "frames": {
                "0": [raw_candidate(0, 0.80)],
                "1": [raw_candidate(1000, 0.80)],
            },
        }
        selected, _ = ns["decode_candidate_tracklet"](
            video, {"yolo": 1.0, "siamese": 0.0, "color": 0.0},
            0.54, graph_config(),
        )
        self.assertEqual(len(selected), 1)

    def test_candidate_translation_proxy_uses_mutual_nearest_median(self):
        ns = graph_namespace()
        previous = [
            {"bbox": [x, 0, x + 10, 10]} for x in (0, 100, 200)
        ]
        current = [
            {"bbox": [x + 5, 3, x + 15, 13]} for x in (0, 100, 200)
        ]
        dx, dy, matches = ns["estimate_candidate_global_shift"](
            previous, current, max_shift=20, min_matches=3
        )
        self.assertEqual(matches, 3)
        self.assertAlmostEqual(dx, 5.0)
        self.assertAlmostEqual(dy, 3.0)

    def test_inference_functions_do_not_read_identity_or_gt(self):
        for name in [
            "estimate_candidate_global_shift",
            "tracklet_transition_score",
            "decode_candidate_tracklet",
        ]:
            source = NODES[name]
            self.assertNotIn("identity", source)
            self.assertNotIn("gt_map", source)
            self.assertNotIn("Cardboard", source)

    def test_tracklet_experiment_is_disabled_after_completed_run(self):
        assignments = {}
        for node in TREE.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                if isinstance(target, ast.Name) and target.id.startswith(("RUN_", "USE_TRACKLET")):
                    try:
                        assignments[target.id] = ast.literal_eval(node.value)
                    except (ValueError, TypeError):
                        pass
        self.assertFalse(assignments["RUN_TRACKLET_GRAPH_EXPERIMENT"])
        self.assertFalse(assignments["USE_TRACKLET_GRAPH"])

    def test_identity_cv_runner_holds_pairs_and_requires_stable_global_vote(self):
        video_ids = [
            "BlackBox_0", "BlackBox_1", "CardboardBox_0",
            "CardboardBox_1", "LifeJacket_0", "LifeJacket_1",
        ]
        baseline = {
            "mean_st_iou": 0.50,
            "detected_frames": 6,
            "per_video": {video_id: 0.50 for video_id in video_ids},
            "predictions": [{"control": True}],
        }
        config_good = graph_config(camera=False)
        config_bad = {**config_good, "motion_weight": 0.10, "camera_compensation": True}

        namespace = {
            "json": json,
            "tqdm": lambda values, desc=None: values,
            "TRACKLET_GRAPH_MIN_MEAN_CV_DELTA": 0.003,
            "TRACKLET_GRAPH_MIN_IMPROVED_IDENTITIES": 2,
            "TRACKLET_GRAPH_MAX_WORST_IDENTITY_DROP": 0.010,
            "TRACKLET_GRAPH_MIN_CONFIG_VOTES": 2,
            "TRACKLET_GRAPH_MIN_REPLAY_FPS": 25.0,
            "tracklet_graph_config_grid": lambda: [config_good, config_bad],
            "evaluate_config": lambda *args, **kwargs: baseline,
        }
        for name in [
            "identity_id_from_video_id",
            "build_identity_folds",
            "mean_row_score",
            "tracklet_graph_config_key",
            "tracklet_graph_result_row",
            "tracklet_graph_config_from_row",
            "select_tracklet_graph_row",
            "select_voted_tracklet_graph_config",
            "run_tracklet_graph_identity_cv",
        ]:
            exec(NODES[name], namespace)

        def fake_graph_evaluation(cache, gt, weights, threshold, config, temporal):
            score = 0.51 if config["motion_weight"] == 0.05 else 0.49
            per_video_diagnostics = {
                video_id: {
                    "total_frames": 1,
                    "candidate_frames": 1,
                    "absent_frames": 0,
                    "rescued_below_threshold": 1,
                    "suppressed_above_threshold": 0,
                    "camera_shift_frames": 0,
                    "mean_camera_shift_pixels": 0.0,
                }
                for video_id in video_ids
            }
            return {
                "mean_st_iou": score,
                "detected_frames": 6,
                "per_video": {video_id: score for video_id in video_ids},
                "predictions": [{"score": score}],
                "graph_diagnostics": {
                    "replay_fps": 100.0,
                    "candidate_frames_before_temporal": 6,
                    "rescued_below_threshold": 6,
                    "suppressed_above_threshold": 0,
                    "camera_shift_frames": 0,
                    "per_video": per_video_diagnostics,
                },
            }

        namespace["evaluate_tracklet_graph_config"] = fake_graph_evaluation
        cache = {
            "signature": {"video_folders": video_ids},
            "videos": {video_id: {} for video_id in video_ids},
        }
        _, folds, videos, diagnostics, recommended = namespace[
            "run_tracklet_graph_identity_cv"
        ](cache, {}, {}, 0.54, baseline, {"max_gap": 7})
        self.assertEqual(len(folds), 3)
        self.assertEqual(len(videos), 6)
        self.assertTrue(diagnostics["production_control_exact_prediction_parity"])
        self.assertEqual(diagnostics["recommendation_votes"], 3)
        self.assertTrue(diagnostics["promotion_passed"])
        self.assertEqual(recommended["mean_st_iou"], 0.51)
        for fold in folds:
            self.assertEqual(len(fold["heldout_videos"].split("|")), 2)


if __name__ == "__main__":
    unittest.main()
