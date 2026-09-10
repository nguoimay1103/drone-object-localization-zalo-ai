import ast
import json
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


def calibration_namespace():
    namespace = {
        "SCORE_CAL_YOLO_POWERS": [0.75, 1.0, 1.25, 1.5],
        "SCORE_CAL_YOLO_WEIGHTS": [0.25, 0.35, 0.425, 0.50, 0.60],
        "SCORE_CAL_COLOR_WEIGHTS": [0.0, 0.05, 0.10, 0.15, 0.20],
        "SCORE_CAL_THRESHOLDS": [round(value / 100, 2) for value in range(44, 65, 2)],
        "SCORE_CAL_MIN_SIAMESE_WEIGHT": 0.20,
        "SCORE_CAL_MIN_MEAN_CV_DELTA_VS_PRODUCTION": 0.003,
        "SCORE_CAL_MIN_IMPROVED_IDENTITIES": 2,
        "SCORE_CAL_MAX_WORST_IDENTITY_DROP": 0.010,
        "SCORE_CAL_MIN_CONFIG_VOTES": 2,
        "SCORE_CAL_MIN_AGGREGATE_FPS": 25.0,
        "WEIGHT_YOLO": 0.425,
        "WEIGHT_SIAMESE": 0.475,
        "WEIGHT_COLOR": 0.1,
        "MATCHING_THRESHOLD": 0.54,
    }
    order = [
        "identity_id_from_video_id",
        "build_identity_folds",
        "mean_row_score",
        "transform_yolo_score",
        "fuse_calibrated_candidate_score",
        "score_calibration_config_key",
        "score_calibration_config_grid",
        "score_calibration_config_from_row",
        "score_calibration_distance_from_production",
        "select_score_calibration_row",
        "select_voted_score_calibration_config",
    ]
    for name in order:
        exec(NODES[name], namespace)
    return namespace


class CheckpointScoreCalibrationTests(unittest.TestCase):
    def test_experiment_is_active_and_other_detector_ab_is_inactive(self):
        assignments = {}
        for node in TREE.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                if isinstance(target, ast.Name):
                    try:
                        assignments[target.id] = ast.literal_eval(node.value)
                    except (ValueError, TypeError):
                        pass
        self.assertFalse(assignments["RUN_CHECKPOINT_SCORE_CALIBRATION"])
        self.assertFalse(assignments["RUN_P2_DETECTOR_AB"])
        self.assertFalse(assignments["RUN_CALIBRATION"])
        self.assertFalse(assignments["RUN_TEMPORAL_CALIBRATION"])

    def test_fixed_grid_has_expected_size_and_exact_production_control(self):
        ns = calibration_namespace()
        grid = ns["score_calibration_config_grid"]()
        self.assertEqual(len(grid), 1100)
        keys = [ns["score_calibration_config_key"](config) for config in grid]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertIn((1.0, 0.425, 0.475, 0.1, 0.54), set(keys))

    def test_gamma_one_fusion_is_exact_legacy_formula(self):
        ns = calibration_namespace()
        weights = {"yolo": 0.425, "siamese": 0.475, "color": 0.1}
        scores = [0.73, 0.64, 0.31]
        expected = sum(weight * score for weight, score in zip(weights.values(), scores))
        actual = ns["fuse_calibrated_candidate_score"](
            scores, weights, 1.0, True, True
        )
        self.assertAlmostEqual(actual, expected, places=15)

    def test_score_transform_is_global_and_does_not_read_identity(self):
        for name in [
            "transform_yolo_score",
            "fuse_calibrated_candidate_score",
            "rank_video_frames_score_calibrated",
        ]:
            source = NODES[name]
            self.assertNotIn("video_id", source)
            self.assertNotIn("identity", source)
            self.assertNotIn("Cardboard", source)

    def test_fold_selection_holds_both_videos_of_each_identity(self):
        ns = calibration_namespace()
        video_ids = [
            "BlackBox_0", "BlackBox_1", "CardboardBox_0",
            "CardboardBox_1", "LifeJacket_0", "LifeJacket_1",
        ]
        folds = ns["build_identity_folds"](video_ids)
        self.assertEqual(set(folds), {"BlackBox", "CardboardBox", "LifeJacket"})
        self.assertTrue(all(len(videos) == 2 for videos in folds.values()))

    def test_vote_requires_identical_global_configuration(self):
        ns = calibration_namespace()
        common = {
            "yolo_power": 1.25,
            "weight_yolo": 0.35,
            "weight_siamese": 0.55,
            "weight_color": 0.10,
            "matching_threshold": 0.52,
        }
        folds = [
            {"heldout_identity": "BlackBox", **common},
            {"heldout_identity": "CardboardBox", **common},
            {
                "heldout_identity": "LifeJacket",
                **common,
                "matching_threshold": 0.54,
            },
        ]
        config, votes = ns["select_voted_score_calibration_config"](folds)
        self.assertEqual(votes, 2)
        self.assertEqual(config["matching_threshold"], 0.52)
        self.assertEqual(config["weights"]["yolo"], 0.35)

    def test_full_runner_promotes_only_cross_identity_stable_checkpoint(self):
        ns = calibration_namespace()
        exec(NODES["run_checkpoint_score_calibration"], ns)
        video_ids = [
            "BlackBox_0", "BlackBox_1", "CardboardBox_0",
            "CardboardBox_1", "LifeJacket_0", "LifeJacket_1",
        ]

        def make_result(score, marker):
            return {
                "mean_st_iou": score,
                "detected_frames": 6,
                "per_video": {video_id: score for video_id in video_ids},
                "predictions": [{"marker": marker}],
            }

        locked = {
            "production": make_result(0.50, "production_locked"),
            "p3_control": make_result(0.45, "p3_locked"),
            "p2_stride4": make_result(0.44, "p2_locked"),
        }
        caches = {
            "production": {"name": "production", "variant_runtime": None},
            "p3_control": {
                "name": "p3_control", "variant_runtime": {"aggregate_fps": 30.0},
            },
            "p2_stride4": {
                "name": "p2_stride4", "variant_runtime": {"aggregate_fps": 27.0},
            },
        }
        ns["load_score_calibration_variant_caches"] = lambda cache, signature: (
            caches, {name: f"/{name}.json.gz" for name in caches}
        )
        ns["evaluate_config"] = lambda cache, gt, weights, threshold, temporal_config: (
            locked[cache["name"]]
        )

        tuned_scores = {"production": 0.501, "p3_control": 0.52, "p2_stride4": 0.49}
        tuned_config = {
            "yolo_power": 1.25,
            "weights": {"yolo": 0.35, "siamese": 0.55, "color": 0.10},
            "matching_threshold": 0.52,
        }

        def fake_evaluate(cache, gt, config, temporal):
            is_control = ns["score_calibration_config_key"](config) == (
                1.0, 0.425, 0.475, 0.1, 0.54
            )
            if is_control:
                return locked[cache["name"]]
            return make_result(tuned_scores[cache["name"]], cache["name"] + "_tuned")

        ns["evaluate_score_calibration_config"] = fake_evaluate

        def fake_rows(variant, cache, gt, temporal):
            rows = []
            for config, score in [
                ({
                    "yolo_power": 1.0,
                    "weights": {"yolo": 0.425, "siamese": 0.475, "color": 0.1},
                    "matching_threshold": 0.54,
                }, locked[variant]["mean_st_iou"]),
                (tuned_config, tuned_scores[variant]),
            ]:
                row = {
                    "variant": variant,
                    "yolo_power": config["yolo_power"],
                    "weight_yolo": config["weights"]["yolo"],
                    "weight_siamese": config["weights"]["siamese"],
                    "weight_color": config["weights"]["color"],
                    "matching_threshold": config["matching_threshold"],
                    "mean_st_iou": score,
                    "detected_frames": 6,
                }
                row.update({f"st_iou__{video_id}": score for video_id in video_ids})
                rows.append(row)
            return rows

        ns["score_calibration_rows"] = fake_rows
        base_signature = {"video_folders": video_ids}
        rows, folds, diagnostics, recommended = ns[
            "run_checkpoint_score_calibration"
        ](
            caches["production"], base_signature, {}, locked["production"],
            {"max_gap": 7},
        )
        self.assertEqual(len(rows), 6)
        self.assertEqual(len(folds), 9)
        self.assertTrue(diagnostics["control_prediction_parity"])
        self.assertTrue(diagnostics["variants"]["p3_control"]["promotion_passed"])
        self.assertFalse(diagnostics["variants"]["p2_stride4"]["promotion_passed"])
        self.assertEqual(diagnostics["recommended_variant"], "p3_control")
        self.assertEqual(recommended["p3_control"]["mean_st_iou"], 0.52)


if __name__ == "__main__":
    unittest.main()
