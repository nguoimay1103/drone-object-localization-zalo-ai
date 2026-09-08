import ast
import json
import unittest
from pathlib import Path

import numpy as np


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
        "json": json,
        "np": np,
        "YOLO_ONLY_THRESHOLD": 0.2,
        "ORACLE_GOOD_IOU": 0.5,
        "MARGIN_WEAK_SCORE_THRESHOLD": 0.54,
        "MARGIN_FEATURES": [
            "selected_score",
            "score_margin",
            "score_ratio",
            "selected_yolo",
            "selected_siamese",
            "selected_color",
            "negative_candidate_count",
        ],
    }
    order = [
        "iou_box",
        "compute_st_iou_video",
        "fuse_candidate_score",
        "identity_id_from_video_id",
        "prediction_frame_maps",
        "oracle_candidate_for_gt",
        "classify_oracle_gt_frame",
        "score_candidate_details",
        "binary_metrics_at_threshold",
        "binary_roc_auc",
        "binary_average_precision",
        "feature_separability",
        "select_best_f1_threshold",
        "margin_frame_rows_for_video",
        "run_margin_diagnostics",
    ]
    for name in order + list(extra_names):
        if name not in namespace:
            exec(NODES[name], namespace)
    return namespace


def candidate(bbox, yolo, siamese, color):
    return {"bbox": bbox, "scores": [yolo, siamese, color]}


class MarginDiagnosticsTests(unittest.TestCase):
    def test_binary_metrics_are_correct_for_perfect_ranking(self):
        ns = load_namespace()
        labels = [0, 0, 1, 1]
        scores = [0.1, 0.2, 0.8, 0.9]
        self.assertEqual(ns["binary_roc_auc"](labels, scores), 1.0)
        self.assertEqual(ns["binary_average_precision"](labels, scores), 1.0)
        metrics = ns["binary_metrics_at_threshold"](labels, scores, 0.5)
        self.assertEqual((metrics["tp"], metrics["fp"], metrics["fn"], metrics["tn"]), (2, 0, 0, 2))
        self.assertEqual(metrics["f1"], 1.0)

    def test_auc_ties_receive_average_rank(self):
        ns = load_namespace()
        self.assertEqual(ns["binary_roc_auc"]([0, 1], [0.5, 0.5]), 0.5)

    def test_candidate_details_preserve_first_candidate_on_score_tie(self):
        ns = load_namespace()
        candidates = [
            candidate([0, 0, 10, 10], 0.8, 0.2, 0.1),
            candidate([20, 0, 30, 10], 0.8, 0.2, 0.1),
        ]
        details = ns["score_candidate_details"](
            candidates,
            {"yolo": 1.0, "siamese": 0.0, "color": 0.0},
            True,
            True,
        )
        self.assertEqual([row["candidate_idx"] for row in details], [0, 1])

    def test_frame_features_include_margin_components_and_correct_label(self):
        ns = load_namespace()
        good = [0, 0, 10, 10]
        bad = [100, 100, 110, 110]
        video_cache = {
            "has_siamese": True,
            "has_color": True,
            "frames": {
                "0": [
                    candidate(good, 0.4, 0.8, 0.5),
                    candidate(bad, 0.3, 0.2, 0.1),
                ]
            },
        }
        weights = {"yolo": 0.425, "siamese": 0.475, "color": 0.1}
        rows = ns["margin_frame_rows_for_video"](
            "Object_0", video_cache, {0: good}, {}, weights, 0.54
        )
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["is_correct_target"], 1)
        self.assertEqual(row["selected_candidate_idx"], 0)
        self.assertEqual(row["selected_yolo"], 0.4)
        self.assertEqual(row["selected_siamese"], 0.8)
        self.assertEqual(row["selected_color"], 0.5)
        self.assertGreater(row["score_margin"], 0)
        self.assertGreater(row["score_ratio"], 1)

    def test_best_f1_threshold_is_selected_from_weak_training_rows(self):
        ns = load_namespace()
        rows = [
            {"is_weak_region": True, "is_correct_target": 1, "score_margin": 0.9},
            {"is_weak_region": True, "is_correct_target": 1, "score_margin": 0.8},
            {"is_weak_region": True, "is_correct_target": 0, "score_margin": 0.2},
            {"is_weak_region": True, "is_correct_target": 0, "score_margin": 0.1},
            {"is_weak_region": False, "is_correct_target": 0, "score_margin": 1.0},
        ]
        threshold, metrics = ns["select_best_f1_threshold"](rows, "score_margin")
        self.assertEqual(threshold, 0.8)
        self.assertEqual(metrics["f1"], 1.0)

    def test_full_runner_preserves_predictions_and_builds_loio_reports(self):
        ns = load_namespace()
        video_ids = [
            "BlackBox_0",
            "BlackBox_1",
            "CardboardBox_0",
            "CardboardBox_1",
            "LifeJacket_0",
            "LifeJacket_1",
        ]
        good = [0, 0, 10, 10]
        bad = [100, 100, 110, 110]
        cache = {
            "signature": {"video_folders": video_ids},
            "videos": {
                video_id: {
                    "has_siamese": True,
                    "has_color": True,
                    "frames": {
                        "0": [
                            candidate(good, 0.5, 0.5, 0.5),
                            candidate(bad, 0.1, 0.1, 0.1),
                        ],
                        "1": [
                            candidate(bad, 0.2, 0.2, 0.2),
                            candidate(good, 0.1, 0.1, 0.1),
                        ],
                    },
                }
                for video_id in video_ids
            },
        }
        gt = {video_id: {0: good} for video_id in video_ids}
        predictions = [
            {
                "video_id": video_id,
                "detections": [
                    {"bboxes": [{"frame": 0, "x1": 0, "y1": 0, "x2": 10, "y2": 10}]}
                ],
            }
            for video_id in video_ids
        ]
        production = {
            "mean_st_iou": 1.0,
            "detected_frames": 6,
            "per_video": {video_id: 1.0 for video_id in video_ids},
            "predictions": predictions,
        }
        before = json.loads(json.dumps(predictions))
        frame_rows, video_rows, diagnostics = ns["run_margin_diagnostics"](
            cache,
            gt,
            {"yolo": 1.0, "siamese": 0.0, "color": 0.0},
            0.54,
            production,
        )
        self.assertEqual(predictions, before)
        self.assertEqual(len(frame_rows), 12)
        self.assertEqual(len(video_rows), 6)
        self.assertTrue(diagnostics["production_predictions_unchanged"])
        self.assertEqual(
            diagnostics["feature_reports"]["selected_score"]["weak_region"]["roc_auc"],
            1.0,
        )
        self.assertEqual(
            set(diagnostics["leave_one_identity_out_weak_f1_transfer"]["score_margin"]),
            {"BlackBox", "CardboardBox", "LifeJacket"},
        )

    def test_margin_diagnostics_is_only_active_experiment(self):
        tracked = {
            "RUN_CALIBRATION",
            "RUN_TEMPORAL_CALIBRATION",
            "RUN_RERANKING_EXPERIMENT",
            "RUN_ORACLE_DIAGNOSTICS",
            "RUN_HYSTERESIS_EXPERIMENT",
            "RUN_MARGIN_DIAGNOSTICS",
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
        self.assertFalse(assignments["RUN_HYSTERESIS_EXPERIMENT"])
        self.assertFalse(assignments["RUN_MARGIN_DIAGNOSTICS"])


if __name__ == "__main__":
    unittest.main()
