import ast
import copy
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
    for node in TREE.body
    if isinstance(node, ast.FunctionDef)
}


def diagnostic_namespace():
    namespace = {
        "json": json,
        "YOLO_ONLY_THRESHOLD": 0.2,
        "ORACLE_IOU_THRESHOLDS": [0.3, 0.5, 0.7],
        "ORACLE_GOOD_IOU": 0.5,
        "ORACLE_GT_FAILURE_CATEGORIES": [
            "no_candidate",
            "detector_localization_miss",
            "matching_error",
            "threshold_rejection",
            "temporal_removal",
            "temporal_localization_degradation",
            "success",
        ],
    }
    function_order = [
        "iou_box",
        "compute_st_iou_video",
        "fuse_candidate_score",
        "rank_video_frames",
        "identity_id_from_video_id",
        "prediction_frame_maps",
        "oracle_candidate_for_gt",
        "classify_oracle_gt_frame",
        "analyze_oracle_video",
        "aggregate_identity_oracle_rows",
        "run_oracle_diagnostics",
    ]
    for name in function_order:
        exec(NODES[name], namespace)
    return namespace


def candidate(bbox, score):
    return {"bbox": bbox, "scores": [score, 0.0, 0.0]}


class OracleDiagnosticsTests(unittest.TestCase):
    def test_first_failure_hierarchy_is_mutually_exclusive(self):
        ns = diagnostic_namespace()
        classify = ns["classify_oracle_gt_frame"]
        cases = [
            ((0, 0.0, 0.0, False, False, 0.0), "no_candidate"),
            ((1, 0.2, 0.0, False, False, 0.0), "detector_localization_miss"),
            ((2, 0.8, 0.2, True, True, 0.2), "matching_error"),
            ((1, 0.8, 0.8, False, False, 0.0), "threshold_rejection"),
            ((1, 0.8, 0.8, True, False, 0.0), "temporal_removal"),
            ((1, 0.8, 0.8, True, True, 0.2), "temporal_localization_degradation"),
            ((1, 0.8, 0.8, True, True, 0.8), "success"),
        ]
        for args, expected in cases:
            self.assertEqual(classify(*args), expected)

    def test_video_decomposition_covers_every_gt_category_and_frame_index(self):
        ns = diagnostic_namespace()
        good = [0, 0, 10, 10]
        bad = [100, 100, 110, 110]
        video_cache = {
            "has_siamese": True,
            "has_color": False,
            "frames": {
                "1": [candidate(bad, 0.9)],
                "2": [candidate(bad, 0.9), candidate(good, 0.8)],
                "3": [candidate(good, 0.5)],
                "4": [candidate(good, 0.8)],
                "5": [candidate(good, 0.8)],
                "6": [candidate(good, 0.8)],
            },
        }
        ranked = {
            1: {"bbox": bad, "score": 0.9},
            2: {"bbox": bad, "score": 0.9},
            3: {"bbox": good, "score": 0.5},
            4: {"bbox": good, "score": 0.8},
            5: {"bbox": good, "score": 0.8},
            6: {"bbox": good, "score": 0.8},
        }
        gt = {frame: good for frame in range(7)}
        final = {5: bad, 6: good}
        frame_rows, video_row = ns["analyze_oracle_video"](
            "Object_0",
            video_cache,
            gt,
            ranked,
            final,
            {"yolo": 1.0, "siamese": 0.0, "color": 0.0},
            0.54,
        )
        self.assertEqual([row["frame"] for row in frame_rows], list(range(7)))
        self.assertEqual(sum(video_row[f"failure__{name}"] for name in ns["ORACLE_GT_FAILURE_CATEGORIES"]), 7)
        for category in ns["ORACLE_GT_FAILURE_CATEGORIES"]:
            self.assertEqual(video_row[f"failure__{category}"], 1)
        self.assertGreaterEqual(
            video_row["oracle_st_iou_upper_bound"],
            video_row["accepted_pretemporal_st_iou"],
        )
        self.assertGreaterEqual(
            video_row["oracle_st_iou_upper_bound"],
            video_row["final_production_st_iou"],
        )

    def test_runner_does_not_mutate_production_predictions(self):
        ns = diagnostic_namespace()
        video_id = "Object_0"
        cache = {
            "signature": {"video_folders": [video_id]},
            "videos": {
                video_id: {
                    "has_siamese": True,
                    "has_color": False,
                    "frames": {"0": [candidate([0, 0, 10, 10], 0.8)]},
                }
            },
        }
        gt = {video_id: {0: [0, 0, 10, 10]}}
        production = {
            "mean_st_iou": 1.0,
            "detected_frames": 1,
            "per_video": {video_id: 1.0},
            "predictions": [
                {
                    "video_id": video_id,
                    "detections": [
                        {"bboxes": [{"frame": 0, "x1": 0, "y1": 0, "x2": 10, "y2": 10}]}
                    ],
                }
            ],
        }
        before = copy.deepcopy(production["predictions"])
        frame_rows, video_rows, diagnostics = ns["run_oracle_diagnostics"](
            cache,
            gt,
            {"yolo": 1.0, "siamese": 0.0, "color": 0.0},
            0.54,
            production,
        )
        self.assertEqual(production["predictions"], before)
        self.assertTrue(diagnostics["production_predictions_unchanged"])
        self.assertEqual(len(frame_rows), 1)
        self.assertEqual(video_rows[0]["failure__success"], 1)
        self.assertIsNone(diagnostics["global"]["dominant_non_success_failure"])

    def test_oracle_diagnostics_is_the_only_active_experiment(self):
        assignments = {}
        tracked = {
            "RUN_CALIBRATION",
            "RUN_TEMPORAL_CALIBRATION",
            "RUN_RERANKING_EXPERIMENT",
            "RUN_ORACLE_DIAGNOSTICS",
            "USE_TEMPORAL_RERANKING",
        }
        for node in TREE.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                if isinstance(target, ast.Name) and target.id in tracked:
                    assignments[target.id] = ast.literal_eval(node.value)
        self.assertFalse(assignments["RUN_CALIBRATION"])
        self.assertFalse(assignments["RUN_TEMPORAL_CALIBRATION"])
        self.assertFalse(assignments["RUN_RERANKING_EXPERIMENT"])
        self.assertFalse(assignments["USE_TEMPORAL_RERANKING"])
        self.assertFalse(assignments["RUN_ORACLE_DIAGNOSTICS"])


if __name__ == "__main__":
    unittest.main()
