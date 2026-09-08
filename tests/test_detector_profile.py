import ast
import copy
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


def profile_namespace():
    namespace = {
        "json": json,
        "math": math,
        "DETECTOR_PROFILE_REFERENCE_IMGSZ": 640,
        "DETECTOR_PROFILE_IOU_THRESHOLDS": [0.3, 0.5, 0.7],
        "DETECTOR_PROFILE_SIZE_BINS": [
            ("tiny_lt_8", 0.0, 8.0),
            ("very_small_8_16", 8.0, 16.0),
            ("small_16_32", 16.0, 32.0),
            ("medium_32_64", 32.0, 64.0),
            ("large_ge_64", 64.0, None),
        ],
        "CONFIDENCE_THRESHOLD": 0.05,
        "USE_TTA": True,
    }
    order = [
        "iou_box",
        "compute_st_iou_video",
        "identity_id_from_video_id",
        "prediction_frame_maps",
        "oracle_candidate_for_gt",
        "detector_profile_size_bin",
        "aggregate_detector_profile_rows",
        "group_detector_profile_rows",
        "detector_profile_video_rows",
        "run_detector_profile",
    ]
    for name in order:
        exec(NODES[name], namespace)
    return namespace


def candidate(bbox, confidence=0.8):
    return {"bbox": bbox, "scores": [confidence, 0.0, 0.0]}


class DetectorProfileTests(unittest.TestCase):
    def test_projected_min_side_bins_have_exact_boundaries(self):
        classify = profile_namespace()["detector_profile_size_bin"]
        self.assertEqual(classify(0), "tiny_lt_8")
        self.assertEqual(classify(7.999), "tiny_lt_8")
        self.assertEqual(classify(8), "very_small_8_16")
        self.assertEqual(classify(16), "small_16_32")
        self.assertEqual(classify(32), "medium_32_64")
        self.assertEqual(classify(64), "large_ge_64")

    def test_profile_measures_candidate_recall_without_mutating_predictions(self):
        ns = profile_namespace()
        video_id = "Object_0"
        cache = {
            "signature": {"video_folders": [video_id]},
            "videos": {
                video_id: {
                    "max_frame_idx": 1,
                    "has_siamese": True,
                    "has_color": True,
                    "frames": {
                        "0": [candidate([0, 0, 12, 12])],
                        "1": [candidate([100, 100, 112, 112])],
                    },
                }
            },
        }
        gt = {
            video_id: {
                0: [0, 0, 12, 12],
                1: [0, 0, 24, 24],
                2: [0, 0, 7, 7],
            }
        }
        predictions = [{"video_id": video_id, "detections": []}]
        production = {
            "mean_st_iou": 0.0,
            "detected_frames": 0,
            "per_video": {video_id: 0.0},
            "predictions": predictions,
        }
        before = copy.deepcopy(predictions)
        metadata = {
            video_id: {"width": 640, "height": 640, "frame_count": 3, "fps": 25.0}
        }
        architecture = {
            "detection_strides": [8.0, 16.0, 32.0],
            "has_p2_stride_4_or_finer": False,
            "has_p3_stride_8_or_finer": True,
        }
        frame_rows, video_rows, diagnostics = ns["run_detector_profile"](
            cache,
            gt,
            production,
            video_metadata=metadata,
            detector_architecture=architecture,
        )
        self.assertEqual(predictions, before)
        self.assertEqual(len(frame_rows), 3)
        self.assertEqual(video_rows[0]["gt_frames"], 3)
        self.assertEqual(diagnostics["global"]["no_candidate_frames"], 1)
        self.assertAlmostEqual(diagnostics["global"]["oracle_recall_iou_05"], 1 / 3)
        self.assertEqual(diagnostics["by_size_bin"]["tiny_lt_8"]["gt_frames"], 1)
        self.assertEqual(diagnostics["by_size_bin"]["very_small_8_16"]["gt_frames"], 1)
        self.assertEqual(diagnostics["by_size_bin"]["small_16_32"]["gt_frames"], 1)
        self.assertTrue(diagnostics["production_predictions_unchanged"])
        self.assertTrue(diagnostics["production_per_video_metric_exact_parity"])
        self.assertTrue(diagnostics["gt_frame_coverage_exact"])
        self.assertFalse(
            diagnostics["detector_inference_call"]["explicit_imgsz_passed_to_yolo_predict"]
        )

    def test_detector_profile_is_disabled_after_completed_run(self):
        tracked = {
            "RUN_CALIBRATION",
            "RUN_TEMPORAL_CALIBRATION",
            "RUN_RERANKING_EXPERIMENT",
            "RUN_ORACLE_DIAGNOSTICS",
            "RUN_HYSTERESIS_EXPERIMENT",
            "RUN_MARGIN_DIAGNOSTICS",
            "RUN_DETECTOR_PROFILE",
        }
        assignments = {}
        for node in TREE.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                if isinstance(target, ast.Name) and target.id in tracked:
                    assignments[target.id] = ast.literal_eval(node.value)
        for name in tracked:
            self.assertFalse(assignments[name], name)


if __name__ == "__main__":
    unittest.main()
