import ast
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
    for node in TREE.body if isinstance(node, ast.FunctionDef)
}


def motion_namespace():
    namespace = {
        "math": math,
        "json": json,
        "DETECTOR_PROFILE_SIZE_BINS": [
            ("tiny_lt_8", 0.0, 8.0),
            ("very_small_8_16", 8.0, 16.0),
            ("small_16_32", 16.0, 32.0),
            ("medium_32_64", 32.0, 64.0),
            ("large_ge_64", 64.0, None),
        ],
        "MOTION_PROFILE_REFERENCE_IMGSZ": 640,
        "MOTION_PROFILE_GATE_THRESHOLDS": [0.4, 0.6],
        "MOTION_PROFILE_GOOD_ENDPOINT_IOU": 0.5,
        "MOTION_PROFILE_PERCENTILES": [0.5, 0.75, 0.9, 0.95, 0.99],
        "MOTION_PROFILE_STAGES": [
            "gt", "selected_top1", "oracle_candidate", "production_final"
        ],
    }
    order = [
        "identity_id_from_video_id",
        "iou_box",
        "compute_st_iou_video",
        "fuse_candidate_score",
        "rank_video_frames",
        "prediction_frame_maps",
        "oracle_candidate_for_gt",
        "gap_motion_metrics",
        "bbox_center",
        "detector_profile_size_bin",
        "linear_percentile",
        "motion_pair_measurements",
        "add_motion_stage_fields",
        "summarize_motion_stage",
        "motion_group_rows",
        "build_motion_profile_frame_rows",
        "run_motion_distribution_diagnostics",
    ]
    for name in order:
        exec(NODES[name], namespace)
    return namespace


def candidate(x, score=0.8):
    return {
        "bbox": [x, 0, x + 10, 10],
        "scores": [score, score, score],
    }


class MotionDistributionDiagnosticsTests(unittest.TestCase):
    def test_linear_percentile_interpolates_and_validates_range(self):
        ns = motion_namespace()
        self.assertEqual(ns["linear_percentile"]([], 0.5), None)
        self.assertAlmostEqual(ns["linear_percentile"]([0, 10], 0.5), 5.0)
        self.assertAlmostEqual(ns["linear_percentile"]([0, 10, 20], 0.75), 15.0)
        with self.assertRaises(ValueError):
            ns["linear_percentile"]([1], 1.1)

    def test_frame_rows_use_absolute_indices_and_only_adjacent_gt_pairs(self):
        ns = motion_namespace()
        gt = {
            36: [0, 0, 10, 10],
            37: [5, 0, 15, 10],
            40: [10, 0, 20, 10],
        }
        video = {
            "has_siamese": True,
            "has_color": False,
            "frames": {
                "36": [candidate(0)],
                "37": [candidate(5)],
                "40": [candidate(10)],
            },
        }
        rows = ns["build_motion_profile_frame_rows"](
            "Object_0", video, gt, {},
            {"width": 640, "height": 480},
            {"yolo": 1.0, "siamese": 0.0, "color": 0.0},
        )
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual((row["previous_frame"], row["current_frame"]), (36, 37))
        self.assertAlmostEqual(row["gt__center_speed"], 0.5)
        self.assertEqual(row["gt__exceeds_gate_04"], 1)
        self.assertEqual(row["gt__exceeds_gate_06"], 0)
        self.assertEqual(row["selected_top1__good_pair_exceeds_gate_04"], 1)

    def test_size_bin_uses_mean_gt_side_projected_to_640(self):
        ns = motion_namespace()
        gt = {0: [0, 0, 20, 20], 1: [1, 0, 21, 20]}
        video = {
            "has_siamese": True,
            "has_color": False,
            "frames": {"0": [candidate(0)], "1": [candidate(1)]},
        }
        row = ns["build_motion_profile_frame_rows"](
            "Object_0", video, gt, {},
            {"width": 1280, "height": 720},
            {"yolo": 1.0, "siamese": 0.0, "color": 0.0},
        )[0]
        self.assertAlmostEqual(row["mean_gt_projected_min_side_px"], 10.0)
        self.assertEqual(row["size_bin"], "very_small_8_16")

    def test_runner_preserves_predictions_and_reports_group_profiles(self):
        ns = motion_namespace()
        video_id = "Object_0"
        gt = {video_id: {0: [0, 0, 10, 10], 1: [1, 0, 11, 10]}}
        predictions = [{
            "video_id": video_id,
            "detections": [{"bboxes": [
                {"frame": 0, "x1": 0, "y1": 0, "x2": 10, "y2": 10},
                {"frame": 1, "x1": 1, "y1": 0, "x2": 11, "y2": 10},
            ]}],
        }]
        production = {
            "predictions": predictions,
            "per_video": {video_id: 1.0},
        }
        cache = {
            "signature": {"video_folders": [video_id]},
            "videos": {video_id: {
                "has_siamese": True,
                "has_color": False,
                "frames": {"0": [candidate(0)], "1": [candidate(1)]},
            }},
        }
        snapshot = json.dumps(predictions, sort_keys=True)
        frames, groups, videos, diagnostics = ns[
            "run_motion_distribution_diagnostics"
        ](
            cache, gt,
            {"yolo": 1.0, "siamese": 0.0, "color": 0.0},
            production,
            video_metadata={video_id: {
                "width": 640, "height": 480, "frame_count": 2, "fps": 30.0
            }},
        )
        self.assertEqual(json.dumps(predictions, sort_keys=True), snapshot)
        self.assertEqual(len(frames), 1)
        self.assertEqual(len(videos), 4)
        self.assertTrue(diagnostics["production_predictions_unchanged"])
        self.assertTrue(diagnostics["adjacent_gt_pair_coverage_exact"])
        self.assertEqual(diagnostics["global"]["gt"]["pair_count"], 1)
        self.assertTrue(any(row["group_type"] == "size_bin" for row in groups))

    def test_motion_profile_is_disabled_after_completed_run(self):
        assignments = {}
        for node in TREE.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                if isinstance(target, ast.Name) and target.id.startswith("RUN_"):
                    try:
                        assignments[target.id] = ast.literal_eval(node.value)
                    except (ValueError, TypeError):
                        pass
        self.assertFalse(assignments["RUN_MOTION_PROFILE"])

    def test_diagnostic_functions_do_not_change_ranking_or_temporal_state(self):
        source = "\n".join(
            NODES[name] for name in [
                "build_motion_profile_frame_rows",
                "run_motion_distribution_diagnostics",
            ]
        )
        self.assertNotIn("temporal_smooth", source)
        self.assertNotIn("matching_threshold", source)
        self.assertNotIn("RUN_TRACKLET", source)


if __name__ == "__main__":
    unittest.main()
