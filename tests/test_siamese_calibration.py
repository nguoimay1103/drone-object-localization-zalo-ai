import ast
import gzip
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ACTIVE = ROOT / "06-inference-main.ipynb"
BASELINE = ROOT / "baselines" / "06-inference-main-baseline.ipynb"
BASELINE_SHA256 = "68cfae281708a8962d785f16c3b5a8b0a0dd58a105a714012689c39907bedae1"


def notebook_source(path):
    nb = json.loads(path.read_text(encoding="utf-8"))
    sources = []
    for cell in nb["cells"]:
        text = "".join(cell.get("source", []))
        if cell.get("cell_type") != "code":
            continue
        if any(line.lstrip().startswith(("!", "%")) for line in text.splitlines()):
            continue
        sources.append(text)
    return nb, "\n".join(sources)


def nodes_by_name(source):
    return {
        node.name: ast.get_source_segment(source, node)
        for node in ast.parse(source).body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
    }


ACTIVE_NB, ACTIVE_SOURCE = notebook_source(ACTIVE)
BASELINE_NB, BASELINE_SOURCE = notebook_source(BASELINE)
ACTIVE_NODES = nodes_by_name(ACTIVE_SOURCE)
BASELINE_NODES = nodes_by_name(BASELINE_SOURCE)


def load_calibration_namespace(*names):
    namespace = {
        "gzip": gzip,
        "json": json,
        "os": os,
        "TEMPORAL_MIN_SEG": 3,
        "TEMPORAL_MAX_GAP": 5,
        "YOLO_ONLY_THRESHOLD": 0.2,
        "WEIGHT_YOLO": 0.4,
        "WEIGHT_SIAMESE": 0.3,
        "WEIGHT_COLOR": 0.3,
        "MATCHING_THRESHOLD": 0.45,
        "CALIBRATION_SIAMESE_WEIGHTS": [0.15, 0.30, 0.50],
        "CALIBRATION_COLOR_WEIGHTS": [0.15, 0.30, 0.45],
        "CALIBRATION_MIN_YOLO_WEIGHT": 0.10,
    }
    dependencies = ["iou_box", "compute_st_iou_video", "temporal_smooth_detections"]
    for name in dependencies + list(names):
        if name not in namespace:
            exec(ACTIVE_NODES[name], namespace)
    return namespace


class NotebookContractTests(unittest.TestCase):
    def test_backup_exact_and_notebook_clean(self):
        self.assertEqual(hashlib.sha256(BASELINE.read_bytes()).hexdigest(), BASELINE_SHA256)
        for cell in ACTIVE_NB["cells"]:
            if cell.get("cell_type") == "code":
                self.assertIsNone(cell.get("execution_count"))
                self.assertEqual(cell.get("outputs"), [])
        compile(ACTIVE_SOURCE, str(ACTIVE), "exec")

    def test_model_preprocessing_metric_and_temporal_are_unchanged(self):
        unchanged = [
            "SiameseMobileNet",
            "get_inference_transforms",
            "load_reference_embeddings",
            "compute_hs_histogram",
            "load_reference_histogram",
            "calculate_color_score",
            "load_gt_annotations",
            "iou_box",
            "compute_st_iou_video",
            "temporal_smooth_detections",
        ]
        for name in unchanged:
            self.assertEqual(
                ast.dump(ast.parse(ACTIVE_NODES[name]), include_attributes=False),
                ast.dump(ast.parse(BASELINE_NODES[name]), include_attributes=False),
                name,
            )

    def test_active_profile_is_the_expected_identity_v1_production_grid(self):
        tree = ast.parse(ACTIVE_SOURCE)
        namespace = {}
        production_names = {
            "RUN_CALIBRATION", "MATCHING_THRESHOLD",
            "WEIGHT_YOLO", "WEIGHT_SIAMESE", "WEIGHT_COLOR",
        }
        for node in tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                if isinstance(target, ast.Name) and target.id in production_names:
                    namespace[target.id] = ast.literal_eval(node.value)
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "CALIBRATION_PROFILE"
                for target in node.targets
            ):
                exec(compile(ast.Module([node], type_ignores=[]), "<profile>", "exec"), namespace)
            if isinstance(node, ast.If) and "CALIBRATION_PROFILE" in ast.unparse(node.test):
                exec(compile(ast.Module([node], type_ignores=[]), "<grid>", "exec"), namespace)
        self.assertEqual(namespace["CALIBRATION_PROFILE"], "fine_identity_v1")
        self.assertEqual(len(namespace["CALIBRATION_THRESHOLDS"]), 21)
        self.assertEqual(len(namespace["CALIBRATION_SIAMESE_WEIGHTS"]), 14)
        self.assertEqual(len(namespace["CALIBRATION_COLOR_WEIGHTS"]), 10)
        self.assertFalse(namespace["RUN_CALIBRATION"])
        self.assertEqual(namespace["MATCHING_THRESHOLD"], 0.54)
        self.assertEqual(
            (namespace["WEIGHT_YOLO"], namespace["WEIGHT_SIAMESE"], namespace["WEIGHT_COLOR"]),
            (0.425, 0.475, 0.1),
        )
        namespace["CALIBRATION_MIN_YOLO_WEIGHT"] = 0.1
        exec(ACTIVE_NODES["calibration_weight_grid"], namespace)
        self.assertEqual(len(namespace["calibration_weight_grid"]()), 140)


class CalibrationTests(unittest.TestCase):
    def test_default_fusion_is_exactly_the_original_formula(self):
        ns = load_calibration_namespace("fuse_candidate_score")
        scores = [0.13, 0.82, 0.41]
        weights = {"yolo": 0.4, "siamese": 0.3, "color": 0.3}
        expected = (0.4 * scores[0] + 0.3 * scores[1] + 0.3 * scores[2]) / 1.0
        self.assertAlmostEqual(ns["fuse_candidate_score"](scores, weights, True, True), expected)
        expected_without_color = (0.4 * scores[0] + 0.3 * scores[1]) / 0.7
        self.assertAlmostEqual(
            ns["fuse_candidate_score"](scores, weights, True, False), expected_without_color
        )

    def test_ranking_preserves_first_candidate_on_exact_tie(self):
        ns = load_calibration_namespace("fuse_candidate_score", "rank_video_frames")
        video = {
            "has_siamese": True,
            "has_color": True,
            "frames": {
                "0": [
                    {"bbox": [0, 0, 10, 10], "scores": [0.5, 0.5, 0.5]},
                    {"bbox": [1, 1, 11, 11], "scores": [0.5, 0.5, 0.5]},
                ]
            },
        }
        ranked, debug = ns["rank_video_frames"](
            video, {"yolo": 0.4, "siamese": 0.3, "color": 0.3}, True
        )
        self.assertEqual(ranked[0]["bbox"], [0, 0, 10, 10])
        self.assertTrue(debug[0][0]["is_best"])
        self.assertFalse(debug[0][1]["is_best"])

    def test_cache_replay_threshold_and_temporal_match_expected_metric(self):
        ns = load_calibration_namespace(
            "fuse_candidate_score",
            "rank_video_frames",
            "render_ranked_predictions",
            "evaluate_config",
        )
        cache = {
            "signature": {"video_folders": ["Video_0"]},
            "videos": {
                "Video_0": {
                    "max_frame_idx": 2,
                    "has_siamese": True,
                    "has_color": True,
                    "frames": {
                        "0": [{"bbox": [0, 0, 10, 10], "scores": [0.8, 0.8, 0.8]}],
                        "2": [{"bbox": [2, 0, 12, 10], "scores": [0.8, 0.8, 0.8]}],
                    },
                }
            },
        }
        gt = {
            "Video_0": {
                0: [0, 0, 10, 10],
                1: [1, 0, 11, 10],
                2: [2, 0, 12, 10],
            }
        }
        result = ns["evaluate_config"](
            cache, gt, {"yolo": 0.4, "siamese": 0.3, "color": 0.3}, 0.45
        )
        self.assertAlmostEqual(result["mean_st_iou"], 1.0)
        self.assertEqual(result["detected_frames"], 3)

    def test_weight_grid_contains_default_and_respects_simplex(self):
        ns = load_calibration_namespace("calibration_weight_grid")
        configs = ns["calibration_weight_grid"]()
        self.assertIn({"yolo": 0.4, "siamese": 0.3, "color": 0.3}, configs)
        for config in configs:
            self.assertAlmostEqual(sum(config.values()), 1.0)
            self.assertGreaterEqual(config["yolo"], 0.1 - 1e-12)

    def test_best_selection_uses_mean_then_baseline_distance(self):
        ns = load_calibration_namespace("select_best_row")
        rows = [
            {"mean_st_iou": 0.7, "weight_yolo": 0.2, "weight_siamese": 0.4,
             "weight_color": 0.4, "matching_threshold": 0.40},
            {"mean_st_iou": 0.7, "weight_yolo": 0.4, "weight_siamese": 0.3,
             "weight_color": 0.3, "matching_threshold": 0.46},
            {"mean_st_iou": 0.69, "weight_yolo": 0.4, "weight_siamese": 0.3,
             "weight_color": 0.3, "matching_threshold": 0.45},
        ]
        self.assertIs(ns["select_best_row"](rows), rows[1])

    def test_stale_cache_signature_is_rejected(self):
        ns = load_calibration_namespace("read_json", "load_or_initialize_cache")
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir) / "cache.json.gz"
            with gzip.open(cache_path, "wt", encoding="utf-8") as f:
                json.dump({"signature": {"checkpoint": "old"}}, f)
            ns["REUSE_CANDIDATE_CACHE"] = True
            ns["CANDIDATE_CACHE_FILE"] = str(cache_path)
            with self.assertRaisesRegex(RuntimeError, "does not match"):
                ns["load_or_initialize_cache"]({"checkpoint": "new"})


if __name__ == "__main__":
    unittest.main()
