import ast
import copy
import json
import math
import os
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


def sahi_namespace():
    namespace = {
        "json": json,
        "math": math,
        "os": os,
        "SAHI_TILE_SIZE": 640,
        "SAHI_TILE_IMGSZ": 640,
        "SAHI_OVERLAP_RATIO": 0.25,
        "SAHI_FRAME_BATCH_SIZE": 8,
        "SAHI_TILE_INFERENCE_BATCH_SIZE": 16,
        "SAHI_TILE_CACHE_FILE": "/tmp/sahi_tiles.json.gz",
        "SAHI_BASELINE_REFERENCE_FPS": 30.308744167118398,
        "SAHI_MIN_MEAN_ST_IOU_DELTA": 0.003,
        "SAHI_MIN_IMPROVED_IDENTITIES": 2,
        "SAHI_MAX_WORST_IDENTITY_DROP": 0.010,
        "SAHI_MIN_SUB16_RECALL_DELTA": 0.002,
        "SAHI_ORACLE_TOLERANCE": 1e-12,
        "CONFIDENCE_THRESHOLD": 0.05,
        "USE_TTA": True,
        "WEIGHT_YOLO": 0.425,
        "WEIGHT_SIAMESE": 0.475,
        "WEIGHT_COLOR": 0.1,
        "MATCHING_THRESHOLD": 0.54,
        "YOLO_ONLY_THRESHOLD": 0.2,
    }
    for name in [
        "fuse_candidate_score",
        "identity_id_from_video_id",
        "sahi_axis_partitions",
        "build_sahi_tile_layout",
        "sahi_center_owned",
        "build_sahi_tile_cache_signature",
        "build_hybrid_sahi_cache",
        "candidate_source_selection_counts",
        "resolution_identity_scores",
        "profile_sub16_recall",
        "run_hybrid_sahi_experiment",
    ]:
        exec(NODES[name], namespace)
    return namespace


def candidate(bbox, score, source=None):
    value = {"bbox": bbox, "scores": [score, 0.0, 0.0]}
    if source:
        value["source"] = source
    return value


class HybridSahiTests(unittest.TestCase):
    def test_layout_splits_overlap_into_unique_ownership(self):
        ns = sahi_namespace()
        x_parts = ns["sahi_axis_partitions"](1024, 640, 0.25)
        self.assertEqual([row["start"] for row in x_parts], [0, 384])
        self.assertEqual(x_parts[0]["owner_end"], 512)
        self.assertEqual(x_parts[1]["owner_start"], 512)
        layout = ns["build_sahi_tile_layout"](1024, 576)
        self.assertEqual(len(layout), 2)
        left, right = layout
        seam_box = [507, 100, 517, 110]
        self.assertFalse(ns["sahi_center_owned"](seam_box, left))
        self.assertTrue(ns["sahi_center_owned"](seam_box, right))
        self.assertEqual(ns["build_sahi_tile_layout"](640, 576), [])

    def test_hybrid_union_preserves_legacy_candidates_and_input_cache(self):
        ns = sahi_namespace()
        video_id = "Object_0"
        base = {
            "complete": True,
            "signature": {"video_folders": [video_id]},
            "videos": {
                video_id: {
                    "max_frame_idx": 1,
                    "has_siamese": True,
                    "has_color": True,
                    "frames": {"0": [candidate([0, 0, 10, 10], 0.8)]},
                }
            },
        }
        tiles = {
            "complete": True,
            "signature": {
                "video_folders": [video_id],
                "candidate_generation": {"mode": "tiles"},
            },
            "videos": {
                video_id: {
                    "max_frame_idx": 1,
                    "has_siamese": True,
                    "has_color": True,
                    "frames": {
                        "0": [candidate([1, 1, 11, 11], 0.9, "sahi_tile")],
                        "1": [candidate([20, 20, 30, 30], 0.7, "sahi_tile")],
                    },
                }
            },
        }
        before = copy.deepcopy(base)
        hybrid = ns["build_hybrid_sahi_cache"](base, tiles)
        self.assertEqual(base, before)
        self.assertEqual(len(hybrid["videos"][video_id]["frames"]["0"]), 2)
        self.assertEqual(
            hybrid["videos"][video_id]["frames"]["0"][0]["source"],
            "legacy_full_frame",
        )
        self.assertEqual(
            hybrid["videos"][video_id]["frames"]["0"][1]["source"],
            "sahi_tile",
        )
        self.assertEqual(hybrid["videos"][video_id]["legacy_candidate_count"], 1)
        self.assertEqual(hybrid["videos"][video_id]["tile_candidate_count"], 2)

    def test_batched_tile_inference_passes_batch_and_imgsz(self):
        node = next(
            node for node in TREE.body
            if isinstance(node, ast.FunctionDef) and node.name == "infer_sahi_frame_batch"
        )
        calls = [
            item for item in ast.walk(node)
            if isinstance(item, ast.Call)
            and isinstance(item.func, ast.Attribute)
            and item.func.attr == "predict"
        ]
        self.assertEqual(len(calls), 1)
        keywords = {keyword.arg for keyword in calls[0].keywords}
        self.assertTrue({"batch", "imgsz", "augment", "conf"}.issubset(keywords))

    def test_full_runner_enforces_oracle_monotonicity_and_generalization_gate(self):
        ns = sahi_namespace()
        video_ids = [
            "BlackBox_0", "BlackBox_1",
            "CardboardBox_0", "CardboardBox_1",
            "LifeJacket_0", "LifeJacket_1",
        ]
        signature = {"schema_version": 1, "video_folders": video_ids}
        base_cache = {
            "complete": True,
            "signature": signature,
            "videos": {
                video_id: {
                    "max_frame_idx": 0,
                    "has_siamese": True,
                    "has_color": False,
                    "frames": {"0": [candidate([0, 0, 10, 10], 0.8)]},
                }
                for video_id in video_ids
            },
        }
        predictions = [{"video_id": video_id, "detections": []} for video_id in video_ids]
        baseline = {
            "mean_st_iou": 0.5,
            "detected_frames": 0,
            "per_video": {video_id: 0.5 for video_id in video_ids},
            "predictions": predictions,
        }
        before = copy.deepcopy(predictions)
        tile_cache = {
            "complete": True,
            "signature": ns["build_sahi_tile_cache_signature"](signature),
            "variant_runtime": {"tile_branch_fps": 15.0},
            "videos": {
                video_id: {
                    "max_frame_idx": 0,
                    "has_siamese": True,
                    "has_color": False,
                    "frames": {"0": [candidate([0, 0, 10, 10], 0.9, "sahi_tile")]},
                }
                for video_id in video_ids
            },
        }
        ns["load_or_initialize_sahi_tile_cache"] = lambda sig: tile_cache
        ns["extract_sahi_tile_cache"] = lambda cache, ids: cache

        def fake_evaluate(cache, gt, weights, threshold, include_debug, temporal_config):
            score = 0.51
            return {
                "mean_st_iou": score,
                "detected_frames": 0,
                "per_video": {video_id: score for video_id in video_ids},
                "predictions": [
                    {"video_id": video_id, "detections": []} for video_id in video_ids
                ],
            }

        ns["evaluate_config"] = fake_evaluate
        ns["collect_video_metadata"] = lambda ids: {
            video_id: {"width": 640, "height": 640, "frame_count": 10, "fps": 25.0}
            for video_id in ids
        }
        ns["inspect_detector_architecture"] = lambda: {
            "detection_strides": [8.0, 16.0, 32.0]
        }

        def fake_profile(cache, gt, result, video_metadata, detector_architecture):
            is_hybrid = cache["signature"].get("schema_version") == 4
            positive_count = 10 if is_hybrid else 9
            oracle_iou = 0.9 if is_hybrid else 0.8
            frame_rows, video_rows = [], []
            for video_id in video_ids:
                identity_id = video_id.rsplit("_", 1)[0]
                for index in range(10):
                    frame_rows.append({
                        "video_id": video_id,
                        "frame": index,
                        "identity_id": identity_id,
                        "projected_min_side_px": 10.0,
                        "oracle_iou": oracle_iou,
                        "oracle_iou_ge_05": int(index < positive_count),
                    })
                video_rows.append({
                    "video_id": video_id,
                    "identity_id": identity_id,
                    "no_candidate_frames": 0,
                    "oracle_recall_iou_05": positive_count / 10,
                    "mean_oracle_iou": oracle_iou,
                })
            diagnostics = {
                "global": {
                    "no_candidate_frames": 0,
                    "mean_oracle_iou": oracle_iou,
                    "oracle_recall_iou_03": positive_count / 10,
                    "oracle_recall_iou_05": positive_count / 10,
                    "oracle_recall_iou_07": positive_count / 10,
                },
                "by_size_bin": {},
                "by_identity": {},
            }
            return frame_rows, video_rows, diagnostics

        ns["run_detector_profile"] = fake_profile
        rows, video_rows, diagnostics, result = ns["run_hybrid_sahi_experiment"](
            base_cache, signature, {}, baseline, {"max_gap": 7}
        )
        self.assertEqual(predictions, before)
        self.assertEqual(len(rows), 2)
        self.assertEqual(len(video_rows), 6)
        self.assertTrue(diagnostics["legacy_predictions_unchanged"])
        self.assertTrue(diagnostics["oracle_monotonic_per_gt_frame"])
        self.assertEqual(diagnostics["oracle_regression_count"], 0)
        self.assertTrue(diagnostics["promotion"]["passed"])
        self.assertEqual(diagnostics["promotion"]["improved_identity_count"], 3)
        self.assertGreater(diagnostics["source_selection_counts"]["top1_sahi_tile"], 0)
        self.assertEqual(result["mean_st_iou"], 0.51)

    def test_hybrid_sahi_is_disabled_after_completed_run(self):
        tracked = {
            "RUN_CALIBRATION", "RUN_TEMPORAL_CALIBRATION",
            "RUN_RERANKING_EXPERIMENT", "RUN_ORACLE_DIAGNOSTICS",
            "RUN_HYSTERESIS_EXPERIMENT", "RUN_MARGIN_DIAGNOSTICS",
            "RUN_DETECTOR_PROFILE", "RUN_RESOLUTION_AB", "RUN_HYBRID_SAHI",
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
