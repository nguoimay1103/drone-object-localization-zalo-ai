import ast
import copy
import json
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
    for node in TREE.body if isinstance(node, ast.FunctionDef)
}


def locked_eval_namespace():
    namespace = {
        "json": json,
        "os": os,
        "P2_AB_IMGSZ": 640,
        "P2_AB_OUTPUT_DIR": "/tmp/p2_ab",
        "P2_AB_PRECOMPUTED_CACHES": {"p3_control": "", "p2_stride4": ""},
        "P2_AB_MIN_MEAN_ST_IOU_DELTA": 0.003,
        "P2_AB_MIN_IMPROVED_IDENTITIES": 2,
        "P2_AB_MAX_WORST_IDENTITY_DROP": 0.010,
        "P2_AB_MIN_SUB16_RECALL_DELTA": 0.002,
        "P2_AB_MIN_AGGREGATE_FPS": 25.0,
        "P2_AB_VARIANTS": {
            "p3_control": {"model_path": "/weights/p3.pt"},
            "p2_stride4": {"model_path": "/weights/p2.pt"},
        },
        "YOLO_MODEL_PATH": "/weights/production.pt",
        "CANDIDATE_CACHE_FILE": "/cache/production.json.gz",
        "CONFIDENCE_THRESHOLD": 0.05,
        "USE_TTA": True,
        "WEIGHT_YOLO": 0.425,
        "WEIGHT_SIAMESE": 0.475,
        "WEIGHT_COLOR": 0.1,
        "MATCHING_THRESHOLD": 0.54,
    }
    for name in [
        "identity_id_from_video_id",
        "resolution_identity_scores",
        "profile_sub16_recall",
        "resolution_variant_row",
        "resolution_video_rows",
        "build_detector_checkpoint_cache_signature",
        "detector_checkpoint_cache_file",
        "detector_ab_variant_gate",
        "run_detector_checkpoint_ab",
    ]:
        exec(NODES[name], namespace)
    return namespace


def result(video_ids, score):
    return {
        "mean_st_iou": score,
        "detected_frames": 10,
        "per_video": {video_id: score for video_id in video_ids},
        "predictions": [
            {"video_id": video_id, "detections": []} for video_id in video_ids
        ],
    }


class P2DetectorLockedEvalTests(unittest.TestCase):
    def test_locked_checkpoint_hashes_and_strides_are_pinned(self):
        assignments = {}
        for node in TREE.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                if isinstance(target, ast.Name):
                    try:
                        assignments[target.id] = ast.literal_eval(node.value)
                    except (ValueError, TypeError):
                        pass
        config_source = next(
            "".join(cell.get("source", []))
            for cell in json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
            if "RUN_P2_DETECTOR_AB" in "".join(cell.get("source", []))
        )
        self.assertIn("d317a81a494113ddc789df116a7adb9ef90be362aa4ef7a63a06242f47843234", config_source)
        self.assertIn("8f8279f178dc7ec1f57a8b19a28ff7b470a4652398face895a00deef67d60e16", config_source)
        self.assertFalse(assignments["RUN_P2_DETECTOR_AB"])
        self.assertEqual(assignments["P2_AB_IMGSZ"], 640)
        self.assertEqual(assignments["P2_AB_MIN_AGGREGATE_FPS"], 25.0)

    def test_extractor_accepts_explicit_checkpoint_path(self):
        node = next(
            node for node in TREE.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "extract_explicit_resolution_cache"
        )
        self.assertIn("model_path", [argument.arg for argument in node.args.args])
        source = ast.get_source_segment(SOURCE, node)
        self.assertIn("selected_model_path = model_path or YOLO_MODEL_PATH", source)
        self.assertIn("yolo = YOLO(selected_model_path)", source)

    def test_full_locked_runner_can_recommend_p2_without_mutating_production(self):
        ns = locked_eval_namespace()
        video_ids = [
            "BlackBox_0", "BlackBox_1", "CardboardBox_0",
            "CardboardBox_1", "LifeJacket_0", "LifeJacket_1",
        ]
        signature = {"video_folders": video_ids, "yolo_sha256": "production"}
        base_cache = {
            "signature": signature,
            "videos": {
                video_id: {
                    "frames": {"0": [{"bbox": [0, 0, 10, 10], "scores": [0.8, 0, 0]}]},
                    "max_frame_idx": 0,
                }
                for video_id in video_ids
            },
        }
        production = result(video_ids, 0.50)
        before = copy.deepcopy(production["predictions"])

        def architecture(path):
            strides = [4.0, 8.0, 16.0, 32.0] if "p2" in path else [8.0, 16.0, 32.0]
            return {
                "checkpoint_path": path,
                "checkpoint_sha256": path,
                "detection_strides": strides,
            }

        ns["inspect_detector_architecture"] = architecture
        ns["validate_detector_ab_checkpoint"] = lambda name, cfg: architecture(cfg["model_path"])
        ns["collect_video_metadata"] = lambda ids: {video_id: {} for video_id in ids}
        ns["file_sha256"] = lambda path: path
        ns["load_or_initialize_resolution_cache"] = lambda sig, path: {
            "signature": sig, "complete": False, "videos": {}
        }

        def fake_extract(cache, ids, imgsz, path, model_path=None):
            cache["complete"] = True
            cache["variant_runtime"] = {"aggregate_fps": 30.0, "total_video_seconds": 1.0}
            cache["videos"] = copy.deepcopy(base_cache["videos"])
            return cache

        ns["extract_explicit_resolution_cache"] = fake_extract

        def fake_evaluate(cache, gt, weights, threshold, include_debug, temporal_config):
            variant = cache["signature"]["candidate_generation"]["variant"]
            return result(video_ids, 0.51 if variant == "p3_control" else 0.52)

        ns["evaluate_config"] = fake_evaluate

        def fake_profile(cache, gt, evaluated, video_metadata, detector_architecture):
            score = evaluated["mean_st_iou"]
            recall = 0.80 if score == 0.50 else (0.85 if score == 0.51 else 0.90)
            frame_rows, video_rows = [], []
            for video_id in video_ids:
                identity = video_id.rsplit("_", 1)[0]
                for index in range(10):
                    frame_rows.append({
                        "identity_id": identity,
                        "projected_min_side_px": 10.0,
                        "oracle_iou_ge_05": int(index < round(recall * 10)),
                    })
                video_rows.append({
                    "video_id": video_id, "identity_id": identity, "gt_frames": 10,
                    "no_candidate_frames": 0, "mean_candidate_count": 1.0,
                    "mean_oracle_iou": recall, "oracle_recall_iou_03": recall,
                    "oracle_recall_iou_05": recall, "oracle_recall_iou_07": recall,
                })
            diagnostics = {
                "global": {
                    "oracle_recall_iou_03": recall,
                    "oracle_recall_iou_05": recall,
                    "oracle_recall_iou_07": recall,
                    "no_candidate_frames": 0,
                    "mean_candidate_count": 1.0,
                },
                "by_size_bin": {}, "by_identity": {},
            }
            return frame_rows, video_rows, diagnostics

        ns["run_detector_profile"] = fake_profile
        rows, videos, diagnostics, variants = ns["run_detector_checkpoint_ab"](
            base_cache, signature, {}, production, {"max_gap": 7}
        )
        self.assertEqual(production["predictions"], before)
        self.assertEqual([row["variant"] for row in rows], [
            "production", "p3_control", "p2_stride4"
        ])
        self.assertEqual(len(videos), 18)
        self.assertTrue(diagnostics["production_predictions_unchanged"])
        self.assertEqual(diagnostics["recommended_variant"], "p2_stride4")
        self.assertTrue(diagnostics["promotion_passed"])
        self.assertAlmostEqual(diagnostics["p2_vs_paired_p3"]["mean_st_iou_delta"], 0.01)
        self.assertIn("p2_stride4", variants)


if __name__ == "__main__":
    unittest.main()
