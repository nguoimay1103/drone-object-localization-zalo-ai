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
    for node in TREE.body
    if isinstance(node, ast.FunctionDef)
}


def resolution_namespace():
    namespace = {
        "json": json,
        "os": os,
        "RESOLUTION_OUTPUT_DIR": "/tmp/resolution_ab",
        "RESOLUTION_PROFILE_REFERENCE_IMGSZ": 640,
        "RESOLUTION_VARIANTS": [640, 960],
        "RESOLUTION_CONTROL_TOLERANCE": 1e-6,
        "RESOLUTION_MIN_MEAN_ST_IOU_DELTA": 0.003,
        "RESOLUTION_MIN_IMPROVED_IDENTITIES": 2,
        "RESOLUTION_MAX_WORST_IDENTITY_DROP": 0.010,
        "RESOLUTION_MIN_SUB16_RECALL_DELTA": 0.002,
        "CONFIDENCE_THRESHOLD": 0.05,
        "USE_TTA": True,
        "WEIGHT_YOLO": 0.425,
        "WEIGHT_SIAMESE": 0.475,
        "WEIGHT_COLOR": 0.1,
        "MATCHING_THRESHOLD": 0.54,
        "CANDIDATE_CACHE_FILE": "/tmp/legacy.json.gz",
    }
    for name in [
        "identity_id_from_video_id",
        "build_resolution_cache_signature",
        "resolution_cache_file",
        "resolution_identity_scores",
        "profile_sub16_recall",
        "resolution_variant_row",
        "resolution_video_rows",
        "run_resolution_candidate_ab",
    ]:
        exec(NODES[name], namespace)
    return namespace


class ResolutionCandidateABTests(unittest.TestCase):
    def test_variant_signature_locks_explicit_imgsz(self):
        ns = resolution_namespace()
        base = {"schema_version": 1, "video_folders": ["Object_0"]}
        signature = ns["build_resolution_cache_signature"](base, 960)
        self.assertEqual(base["schema_version"], 1)
        self.assertEqual(signature["schema_version"], 2)
        self.assertEqual(signature["candidate_generation"]["imgsz"], 960)
        self.assertEqual(signature["candidate_generation"]["confidence_threshold"], 0.05)
        self.assertTrue(signature["candidate_generation"]["augment_tta"])

    def test_extractor_passes_explicit_imgsz_to_yolo(self):
        function_node = next(
            node for node in TREE.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "extract_explicit_resolution_cache"
        )
        predict_calls = [
            node for node in ast.walk(function_node)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "predict"
        ]
        self.assertEqual(len(predict_calls), 1)
        keyword_names = {keyword.arg for keyword in predict_calls[0].keywords}
        self.assertIn("imgsz", keyword_names)
        self.assertIn("augment", keyword_names)
        self.assertIn("conf", keyword_names)

    def test_full_runner_requires_control_and_passes_generalized_960_gain(self):
        ns = resolution_namespace()
        video_ids = [
            "BlackBox_0", "BlackBox_1",
            "CardboardBox_0", "CardboardBox_1",
            "LifeJacket_0", "LifeJacket_1",
        ]
        base_signature = {"schema_version": 1, "video_folders": video_ids}
        base_cache = {
            "signature": base_signature,
            "videos": {
                video_id: {
                    "max_frame_idx": 0,
                    "frames": {"0": [{"bbox": [0, 0, 10, 10], "scores": [0.8, 0, 0]}]},
                }
                for video_id in video_ids
            },
        }
        legacy_predictions = [
            {"video_id": video_id, "detections": []} for video_id in video_ids
        ]
        legacy_result = {
            "mean_st_iou": 0.5,
            "detected_frames": 0,
            "per_video": {video_id: 0.5 for video_id in video_ids},
            "predictions": legacy_predictions,
        }
        before = copy.deepcopy(legacy_predictions)

        ns["collect_video_metadata"] = lambda ids: {
            video_id: {"width": 640, "height": 640, "frame_count": 10, "fps": 25.0}
            for video_id in ids
        }
        ns["inspect_detector_architecture"] = lambda: {
            "detection_strides": [8.0, 16.0, 32.0]
        }
        ns["load_or_initialize_resolution_cache"] = lambda signature, path: {
            "signature": signature,
            "videos": {},
            "complete": False,
        }

        def fake_extract(cache, ids, imgsz, path):
            cache["complete"] = True
            cache["variant_runtime"] = {
                "total_video_seconds": 10.0,
                "aggregate_fps": 6.0,
                "peak_cuda_memory_mb_this_invocation": 100.0,
            }
            cache["videos"] = {
                video_id: {
                    "max_frame_idx": 0,
                    "frames": {
                        "0": [{"bbox": [0, 0, 10, 10], "scores": [0.8, 0, 0]}]
                    },
                }
                for video_id in ids
            }
            return cache

        ns["extract_explicit_resolution_cache"] = fake_extract

        def fake_evaluate(cache, gt, weights, threshold, include_debug, temporal_config):
            imgsz = cache["signature"]["candidate_generation"]["imgsz"]
            score = 0.5 if imgsz == 640 else 0.51
            return {
                "mean_st_iou": score,
                "detected_frames": 0,
                "per_video": {video_id: score for video_id in video_ids},
                "predictions": [
                    {"video_id": video_id, "detections": []} for video_id in video_ids
                ],
            }

        ns["evaluate_config"] = fake_evaluate

        def fake_profile(cache, gt, result, video_metadata, detector_architecture):
            generation = cache["signature"].get("candidate_generation", {})
            imgsz = generation.get("imgsz")
            positives_per_video = 10 if imgsz == 960 else 9
            frame_rows = []
            profile_videos = []
            for video_id in video_ids:
                identity_id = video_id.rsplit("_", 1)[0]
                for index in range(10):
                    frame_rows.append({
                        "identity_id": identity_id,
                        "projected_min_side_px": 10.0,
                        "oracle_iou_ge_05": int(index < positives_per_video),
                    })
                profile_videos.append({
                    "video_id": video_id,
                    "identity_id": identity_id,
                    "gt_frames": 10,
                    "no_candidate_frames": 0,
                    "mean_candidate_count": 1.0,
                    "mean_oracle_iou": 0.9,
                    "oracle_recall_iou_03": positives_per_video / 10,
                    "oracle_recall_iou_05": positives_per_video / 10,
                    "oracle_recall_iou_07": positives_per_video / 10,
                })
            recall = positives_per_video / 10
            diagnostics = {
                "global": {
                    "oracle_recall_iou_03": recall,
                    "oracle_recall_iou_05": recall,
                    "oracle_recall_iou_07": recall,
                    "no_candidate_frames": 0,
                    "mean_candidate_count": 1.0,
                },
                "by_size_bin": {},
                "by_identity": {},
            }
            return frame_rows, profile_videos, diagnostics

        ns["run_detector_profile"] = fake_profile
        result_rows, video_rows, diagnostics, variants = ns["run_resolution_candidate_ab"](
            base_cache,
            base_signature,
            {},
            legacy_result,
            {"max_gap": 7},
        )
        self.assertEqual(legacy_predictions, before)
        self.assertEqual([row["variant"] for row in result_rows], [
            "legacy", "explicit_640", "explicit_960"
        ])
        self.assertEqual(len(video_rows), 18)
        self.assertTrue(diagnostics["legacy_predictions_unchanged"])
        self.assertTrue(diagnostics["control"]["passed"])
        self.assertTrue(diagnostics["control"]["total_candidate_count_equal"])
        self.assertTrue(diagnostics["promotion"]["passed"])
        self.assertEqual(diagnostics["promotion"]["improved_identity_count"], 3)
        self.assertAlmostEqual(diagnostics["promotion"]["sub16_recall_iou_05_delta"], 0.1)
        self.assertIn("explicit_960", variants)

    def test_resolution_ab_is_disabled_after_completed_run(self):
        tracked = {
            "RUN_CALIBRATION",
            "RUN_TEMPORAL_CALIBRATION",
            "RUN_RERANKING_EXPERIMENT",
            "RUN_ORACLE_DIAGNOSTICS",
            "RUN_HYSTERESIS_EXPERIMENT",
            "RUN_MARGIN_DIAGNOSTICS",
            "RUN_DETECTOR_PROFILE",
            "RUN_RESOLUTION_AB",
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
