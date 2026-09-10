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


def admission_namespace():
    namespace = {
        "json": json,
        "os": os,
        "SAHI_ADMISSION_POLICIES": [
            "legacy_only", "missing_legacy_only", "weak_legacy_rescue"
        ],
        "SAHI_ADMISSION_MIN_MEAN_CV_DELTA": 0.003,
        "SAHI_ADMISSION_MIN_IMPROVED_IDENTITIES": 2,
        "SAHI_ADMISSION_MAX_WORST_IDENTITY_DROP": 0.010,
        "SAHI_ADMISSION_MIN_POLICY_FOLD_COUNT": 2,
        "SAHI_ADMISSION_ORACLE_TOLERANCE": 1e-12,
        "WEIGHT_YOLO": 1.0,
        "WEIGHT_SIAMESE": 0.0,
        "WEIGHT_COLOR": 0.0,
        "MATCHING_THRESHOLD": 0.54,
        "YOLO_ONLY_THRESHOLD": 0.2,
    }
    for name in [
        "fuse_candidate_score",
        "identity_id_from_video_id",
        "resolution_identity_scores",
        "profile_sub16_recall",
        "candidate_source_selection_counts",
        "build_sahi_admission_cache",
        "sahi_admission_identity_cv",
        "run_sahi_admission_experiment",
    ]:
        exec(NODES[name], namespace)
    return namespace


def candidate(score, source=None):
    row = {"bbox": [0, 0, 10, 10], "scores": [score, 0.0, 0.0]}
    if source:
        row["source"] = source
    return row


class SahiAdmissionTests(unittest.TestCase):
    def make_caches(self):
        video_id = "Object_0"
        base = {
            "complete": True,
            "signature": {"video_folders": [video_id]},
            "videos": {
                video_id: {
                    "max_frame_idx": 2,
                    "has_siamese": True,
                    "has_color": False,
                    "frames": {
                        "0": [candidate(0.8)],
                        "1": [candidate(0.4)],
                    },
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
                    "max_frame_idx": 2,
                    "has_siamese": True,
                    "has_color": False,
                    "frames": {
                        "0": [candidate(0.9, "sahi_tile")],
                        "1": [candidate(0.9, "sahi_tile")],
                        "2": [candidate(0.9, "sahi_tile")],
                    },
                }
            },
        }
        return base, tiles

    def test_missing_only_and_weak_rescue_have_distinct_fixed_behavior(self):
        ns = admission_namespace()
        base, tiles = self.make_caches()
        before = copy.deepcopy(base)
        weights = {"yolo": 1.0, "siamese": 0.0, "color": 0.0}
        missing = ns["build_sahi_admission_cache"](
            base, tiles, "missing_legacy_only", weights, 0.54
        )
        weak = ns["build_sahi_admission_cache"](
            base, tiles, "weak_legacy_rescue", weights, 0.54
        )
        self.assertEqual(base, before)
        self.assertEqual(len(missing["videos"]["Object_0"]["frames"]["0"]), 1)
        self.assertEqual(len(missing["videos"]["Object_0"]["frames"]["1"]), 1)
        self.assertEqual(len(missing["videos"]["Object_0"]["frames"]["2"]), 1)
        self.assertEqual(len(weak["videos"]["Object_0"]["frames"]["0"]), 1)
        self.assertEqual(len(weak["videos"]["Object_0"]["frames"]["1"]), 2)
        self.assertEqual(len(weak["videos"]["Object_0"]["frames"]["2"]), 1)
        self.assertEqual(missing["videos"]["Object_0"]["tile_admitted_count"], 1)
        self.assertEqual(weak["videos"]["Object_0"]["tile_admitted_count"], 2)

    def test_identity_cv_selects_consistent_general_policy(self):
        ns = admission_namespace()
        video_ids = [
            "BlackBox_0", "BlackBox_1",
            "CardboardBox_0", "CardboardBox_1",
            "LifeJacket_0", "LifeJacket_1",
        ]
        results = {
            "legacy_only": {
                "mean_st_iou": 0.5,
                "per_video": {video_id: 0.5 for video_id in video_ids},
            },
            "missing_legacy_only": {
                "mean_st_iou": 0.51,
                "per_video": {video_id: 0.51 for video_id in video_ids},
            },
            "weak_legacy_rescue": {
                "mean_st_iou": 0.505,
                "per_video": {video_id: 0.505 for video_id in video_ids},
            },
        }
        folds, summary = ns["sahi_admission_identity_cv"](results)
        self.assertEqual(len(folds), 3)
        self.assertEqual({row["selected_policy"] for row in folds}, {"missing_legacy_only"})
        self.assertEqual(summary["dominant_policy_across_folds"], "missing_legacy_only")
        self.assertEqual(summary["selected_policy_counts"]["missing_legacy_only"], 3)
        self.assertTrue(summary["policy_consistent"])
        self.assertTrue(summary["promotion_passed"])

    def test_full_runner_is_cache_only_and_preserves_oracle(self):
        ns = admission_namespace()
        video_ids = [
            "BlackBox_0", "BlackBox_1",
            "CardboardBox_0", "CardboardBox_1",
            "LifeJacket_0", "LifeJacket_1",
        ]
        signature = {"video_folders": video_ids}
        base = {
            "complete": True,
            "signature": signature,
            "videos": {
                video_id: {
                    "max_frame_idx": 1,
                    "has_siamese": True,
                    "has_color": False,
                    "frames": {"0": [candidate(0.8)]},
                }
                for video_id in video_ids
            },
        }
        tiles = {
            "complete": True,
            "signature": {
                "video_folders": video_ids,
                "candidate_generation": {"mode": "tiles"},
            },
            "videos": {
                video_id: {
                    "max_frame_idx": 1,
                    "has_siamese": True,
                    "has_color": False,
                    "frames": {"1": [candidate(0.9, "sahi_tile")]},
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

        def fake_evaluate(cache, gt, weights, threshold, include_debug, temporal_config):
            policy = cache["signature"]["candidate_generation"]["policy"]
            score = 0.51 if policy == "missing_legacy_only" else 0.505
            return {
                "mean_st_iou": score,
                "detected_frames": 1,
                "per_video": {video_id: score for video_id in video_ids},
                "predictions": [
                    {"video_id": video_id, "detections": []} for video_id in video_ids
                ],
            }

        ns["evaluate_config"] = fake_evaluate
        ns["collect_video_metadata"] = lambda ids: {
            video_id: {"width": 640, "height": 640, "frame_count": 2, "fps": 25.0}
            for video_id in ids
        }
        ns["inspect_detector_architecture"] = lambda: {"detection_strides": [8, 16, 32]}

        def fake_profile(cache, gt, result, video_metadata, detector_architecture):
            policy = cache["signature"].get("candidate_generation", {}).get(
                "policy", "legacy_only"
            )
            improved = policy != "legacy_only"
            oracle = 0.9 if improved else 0.8
            rows, videos = [], []
            for video_id in video_ids:
                identity_id = video_id.rsplit("_", 1)[0]
                for frame in range(2):
                    rows.append({
                        "video_id": video_id,
                        "frame": frame,
                        "identity_id": identity_id,
                        "oracle_iou": oracle,
                        "oracle_iou_ge_05": 1,
                        "projected_min_side_px": 10.0,
                    })
                videos.append({
                    "video_id": video_id,
                    "no_candidate_frames": 0,
                    "oracle_recall_iou_05": 1.0,
                    "mean_oracle_iou": oracle,
                })
            diagnostics = {
                "global": {
                    "no_candidate_frames": 0,
                    "mean_oracle_iou": oracle,
                    "oracle_recall_iou_05": 1.0,
                },
                "by_size_bin": {},
                "by_identity": {},
            }
            return rows, videos, diagnostics

        ns["run_detector_profile"] = fake_profile
        result_rows, video_rows, folds, summary, results = ns[
            "run_sahi_admission_experiment"
        ](base, tiles, {}, baseline, {"max_gap": 7})
        self.assertEqual(predictions, before)
        self.assertEqual(len(result_rows), 3)
        self.assertEqual(len(video_rows), 18)
        self.assertEqual(len(folds), 3)
        self.assertTrue(summary["legacy_predictions_unchanged"])
        self.assertTrue(summary["oracle_monotonic_for_all_policies"])
        self.assertTrue(summary["cross_validation"]["promotion_passed"])
        self.assertEqual(
            summary["cross_validation"]["recommended_policy_descriptive_full_public"],
            "missing_legacy_only",
        )
        self.assertIn("weak_legacy_rescue", results)

    def test_sahi_admission_is_disabled_after_completed_experiment(self):
        tracked = {
            "RUN_CALIBRATION", "RUN_TEMPORAL_CALIBRATION",
            "RUN_RERANKING_EXPERIMENT", "RUN_ORACLE_DIAGNOSTICS",
            "RUN_HYSTERESIS_EXPERIMENT", "RUN_MARGIN_DIAGNOSTICS",
            "RUN_DETECTOR_PROFILE", "RUN_RESOLUTION_AB", "RUN_HYBRID_SAHI",
            "RUN_SAHI_ADMISSION",
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
