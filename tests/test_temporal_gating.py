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
    for node in TREE.body
    if isinstance(node, (ast.FunctionDef, ast.ClassDef))
}


def load_namespace(*names):
    namespace = {
        "math": math,
        "TEMPORAL_MAX_GAP": 5,
        "TEMPORAL_MIN_SEG": 3,
        "TEMPORAL_MAX_GAP_GRID": [4, 5, 6, 7, 8, 10, 12, 15, 20],
        "TEMPORAL_MIN_SEG_GRID": [3, 22, 23, 24, 25, 26, 27, 28, 29, 30],
        "TEMPORAL_CENTER_SPEED_GRID": [None, 0.4, 0.45, 0.5, 0.6, 0.75, 1.0],
        "TEMPORAL_LOG_SCALE_SPEED_GRID": [None],
    }
    dependencies = [
        "temporal_smooth_detections",
        "gap_motion_metrics",
        "gap_is_plausible",
        "temporal_smooth_detections_gated",
    ]
    for name in dependencies + list(names):
        if name not in namespace:
            exec(NODES[name], namespace)
    return namespace


def load_reranking_namespace(*names):
    namespace = {
        "math": math,
        "RERANK_MOTION_SCALE": 0.4,
        "RERANK_TOP_K_GRID": [3, 5],
        "RERANK_NEIGHBOR_RADIUS_GRID": [1, 3, 5],
        "RERANK_WEIGHT_GRID": [0.0, 0.03, 0.06, 0.10],
    }
    dependencies = [
        "gap_motion_metrics",
        "fuse_candidate_score",
        "rank_video_frames",
        "rank_top_k_video_frames",
        "candidate_motion_affinity",
        "candidate_directional_support",
        "rerank_video_candidates",
        "identity_id_from_video_id",
        "build_identity_folds",
        "reranking_config_grid",
    ]
    for name in dependencies + list(names):
        if name not in namespace:
            exec(NODES[name], namespace)
    return namespace


class TemporalGatingTests(unittest.TestCase):
    def test_disabled_gates_match_legacy_exactly(self):
        ns = load_namespace()
        cases = [
            {},
            {0: {"bbox": [0, 0, 10, 10], "score": 0.8}},
            {
                0: {"bbox": [0, 0, 10, 10], "score": 0.8},
                3: {"bbox": [6, 0, 16, 10], "score": 0.7},
                8: {"bbox": [16, 2, 28, 14], "score": 0.9},
            },
        ]
        for frame_best in cases:
            for max_gap in (2, 5):
                for min_seg_len in (1, 3, 5):
                    legacy = ns["temporal_smooth_detections"](
                        frame_best, 10, min_seg_len=min_seg_len, max_gap=max_gap
                    )
                    gated = ns["temporal_smooth_detections_gated"](
                        frame_best,
                        10,
                        min_seg_len=min_seg_len,
                        max_gap=max_gap,
                        max_center_speed=None,
                        max_log_scale_speed=None,
                    )
                    self.assertEqual(gated, legacy)

    def test_large_spatial_jump_is_not_interpolated(self):
        ns = load_namespace()
        frame_best = {
            0: {"bbox": [0, 0, 10, 10], "score": 0.9},
            2: {"bbox": [1000, 0, 1010, 10], "score": 0.9},
        }
        legacy = ns["temporal_smooth_detections"](
            frame_best, 2, min_seg_len=3, max_gap=5
        )
        gated = ns["temporal_smooth_detections_gated"](
            frame_best,
            2,
            min_seg_len=3,
            max_gap=5,
            max_center_speed=8.0,
            max_log_scale_speed=None,
        )
        self.assertEqual([row["frame"] for row in legacy], [0, 1, 2])
        self.assertEqual(gated, [])

    def test_large_scale_jump_is_not_interpolated(self):
        ns = load_namespace()
        frame_best = {
            0: {"bbox": [-5, -5, 5, 5], "score": 0.9},
            2: {"bbox": [-50, -50, 50, 50], "score": 0.9},
        }
        gated = ns["temporal_smooth_detections_gated"](
            frame_best,
            2,
            min_seg_len=1,
            max_gap=5,
            max_center_speed=None,
            max_log_scale_speed=0.3,
        )
        self.assertEqual([row["frame"] for row in gated], [0, 2])

    def test_motion_metrics_are_size_and_time_normalized(self):
        ns = load_namespace()
        center_speed, log_scale_speed = ns["gap_motion_metrics"](
            [0, 0, 10, 10], [20, 0, 30, 10], steps=2
        )
        self.assertAlmostEqual(center_speed, 1.0)
        self.assertAlmostEqual(log_scale_speed, 0.0)

    def test_grid_contains_legacy_control_and_has_expected_size(self):
        ns = load_namespace("temporal_config_grid")
        grid = ns["temporal_config_grid"]()
        self.assertEqual(len(grid), 630)
        self.assertIn(
            {
                "max_gap": 5,
                "min_seg_len": 3,
                "max_center_speed": None,
                "max_log_scale_speed": None,
            },
            grid,
        )
        self.assertEqual(len({tuple(config.items()) for config in grid}), len(grid))

    def test_temporal_production_is_locked_without_calibration_sweeps(self):
        assignments = {}
        for node in TREE.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                if isinstance(target, ast.Name) and target.id in {
                    "RUN_CALIBRATION",
                    "RUN_TEMPORAL_CALIBRATION",
                    "RUN_RERANKING_EXPERIMENT",
                    "USE_TEMPORAL_RERANKING",
                    "USE_TEMPORAL_GATING",
                    "PRODUCTION_TEMPORAL_MAX_GAP",
                    "PRODUCTION_TEMPORAL_MIN_SEG",
                    "PRODUCTION_TEMPORAL_MAX_CENTER_SPEED",
                    "PRODUCTION_TEMPORAL_MAX_LOG_SCALE_SPEED",
                    "WEIGHT_YOLO",
                    "WEIGHT_SIAMESE",
                    "WEIGHT_COLOR",
                    "MATCHING_THRESHOLD",
                }:
                    assignments[target.id] = ast.literal_eval(node.value)
        self.assertFalse(assignments["RUN_CALIBRATION"])
        self.assertFalse(assignments["RUN_TEMPORAL_CALIBRATION"])
        self.assertFalse(assignments["RUN_RERANKING_EXPERIMENT"])
        self.assertFalse(assignments["USE_TEMPORAL_RERANKING"])
        self.assertTrue(assignments["USE_TEMPORAL_GATING"])
        self.assertEqual(assignments["PRODUCTION_TEMPORAL_MAX_GAP"], 7)
        self.assertEqual(assignments["PRODUCTION_TEMPORAL_MIN_SEG"], 24)
        self.assertEqual(assignments["PRODUCTION_TEMPORAL_MAX_CENTER_SPEED"], 0.4)
        self.assertIsNone(assignments["PRODUCTION_TEMPORAL_MAX_LOG_SCALE_SPEED"])
        self.assertEqual(
            (
                assignments["WEIGHT_YOLO"],
                assignments["WEIGHT_SIAMESE"],
                assignments["WEIGHT_COLOR"],
                assignments["MATCHING_THRESHOLD"],
            ),
            (0.425, 0.475, 0.1, 0.54),
        )

    def test_evaluate_config_forwards_temporal_configuration(self):
        function_source = NODES["evaluate_config"]
        self.assertIn("temporal_config=None", function_source)
        self.assertIn("temporal_config=temporal_config", function_source)


class TemporalCandidateRerankingTests(unittest.TestCase):
    @staticmethod
    def video_cache(frames):
        return {
            "has_siamese": False,
            "has_color": False,
            "frames": {
                str(frame): [
                    {"bbox": bbox, "scores": [score, 0.0, 0.0]}
                    for bbox, score in candidates
                ]
                for frame, candidates in frames.items()
            },
        }

    def test_zero_rerank_weight_matches_legacy_top1(self):
        ns = load_reranking_namespace()
        cache = self.video_cache(
            {
                0: [([0, 0, 10, 10], 0.8), ([100, 0, 110, 10], 0.8)],
                1: [([1, 0, 11, 10], 0.7), ([101, 0, 111, 10], 0.9)],
            }
        )
        weights = {"yolo": 1.0, "siamese": 0.0, "color": 0.0}
        legacy, _ = ns["rank_video_frames"](cache, weights)
        top_k = ns["rank_top_k_video_frames"](cache, weights, top_k=3)
        reranked = ns["rerank_video_candidates"](
            top_k, neighbor_radius=3, rerank_weight=0.0
        )
        self.assertEqual(
            {frame: (row["bbox"], row["score"]) for frame, row in reranked.items()},
            {frame: (row["bbox"], row["score"]) for frame, row in legacy.items()},
        )

    def test_bidirectional_support_can_recover_coherent_second_candidate(self):
        ns = load_reranking_namespace()
        cache = self.video_cache(
            {
                0: [([0, 0, 10, 10], 0.80), ([100, 0, 110, 10], 0.40)],
                1: [([100, 0, 110, 10], 0.81), ([1, 0, 11, 10], 0.79)],
                2: [([2, 0, 12, 10], 0.80), ([100, 0, 110, 10], 0.40)],
            }
        )
        weights = {"yolo": 1.0, "siamese": 0.0, "color": 0.0}
        top_k = ns["rank_top_k_video_frames"](cache, weights, top_k=3)
        reranked = ns["rerank_video_candidates"](
            top_k, neighbor_radius=1, rerank_weight=0.10
        )
        self.assertEqual(reranked[1]["bbox"], [1, 0, 11, 10])
        self.assertEqual(reranked[1]["score"], 0.79)
        self.assertGreater(reranked[1]["contextual_score"], reranked[1]["score"])

    def test_identity_pairs_are_kept_in_the_same_fold(self):
        ns = load_reranking_namespace()
        folds = ns["build_identity_folds"](
            [
                "BlackBox_0",
                "LifeJacket_1",
                "CardboardBox_0",
                "BlackBox_1",
                "CardboardBox_1",
                "LifeJacket_0",
            ]
        )
        self.assertEqual(folds["BlackBox"], ["BlackBox_0", "BlackBox_1"])
        self.assertEqual(
            folds["CardboardBox"], ["CardboardBox_0", "CardboardBox_1"]
        )
        self.assertEqual(folds["LifeJacket"], ["LifeJacket_0", "LifeJacket_1"])

    def test_scoring_functions_do_not_read_video_or_identity_names(self):
        for name in (
            "candidate_motion_affinity",
            "candidate_directional_support",
            "rerank_video_candidates",
        ):
            self.assertNotIn("video_id", NODES[name])
            self.assertNotIn("identity", NODES[name])

    def test_reranking_grid_is_small_and_contains_exact_control(self):
        ns = load_reranking_namespace()
        grid = ns["reranking_config_grid"]()
        self.assertEqual(len(grid), 24)
        self.assertEqual(len({tuple(config.items()) for config in grid}), 24)
        self.assertIn(
            {
                "top_k": 3,
                "neighbor_radius": 1,
                "rerank_weight": 0.0,
                "motion_scale": 0.4,
            },
            grid,
        )

    def test_identity_cv_selects_without_splitting_identity_pairs(self):
        video_ids = [
            "BlackBox_0",
            "BlackBox_1",
            "CardboardBox_0",
            "CardboardBox_1",
            "LifeJacket_0",
            "LifeJacket_1",
        ]
        baseline = {
            "mean_st_iou": 0.5,
            "detected_frames": 6,
            "per_video": {video_id: 0.5 for video_id in video_ids},
            "predictions": [{"control": True}],
        }

        def fake_evaluate(cache, gt_map, weights, matching_threshold, config):
            if config["rerank_weight"] == 0.0:
                return baseline
            return {
                "mean_st_iou": 0.51,
                "detected_frames": 6,
                "per_video": {video_id: 0.51 for video_id in video_ids},
                "predictions": [{"control": False}],
            }

        namespace = {
            "RERANK_TOP_K_GRID": [3],
            "RERANK_NEIGHBOR_RADIUS_GRID": [1],
            "RERANK_WEIGHT_GRID": [0.0, 0.1],
            "RERANK_MOTION_SCALE": 0.4,
            "RERANK_MIN_MEAN_CV_DELTA": 0.003,
            "RERANK_MIN_IMPROVED_IDENTITIES": 2,
            "RERANK_MAX_WORST_IDENTITY_DROP": 0.010,
            "tqdm": lambda iterable, desc=None: iterable,
            "evaluate_reranking_config": fake_evaluate,
        }
        function_order = [
            "identity_id_from_video_id",
            "build_identity_folds",
            "reranking_config_grid",
            "reranking_result_row",
            "mean_row_score",
            "reranking_config_from_row",
            "select_reranking_row",
            "select_voted_reranking_config",
            "run_reranking_identity_cv",
        ]
        for name in function_order:
            if name != "evaluate_reranking_config":
                exec(NODES[name], namespace)

        cache = {
            "signature": {"video_folders": video_ids},
            "videos": {video_id: {} for video_id in video_ids},
        }
        _, folds, diagnostics, _ = namespace["run_reranking_identity_cv"](
            cache, {}, {}, 0.54, baseline
        )
        self.assertEqual(len(folds), 3)
        self.assertTrue(diagnostics["weight_zero_control_exact_prediction_parity"])
        self.assertTrue(diagnostics["promotion_passed"])
        self.assertEqual(diagnostics["recommendation_votes"], 3)
        self.assertEqual(diagnostics["recommended_config"]["rerank_weight"], 0.1)
        for fold in folds:
            heldout = fold["heldout_videos"].split("|")
            self.assertEqual(len(heldout), 2)
            self.assertEqual(
                {
                    namespace["identity_id_from_video_id"](video_id)
                    for video_id in heldout
                },
                {fold["heldout_identity"]},
            )


if __name__ == "__main__":
    unittest.main()
