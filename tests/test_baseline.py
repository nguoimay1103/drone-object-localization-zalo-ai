"""CPU-only regression checks against the actual NB06 source and CLI contracts.

These do not certify tensor/GPU/video parity; a real golden-video run is required.
"""
import ast
from dataclasses import replace
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from drone_localization.artifacts import sha256_file, write_json
from drone_localization.config import Config, Inference, Paths, load_config
from drone_localization.evaluation import evaluate_predictions, load_ground_truth, prediction_maps
from drone_localization.merge import merge_runs
from drone_localization.metrics import compute_st_iou_video
from drone_localization.preflight import preflight
from drone_localization.scoring import select_best_candidate
from drone_localization.temporal import temporal_smooth_detections

NB = json.loads((ROOT / "06-inference-main.ipynb").read_text(encoding="utf-8"))
NB_SOURCE = "\n".join(
    "".join(cell.get("source", []))
    for cell in NB["cells"]
    if cell.get("cell_type") == "code"
    and not any(
        line.lstrip().startswith(("!", "%"))
        for line in "".join(cell.get("source", [])).splitlines()
    )
)
NB_TREE = ast.parse(NB_SOURCE)
NB_NODES = {n.name: n for n in NB_TREE.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}


def original_functions(names):
    namespace = {"TEMPORAL_MIN_SEG": 3, "TEMPORAL_MAX_GAP": 5}
    exec(compile(ast.Module(body=[NB_NODES[n] for n in names], type_ignores=[]), "NB06", "exec"), namespace)
    return namespace


class Rename(ast.NodeTransformer):
    def __init__(self, names):
        self.names = names

    def visit_Name(self, node):
        return ast.copy_location(ast.Name(id=self.names.get(node.id, node.id), ctx=node.ctx), node)


class Box(list):
    def tolist(self):
        return list(self)


class LegacyParityTests(unittest.TestCase):
    def test_metric_matches_original_on_random_frame_sets(self):
        original = original_functions(["iou_box", "compute_st_iou_video"])["compute_st_iou_video"]
        rng = random.Random(42)
        for _ in range(100):
            gt = {f: [0, 0, 10, 10] for f in range(12) if rng.random() < 0.6}
            pred = {f: [rng.randrange(8), 0, 12, 10] for f in range(12) if rng.random() < 0.6}
            self.assertEqual(compute_st_iou_video(gt, pred), original(gt, pred))

    def test_temporal_matches_original_including_rounding_and_custom_gaps(self):
        original = original_functions(["temporal_smooth_detections"])["temporal_smooth_detections"]
        rng = random.Random(17)
        for gap, minimum in ((5, 3), (4, 5), (0, 1)):
            for _ in range(30):
                detections = {f: {"bbox": [f + 0.5, 1.5, f + 10.5, 9.5], "score": rng.random()}
                              for f in range(20) if rng.random() < 0.3}
                self.assertEqual(temporal_smooth_detections(detections, 19, minimum, gap), original(detections, 19, minimum, gap))
        self.assertEqual(temporal_smooth_detections({}, -1), [])

    def test_motion_gate_blocks_implausible_interpolation(self):
        detections = {
            0: {"bbox": [0, 0, 10, 10], "score": 0.9},
            2: {"bbox": [1000, 0, 1010, 10], "score": 0.9},
        }
        legacy = temporal_smooth_detections(
            detections, 2, min_seg_len=3, max_gap=5
        )
        gated = temporal_smooth_detections(
            detections, 2, min_seg_len=3, max_gap=5,
            max_center_speed=0.4,
        )
        self.assertEqual([row["frame"] for row in legacy], [0, 1, 2])
        self.assertEqual(gated, [])

    def test_scoring_handles_missing_features_and_stable_ties(self):
        rng = random.Random(23)
        settings = Inference()
        for has_ref in (False, True):
            for has_hist in (False, True):
                for repeated in (False, True):
                    boxes = [Box([i, 0, i + 10, 10]) for i in range(5)]
                    scores = [0.5] * 5 if repeated else [rng.random() for _ in range(5)]
                    valid = [4, 1, 3]
                    box, score = select_best_candidate(boxes, scores, scores[:3], scores[2:], valid, has_ref, has_hist, settings)
                    self.assertIn(box, boxes)
                    self.assertGreaterEqual(score, 0.0)
                    if repeated and has_ref and has_hist:
                        self.assertIs(box, boxes[valid[0]])
        self.assertEqual(select_best_candidate([], [], [], [], [], False, False, settings), (None, -1.0))

    def test_extracted_model_reference_and_temporal_bodies_match_notebook(self):
        groups = {
            "models/siamese.py": ["SiameseMobileNet", "get_inference_transforms"],
            "matching.py": ["load_reference_embeddings", "load_reference_histogram", "compute_hs_histogram", "calculate_color_score"],
        }
        names = {"samples_dir": "TEST_DATA_DIR", "device": "DEVICE", "use_multiscale_ref": "USE_MULTISCALE_REF", "ref_scales": "REF_SCALES"}
        for file, functions in groups.items():
            tree = ast.parse((ROOT / "src/drone_localization" / file).read_text(encoding="utf-8"))
            extracted = {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
            for name in functions:
                original = ast.Module(body=NB_NODES[name].body, type_ignores=[])
                current = Rename(names).visit(ast.Module(body=extracted[name].body, type_ignores=[]))
                self.assertEqual(ast.dump(original), ast.dump(current), name)

class RunnerContractsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.samples = self.root / "samples"
        self.samples.mkdir()
        for video in ("A", "B", "C"):
            folder = self.samples / video
            folder.mkdir()
            (folder / "drone_video.mp4").write_bytes(b"preflight does not decode this fixture")
        self.yolo = self.root / "detector.pt"
        self.siam = self.root / "matcher.pth"
        self.yolo.write_bytes(b"detector fixture")
        self.siam.write_bytes(b"matcher fixture")
        self.gt_path = self.root / "gt.json"
        box = {"frame": 0, "x1": 0, "y1": 0, "x2": 10, "y2": 10}
        write_json(self.gt_path, [{"video_id": v, "annotations": [{"bboxes": [box]}]} for v in ("A", "B", "C")])
        self.config = Config(Paths(self.samples, self.yolo, self.siam, self.root / "out", self.gt_path))

    def test_shards_are_disjoint_and_cover_all_videos(self):
        a, _ = preflight(self.config, 0, 2)
        b, _ = preflight(self.config, 1, 2)
        self.assertFalse(set(a["video_ids"]) & set(b["video_ids"]))
        self.assertEqual(set(a["video_ids"]) | set(b["video_ids"]), {"A", "B", "C"})
        with self.assertRaises(ValueError):
            preflight(self.config, 2, 2)

    def test_preflight_rejects_missing_gt_and_wrong_weight_hash(self):
        write_json(self.gt_path, [{"video_id": "A", "annotations": []}])
        with self.assertRaisesRegex(ValueError, "Missing GT"):
            preflight(self.config)
        from drone_localization.config import Checkpoints
        cfg = replace(self.config, checkpoints=Checkpoints(yolo_sha256="0" * 64))
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            preflight(cfg)

    def test_annotation_frame_origin_and_duplicates(self):
        gt = load_ground_truth(self.gt_path)
        self.assertIn(0, gt["A"])
        duplicate = {"frame": 0, "x1": 0, "y1": 0, "x2": 10, "y2": 10}
        write_json(self.gt_path, [{"video_id": "A", "annotations": [{"bboxes": [duplicate, duplicate]}]}])
        with self.assertRaisesRegex(ValueError, "duplicate frame"):
            load_ground_truth(self.gt_path)

    def test_evaluation_keeps_empty_predictions_and_rejects_missing_video(self):
        records = [{"video_id": v, "detections": []} for v in ("A", "B", "C")]
        result = evaluate_predictions(records, load_ground_truth(self.gt_path), ["A", "B", "C"])
        self.assertEqual(result["mean_st_iou"], 0.0)
        self.assertEqual(result["video_count"], 3)
        with self.assertRaisesRegex(ValueError, "video set mismatch"):
            evaluate_predictions(records[:-1], load_ground_truth(self.gt_path), ["A", "B", "C"])
        with self.assertRaisesRegex(ValueError, "duplicate prediction"):
            prediction_maps(records + records[:1])

    def write_yaml(self):
        # JSON is a YAML subset, so this also tests the YAML loader without a writer dependency.
        path = self.root / "runner.yaml"
        write_json(path, {"paths": {"samples_dir": "samples", "annotations": "gt.json", "yolo_weights": "detector.pt",
                                    "siamese_weights": "matcher.pth", "output_dir": "out"}})
        return path

    def test_config_resolves_relative_paths_and_no_eval(self):
        cfg = load_config(self.write_yaml())
        self.assertEqual(cfg.paths.samples_dir, self.samples)
        cfg = load_config(self.write_yaml(), device="cuda:1", no_eval=True)
        selection, gt = preflight(cfg)
        self.assertIsNone(gt)
        self.assertIsNone(selection["annotation_sha256"])
        for bad in (
            {"use_tta": "false"}, {"temporal_max_gap": -1},
            {"temporal_max_center_speed": 0}, {"device": "cuda:0,1"},
            {"matching_threshold": float("nan")},
        ):
            with self.assertRaises(ValueError):
                Inference(**bad)

    def test_locked_production_config_contains_best_global_parameters(self):
        config = load_config(ROOT / "configs/production_0_740013.yaml")
        settings = config.inference
        self.assertEqual(
            (settings.weight_yolo, settings.weight_siamese, settings.weight_color),
            (0.425, 0.475, 0.10),
        )
        self.assertEqual(settings.matching_threshold, 0.54)
        self.assertEqual(settings.temporal_max_gap, 7)
        self.assertEqual(settings.temporal_min_seg, 24)
        self.assertEqual(settings.temporal_max_center_speed, 0.4)
        self.assertIsNone(settings.temporal_max_log_scale_speed)

    def test_cli_dry_run_is_read_only_and_does_not_need_gpu_libraries(self):
        proc = subprocess.run([sys.executable, "-B", str(ROOT / "scripts/run_inference.py"),
                               "--config", str(self.write_yaml()), "--dry-run"], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        result = json.loads(proc.stdout)
        self.assertEqual(result["selection"]["video_ids"], ["A", "B", "C"])
        self.assertFalse(result["gpu_or_video_decode_tested"])
        self.assertFalse(self.config.paths.output_dir.exists())

    def test_merge_recomputes_global_mean_and_rejects_duplicates_and_changed_gt(self):
        gt = load_ground_truth(self.gt_path)
        good = {"video_id": "A", "detections": [{"bboxes": [{"frame": 0, "x1": 0, "y1": 0, "x2": 10, "y2": 10}]}]}
        empty = lambda v: {"video_id": v, "detections": []}
        runtime = {k: "fixture" for k in ("python", "ultralytics", "torch", "torchvision", "opencv", "numpy", "pillow", "cuda", "cudnn")}
        dirs = []
        for i, records in enumerate(([good, empty("C")], [empty("B")])):
            folder = self.root / f"worker{i}"
            folder.mkdir()
            write_json(folder / "predictions.json", records)
            write_json(folder / "run_manifest.json", {"status": "completed", "config": self.config.to_dict(), "runtime": runtime,
                       "selection": {"checkpoint_hashes": {"yolo": "a", "siamese": "b"}, "annotation_sha256": sha256_file(self.gt_path),
                                     "video_ids": [r["video_id"] for r in records]}, "source_hashes": {"code": "same"}, "frame_index_origin": 0,
                       "predictions_sha256": sha256_file(folder / "predictions.json")})
            dirs.append(folder)
        summary = merge_runs(dirs, self.root / "merged", ["A", "B", "C"], gt,
                             expected_annotation_sha256=sha256_file(self.gt_path))
        self.assertEqual(summary["evaluation"]["mean_st_iou"], 1/3)
        self.assertIsNone(summary["parallel_pipeline_fps"])
        with self.assertRaisesRegex(ValueError, "duplicate prediction"):
            merge_runs([dirs[0], dirs[0]], self.root / "duplicate", ["A", "C"])
        with self.assertRaisesRegex(ValueError, "Merge GT differs"):
            merge_runs(dirs, self.root / "wrong_gt", ["A", "B", "C"], gt, expected_annotation_sha256="different")


if __name__ == "__main__":
    unittest.main()
