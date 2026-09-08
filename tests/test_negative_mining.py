import ast
import collections
import hashlib
import json
import os
import random
import re
import shutil
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ACTIVE = ROOT / "04-data-prep-matching.ipynb"
BASELINE = ROOT / "baselines" / "04-data-prep-matching-baseline.ipynb"
BASELINE_SHA256 = "958e3a4fd7b304dfd5563e697d04abfc0edb03e71a2104582513c01c58b9f61a"


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


NB, SOURCE = notebook_source(ACTIVE)
TREE = ast.parse(SOURCE)
NODES = {
    node.name: ast.get_source_segment(SOURCE, node)
    for node in TREE.body
    if isinstance(node, ast.FunctionDef)
}


def literal_assignments():
    values = {}
    for node in TREE.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                values[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                pass
    return values


def mining_namespace(*function_names):
    namespace = {
        "collections": collections,
        "json": json,
        "os": os,
        "Path": Path,
        "random": random,
        "re": re,
        "shutil": shutil,
        "GT_EXCLUSION_MARGIN_RATIO": 0.15,
        "GT_EXCLUSION_MARGIN_PX": 4,
        "MIN_CROP_SIZE": 10,
        "BACKGROUND_SCALE_RANGE": (0.75, 1.50),
        "BACKGROUND_FALLBACK_SIZE_RANGE": (50, 200),
        "BACKGROUND_MAX_ATTEMPTS": 100,
        "BACKGROUND_MAX_MUTUAL_IOU": 0.50,
        "IDENTITY_OVERRIDES": {},
    }
    dependencies = [
        "compute_iou", "intersection_area", "clip_box", "valid_box", "expand_gt_box",
        "negative_quality", "propose_background_box", "random_crop_background",
    ]
    for name in dependencies + list(function_names):
        if name not in namespace:
            exec(NODES[name], namespace)
    return namespace


class DummyFrame:
    shape = (400, 600, 3)


class NotebookContractTests(unittest.TestCase):
    def test_baseline_backup_is_exact_and_active_notebook_is_clean(self):
        self.assertEqual(hashlib.sha256(BASELINE.read_bytes()).hexdigest(), BASELINE_SHA256)
        for cell in NB["cells"]:
            if cell.get("cell_type") == "code":
                self.assertIsNone(cell.get("execution_count"))
                self.assertEqual(cell.get("outputs"), [])
        compile(SOURCE, str(ACTIVE), "exec")

    def test_reproducible_defaults_preserve_sampling_and_use_task_detector(self):
        values = literal_assignments()
        self.assertEqual(values["SAMPLING_RATE"], 5)
        self.assertEqual(values["MAX_NEG_PER_POS"], 2)
        self.assertEqual(values["NUM_BG_PER_FRAME"], 2)
        self.assertEqual(values["CONF_THRESHOLD"], 0.05)
        self.assertEqual(values["RANDOM_SEED"], 42)
        self.assertFalse(values["RUN_MINING"])
        self.assertFalse(values["ALLOW_OVERWRITE"])
        self.assertTrue(values["REUSE_FROZEN_POSITIVES"])
        self.assertIn("yolo_drone", values["MINING_MODEL_PATH"])
        self.assertNotIn("yolov8n.pt", SOURCE)
        install_source = "".join(NB["cells"][1]["source"])
        self.assertIn("ultralytics==8.3.221", install_source)

    def test_no_silent_exception_handler_remains(self):
        silent_handlers = []
        for node in ast.walk(TREE):
            if isinstance(node, ast.ExceptHandler) and any(isinstance(item, ast.Pass) for item in node.body):
                silent_handlers.append(node.lineno)
        self.assertEqual(silent_handlers, [])

    def test_negative_filenames_remain_compatible_with_nb05_provenance(self):
        pattern = re.compile(r"(.+)_f(\d+)_(h|bg)\d+_\d+_\d+\.jpg")
        self.assertIsNotNone(pattern.fullmatch("Backpack_0_f125_h0_10_20.jpg"))
        self.assertIsNotNone(pattern.fullmatch("Backpack_0_f125_bg1_10_20.jpg"))
        self.assertIn("prefix=f'{video_id}_f{frame_idx}_h{hard_index}'", SOURCE)
        self.assertIn("prefix=f'{video_id}_f{frame_idx}_bg{background_index}'", SOURCE)


class NegativeQualityTests(unittest.TestCase):
    def test_large_candidate_containing_small_target_is_rejected_despite_tiny_iou(self):
        ns = mining_namespace()
        target = [100, 100, 110, 110]
        candidate = [0, 0, 500, 500]
        self.assertLess(ns["compute_iou"](candidate, target), 0.05)
        clean, reason, metrics = ns["negative_quality"](candidate, [target], 500, 500)
        self.assertFalse(clean)
        self.assertEqual(reason, "expanded_gt_overlap")
        self.assertAlmostEqual(metrics["max_gt_coverage"], 1.0)

    def test_margin_rejects_near_target_and_disjoint_candidate_is_clean(self):
        ns = mining_namespace()
        target = [100, 100, 120, 120]
        near = [88, 102, 98, 118]
        clean, reason, _ = ns["negative_quality"](near, [target], 500, 500)
        self.assertFalse(clean)
        self.assertEqual(reason, "expanded_gt_overlap")
        clean, reason, metrics = ns["negative_quality"]([10, 10, 40, 40], [target], 500, 500)
        self.assertTrue(clean)
        self.assertEqual(reason, "clean")
        self.assertEqual(metrics["max_gt_coverage"], 0.0)

    def test_background_sampling_is_seeded_target_sized_and_clean(self):
        ns = mining_namespace()
        target = [250, 150, 290, 170]
        target_sizes = [(40, 20)]
        first, rejected_first = ns["random_crop_background"](
            DummyFrame(), [target], target_sizes, random.Random(123), num_crops=2
        )
        second, rejected_second = ns["random_crop_background"](
            DummyFrame(), [target], target_sizes, random.Random(123), num_crops=2
        )
        self.assertEqual(first, second)
        self.assertEqual(rejected_first, rejected_second)
        self.assertEqual(len(first), 2)
        for box, _ in first:
            self.assertGreaterEqual(box[2] - box[0], 30)
            self.assertLessEqual(box[2] - box[0], 60)
            clean, _, _ = ns["negative_quality"](box, [target], 600, 400)
            self.assertTrue(clean)

    def test_identity_rule_preserves_entire_prefix(self):
        ns = mining_namespace("resolve_identity")
        self.assertEqual(ns["resolve_identity"]("Red_Backpack_0"), "Red_Backpack")
        self.assertEqual(ns["resolve_identity"]("Person1_1"), "Person1")
        self.assertEqual(ns["resolve_identity"]("exception_name"), "exception_name")

    def test_existing_output_is_refused_without_explicit_overwrite(self):
        ns = mining_namespace("prepare_output_dir")
        with tempfile.TemporaryDirectory() as temp_dir:
            Path(temp_dir, "existing.txt").write_text("do not delete", encoding="utf-8")
            ns["OUTPUT_DIR"] = temp_dir
            ns["ALLOW_OVERWRITE"] = False
            with self.assertRaisesRegex(FileExistsError, "not empty"):
                ns["prepare_output_dir"]()
            self.assertTrue(Path(temp_dir, "existing.txt").exists())

    def test_output_integrity_checks_manifest_counts_provenance_and_overlap(self):
        ns = mining_namespace("validate_mining_output")
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for folder in ("positives/Video_0", "negatives/hard", "negatives/background", "anchors"):
                (root / folder).mkdir(parents=True)
            (root / "positives/Video_0/f0_p0_1_1.jpg").write_bytes(b"jpg")
            (root / "anchors/Video_0_ref_0.jpg").write_bytes(b"jpg")
            relative = "negatives/hard/Video_0_f0_h0_20_20.jpg"
            (root / relative).write_bytes(b"jpg")
            record = {
                "path": relative,
                "max_iou": 0.0,
                "max_gt_coverage": 0.0,
            }
            manifest = root / "negative_manifest.jsonl"
            manifest.write_text(json.dumps(record) + "\n", encoding="utf-8")
            ns["OUTPUT_DIR"] = temp_dir
            summary = {"totals": {"hard_negatives": 1, "background_negatives": 0, "anchors": 1}}
            diagnostics = [{
                "sampled_positive_boxes": 2,
                "frozen_positive_files": 1,
                "frozen_anchor_files": 1,
            }]
            result = ns["validate_mining_output"](summary, diagnostics)
            self.assertEqual(result["status"], "passed")
            record["max_gt_coverage"] = 1.0
            manifest.write_text(json.dumps(record) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "overlaps GT"):
                ns["validate_mining_output"](summary, diagnostics)

    def test_frozen_anchors_and_positives_are_copied_byte_for_byte(self):
        ns = mining_namespace("stage_frozen_positives")
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            source = temp / "source"
            output = temp / "output"
            (source / "anchors").mkdir(parents=True)
            (source / "positives/Video_0").mkdir(parents=True)
            anchor_bytes = b"anchor exact bytes\x00\x01"
            positive_bytes = b"positive exact bytes\x02\x03"
            (source / "anchors/Video_0_ref_0.jpg").write_bytes(anchor_bytes)
            (source / "positives/Video_0/f0_p0_1_1.jpg").write_bytes(positive_bytes)
            output.mkdir()
            ns["FROZEN_POSITIVE_DATASET_DIR"] = str(source)
            ns["OUTPUT_DIR"] = str(output)
            ns["REUSE_FROZEN_POSITIVES"] = True
            ns["stage_frozen_positives"]()
            self.assertEqual((output / "anchors/Video_0_ref_0.jpg").read_bytes(), anchor_bytes)
            self.assertEqual((output / "positives/Video_0/f0_p0_1_1.jpg").read_bytes(), positive_bytes)


if __name__ == "__main__":
    unittest.main()
