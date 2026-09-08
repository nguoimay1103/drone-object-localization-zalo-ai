"""Exercise the standalone NB05 data code without importing torch or reading images."""
import ast
from collections import Counter
from contextlib import nullcontext, redirect_stdout
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import random
import re
import tempfile
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]


def notebook_tree(path):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code")
    return notebook, ast.parse(source)


NOTEBOOK, TREE = notebook_tree(ROOT / "05-train-siamese.ipynb")
_, BASELINE_TREE = notebook_tree(ROOT / "baselines/05-train-siamese-baseline.ipynb")


def load_data_code():
    namespace = {"Path": Path, "random": random, "re": re, "math": math, "json": json,
                 "hashlib": hashlib, "Counter": Counter, "Dataset": object, "csv": csv}
    names = {"canonical_hash", "resolve_identity_map", "build_data_index", "split_by_identity",
             "SiameseDroneDataset", "split_paths", "audit_splits", "write_locked_json",
             "write_result_json", "prepare_experiment"}
    nodes = [node for node in TREE.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "NB05 data", "exec"), namespace)
    return namespace


def trainer_method(name):
    trainer = next(node for node in TREE.body if isinstance(node, ast.ClassDef) and node.name == "SiameseTrainer")
    return next(node for node in trainer.body if isinstance(node, ast.FunctionDef) and node.name == name)


class NotebookContractTests(unittest.TestCase):
    def test_model_augmentation_train_step_and_loading_are_unchanged(self):
        original = {n.name: n for n in BASELINE_TREE.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        current = {n.name: n for n in TREE.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        for name in ("DroneDegradation", "get_transforms", "SiameseMobileNet"):
            self.assertEqual(ast.dump(original[name]), ast.dump(current[name]), name)
        for class_name, methods in (("SiameseTrainer", ["__init__", "calc_accuracy", "train_epoch"]),
                                     ("SiameseDroneDataset", ["_load_image"])):
            before = {n.name: n for n in original[class_name].body if isinstance(n, ast.FunctionDef)}
            after = {n.name: n for n in current[class_name].body if isinstance(n, ast.FunctionDef)}
            for name in methods:
                self.assertEqual(ast.dump(before[name]), ast.dump(after[name]), f"{class_name}.{name}")

    def test_notebook_is_standalone_and_has_no_old_results(self):
        for cell in NOTEBOOK["cells"]:
            if cell["cell_type"] == "code":
                self.assertEqual(cell["outputs"], [])
                self.assertIsNone(cell["execution_count"])
        self.assertNotIn("widgets", NOTEBOOK["metadata"])
        self.assertNotIn("papermill", NOTEBOOK["metadata"])
        for node in ast.walk(TREE):
            if isinstance(node, ast.ImportFrom):
                self.assertFalse((node.module or "").startswith("drone_localization"))
        self.assertEqual(hashlib.sha256((ROOT / "baselines/05-train-siamese-baseline.ipynb").read_bytes()).hexdigest(),
                         "b3fd43fe88e334cf7c587ebeb5a7765d9fed4c8b666ee8667cc54e41dd272f75")

    def test_validation_averages_by_samples_including_short_final_batch(self):
        class Batch:
            def __init__(self, size):
                self.shape = (size, 3, 224, 224)

            def to(self, device):
                return self

        class Scalar:
            def __init__(self, value):
                self.value = value

            def item(self):
                return self.value

        class Model:
            def eval(self):
                pass

            def __call__(self, a, p, n):
                return a, p, n

        namespace = {"torch": SimpleNamespace(no_grad=nullcontext), "autocast": nullcontext}
        exec(compile(ast.Module(body=[trainer_method("validate")], type_ignores=[]), "NB05 validate", "exec"), namespace)
        batches = [(Batch(n), Batch(n), Batch(n)) for n in (2, 1)]
        trainer = SimpleNamespace(model=Model(), device="cpu", val_loader=batches,
                                  criterion=lambda a, p, n: Scalar(1.0 if a.shape[0] == 2 else 4.0),
                                  calc_accuracy=lambda a, p, n: Scalar(0.5 if a.shape[0] == 2 else 1.0))
        loss, accuracy = namespace["validate"](trainer)
        self.assertEqual(loss, 2.0)
        self.assertAlmostEqual(accuracy, 2/3)
        trainer.val_loader = []
        with self.assertRaises(ValueError):
            namespace["validate"](trainer)


class SamplingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "dataset"
        self.root.mkdir()
        self.ns = load_data_code()
        for relative in ("anchors", "positives", "negatives/hard", "negatives/background"):
            (self.root / relative).mkdir(parents=True, exist_ok=True)
        self.videos = [f"{identity}_{i}" for identity in ("Work_Backpack", "Coat", "Box", "Bag") for i in (0, 1)]
        for video in self.videos:
            (self.root / "positives" / video).mkdir()
            for i in (0, 1):
                self.touch(f"anchors/{video}_ref_{i}.jpg")
            for i in range(3):
                self.touch(f"positives/{video}/f{i*5}_p0_10_20.jpg")
            self.touch(f"negatives/hard/{video}_f0_h0_10_20.jpg")
            self.touch(f"negatives/background/{video}_f0_bg0_30_40.jpg")
        self.index = self.ns["build_data_index"](self.root)
        self.identities = self.ns["resolve_identity_map"](self.index["valid_video_ids"])
        self.train_videos, self.val_videos, _, _ = self.ns["split_by_identity"](
            self.identities, val_identity_ids=["Box", "Coat"],
        )

    def touch(self, relative):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"synthetic filename fixture, not a decodable image")
        return path

    def dataset(self, videos=None, train=True, probability=0.7, seed=2026, index=None):
        return self.ns["SiameseDroneDataset"](
            self.root, self.index if index is None else index, self.identities,
            self.train_videos if videos is None else videos,
            is_train=train, val_seed=seed, pool_probability=probability,
        )

    def test_identity_preserves_full_prefix_and_requires_explicit_exceptions(self):
        self.assertEqual(self.identities["Work_Backpack_0"], "Work_Backpack")
        self.assertEqual(self.identities["Work_Backpack_1"], "Work_Backpack")
        with self.assertRaisesRegex(ValueError, "IDENTITY_OVERRIDES"):
            self.ns["resolve_identity_map"](["clip_without_suffix"])
        self.assertEqual(self.ns["resolve_identity_map"](["clip_without_suffix"], {"clip_without_suffix": "Bag"}),
                         {"clip_without_suffix": "Bag"})

    def test_identity_groups_never_cross_split_and_split_is_deterministic(self):
        fn = self.ns["split_by_identity"]
        a = fn(self.identities, seed=42)
        random.seed(901)
        b = fn(self.identities, seed=42)
        self.assertEqual(a, b)
        train, val, train_ids, val_ids = a
        self.assertFalse(set(train_ids) & set(val_ids))
        self.assertEqual(set(train) | set(val), set(self.videos))
        for identity in set(self.identities.values()):
            videos = {v for v in self.videos if self.identities[v] == identity}
            self.assertTrue(videos <= set(train) or videos <= set(val))
        with self.assertRaises(ValueError):
            fn({"Bag_0": "Bag", "Bag_1": "Bag"})
        with self.assertRaises(ValueError):
            fn(self.identities, val_identity_ids=list(set(self.identities.values())))

    def test_other_positive_negatives_always_have_a_different_identity(self):
        dataset = self.dataset(probability=0.0)
        rng = random.Random(42)
        for _ in range(500):
            record = dataset.sample_triplet(rng)
            dataset.validate_triplet(record)
            self.assertEqual(record["negative_source"], "other_identity")
            self.assertNotEqual(record["identity_id"], record["negative_source_identity"])
            self.assertIn(record["positive"], self.index["positives"][record["video_id"]])

    def test_negative_pool_and_all_image_paths_are_disjoint_between_splits(self):
        train = self.dataset(probability=1.0)
        val = self.dataset(self.val_videos, train=False, probability=1.0)
        self.assertFalse(set(train.negatives) & set(val.negatives))
        self.assertEqual(len(train.negatives) + len(val.negatives), len(self.index["negatives"]))
        self.assertEqual({r["video_id"] for r in train.negative_records}, set(self.train_videos))
        report = self.ns["audit_splits"](train, val, sample_count=500)
        self.assertEqual(report["same_identity_other_positive_negatives"], 0)
        self.assertEqual(report["train_sampled_negative_sources"], {"pool": 500})

    def test_validation_is_fixed_covers_each_positive_and_does_not_consume_global_rng(self):
        before = random.getstate()
        a = self.dataset(self.val_videos, train=False)
        self.assertEqual(before, random.getstate())
        b = self.dataset(self.val_videos, train=False)
        self.assertEqual(a.fixed_triplets, b.fixed_triplets)
        positives = [r["positive"] for r in a.fixed_triplets]
        expected = [p for video in self.val_videos for p in self.index["positives"][video]]
        self.assertEqual(Counter(positives), Counter(expected))
        a._load_image = lambda path: path
        self.assertEqual(a[0], a[0])
        self.assertEqual(before, random.getstate())
        c = self.dataset(self.val_videos, train=False, seed=17)
        self.assertNotEqual(a.fixed_triplets, c.fixed_triplets)

    def test_single_identity_uses_only_pool_and_no_legal_negative_fails(self):
        videos = ["Bag_0", "Bag_1"]
        dataset = self.dataset(videos, probability=0.0)
        record = dataset.sample_triplet(random.Random(1))
        self.assertEqual(record["negative_source"], "pool")
        dataset.validate_triplet(record)
        no_pool = dict(self.index, negatives=[])
        with self.assertRaisesRegex(ValueError, "Không có negative"):
            self.dataset(videos, index=no_pool)
        multi_identity = self.dataset(index=no_pool, probability=1.0)
        self.assertEqual(multi_identity.sample_triplet(random.Random(1))["negative_source"], "other_identity")

    def test_unknown_negative_provenance_is_rejected_until_explicitly_mapped(self):
        relative = "negatives/hard/custom.jpg"
        self.touch(relative)
        with self.assertRaisesRegex(ValueError, "provenance"):
            self.ns["build_data_index"](self.root)
        index = self.ns["build_data_index"](self.root, {relative: "Bag_0"})
        self.assertEqual(next(r["video_id"] for r in index["negatives"] if r["path"] == relative), "Bag_0")

    def test_integrity_checker_rejects_tampered_cross_split_and_same_identity_negative(self):
        dataset = self.dataset(probability=0.0)
        record = dataset.sample_triplet(random.Random(1))
        other_same = next(v for v in self.train_videos if v != record["video_id"] and self.identities[v] == record["identity_id"])
        bad = dict(record, negative_video_id=other_same, negative=self.index["positives"][other_same][0])
        with self.assertRaisesRegex(ValueError, "Cùng physical identity"):
            dataset.validate_triplet(bad)
        bad = dict(record, negative_video_id=self.val_videos[0])
        with self.assertRaisesRegex(ValueError, "ngoài split"):
            dataset.validate_triplet(bad)

    def test_prepare_can_be_repeated_but_refuses_changed_validation_or_unrelated_outputs(self):
        output = Path(self.temp.name) / "prepared"
        self.ns.update(ROOT_FOLDER=str(self.root), SAVE_DIR=str(output), EXPERIMENT_NAME='test_mining_v2', NEGATIVE_VIDEO_OVERRIDES={},
                       IDENTITY_OVERRIDES={}, VAL_FRACTION=0.2, SPLIT_SEED=42, TRAIN_SEED=42,
                       VAL_TRIPLET_SEED=2026, VAL_IDENTITY_IDS=["Box", "Coat"], POOL_NEGATIVE_PROBABILITY=0.7,
                       BATCH_SIZE=128, LEARNING_RATE=1e-4, NUM_EPOCHS=15, MARGIN=0.5,
                       get_transforms=lambda train: (None, None))
        with redirect_stdout(io.StringIO()):
            self.ns["prepare_experiment"]()
            self.ns["prepare_experiment"]()
        manifest = json.loads((output / "split_manifest.json").read_text(encoding="utf-8"))
        triplets = json.loads((output / "val_triplets.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["val_triplets_sha256"], self.ns["canonical_hash"](triplets))
        self.ns["VAL_TRIPLET_SEED"] = 17
        with redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "SAVE_DIR mới"):
            self.ns["prepare_experiment"]()
        unrelated = Path(self.temp.name) / "old_baseline"
        unrelated.mkdir()
        (unrelated / "siamese_mobilenet_best.pth").write_bytes(b"do not overwrite")
        self.ns["SAVE_DIR"] = str(unrelated)
        with redirect_stdout(io.StringIO()), self.assertRaises(FileExistsError):
            self.ns["prepare_experiment"]()
        self.assertEqual((unrelated / "siamese_mobilenet_best.pth").read_bytes(), b"do not overwrite")

    def test_fit_logs_history_and_selects_best_fixed_validation_loss(self):
        output = Path(self.temp.name) / "fit"
        output.mkdir()
        losses = iter([0.3, 0.2, 0.4])
        fake_torch = SimpleNamespace(save=lambda state, path: path.write_bytes(json.dumps(state).encode()))
        self.ns.update(torch=fake_torch, BATCH_SIZE=128, LEARNING_RATE=1e-4,
                       EXPERIMENT_NAME='test_mining_v2')
        exec(compile(ast.Module(body=[trainer_method("fit")], type_ignores=[]), "NB05 fit", "exec"), self.ns)
        trainer = SimpleNamespace(save_dir=output, best_val_loss=float("inf"),
                                  optimizer=SimpleNamespace(param_groups=[{"lr": 1e-4}]),
                                  scheduler=SimpleNamespace(step=lambda: None),
                                  train_epoch=lambda epoch: (0.5 / epoch, 0.8), validate=lambda: (next(losses), 0.9),
                                  model=SimpleNamespace(state_dict=lambda: {"test_fixture": True}))
        with redirect_stdout(io.StringIO()):
            self.ns["fit"](trainer, 3)
        report = json.loads((output / "run_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(report["best_epoch"], 2)
        self.assertEqual(report["best_val_loss"], 0.2)
        self.assertIsNone(report["local_st_iou"])
        with (output / "training_history.csv").open(encoding="utf-8") as handle:
            self.assertEqual(len(list(csv.DictReader(handle))), 3)


if __name__ == "__main__":
    unittest.main()
