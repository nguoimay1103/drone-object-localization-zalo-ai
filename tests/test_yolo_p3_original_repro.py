import ast
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "03_train_yolo.ipynb"

notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
sources = ["".join(cell.get("source", [])) for cell in notebook["cells"]]
python_source = "\n".join(
    source for source in sources
    if not any(line.lstrip().startswith(("!", "%")) for line in source.splitlines())
)
tree = ast.parse(python_source)
functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
config = {}
exec(sources[2], config)


def function_source(name):
    return ast.get_source_segment(python_source, functions[name])


class YoloP3OriginalReproductionTests(unittest.TestCase):
    def test_reproduction_is_the_only_active_mode(self):
        self.assertTrue(config["RUN_ORIGINAL_P3_REPRODUCTION"])
        self.assertFalse(config["RUN_PAIRED_TRAINING"])
        dispatcher = function_source("main")
        self.assertIn("active_modes != 1", dispatcher)
        self.assertIn("return run_original_p3_reproduction()", dispatcher)

    def test_original_seed_single_gpu_and_batch_are_locked(self):
        self.assertEqual(config["REPRO_SEED"], 0)
        self.assertEqual(config["REPRO_DEVICE"], "0")
        self.assertEqual(config["REPRO_BATCH"], 16)
        self.assertEqual(config["REPRO_WORKERS"], 2)
        self.assertEqual(config["EPOCHS"], 70)
        self.assertEqual(config["IMGSZ"], 640)

    def test_original_initialization_contract_is_restored(self):
        original_init = config["P3_ORIGINAL_INIT_MODEL_YAML"]
        paired_p3 = config["P3_MODEL_YAML"]
        expected = paired_p3.replace("nc: 1\nscale: n\n", "nc: 80\n", 1)
        self.assertEqual(original_init, expected)
        self.assertIn("\nnc: 80\nscales:\n", original_init)
        self.assertNotIn("\nscale: n\n", original_init)
        self.assertIn("[[16, 19, 22], 1, Detect, [nc]]", original_init)

    def test_explicit_training_knobs_match_original_notebook(self):
        source = function_source("original_reproduction_training_arguments")
        expected = {
            '"epochs": EPOCHS', '"imgsz": IMGSZ', '"batch": REPRO_BATCH',
            '"lr0": 0.003', '"lrf": 0.01', '"workers": REPRO_WORKERS',
            '"amp": True', '"mixup": 0.05', '"degrees": 5.0',
            '"shear": 2.0', '"device": REPRO_DEVICE', '"seed": REPRO_SEED',
        }
        for fragment in expected:
            self.assertIn(fragment, source)
        self.assertNotIn("TRAINING_HYPERPARAMETERS", source)

    def test_known_data_and_weight_artifacts_are_hard_guarded(self):
        validate = function_source("validate_original_reproduction_inputs")
        self.assertIn("REPRO_EXPECTED_PRETRAINED_SHA256", validate)
        self.assertIn("REPRO_EXPECTED_SOURCE_YAML_SHA256", validate)
        self.assertIn("REPRO_EXPECTED_SPLITS", validate)
        self.assertIn("REPRO_EXPECTED_VAL_UNIQUE_INSTANCES", validate)
        self.assertEqual(config["REPRO_EXPECTED_SPLITS"]["train"]["image_count"], 18965)
        self.assertEqual(config["REPRO_EXPECTED_SPLITS"]["val"]["image_count"], 6041)
        self.assertEqual(config["REPRO_EXPECTED_VAL_UNIQUE_INSTANCES"], 6053)

    def test_pretrained_load_precedes_training_without_scratch_fallback(self):
        source = function_source("run_original_p3_reproduction")
        load_position = source.index("initial_model = initial_model.load(str(pretrained_path))")
        train_position = source.index("model.train(**arguments)")
        self.assertLess(load_position, train_position)
        self.assertNotIn("train from scratch", source.lower())

    def test_saved_checkpoint_topology_is_verified_and_fully_validated(self):
        source = function_source("run_original_p3_reproduction")
        self.assertIn("REPRO_EXPECTED_FINAL_STRIDES", source)
        self.assertIn("REPRO_EXPECTED_FINAL_PARAMETER_COUNT", source)
        self.assertIn("trained_model.val(", source)
        self.assertIn('name="best_full_val_single_gpu"', source)
        self.assertIn('"input_integrity_passed"', source)

    def test_outputs_are_auditable_and_do_not_use_public_gt(self):
        source = function_source("run_original_p3_reproduction")
        for artifact in (
            "environment.json", "label_semantic_stats.json", "run_summary.json",
            "training_history.csv", "p3_original_reproduction_best.pt",
        ):
            self.assertIn(artifact, source)
        self.assertIn('"public_test_used_for_training_or_selection": False', source)
        self.assertNotIn("public_test", sources[2].lower())
        self.assertNotIn("gt_ann", sources[2].lower())


if __name__ == "__main__":
    unittest.main()
