import ast
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "03_train_yolo.ipynb"
P3_YAML = ROOT / "configs/models/yolo_drone_ghost_p3.yaml"
P2_YAML = ROOT / "configs/models/yolo_drone_ghost_p2.yaml"

notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
sources = ["".join(cell.get("source", [])) for cell in notebook["cells"]]
python_source = "\n".join(
    source for source in sources
    if not any(line.lstrip().startswith(("!", "%")) for line in source.splitlines())
)
tree = ast.parse(python_source)


def literal_assignments():
    assignments = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                try:
                    assignments[target.id] = ast.literal_eval(node.value)
                except (ValueError, TypeError):
                    pass
    return assignments


ASSIGNMENTS = literal_assignments()
CONFIG = {}
exec(sources[2], CONFIG)
FUNCTIONS = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}


class YoloP2TrainingABTests(unittest.TestCase):
    def test_runtime_and_training_protocol_are_locked(self):
        self.assertIn("ultralytics==8.3.221", sources[0])
        self.assertEqual(ASSIGNMENTS["EXPECTED_ULTRALYTICS_VERSION"], "8.3.221")
        self.assertEqual(ASSIGNMENTS["SEED"], 2026)
        self.assertEqual(ASSIGNMENTS["EPOCHS"], 70)
        self.assertEqual(ASSIGNMENTS["IMGSZ"], 640)
        self.assertEqual(ASSIGNMENTS["BATCH_PER_GPU"], 16)
        self.assertFalse(ASSIGNMENTS["RUN_PAIRED_TRAINING"])
        self.assertTrue(ASSIGNMENTS["RUN_ORIGINAL_P3_REPRODUCTION"])

    def test_embedded_architectures_match_canonical_configs(self):
        self.assertEqual(ASSIGNMENTS["P3_MODEL_YAML"], P3_YAML.read_text(encoding="utf-8"))
        self.assertEqual(ASSIGNMENTS["P2_MODEL_YAML"], P2_YAML.read_text(encoding="utf-8"))

    def test_backbone_is_identical_and_only_p2_has_stride4_head(self):
        p3 = P3_YAML.read_text(encoding="utf-8")
        p2 = P2_YAML.read_text(encoding="utf-8")
        p3_backbone = p3.split("backbone:\n", 1)[1].split("\nhead:\n", 1)[0]
        p2_backbone = p2.split("backbone:\n", 1)[1].split("\nhead:\n", 1)[0]
        self.assertEqual(p3_backbone, p2_backbone)
        self.assertIn("[[16, 19, 22], 1, Detect, [nc]]", p3)
        self.assertIn("[[19, 22, 25, 28], 1, Detect, [nc]]", p2)
        self.assertIn("[[-1, 2], 1, Concat, [1]]", p2)
        self.assertIn("[-1, 2, C2f, [128]]", p2)

    def test_variants_are_paired_and_expected_strides_are_explicit(self):
        variants = CONFIG["VARIANTS"]
        self.assertEqual([variant["name"] for variant in variants], ["p3_control", "p2_stride4"])
        self.assertEqual(variants[0]["expected_strides"], [8, 16, 32])
        self.assertEqual(variants[1]["expected_strides"], [4, 8, 16, 32])
        self.assertEqual(variants[0]["model_yaml"], ASSIGNMENTS["P3_MODEL_YAML"])
        self.assertEqual(variants[1]["model_yaml"], ASSIGNMENTS["P2_MODEL_YAML"])

    def test_training_arguments_lock_seed_determinism_and_effective_batch(self):
        source = ast.get_source_segment(python_source, FUNCTIONS["training_arguments"])
        self.assertIn('"seed": SEED', source)
        self.assertIn('"deterministic": True', source)
        self.assertIn('"batch": total_batch', source)
        self.assertIn("arguments.update(TRAINING_HYPERPARAMETERS)", source)

    def test_pretrained_failure_cannot_silently_fall_back_to_scratch(self):
        source = ast.get_source_segment(python_source, FUNCTIONS["run_variant"])
        self.assertIn("model = model.load(str(pretrained_path))", source)
        self.assertLess(
            source.index("model = model.load(str(pretrained_path))"), source.index("try:")
        )
        self.assertNotIn("train from scratch", source.lower())

    def test_both_variants_are_preflighted_before_training_loop(self):
        main_source = ast.get_source_segment(
            python_source, FUNCTIONS["run_paired_training_main"]
        )
        preflight_position = main_source.index("preflight_profiles =")
        training_loop_position = main_source.index("for variant in VARIANTS:", preflight_position + 1)
        self.assertLess(preflight_position, training_loop_position)
        self.assertIn("preflight_variant(variant, output_root, pretrained_path)", main_source)
        preflight_source = ast.get_source_segment(python_source, FUNCTIONS["preflight_variant"])
        self.assertIn("model = model.load(str(pretrained_path))", preflight_source)

    def test_dataset_and_weight_provenance_are_persisted(self):
        self.assertIn("dataset_manifest.json", python_source)
        self.assertIn("environment.json", python_source)
        self.assertIn("best_weight_sha256", python_source)
        self.assertIn("run_signature", python_source)
        self.assertIn('"public_test_used_for_training_or_selection": False', python_source)

    def test_public_test_is_not_a_training_input(self):
        config_source = sources[2].lower()
        self.assertNotIn("public_test", config_source)
        self.assertNotIn("gt_ann", config_source)


if __name__ == "__main__":
    unittest.main()
