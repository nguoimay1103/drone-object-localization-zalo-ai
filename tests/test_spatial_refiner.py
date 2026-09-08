import ast
import copy
import json
import math
import random
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "06-inference-main.ipynb"
TRAINER = ROOT / "scripts" / "train_spatial_refiner_v2.py"
TRAINING_NOTEBOOK = ROOT / "08-train-spatial-refiner-v2.ipynb"


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


def namespace():
    ns = {
        "math": math,
        "copy": copy,
        "SPATIAL_REFINER_V2_SEARCH_FACTOR": 4.0,
        "SPATIAL_REFINER_V2_MIN_SEARCH_SIDE": 96.0,
        "SPATIAL_REFINER_V2_MAX_CENTER_SHIFT": 0.5,
        "SPATIAL_REFINER_V2_MAX_LOG_SCALE_CHANGE": 0.7,
    }
    for name in (
        "spatial_bbox_geometry",
        "spatial_refiner_v2_crop_spec",
        "spatial_refiner_v2_box_from_normalized",
        "spatial_refiner_v2_box_to_normalized",
        "spatial_refiner_v2_plausibility",
        "replace_spatial_refiner_v2_boxes",
    ):
        exec(NODES[name], ns)
    return ns


class SpatialRefinerTests(unittest.TestCase):
    def test_only_locked_refiner_ab_is_active(self):
        assignments = {}
        for node in TREE.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                if isinstance(target, ast.Name) and target.id.startswith("RUN_"):
                    try:
                        assignments[target.id] = ast.literal_eval(node.value)
                    except (ValueError, TypeError):
                        pass
        self.assertEqual(
            [name for name, enabled in assignments.items() if enabled],
            ["RUN_SPATIAL_REFINER_V2_AB"],
        )
        self.assertFalse(assignments["RUN_SPATIAL_HEADROOM"])

    def test_crop_mapping_is_invertible_and_clipped(self):
        ns = namespace()
        spec = ns["spatial_refiner_v2_crop_spec"]([0, 0, 10, 20])
        self.assertEqual(spec, (-43.0, -38.0, 96.0))
        mapped = ns["spatial_refiner_v2_box_from_normalized"](
            [0.25, 0.25, 0.75, 0.75], spec, 100, 100
        )
        self.assertEqual(mapped, [0.0, 0.0, 29.0, 34.0])
        self.assertIsNone(ns["spatial_refiner_v2_box_from_normalized"](
            [0.5, 0.5, 0.5, 0.5], spec, 100, 100
        ))
        normalized = ns["spatial_refiner_v2_box_to_normalized"](
            [0, 0, 10, 20], spec
        )
        restored = ns["spatial_refiner_v2_box_from_normalized"](
            normalized, spec, 100, 100
        )
        self.assertTrue(all(abs(a - b) < 1e-9 for a, b in zip(restored, [0, 0, 10, 20])))

    def test_plausibility_falls_back_on_large_shift_or_scale(self):
        ns = namespace()
        accepted, reason, _ = ns["spatial_refiner_v2_plausibility"](
            [0, 0, 10, 10], [1, 0, 11, 10]
        )
        self.assertTrue(accepted)
        self.assertEqual(reason, "accepted")
        self.assertEqual(
            ns["spatial_refiner_v2_plausibility"](
                [0, 0, 10, 10], [100, 0, 110, 10]
            )[1],
            "center_shift",
        )
        self.assertEqual(
            ns["spatial_refiner_v2_plausibility"](
                [0, 0, 10, 10], [-10, -10, 20, 20]
            )[1],
            "scale_change",
        )

    def test_replacement_preserves_support_and_input_payload(self):
        ns = namespace()
        production = [{
            "video_id": "Object_0",
            "detections": [{"bboxes": [
                {"frame": 36, "x1": 0, "y1": 0, "x2": 10, "y2": 10},
                {"frame": 37, "x1": 1, "y1": 0, "x2": 11, "y2": 10},
            ]}],
        }]
        before = copy.deepcopy(production)
        output = ns["replace_spatial_refiner_v2_boxes"](
            production, {"Object_0": {36: [0.4, 0.4, 10.6, 10.6]}}
        )
        self.assertEqual(production, before)
        boxes = output[0]["detections"][0]["bboxes"]
        self.assertEqual([row["frame"] for row in boxes], [36, 37])
        self.assertEqual(boxes[0]["x2"], 11)
        self.assertEqual(boxes[1], before[0]["detections"][0]["bboxes"][1])

    def test_refiner_selection_never_uses_gt(self):
        source = "\n".join(NODES[name] for name in (
            "spatial_refiner_v2_box_from_normalized",
            "spatial_refiner_v2_plausibility",
            "replace_spatial_refiner_v2_boxes",
        ))
        self.assertNotIn("gt_map", source)
        self.assertNotIn("Cardboard", source)
        self.assertNotIn("LifeJacket", source)

    def test_training_script_does_not_accept_public_annotations(self):
        source = TRAINER.read_text(encoding="utf-8")
        tree = ast.parse(source)
        parse_args = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "parse_args"
        )
        parse_source = ast.get_source_segment(source, parse_args)
        self.assertNotIn("public", parse_source.lower())
        self.assertIn("group K-fold by physical identity", source)

    def test_training_folds_hold_both_videos_of_an_identity_together(self):
        source = TRAINER.read_text(encoding="utf-8")
        tree = ast.parse(source)
        functions = {
            node.name: ast.get_source_segment(source, node)
            for node in tree.body if isinstance(node, ast.FunctionDef)
        }
        ns = {"defaultdict": defaultdict, "random": random}
        exec(functions["build_identity_folds"], ns)
        records = [
            {"identity_id": identity, "video_id": f"{identity}_{view}"}
            for identity in ("Black_Box", "Cardboard_Box", "Life_Jacket", "Bag")
            for view in (0, 1)
            for _ in range(view + 1)
        ]
        folds = ns["build_identity_folds"](records, fold_count=3, seed=42)
        flattened = [identity for fold in folds for identity in fold]
        self.assertEqual(sorted(flattened), sorted(set(flattened)))
        self.assertEqual(len(folds), 3)

    def test_multi_box_annotation_frames_keep_distinct_targets(self):
        source = TRAINER.read_text(encoding="utf-8")
        tree = ast.parse(source)
        functions = {
            node.name: ast.get_source_segment(source, node)
            for node in tree.body if isinstance(node, ast.FunctionDef)
        }
        ns = {"math": math}
        exec(functions["valid_box"], ns)
        exec(functions["box_iou_xyxy"], ns)
        ns.update({"json": json, "statistics": __import__("statistics")})
        exec(functions["load_annotations"], ns)
        records = [{
            "video_id": "MobilePhone_0",
            "annotations": [
                {"bboxes": [
                    {"frame": 579, "x1": 10, "y1": 20, "x2": 30, "y2": 40},
                    {"frame": 580, "x1": 11, "y1": 20, "x2": 31, "y2": 40},
                ]},
                {"bboxes": [
                    {"frame": 579, "x1": 12, "y1": 21, "x2": 32, "y2": 41},
                    {"frame": 580, "x1": 11, "y1": 20, "x2": 31, "y2": 40},
                ]},
            ],
        }]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "annotations.json"
            path.write_text(json.dumps(records), encoding="utf-8")
            gt_map, diagnostics = ns["load_annotations"](path)
        self.assertEqual(gt_map["MobilePhone_0"][579], [
            [10.0, 20.0, 30.0, 40.0],
            [12.0, 21.0, 32.0, 41.0],
        ])
        self.assertEqual(diagnostics["multi_box_frame_count"], 1)
        self.assertEqual(diagnostics["extra_distinct_box_count"], 1)
        self.assertEqual(diagnostics["exact_duplicates_ignored"], 1)
        self.assertEqual(
            diagnostics["policy"],
            "all_distinct_boxes_are_independent_targets",
        )

    def test_real_proposal_matching_is_one_to_one(self):
        source = TRAINER.read_text(encoding="utf-8")
        tree = ast.parse(source)
        functions = {
            node.name: ast.get_source_segment(source, node)
            for node in tree.body if isinstance(node, ast.FunctionDef)
        }
        ns = {}
        exec(functions["box_iou_xyxy"], ns)
        exec(functions["greedy_match_boxes"], ns)
        matches = ns["greedy_match_boxes"](
            [[0, 0, 10, 10], [20, 0, 30, 10]],
            [[1, 0, 11, 10], [19, 0, 29, 10]],
        )
        self.assertEqual(set(matches), {0, 1})
        self.assertEqual(matches[0][0], [1, 0, 11, 10])
        self.assertEqual(matches[1][0], [19, 0, 29, 10])
        self.assertEqual(len({tuple(row[0]) for row in matches.values()}), 2)

    def test_training_notebook_is_self_contained(self):
        notebook = json.loads(TRAINING_NOTEBOOK.read_text(encoding="utf-8"))
        sources = ["".join(cell.get("source", [])) for cell in notebook["cells"]]
        combined = "\n".join(sources)
        self.assertIn("class AnchorPreservingResidualRefiner", combined)
        self.assertIn("def main(args=None)", combined)
        self.assertIn("main(args)", combined)
        self.assertNotIn("TRAINER_SCRIPT", combined)
        self.assertNotIn("subprocess.run", combined)

    def test_v2_is_zero_initialized_and_consumes_coarse_box(self):
        source = TRAINER.read_text(encoding="utf-8")
        tree = ast.parse(source)
        model = next(
            node for node in tree.body
            if isinstance(node, ast.ClassDef)
            and node.name == "AnchorPreservingResidualRefiner"
        )
        model_source = ast.get_source_segment(source, model)
        forward = next(
            node for node in model.body
            if isinstance(node, ast.FunctionDef) and node.name == "forward"
        )
        geometry = next(
            node for node in model.body
            if isinstance(node, ast.FunctionDef) and node.name == "_geometry"
        )
        self.assertEqual([arg.arg for arg in forward.args.args], [
            "self", "reference", "search", "coarse_box",
        ])
        self.assertEqual(ast.unparse(geometry.args.args[1].annotation), "int")
        self.assertEqual(ast.unparse(geometry.args.args[2].annotation), "int")
        self.assertIn("nn.init.zeros_(self.residual_head[-1].weight)", model_source)
        self.assertIn("nn.init.zeros_(self.residual_head[-1].bias)", model_source)
        self.assertIn("return self._decode(coarse_box, bounded_residual)", model_source)


if __name__ == "__main__":
    unittest.main()
