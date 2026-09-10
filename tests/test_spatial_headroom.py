import csv
import json
import math
import os
import tempfile
import unittest
from pathlib import Path

from tests.test_motion_distribution_diagnostics import NODES, motion_namespace


def spatial_namespace():
    ns = motion_namespace()
    ns.update({
        'SPATIAL_HEADROOM_REFERENCE_IMGSZ': 640,
        'SPATIAL_HEADROOM_GOOD_IOU': 0.5,
        'SPATIAL_HEADROOM_LOCAL_OVERLAP': 0.5,
        'SPATIAL_HEADROOM_MIN_REFINER_CEILING_DELTA': 0.010,
        'SPATIAL_HEADROOM_MIN_IDENTITY_CEILING_DELTA': 0.005,
        'SPATIAL_HEADROOM_MIN_QUALIFYING_IDENTITIES': 2,
        'SPATIAL_HEADROOM_SCENARIOS': (
            'perfect_shared', 'perfect_good_overlap', 'candidate_keep_best',
            'local_candidate_keep_best', 'center_only_good_overlap', 'size_only_good_overlap',
        ),
    })
    for name in ('spatial_bbox_geometry', 'spatial_box_from_geometry',
                 'build_spatial_headroom_rows', 'summarize_spatial_rows',
                 'run_spatial_headroom_diagnostics'):
        exec(NODES[name], ns)
    return ns


def fixture(ns):
    gt = {'Object_0': {36: [0, 0, 10, 10], 37: [100, 0, 110, 10],
                       38: [0, 0, 10, 10]}, 'Other_0': {0: [0, 0, 10, 10]}}
    maps = {'Object_0': {36: [2, 0, 12, 10], 37: [0, 0, 10, 10],
                         39: [0, 0, 10, 10]}, 'Other_0': {0: [1, 0, 11, 10]}}
    predictions = [{
        'video_id': vid, 'detections': [{'bboxes': [
            dict(frame=frame, **dict(zip(('x1', 'y1', 'x2', 'y2'), box)))
            for frame, box in frames.items()]}],
    } for vid, frames in maps.items()]
    per_video = {vid: ns['compute_st_iou_video'](gt[vid], maps[vid]) for vid in gt}
    production = {'predictions': predictions, 'per_video': per_video,
                  'mean_st_iou': sum(per_video.values()) / 2, 'detected_frames': 4}
    cache = {'signature': {'video_folders': list(gt)}, 'videos': {
        vid: {'frames': {str(frame): [{'bbox': box, 'scores': [.8, .8, .8]}]
                        for frame, box in frames.items()}}
        for vid, frames in gt.items()}}
    metadata = {vid: {'width': 1280, 'height': 720} for vid in gt}
    return cache, gt, production, metadata


class SpatialHeadroomTests(unittest.TestCase):
    def test_fixed_support_upper_bound_counts_fp_fn_and_uses_macro_average(self):
        ns = spatial_namespace()
        cache, gt, production, metadata = fixture(ns)
        before = json.dumps([cache, gt, production], sort_keys=True)
        rows, groups, videos, result = ns['run_spatial_headroom_diagnostics'](
            cache, gt, production, metadata)
        self.assertEqual(json.dumps([cache, gt, production], sort_keys=True), before)
        self.assertEqual(result['fixed_prediction_frame_count'], 4)
        self.assertEqual(result['global']['missed_gt_count'], 1)
        self.assertEqual(result['global']['background_prediction_count'], 1)
        # Per-video oracle=(2/4, 1/1), NOT pooled 3/5.
        self.assertAlmostEqual(result['scenarios']['perfect_shared']['mean_st_iou'], .75)
        self.assertTrue(result['refiner_ab_decision']['run_alpha_refine_ab'])
        self.assertEqual(result['refiner_ab_decision']['qualifying_identity_count'], 2)
        for row in rows:
            if row['cohort'] in ('missed_gt', 'background_prediction'):
                for name in ns['SPATIAL_HEADROOM_SCENARIOS']:
                    self.assertEqual(row[f'{name}_iou'], 0)
        for name in ns['SPATIAL_HEADROOM_SCENARIOS']:
            delta = result['scenarios'][name]['macro_delta']
            for field in ('cohort', 'size_bin', 'identity_id'):
                self.assertAlmostEqual(delta, sum(
                    row[f'{name}_macro_delta_contribution']
                    for row in groups if row['group_type'] == field))
        self.assertEqual([r['frame'] for r in rows if r['video_id'] == 'Object_0'], [36, 37, 38, 39])

    def test_local_oracle_does_not_relabel_remote_candidate_as_refinement(self):
        ns = spatial_namespace()
        rows, _, _, result = ns['run_spatial_headroom_diagnostics'](*fixture(ns))
        row = next(r for r in rows if r['frame'] == 37)
        self.assertEqual(row['best_candidate_iou'], 1)
        self.assertEqual(row['local_candidate_keep_best_iou'], 0)
        self.assertEqual(row['perfect_good_overlap_iou'], 0)
        self.assertGreater(result['scenarios']['candidate_keep_best']['macro_delta'],
                           result['scenarios']['local_candidate_keep_best']['macro_delta'])

    def test_candidate_oracle_keeps_better_production_and_rounds_like_export(self):
        ns = spatial_namespace()
        cache, gt, production, metadata = fixture(ns)
        cache['videos']['Other_0']['frames']['0'] = [{'bbox': [.49, 0, 10.49, 10]}]
        cache['videos']['Object_0']['frames']['36'] = []
        rows, _, _, _ = ns['run_spatial_headroom_diagnostics'](cache, gt, production, metadata)
        self.assertEqual(next(r for r in rows if r['video_id'] == 'Other_0')['best_candidate_iou'], 1)
        row = next(r for r in rows if r['frame'] == 36)
        self.assertEqual(row['candidate_keep_best_iou'], row['production_iou'])
        self.assertEqual(row['size_bin'], 'tiny_lt_8')  # 10px * 640/1280

    def test_size_only_counterfactual_reports_negative_delta(self):
        ns = spatial_namespace()
        rows = ns['build_spatial_headroom_rows'](
            'X_0', {'frames': {}}, {0: [0, 0, 10, 10]}, {0: [0, 0, 20, 10]},
            {'width': 640, 'height': 480}, 1)
        self.assertAlmostEqual(rows[0]['production_iou'], .5)
        self.assertLess(rows[0]['size_only_good_overlap_macro_delta_contribution'], 0)
        self.assertAlmostEqual(rows[0]['log_width_ratio'], math.log(2))

    def test_residual_jitter_subtracts_true_motion_and_never_bridges_gap(self):
        ns = spatial_namespace()
        rows = ns['build_spatial_headroom_rows'](
            'X_0', {'frames': {}}, {0: [0, 0, 10, 10], 1: [100, 0, 110, 10],
                                    3: [200, 0, 210, 10]},
            {0: [1, 0, 11, 10], 1: [101, 0, 111, 10], 3: [202, 0, 212, 10]},
            {'width': 640, 'height': 480}, 1)
        self.assertEqual(rows[1]['center_residual_jitter'], 0)
        self.assertEqual(rows[1]['gt_center_speed'], 10)
        self.assertIsNone(rows[2]['center_residual_jitter'])

    def test_mismatch_and_invalid_geometry_fail_instead_of_silent_reporting(self):
        ns = spatial_namespace()
        cache, gt, production, metadata = fixture(ns)
        production['mean_st_iou'] += .01
        with self.assertRaisesRegex(RuntimeError, 'macro production'):
            ns['run_spatial_headroom_diagnostics'](cache, gt, production, metadata)
        for bbox in ([0, 0, 0, 5], [0, 0, float('nan'), 5]):
            with self.assertRaises(ValueError):
                ns['spatial_bbox_geometry'](bbox)

    def test_empty_video_keeps_legacy_zero_metric_and_null_distribution(self):
        ns = spatial_namespace()
        cache = {'signature': {'video_folders': ['X_0']}, 'videos': {'X_0': {'frames': {}}}}
        production = {'predictions': [{'video_id': 'X_0', 'detections': []}],
                      'per_video': {'X_0': 0.0}, 'mean_st_iou': 0.0, 'detected_frames': 0}
        rows, _, _, result = ns['run_spatial_headroom_diagnostics'](
            cache, {'X_0': {}}, production, {'X_0': {'width': 640, 'height': 480}})
        self.assertEqual(rows, [])
        self.assertEqual(result['scenarios']['perfect_shared']['mean_st_iou'], 0)
        self.assertIsNone(result['global']['mean_iou_on_shared'])

    def test_main_writes_all_diagnostics_and_only_production_predictions(self):
        import ast
        from tests.test_motion_distribution_diagnostics import TREE
        ns = spatial_namespace()
        cache, gt, production, metadata = fixture(ns)
        for node in TREE.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                name = getattr(node.targets[0], 'id', '')
                if name.startswith('RUN_'):
                    ns[name] = ast.literal_eval(node.value)
        self.assertEqual(
            [k for k, v in ns.items() if k.startswith('RUN_') and v],
            ['RUN_SPATIAL_REFINER_V2_AB'],
        )
        # Exercise the archived diagnostic explicitly; the active experiment is now
        # the locked refiner A/B and is tested separately.
        ns['RUN_SPATIAL_REFINER_V2_AB'] = False
        ns['RUN_SPATIAL_HEADROOM'] = True
        with tempfile.TemporaryDirectory() as folder:
            ns.update({
                'os': os, 'csv': csv, 'CALIBRATION_DIR': folder,
                'SPATIAL_HEADROOM_OUTPUT_DIR': folder,
                'SPATIAL_HEADROOM_EXPERIMENT_NAME': 'spatial_localization_headroom_v1',
                'OUTPUT_FILE': str(Path(folder) / 'predictions.json'), 'GT_ANN_PATH': '',
                'WEIGHT_YOLO': .425, 'WEIGHT_SIAMESE': .475, 'WEIGHT_COLOR': .1,
                'MATCHING_THRESHOLD': .54, 'USE_TEMPORAL_GATING': True,
                'SPATIAL_HEADROOM_MIN_REFINER_CEILING_DELTA': .010,
                'SPATIAL_HEADROOM_MIN_IDENTITY_CEILING_DELTA': .005,
                'SPATIAL_HEADROOM_MIN_QUALIFYING_IDENTITIES': 2,
                'PRODUCTION_TEMPORAL_CONFIG': {'max_gap': 7},
                'list_video_folders': lambda: list(gt),
                'build_cache_signature': lambda _: cache['signature'],
                'load_or_initialize_cache': lambda _: cache,
                'extract_candidate_cache': lambda *args: cache,
                'load_gt_annotations': lambda _: gt,
                'collect_video_metadata': lambda _: metadata,
                'evaluate_config': lambda *args, **kwargs: production,
                'print_evaluation': lambda *args: None,
                'print': lambda *args: None,
                'atomic_write_json': lambda path, value: Path(path).write_text(
                    json.dumps(value, allow_nan=False), encoding='utf-8'),
            })
            exec(NODES['main'], ns)
            ns['main']()
            files = list(Path(folder).glob('*'))
            self.assertEqual(len(files), 5)
            summary_path = next(Path(folder).glob('*_summary.json'))
            summary = json.loads(summary_path.read_text())
            for path in summary['outputs'].values():
                with open(path, newline='', encoding='utf-8') as handle:
                    self.assertTrue(list(csv.DictReader(handle)))
            self.assertEqual(json.loads(Path(ns['OUTPUT_FILE']).read_text()), production['predictions'])


if __name__ == '__main__':
    unittest.main()
