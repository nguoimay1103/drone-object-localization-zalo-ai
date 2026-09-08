"""Combine disjoint video shards; never average worker means or sum their FPS."""
import json
from pathlib import Path
from .artifacts import sha256_file, write_json
from .evaluation import evaluate_predictions, prediction_maps


def _signature(manifest):
    settings = dict(manifest["config"]["inference"])
    settings.pop("device")
    runtime = manifest["runtime"]
    return {"inference": settings, "weights": manifest["selection"]["checkpoint_hashes"],
            "annotation": manifest["selection"]["annotation_sha256"],
            "source": manifest["source_hashes"], "frame_origin": manifest["frame_index_origin"],
            "runtime": {k: runtime[k] for k in ("python", "ultralytics", "torch", "torchvision", "opencv", "numpy", "pillow", "cuda", "cudnn")}}


def merge_runs(run_dirs, output_dir, expected_video_ids, gt_map=None, expected_checkpoint_hashes=None,
               expected_annotation_sha256=None):
    if not run_dirs:
        raise ValueError("At least one input run is required")
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Run directory already exists: {output_dir}")
    records = []
    signatures = []
    source_runs = []
    for directory in run_dirs:
        directory = Path(directory)
        with (directory / "run_manifest.json").open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        if manifest["status"] != "completed":
            raise ValueError(f"Input run is not completed: {directory}")
        path = directory / "predictions.json"
        if sha256_file(path) != manifest["predictions_sha256"]:
            raise ValueError(f"Prediction hash mismatch: {directory}")
        signature = _signature(manifest)
        if signatures and signature != signatures[0]:
            raise ValueError("Worker runs differ in settings, weights, GT, source or runtime")
        if expected_checkpoint_hashes is not None and signature["weights"] != expected_checkpoint_hashes:
            raise ValueError("Worker checkpoints differ from merge configuration")
        if gt_map is not None and signature["annotation"] != expected_annotation_sha256:
            raise ValueError("Merge GT differs from the GT used by workers")
        signatures.append(signature)
        with path.open(encoding="utf-8") as handle:
            shard_records = json.load(handle)
        shard_maps = prediction_maps(shard_records)
        if set(shard_maps) != set(manifest["selection"]["video_ids"]):
            raise ValueError(f"Shard video coverage differs from its manifest: {directory}")
        records.extend(shard_records)
        source_runs.append(str(directory.resolve()))
    maps = prediction_maps(records)  # Reject duplicate video IDs across workers.
    if set(maps) != set(expected_video_ids):
        raise ValueError("Merged predictions must cover the exact expected video set")
    records.sort(key=lambda r: r["video_id"])
    evaluation = evaluate_predictions(records, gt_map, expected_video_ids) if gt_map is not None else None
    output_dir.mkdir(parents=True)
    write_json(output_dir / "predictions.json", records)
    summary = {"evaluation": evaluation, "source_runs": source_runs, "parallel_pipeline_fps": None,
               "timing_note": "Measure joint wall-clock for the two workers; do not add worker FPS or assume 2x speedup."}
    write_json(output_dir / "metrics.json", summary)
    write_json(output_dir / "merge_manifest.json", {"status": "completed", "signature": signatures[0],
                "source_runs": source_runs, "video_ids": sorted(expected_video_ids),
                "predictions_sha256": sha256_file(output_dir / "predictions.json")})
    return summary
