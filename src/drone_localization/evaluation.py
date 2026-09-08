"""Strict input validation around the unchanged local ST-IoU formula."""
import json
import math
from pathlib import Path
from .metrics import compute_st_iou_video


def _frame_map(boxes, label):
    frames = {}
    for box in boxes:
        frame = box["frame"]
        if type(frame) is not int or frame < 0:
            raise ValueError(f"{label}: frame must be a nonnegative integer, starting from video frame 0")
        if frame in frames:
            raise ValueError(f"{label}: duplicate frame {frame}")
        coords = [box[k] for k in ("x1", "y1", "x2", "y2")]
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in coords):
            raise ValueError(f"{label}: bbox coordinates must be finite numbers")
        coords = [int(v) for v in coords]  # NB06 GT loader uses integer truncation.
        if coords[2] <= coords[0] or coords[3] <= coords[1]:
            raise ValueError(f"{label}: bbox must have positive area")
        frames[frame] = coords
    return frames


def load_ground_truth(path):
    with Path(path).open(encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise ValueError("Ground truth must be a list of video records")
    result = {}
    for record in records:
        video_id = record["video_id"]
        if not isinstance(video_id, str) or not video_id or video_id in result:
            raise ValueError(f"Missing/duplicate GT video_id: {video_id!r}")
        boxes = [b for segment in record.get("annotations", []) for b in segment.get("bboxes", [])]
        result[video_id] = _frame_map(boxes, f"GT {video_id}")
    return result


def prediction_maps(records):
    if not isinstance(records, list):
        raise ValueError("Predictions must be a list")
    result = {}
    for record in records:
        video_id = record["video_id"]
        if not isinstance(video_id, str) or not video_id or video_id in result:
            raise ValueError(f"Missing/duplicate prediction video_id: {video_id!r}")
        detections = record["detections"]
        if not isinstance(detections, list) or len(detections) > 1:
            raise ValueError("The baseline schema supports at most one target per video")
        boxes = [b for detection in detections for b in detection["bboxes"]]
        result[video_id] = _frame_map(boxes, f"Prediction {video_id}")
    return result


def evaluate_predictions(records, gt_map, expected_video_ids):
    predicted = prediction_maps(records)
    expected = set(expected_video_ids)
    if set(predicted) != expected:
        raise ValueError(f"Prediction video set mismatch: missing={sorted(expected-set(predicted))}, extra={sorted(set(predicted)-expected)}")
    if not expected or expected - set(gt_map):
        raise ValueError(f"No videos or missing GT: {sorted(expected-set(gt_map))}")
    per_video = {v: compute_st_iou_video(gt_map[v], predicted[v]) for v in sorted(expected)}
    return {"metric": "local_mean_st_iou", "mean_st_iou": sum(per_video.values()) / len(per_video),
            "video_count": len(per_video), "per_video": per_video}
