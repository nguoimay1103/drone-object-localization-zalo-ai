"""Historical local ST-IoU metric, not an official competition evaluator."""
import json
import os


def load_gt_annotations(gt_path):
    if not os.path.exists(gt_path):
        return {}
    with open(gt_path, 'r') as f:
        records = json.load(f)
    gt_map = {}
    for rec in records:
        vid = rec.get("video_id")
        if not vid:
            continue
        frame2bbox = {}
        for interval in rec.get("annotations", []):
            for b in interval.get("bboxes", []):
                frame2bbox[int(b["frame"])] = [int(b["x1"]), int(b["y1"]),
                                               int(b["x2"]), int(b["y2"])]
        gt_map[vid] = frame2bbox
    return gt_map


def iou_box(b1, b2):
    ix1, iy1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    ix2, iy2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area1 = max(0, b1[2] - b1[0]) * max(0, b1[3] - b1[1])
    area2 = max(0, b2[2] - b2[0]) * max(0, b2[3] - b2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def compute_st_iou_video(gt_frames, pred_frames):
    inter = set(gt_frames.keys()) & set(pred_frames.keys())
    union = set(gt_frames.keys()) | set(pred_frames.keys())
    if not union:
        return 0.0
    return sum(iou_box(gt_frames[f], pred_frames[f]) for f in inter) / len(union)
