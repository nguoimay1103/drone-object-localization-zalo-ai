"""Train an identity-held-out, Alpha-Refine-inspired bbox refiner on Kaggle.

The public-test annotations are deliberately not accepted by this program.  It
uses only the original training videos, their bounding boxes and reference
images.  The final artifact is TorchScript so NB06 can consume it without
duplicating the training model definition.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import re
import shutil
import statistics
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.models import mobilenet_v3_small


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True,
                        help="Train root containing annotations/ and samples/")
    parser.add_argument("--siamese-weights", required=True,
                        help="Identity-v1 Siamese state_dict used to initialize the encoder")
    parser.add_argument("--output-dir", default="/kaggle/working/spatial_refiner_v1")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--frame-stride", type=int, default=5)
    parser.add_argument("--train-repeats", type=int, default=2)
    parser.add_argument("--search-size", type=int, default=256)
    parser.add_argument("--reference-size", type=int, default=128)
    parser.add_argument("--search-factor", type=float, default=4.0)
    parser.add_argument("--min-search-side", type=float, default=96.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--devices", default="0,1")
    parser.add_argument("--identity-overrides-json", default="")
    parser.add_argument("--reuse-prepared-frames", action="store_true")
    parser.add_argument("--force-final", action="store_true",
                        help="Train final model even when the identity-CV gate fails")
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def worker_seed(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


def canonical_hash(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
    temporary.replace(path)


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def resolve_identity(video_id, overrides=None):
    overrides = overrides or {}
    if video_id in overrides:
        identity = overrides[video_id]
    else:
        match = re.fullmatch(r"(.+)_([01])", video_id)
        if match is None:
            raise ValueError(
                f"Cannot infer physical identity for {video_id}; add an explicit override"
            )
        identity = match.group(1)
    if not isinstance(identity, str) or not identity.strip():
        raise ValueError(f"Invalid identity for {video_id}: {identity!r}")
    return identity


def valid_box(box):
    return (len(box) == 4 and all(math.isfinite(float(value)) for value in box)
            and float(box[2]) > float(box[0]) and float(box[3]) > float(box[1]))


def box_iou_xyxy(first, second):
    ix1, iy1 = max(first[0], second[0]), max(first[1], second[1])
    ix2, iy2 = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def load_annotations(annotation_path):
    """Load every distinct train bbox, matching the original YOLO data builder.

    Train annotations can legitimately contain multiple target boxes in one
    frame.  Each distinct box is an independent refiner sample; exact duplicate
    rows are ignored.  This mirrors notebooks 02/04 and avoids imposing the
    single-box public-evaluation schema on detector/refiner training data.
    """
    with open(annotation_path, encoding="utf-8") as handle:
        records = json.load(handle)
    if isinstance(records, dict):
        records = [records]
    output = {}
    diagnostics = {
        "policy": "all_distinct_boxes_are_independent_targets",
        "exact_duplicates_ignored": 0,
        "multi_box_frame_count": 0,
        "extra_distinct_box_count": 0,
        "multi_box_pair_iou_minimum": None,
        "multi_box_pair_iou_mean": None,
        "multi_box_examples": [],
    }
    pair_ious = []
    for record in records:
        video_id = str(record["video_id"])
        frame_map = output.setdefault(video_id, {})
        for interval in record.get("annotations", []):
            for raw in interval.get("bboxes", []):
                frame = int(raw["frame"])
                box = [float(raw[key]) for key in ("x1", "y1", "x2", "y2")]
                if frame < 0 or not valid_box(box):
                    raise ValueError(f"Invalid annotation: {video_id} frame={frame} box={box}")
                boxes = frame_map.setdefault(frame, [])
                if box in boxes:
                    diagnostics["exact_duplicates_ignored"] += 1
                    continue
                boxes.append(box)
    if not output:
        raise ValueError("Annotation file contains no videos")
    for video_id, frame_map in output.items():
        for frame, boxes in frame_map.items():
            if len(boxes) <= 1:
                continue
            diagnostics["multi_box_frame_count"] += 1
            diagnostics["extra_distinct_box_count"] += len(boxes) - 1
            frame_pair_ious = [
                box_iou_xyxy(boxes[first], boxes[second])
                for first in range(len(boxes))
                for second in range(first + 1, len(boxes))
            ]
            pair_ious.extend(frame_pair_ious)
            if len(diagnostics["multi_box_examples"]) < 20:
                diagnostics["multi_box_examples"].append({
                    "video_id": video_id,
                    "frame": frame,
                    "boxes": boxes,
                    "pair_ious": frame_pair_ious,
                })
    if pair_ious:
        diagnostics["multi_box_pair_iou_minimum"] = min(pair_ious)
        diagnostics["multi_box_pair_iou_mean"] = statistics.fmean(pair_ious)
    diagnostics["video_count"] = len(output)
    diagnostics["unique_annotated_frame_count"] = sum(
        len(frame_map) for frame_map in output.values()
    )
    diagnostics["distinct_annotation_box_count"] = sum(
        len(boxes) for frame_map in output.values() for boxes in frame_map.values()
    )
    return output, diagnostics


def find_annotation_path(data_root):
    candidates = [
        data_root / "annotations" / "annotations.json",
        data_root / "annotations" / "drone_annotations.json",
    ]
    found = [path for path in candidates if path.is_file()]
    if len(found) != 1:
        raise FileNotFoundError(
            f"Expected exactly one training annotation file; found={list(map(str, found))}"
        )
    return found[0]


def reference_paths(samples_dir, video_id):
    folder = samples_dir / video_id / "object_images"
    paths = sorted(
        path for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ) if folder.is_dir() else []
    if not paths:
        raise FileNotFoundError(f"No reference images for {video_id}: {folder}")
    return paths


def prepare_frame_dataset(args, identity_overrides):
    data_root = Path(args.data_root).expanduser().resolve()
    samples_dir = data_root / "samples"
    annotation_path = find_annotation_path(data_root)
    output_root = Path(args.output_dir).expanduser().resolve()
    frames_root = output_root / "prepared_frames"
    manifest_path = output_root / "dataset_manifest.json"
    gt_map, annotation_diagnostics = load_annotations(annotation_path)
    if annotation_diagnostics["multi_box_frame_count"]:
        print(
            "Annotation normalization: retained every distinct bbox across "
            f"{annotation_diagnostics['multi_box_frame_count']} multi-box frames; ignored "
            f"{annotation_diagnostics['exact_duplicates_ignored']} exact duplicate rows."
        )
    preparation_config = {
        "schema_version": 2,
        "data_root": str(data_root),
        "annotation_path": str(annotation_path),
        "annotation_sha256": file_sha256(annotation_path),
        "annotation_multi_box_policy": annotation_diagnostics["policy"],
        "frame_stride": args.frame_stride,
        "identity_overrides": identity_overrides,
        "jpeg_quality": 95,
    }
    preparation_config["config_sha256"] = canonical_hash(preparation_config)

    if args.reuse_prepared_frames and manifest_path.is_file():
        with manifest_path.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        if manifest.get("preparation_config") != preparation_config:
            raise ValueError("Prepared-frame manifest config mismatch; use a new output directory")
        missing = [row["frame_path"] for row in manifest["records"]
                   if not Path(row["frame_path"]).is_file()]
        if missing:
            raise FileNotFoundError(f"Prepared manifest has missing frames: {missing[:5]}")
        return manifest

    if frames_root.exists():
        shutil.rmtree(frames_root)
    frames_root.mkdir(parents=True, exist_ok=True)
    rows = []
    videos = []
    for video_id in sorted(gt_map):
        identity_id = resolve_identity(video_id, identity_overrides)
        video_path = samples_dir / video_id / "drone_video.mp4"
        refs = reference_paths(samples_dir, video_id)
        if not video_path.is_file():
            raise FileNotFoundError(video_path)
        selected = sorted(frame for frame in gt_map[video_id]
                          if frame % args.frame_stride == 0)
        if not selected:
            selected = [min(gt_map[video_id])]
        selected_set = set(selected)
        video_output = frames_root / video_id
        video_output.mkdir(parents=True, exist_ok=True)
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if max(selected) >= total_frames:
            capture.release()
            raise ValueError(f"GT frame outside video: {video_id} max={max(selected)} total={total_frames}")
        frame_index = 0
        saved = 0
        while capture.isOpened() and selected_set:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index in selected_set:
                destination = video_output / f"frame_{frame_index:06d}.jpg"
                if not cv2.imwrite(str(destination), frame,
                                   [cv2.IMWRITE_JPEG_QUALITY, 95]):
                    capture.release()
                    raise IOError(f"Failed to write {destination}")
                for box_index, gt_box in enumerate(gt_map[video_id][frame_index]):
                    rows.append({
                        "video_id": video_id,
                        "identity_id": identity_id,
                        "frame": frame_index,
                        "box_index": box_index,
                        "frame_path": str(destination),
                        "reference_paths": [str(path) for path in refs],
                        "gt_box": gt_box,
                        "width": width,
                        "height": height,
                    })
                selected_set.remove(frame_index)
                saved += 1
            frame_index += 1
        capture.release()
        if selected_set:
            raise RuntimeError(f"Video ended before selected GT frames: {video_id} {sorted(selected_set)[:5]}")
        videos.append({
            "video_id": video_id,
            "identity_id": identity_id,
            "video_frame_count": total_frames,
            "annotated_frame_count": len(gt_map[video_id]),
            "prepared_frame_count": saved,
            "prepared_box_count": sum(
                len(gt_map[video_id][frame]) for frame in selected
            ),
            "reference_count": len(refs),
            "width": width,
            "height": height,
        })
    manifest = {
        "experiment": "spatial_refiner_v1",
        "public_test_used": False,
        "preparation_config": preparation_config,
        "annotation_diagnostics": annotation_diagnostics,
        "record_count": len(rows),
        "identity_count": len({row["identity_id"] for row in rows}),
        "video_count": len(videos),
        "videos": videos,
        "records": rows,
    }
    manifest["record_index_sha256"] = canonical_hash(rows)
    atomic_write_json(manifest_path, manifest)
    return manifest


def build_identity_folds(records, fold_count=3, seed=42):
    counts = defaultdict(int)
    for row in records:
        counts[row["identity_id"]] += 1
    identities = sorted(counts)
    if len(identities) < 2:
        raise ValueError("Identity-held-out validation requires at least two identities")
    fold_count = min(max(2, fold_count), len(identities))
    rng = random.Random(seed)
    shuffled = identities[:]
    rng.shuffle(shuffled)
    # Greedy balancing by sample count after seeded tie-breaking.
    ordered = sorted(shuffled, key=lambda name: counts[name], reverse=True)
    folds = [[] for _ in range(fold_count)]
    totals = [0] * fold_count
    for identity in ordered:
        target = min(range(fold_count), key=lambda index: (totals[index], index))
        folds[target].append(identity)
        totals[target] += counts[identity]
    return [sorted(fold) for fold in folds]


def jitter_box(gt, rng, min_iou=0.35, max_iou=0.98):
    x1, y1, x2, y2 = map(float, gt)
    width, height = x2 - x1, y2 - y1
    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
    scale = math.sqrt(width * height)
    for _ in range(30):
        hard = rng.random() < 0.25
        center_std = 0.16 if hard else 0.07
        log_scale_std = 0.16 if hard else 0.07
        dx = float(np.clip(rng.gauss(0, center_std), -0.40, 0.40)) * scale
        dy = float(np.clip(rng.gauss(0, center_std), -0.40, 0.40)) * scale
        log_width = float(np.clip(rng.gauss(0, log_scale_std), -0.40, 0.40))
        log_height = float(np.clip(rng.gauss(0, log_scale_std), -0.40, 0.40))
        candidate_width = width * math.exp(log_width)
        candidate_height = height * math.exp(log_height)
        candidate_center_x, candidate_center_y = center_x + dx, center_y + dy
        candidate = [
            candidate_center_x - candidate_width / 2,
            candidate_center_y - candidate_height / 2,
            candidate_center_x + candidate_width / 2,
            candidate_center_y + candidate_height / 2,
        ]
        overlap = box_iou_xyxy(gt, candidate)
        if min_iou <= overlap <= max_iou:
            return candidate
    # Deterministic non-perfect fallback, used only for extremely small boxes.
    return [x1 + 0.05 * scale, y1, x2 + 0.05 * scale, y2]


def search_crop_spec(coarse_box, search_factor=4.0, min_search_side=96.0):
    if not valid_box(coarse_box):
        raise ValueError(f"Invalid coarse box: {coarse_box}")
    x1, y1, x2, y2 = map(float, coarse_box)
    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(max(x2 - x1, y2 - y1) * float(search_factor),
               float(min_search_side))
    if not math.isfinite(side) or side <= 0:
        raise ValueError("Invalid search side")
    return center_x - side / 2, center_y - side / 2, side


def map_box_to_crop(box, crop_spec):
    crop_x, crop_y, side = crop_spec
    return [
        (float(box[0]) - crop_x) / side,
        (float(box[1]) - crop_y) / side,
        (float(box[2]) - crop_x) / side,
        (float(box[3]) - crop_y) / side,
    ]


def extract_square_crop(image, crop_spec, output_size):
    crop_x, crop_y, side = crop_spec
    scale = output_size / side
    matrix = np.asarray([
        [scale, 0.0, -crop_x * scale],
        [0.0, scale, -crop_y * scale],
    ], dtype=np.float32)
    return cv2.warpAffine(
        image, matrix, (output_size, output_size), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def image_tensor(image_bgr, size):
    image = cv2.resize(image_bgr, (size, size), interpolation=cv2.INTER_LINEAR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    image = (image - np.asarray(IMAGENET_MEAN, dtype=np.float32)) / np.asarray(
        IMAGENET_STD, dtype=np.float32
    )
    return torch.from_numpy(image.transpose(2, 0, 1).copy())


class SpatialRefinerDataset(Dataset):
    def __init__(self, records, training, search_size, reference_size,
                 search_factor, min_search_side, repeat=1, seed=42):
        self.records = list(records)
        if not self.records:
            raise ValueError("SpatialRefinerDataset cannot be empty")
        self.training = bool(training)
        self.search_size = int(search_size)
        self.reference_size = int(reference_size)
        self.search_factor = float(search_factor)
        self.min_search_side = float(min_search_side)
        self.repeat = max(1, int(repeat)) if training else 1
        self.seed = int(seed)

    def __len__(self):
        return len(self.records) * self.repeat

    def __getitem__(self, index):
        record_index = index % len(self.records)
        record = self.records[record_index]
        if self.training:
            rng = random.Random(random.randrange(2 ** 32))
        else:
            rng = random.Random(self.seed + index * 1009)
        frame = cv2.imread(record["frame_path"])
        if frame is None:
            raise IOError(f"Unreadable prepared frame: {record['frame_path']}")
        references = record["reference_paths"]
        reference_path = references[rng.randrange(len(references))]
        reference = cv2.imread(reference_path)
        if reference is None:
            raise IOError(f"Unreadable reference image: {reference_path}")
        gt_box = list(map(float, record["gt_box"]))
        coarse_box = jitter_box(gt_box, rng)
        crop_spec = search_crop_spec(
            coarse_box, self.search_factor, self.min_search_side
        )
        target_box = np.clip(map_box_to_crop(gt_box, crop_spec), 0.0, 1.0)
        coarse_crop_box = np.clip(map_box_to_crop(coarse_box, crop_spec), 0.0, 1.0)
        if target_box[2] <= target_box[0] or target_box[3] <= target_box[1]:
            raise RuntimeError(f"GT vanished outside search crop: {record['video_id']}")
        search = extract_square_crop(frame, crop_spec, self.search_size)
        return {
            "reference": image_tensor(reference, self.reference_size),
            "search": image_tensor(search, self.search_size),
            "target_box": torch.tensor(target_box, dtype=torch.float32),
            "coarse_box": torch.tensor(coarse_crop_box, dtype=torch.float32),
            "identity_id": record["identity_id"],
            "video_id": record["video_id"],
        }


class PixelCorrelationCornerRefiner(nn.Module):
    """Stride-8 shared encoder + pixel-wise correlation + two corner heads."""

    def __init__(self, siamese_weights=None, freeze_stem=True):
        super().__init__()
        full = mobilenet_v3_small(weights=None)
        if siamese_weights:
            checkpoint = torch.load(siamese_weights, map_location="cpu")
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                checkpoint = checkpoint["state_dict"]
            cleaned = {}
            for key, value in checkpoint.items():
                key = key.removeprefix("module.")
                if key.startswith("features."):
                    cleaned[key[len("features."):]] = value
            missing, unexpected = full.features.load_state_dict(cleaned, strict=False)
            # Projection keys are excluded before this load, so every feature tensor must
            # still match. Do not silently train from partially initialized features.
            if missing:
                raise RuntimeError(f"Siamese feature weights incomplete: {missing[:8]}")
            if unexpected:
                raise RuntimeError(f"Unexpected Siamese feature weights: {unexpected[:8]}")
        self.encoder = nn.Sequential(*list(full.features.children())[:4])
        encoder_training = self.encoder.training
        self.encoder.eval()
        with torch.no_grad():
            feature_channels = int(self.encoder(torch.zeros(1, 3, 64, 64)).shape[1])
        self.encoder.train(encoder_training)
        if freeze_stem:
            for layer in list(self.encoder.children())[:2]:
                for parameter in layer.parameters():
                    parameter.requires_grad = False
        self.fusion = nn.Sequential(
            nn.Conv2d(feature_channels + 4, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.Hardswish(),
            nn.Conv2d(64, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.Hardswish(),
            nn.Conv2d(32, 2, 1),
        )

    def _fused_features(self, reference, search):
        ref_features = F.normalize(self.encoder(reference), dim=1)
        search_features = F.normalize(self.encoder(search), dim=1)
        batch, _, height, width = search_features.shape
        ref_flat = ref_features.flatten(2)
        search_flat = search_features.flatten(2).transpose(1, 2)
        correlation = torch.bmm(search_flat, ref_flat)
        correlation_max = correlation.max(dim=2).values.view(batch, 1, height, width)
        correlation_mean = correlation.mean(dim=2).view(batch, 1, height, width)
        x_grid = torch.linspace(-1.0, 1.0, width, device=search.device,
                                dtype=search.dtype).view(1, 1, 1, width)
        y_grid = torch.linspace(-1.0, 1.0, height, device=search.device,
                                dtype=search.dtype).view(1, 1, height, 1)
        x_grid = x_grid.expand(batch, 1, height, width)
        y_grid = y_grid.expand(batch, 1, height, width)
        return torch.cat(
            [search_features, correlation_max, correlation_mean, x_grid, y_grid], dim=1
        )

    @staticmethod
    def _soft_argmax(logits):
        batch, _, height, width = logits.shape
        probabilities = F.softmax(logits.flatten(2), dim=2)
        x = torch.linspace(0.0, 1.0, width, device=logits.device,
                           dtype=logits.dtype).view(1, 1, 1, width)
        y = torch.linspace(0.0, 1.0, height, device=logits.device,
                           dtype=logits.dtype).view(1, 1, height, 1)
        x = x.expand(batch, 1, height, width).flatten(2)
        y = y.expand(batch, 1, height, width).flatten(2)
        coordinate_x = (probabilities * x).sum(dim=2)
        coordinate_y = (probabilities * y).sum(dim=2)
        return torch.cat([coordinate_x, coordinate_y], dim=1)

    def forward(self, reference, search):
        logits = self.fusion(self._fused_features(reference, search))
        top_left = self._soft_argmax(logits[:, 0:1])
        bottom_right = self._soft_argmax(logits[:, 1:2])
        return torch.cat([top_left, bottom_right], dim=1), logits


def load_refiner_model(siamese_weights, device_ids):
    model = PixelCorrelationCornerRefiner(siamese_weights=siamese_weights)
    primary = torch.device(f"cuda:{device_ids[0]}" if torch.cuda.is_available() else "cpu")
    model = model.to(primary)
    if torch.cuda.is_available() and len(device_ids) > 1:
        model = nn.DataParallel(model, device_ids=device_ids)
    return model, primary


def gaussian_corner_targets(boxes, height, width, sigma_cells=1.5):
    batch = boxes.shape[0]
    x = torch.linspace(0.0, 1.0, width, device=boxes.device,
                       dtype=boxes.dtype).view(1, 1, 1, width)
    y = torch.linspace(0.0, 1.0, height, device=boxes.device,
                       dtype=boxes.dtype).view(1, 1, height, 1)
    sigma_x = sigma_cells / max(width - 1, 1)
    sigma_y = sigma_cells / max(height - 1, 1)
    corners = torch.stack([boxes[:, :2], boxes[:, 2:]], dim=1)
    center_x = corners[:, :, 0].view(batch, 2, 1, 1)
    center_y = corners[:, :, 1].view(batch, 2, 1, 1)
    targets = torch.exp(-0.5 * (((x - center_x) / sigma_x) ** 2
                                + ((y - center_y) / sigma_y) ** 2))
    return targets / targets.flatten(2).sum(dim=2).view(batch, 2, 1, 1).clamp_min(1e-8)


def aligned_iou_torch(first, second):
    intersection_min = torch.maximum(first[:, :2], second[:, :2])
    intersection_max = torch.minimum(first[:, 2:], second[:, 2:])
    intersection = (intersection_max - intersection_min).clamp_min(0).prod(dim=1)
    first_area = (first[:, 2:] - first[:, :2]).clamp_min(0).prod(dim=1)
    second_area = (second[:, 2:] - second[:, :2]).clamp_min(0).prod(dim=1)
    return intersection / (first_area + second_area - intersection).clamp_min(1e-8)


def generalized_iou_loss(predicted, target):
    iou = aligned_iou_torch(predicted, target)
    enclosing_min = torch.minimum(predicted[:, :2], target[:, :2])
    enclosing_max = torch.maximum(predicted[:, 2:], target[:, 2:])
    enclosing_area = (enclosing_max - enclosing_min).clamp_min(0).prod(dim=1)
    first_area = (predicted[:, 2:] - predicted[:, :2]).clamp_min(0).prod(dim=1)
    second_area = (target[:, 2:] - target[:, :2]).clamp_min(0).prod(dim=1)
    intersection_min = torch.maximum(predicted[:, :2], target[:, :2])
    intersection_max = torch.minimum(predicted[:, 2:], target[:, 2:])
    intersection = (intersection_max - intersection_min).clamp_min(0).prod(dim=1)
    union = first_area + second_area - intersection
    giou = iou - (enclosing_area - union) / enclosing_area.clamp_min(1e-8)
    return (1.0 - giou).mean()


def refiner_loss(predicted, logits, target):
    targets = gaussian_corner_targets(target, logits.shape[2], logits.shape[3])
    heatmap_loss = -(targets * F.log_softmax(logits.flatten(2), dim=2).view_as(logits)).sum(
        dim=(1, 2, 3)
    ).mean()
    coordinate_loss = F.smooth_l1_loss(predicted, target, beta=0.02)
    giou_loss = generalized_iou_loss(predicted, target)
    total = heatmap_loss + 5.0 * coordinate_loss + 2.0 * giou_loss
    return total, {
        "heatmap_loss": float(heatmap_loss.detach()),
        "coordinate_loss": float(coordinate_loss.detach()),
        "giou_loss": float(giou_loss.detach()),
    }


def ordered_boxes(boxes, minimum_side=1e-3):
    low = torch.minimum(boxes[:, :2], boxes[:, 2:]).clamp(0, 1)
    high = torch.maximum(boxes[:, :2], boxes[:, 2:]).clamp(0, 1)
    high = torch.maximum(high, low + minimum_side).clamp(max=1)
    low = torch.minimum(low, high - minimum_side).clamp(min=0)
    return torch.cat([low, high], dim=1)


def plausibility_mask(predicted, coarse, max_center_shift=0.5,
                      max_log_scale_change=0.7):
    predicted = ordered_boxes(predicted)
    coarse = ordered_boxes(coarse)
    predicted_size = (predicted[:, 2:] - predicted[:, :2]).clamp_min(1e-6)
    coarse_size = (coarse[:, 2:] - coarse[:, :2]).clamp_min(1e-6)
    predicted_center = (predicted[:, :2] + predicted[:, 2:]) / 2
    coarse_center = (coarse[:, :2] + coarse[:, 2:]) / 2
    coarse_scale = torch.sqrt(coarse_size.prod(dim=1)).clamp_min(1e-6)
    center_shift = torch.linalg.vector_norm(predicted_center - coarse_center, dim=1) / coarse_scale
    log_scale = torch.abs(torch.log(predicted_size / coarse_size)).max(dim=1).values
    return (center_shift <= max_center_shift) & (log_scale <= max_log_scale_change)


def model_module(model):
    return model.module if isinstance(model, nn.DataParallel) else model


def make_loader(dataset, batch_size, workers, shuffle, seed):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
        pin_memory=True, drop_last=shuffle and len(dataset) >= batch_size,
        worker_init_fn=worker_seed, generator=generator,
        persistent_workers=workers > 0,
    )


@torch.no_grad()
def evaluate_model(model, loader, device):
    model.eval()
    totals = defaultdict(float)
    per_identity = defaultdict(lambda: defaultdict(float))
    sample_count = 0
    started = time.perf_counter()
    for batch in loader:
        reference = batch["reference"].to(device, non_blocking=True)
        search = batch["search"].to(device, non_blocking=True)
        target = batch["target_box"].to(device, non_blocking=True)
        coarse = batch["coarse_box"].to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            predicted, logits = model(reference, search)
        predicted = ordered_boxes(predicted.float())
        accepted = plausibility_mask(predicted, coarse)
        deployed = torch.where(accepted[:, None], predicted, coarse)
        coarse_iou = aligned_iou_torch(coarse, target)
        refined_iou = aligned_iou_torch(deployed, target)
        loss, _ = refiner_loss(predicted, logits.float(), target)
        batch_size = target.shape[0]
        sample_count += batch_size
        totals["loss"] += float(loss) * batch_size
        totals["coarse_iou"] += float(coarse_iou.sum())
        totals["refined_iou"] += float(refined_iou.sum())
        totals["improved"] += float((refined_iou > coarse_iou + 1e-8).sum())
        totals["worsened"] += float((refined_iou < coarse_iou - 1e-8).sum())
        totals["accepted"] += float(accepted.sum())
        for index, identity in enumerate(batch["identity_id"]):
            row = per_identity[identity]
            row["samples"] += 1
            row["coarse_iou"] += float(coarse_iou[index])
            row["refined_iou"] += float(refined_iou[index])
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    result = {
        "samples": sample_count,
        "loss": totals["loss"] / sample_count,
        "coarse_mean_iou": totals["coarse_iou"] / sample_count,
        "refined_mean_iou": totals["refined_iou"] / sample_count,
        "mean_iou_delta": (totals["refined_iou"] - totals["coarse_iou"]) / sample_count,
        "improved_fraction": totals["improved"] / sample_count,
        "worsened_fraction": totals["worsened"] / sample_count,
        "accepted_fraction": totals["accepted"] / sample_count,
        "samples_per_second": sample_count / elapsed if elapsed > 0 else None,
        "per_identity": {},
    }
    for identity, values in sorted(per_identity.items()):
        count = int(values["samples"])
        coarse_mean = values["coarse_iou"] / count
        refined_mean = values["refined_iou"] / count
        result["per_identity"][identity] = {
            "samples": count,
            "coarse_mean_iou": coarse_mean,
            "refined_mean_iou": refined_mean,
            "mean_iou_delta": refined_mean - coarse_mean,
        }
    return result


def train_one_epoch(model, loader, optimizer, scaler, device):
    model.train()
    totals = defaultdict(float)
    samples = 0
    for batch in loader:
        reference = batch["reference"].to(device, non_blocking=True)
        search = batch["search"].to(device, non_blocking=True)
        target = batch["target_box"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            predicted, logits = model(reference, search)
            loss, components = refiner_loss(predicted, logits, target)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optimizer)
        scaler.update()
        count = target.shape[0]
        samples += count
        totals["loss"] += float(loss.detach()) * count
        for name, value in components.items():
            totals[name] += value * count
    return {name: value / samples for name, value in totals.items()}


def run_fold(args, records, heldout_identities, fold_index, device_ids):
    heldout = set(heldout_identities)
    train_records = [row for row in records if row["identity_id"] not in heldout]
    val_records = [row for row in records if row["identity_id"] in heldout]
    if not train_records or not val_records:
        raise RuntimeError(f"Empty fold {fold_index}")
    train_dataset = SpatialRefinerDataset(
        train_records, True, args.search_size, args.reference_size,
        args.search_factor, args.min_search_side, args.train_repeats,
        args.seed + fold_index,
    )
    val_dataset = SpatialRefinerDataset(
        val_records, False, args.search_size, args.reference_size,
        args.search_factor, args.min_search_side, 1,
        args.seed + 10000 + fold_index,
    )
    train_loader = make_loader(train_dataset, args.batch_size, args.workers,
                               True, args.seed + fold_index)
    val_loader = make_loader(val_dataset, args.batch_size, args.workers,
                             False, args.seed + 1000 + fold_index)
    model, device = load_refiner_model(args.siamese_weights, device_ids)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate, weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    fold_dir = Path(args.output_dir) / "folds" / f"fold_{fold_index}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    best_delta = -float("inf")
    best_epoch = 0
    best_metrics = None
    history = []
    patience, stale = 4, 0
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, scaler, device)
        val_metrics = evaluate_model(model, val_loader, device)
        scheduler.step()
        row = {
            "fold": fold_index,
            "epoch": epoch,
            "heldout_identities": "|".join(sorted(heldout)),
            "learning_rate": optimizer.param_groups[0]["lr"],
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()
               if key != "per_identity"},
        }
        history.append(row)
        print(
            f"Fold {fold_index} epoch {epoch:02d}: "
            f"loss={val_metrics['loss']:.4f} coarse={val_metrics['coarse_mean_iou']:.4f} "
            f"refined={val_metrics['refined_mean_iou']:.4f} "
            f"delta={val_metrics['mean_iou_delta']:+.4f}"
        )
        if val_metrics["mean_iou_delta"] > best_delta + 1e-5:
            best_delta = val_metrics["mean_iou_delta"]
            best_epoch = epoch
            best_metrics = val_metrics
            torch.save(model_module(model).state_dict(), fold_dir / "best.pth")
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    return {
        "fold": fold_index,
        "heldout_identities": sorted(heldout),
        "train_identity_count": len({row["identity_id"] for row in train_records}),
        "train_samples": len(train_records),
        "validation_samples": len(val_records),
        "best_epoch": best_epoch,
        "best_validation": best_metrics,
        "checkpoint": str(fold_dir / "best.pth"),
    }, history


def aggregate_cv(fold_results):
    identity_rows = []
    for fold in fold_results:
        for identity, metrics in fold["best_validation"]["per_identity"].items():
            identity_rows.append({"fold": fold["fold"], "identity_id": identity, **metrics})
    if len({row["identity_id"] for row in identity_rows}) != len(identity_rows):
        raise RuntimeError("An identity appeared in more than one held-out fold")
    total_samples = sum(row["samples"] for row in identity_rows)
    coarse = sum(row["coarse_mean_iou"] * row["samples"] for row in identity_rows) / total_samples
    refined = sum(row["refined_mean_iou"] * row["samples"] for row in identity_rows) / total_samples
    deltas = [row["mean_iou_delta"] for row in identity_rows]
    required_improved = max(2, math.ceil(len(identity_rows) * 2 / 3))
    improved = sum(delta > 0 for delta in deltas)
    summary = {
        "sample_weighted_coarse_mean_iou": coarse,
        "sample_weighted_refined_mean_iou": refined,
        "sample_weighted_mean_iou_delta": refined - coarse,
        "identity_macro_mean_delta": statistics.fmean(deltas),
        "improved_identity_count": improved,
        "identity_count": len(deltas),
        "worst_identity_delta": min(deltas),
        "criteria": {
            "minimum_sample_weighted_mean_iou_delta": 0.005,
            "minimum_improved_identities": required_improved,
            "maximum_worst_identity_drop": 0.005,
        },
    }
    summary["promotion_passed"] = (
        summary["sample_weighted_mean_iou_delta"] >= 0.005
        and improved >= required_improved
        and summary["worst_identity_delta"] >= -0.005
    )
    return summary, identity_rows


def train_final(args, records, epochs, device_ids):
    dataset = SpatialRefinerDataset(
        records, True, args.search_size, args.reference_size,
        args.search_factor, args.min_search_side, args.train_repeats, args.seed + 90000,
    )
    loader = make_loader(dataset, args.batch_size, args.workers, True, args.seed + 90000)
    model, device = load_refiner_model(args.siamese_weights, device_ids)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate, weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    history = []
    for epoch in range(1, epochs + 1):
        metrics = train_one_epoch(model, loader, optimizer, scaler, device)
        scheduler.step()
        history.append({
            "fold": "final", "epoch": epoch,
            "heldout_identities": "", "learning_rate": optimizer.param_groups[0]["lr"],
            **{f"train_{key}": value for key, value in metrics.items()},
        })
        print(f"Final epoch {epoch:02d}: loss={metrics['loss']:.4f}")
    module = model_module(model).eval().cpu()
    weights_dir = Path(args.output_dir) / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    state_path = weights_dir / "spatial_refiner_v1_final.pth"
    script_path = weights_dir / "spatial_refiner_v1_final.ts"
    torch.save(module.state_dict(), state_path)
    scripted = torch.jit.script(module)
    scripted.save(str(script_path))
    return history, state_path, script_path


def validate_args(args):
    if args.epochs <= 0 or args.folds < 2 or args.batch_size <= 0:
        raise ValueError("epochs/batch-size must be positive and folds >= 2")
    if args.frame_stride <= 0 or args.train_repeats <= 0:
        raise ValueError("frame-stride and train-repeats must be positive")
    if args.search_size % 8 or args.reference_size % 8:
        raise ValueError("search/reference sizes must be divisible by encoder stride 8")
    if args.search_factor <= 1 or args.min_search_side <= 0:
        raise ValueError("search-factor must be >1 and min-search-side positive")
    if not Path(args.siamese_weights).is_file():
        raise FileNotFoundError(args.siamese_weights)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for the refiner experiment")


def preflight_torchscript(args, output_dir):
    """Fail before CV training if checkpoint transfer or TorchScript is incompatible."""
    model = PixelCorrelationCornerRefiner(
        siamese_weights=args.siamese_weights
    ).eval().cpu()
    scripted = torch.jit.script(model)
    with torch.no_grad():
        boxes, logits = scripted(
            torch.zeros(2, 3, args.reference_size, args.reference_size),
            torch.zeros(2, 3, args.search_size, args.search_size),
        )
    if tuple(boxes.shape) != (2, 4) or logits.shape[:2] != (2, 2):
        raise RuntimeError(
            f"TorchScript output contract mismatch: boxes={tuple(boxes.shape)}, "
            f"logits={tuple(logits.shape)}"
        )
    if not torch.isfinite(boxes).all() or not torch.isfinite(logits).all():
        raise RuntimeError("TorchScript preflight produced non-finite outputs")
    temporary = Path(output_dir) / "_spatial_refiner_preflight.ts"
    scripted.save(str(temporary))
    reloaded = torch.jit.load(str(temporary), map_location="cpu")
    with torch.no_grad():
        reloaded_boxes, _ = reloaded(
            torch.zeros(1, 3, args.reference_size, args.reference_size),
            torch.zeros(1, 3, args.search_size, args.search_size),
        )
    temporary.unlink()
    if tuple(reloaded_boxes.shape) != (1, 4):
        raise RuntimeError("Reloaded TorchScript model violates output contract")
    del model, scripted, reloaded
    return {
        "status": "passed",
        "reference_shape": [2, 3, args.reference_size, args.reference_size],
        "search_shape": [2, 3, args.search_size, args.search_size],
        "box_output_shape": list(boxes.shape),
        "heatmap_output_shape": list(logits.shape),
        "save_reload_passed": True,
    }


def preflight_dataset(args, records, sample_count=8):
    """Decode and transform a bounded sample before starting multi-fold training."""
    subset = list(records[:min(sample_count, len(records))])
    dataset = SpatialRefinerDataset(
        subset, False, args.search_size, args.reference_size,
        args.search_factor, args.min_search_side, repeat=1, seed=args.seed + 777,
    )
    observed_ious = []
    for index in range(len(dataset)):
        item = dataset[index]
        if tuple(item["reference"].shape) != (3, args.reference_size, args.reference_size):
            raise RuntimeError("Reference tensor shape mismatch in dataset preflight")
        if tuple(item["search"].shape) != (3, args.search_size, args.search_size):
            raise RuntimeError("Search tensor shape mismatch in dataset preflight")
        for name in ("reference", "search", "target_box", "coarse_box"):
            if not torch.isfinite(item[name]).all():
                raise RuntimeError(f"Non-finite tensor in dataset preflight: {name}")
        overlap = box_iou_xyxy(
            item["target_box"].tolist(), item["coarse_box"].tolist()
        )
        if not 0.34 <= overlap <= 0.99:
            raise RuntimeError(f"Jitter IoU outside expected range: {overlap}")
        observed_ious.append(overlap)
    return {
        "status": "passed",
        "sample_count": len(dataset),
        "minimum_coarse_iou": min(observed_ious),
        "maximum_coarse_iou": max(observed_ious),
    }


def main(args=None):
    args = parse_args() if args is None else args
    validate_args(args)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    identity_overrides = {}
    if args.identity_overrides_json:
        with open(args.identity_overrides_json, encoding="utf-8") as handle:
            identity_overrides = json.load(handle)
    seed_everything(args.seed)
    requested_devices = [int(value) for value in args.devices.split(",") if value.strip()]
    device_ids = [value for value in requested_devices
                  if value < torch.cuda.device_count()]
    if not device_ids:
        raise RuntimeError(
            f"No requested CUDA device is available: requested={requested_devices}, "
            f"count={torch.cuda.device_count()}"
        )
    started = time.time()
    torchscript_preflight = preflight_torchscript(args, output_dir)
    print("TorchScript/checkpoint preflight passed:", torchscript_preflight)
    manifest = prepare_frame_dataset(args, identity_overrides)
    records = manifest["records"]
    dataset_preflight = preflight_dataset(args, records)
    print("Dataset preflight passed:", dataset_preflight)
    folds = build_identity_folds(records, args.folds, args.seed)
    split_manifest = {
        "experiment": "spatial_refiner_v1",
        "protocol": "group K-fold by physical identity; terminal _0/_1 stay together",
        "public_test_used": False,
        "identity_folds": folds,
        "record_index_sha256": manifest["record_index_sha256"],
    }
    atomic_write_json(output_dir / "split_manifest.json", split_manifest)
    fold_results, history = [], []
    for fold_index, heldout in enumerate(folds):
        fold_result, fold_history = run_fold(
            args, records, heldout, fold_index, device_ids
        )
        fold_results.append(fold_result)
        history.extend(fold_history)
    cv_summary, identity_rows = aggregate_cv(fold_results)
    write_csv(output_dir / "training_history.csv", history)
    write_csv(output_dir / "cv_identity_results.csv", identity_rows)
    fold_rows = [{
        "fold": row["fold"],
        "heldout_identities": "|".join(row["heldout_identities"]),
        "train_identity_count": row["train_identity_count"],
        "train_samples": row["train_samples"],
        "validation_samples": row["validation_samples"],
        "best_epoch": row["best_epoch"],
        "coarse_mean_iou": row["best_validation"]["coarse_mean_iou"],
        "refined_mean_iou": row["best_validation"]["refined_mean_iou"],
        "mean_iou_delta": row["best_validation"]["mean_iou_delta"],
        "improved_fraction": row["best_validation"]["improved_fraction"],
        "worsened_fraction": row["best_validation"]["worsened_fraction"],
        "accepted_fraction": row["best_validation"]["accepted_fraction"],
        "samples_per_second": row["best_validation"]["samples_per_second"],
        "checkpoint": row["checkpoint"],
    } for row in fold_results]
    write_csv(output_dir / "cv_folds.csv", fold_rows)
    final_history, state_path, script_path = [], None, None
    final_epochs = max(1, round(statistics.median(
        row["best_epoch"] for row in fold_results
    )))
    if cv_summary["promotion_passed"] or args.force_final:
        final_history, state_path, script_path = train_final(
            args, records, final_epochs, device_ids
        )
        combined_history = history + final_history
        write_csv(output_dir / "training_history.csv", combined_history)
    environment = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": __import__("torchvision").__version__,
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_devices": [torch.cuda.get_device_name(index) for index in device_ids],
    }
    atomic_write_json(output_dir / "environment.json", environment)
    run_summary = {
        "experiment": "spatial_refiner_v1_identity_cv",
        "status": "trained_final" if script_path else "cv_rejected_no_final_model",
        "public_test_used_for_training_or_selection": False,
        "dataset_manifest": str(output_dir / "dataset_manifest.json"),
        "dataset_record_index_sha256": manifest["record_index_sha256"],
        "split_manifest": str(output_dir / "split_manifest.json"),
        "siamese_initialization_sha256": file_sha256(args.siamese_weights),
        "architecture": {
            "encoder": "MobileNetV3-small features[0:4], stride 8, initialized from identity-v1 Siamese",
            "fusion": "pixel-wise reference/search correlation max+mean with spatial coordinate channels",
            "head": "top-left and bottom-right corner distributions",
            "mask_head": False,
        },
        "torchscript_preflight": torchscript_preflight,
        "dataset_preflight": dataset_preflight,
        "preprocessing": {
            "search_size": args.search_size,
            "reference_size": args.reference_size,
            "search_factor": args.search_factor,
            "min_search_side": args.min_search_side,
            "normalization_mean": IMAGENET_MEAN,
            "normalization_std": IMAGENET_STD,
        },
        "jitter": {
            "minimum_iou": 0.35,
            "maximum_iou": 0.98,
            "hard_probability": 0.25,
            "regular_center_std_gt_scale": 0.07,
            "hard_center_std_gt_scale": 0.16,
            "regular_log_scale_std": 0.07,
            "hard_log_scale_std": 0.16,
            "clip": 0.40,
        },
        "inference_safety_gate": {
            "maximum_center_shift_gt_scale": 0.5,
            "maximum_absolute_log_scale_change": 0.7,
            "fallback": "keep original production bbox",
        },
        "training": {
            "epochs_requested": args.epochs,
            "final_epochs_from_median_cv_best_epoch": final_epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "train_repeats": args.train_repeats,
            "seed": args.seed,
            "device_ids": device_ids,
        },
        "identity_cv": cv_summary,
        "folds": fold_results,
        "artifacts": {
            "state_dict": str(state_path) if state_path else None,
            "torchscript": str(script_path) if script_path else None,
            "torchscript_sha256": file_sha256(script_path) if script_path else None,
        },
        "elapsed_seconds": time.time() - started,
    }
    atomic_write_json(output_dir / "run_summary.json", run_summary)
    print("\nSPATIAL REFINER V1 IDENTITY-CV")
    print(json.dumps(cv_summary, indent=2))
    if script_path:
        print(f"Final TorchScript: {script_path}")
    else:
        print("CV gate failed: no final checkpoint was exported. Do not run public A/B.")


if __name__ == "__main__":
    main()
