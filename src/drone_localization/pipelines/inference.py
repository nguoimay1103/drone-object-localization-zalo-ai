"""NB06 numerical pipeline, with explicit configuration and bounded debug memory.

Original notebooks remain the reference. GPU/video parity is not certified until
this runner has been compared with golden predictions on the same environment.
"""
import importlib.metadata
import platform
import time

import cv2
import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO

from ..artifacts import package_hashes, sha256_file, write_json
from ..evaluation import evaluate_predictions
from ..matching import load_reference_embeddings, load_reference_histogram, calculate_color_score
from ..models.siamese import SiameseMobileNet, get_inference_transforms
from ..scoring import select_best_candidate
from ..temporal import temporal_smooth_detections


def _synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _json_value(value):
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _video_inputs(samples_dir, video_id):
    folder = samples_dir / video_id
    refs = folder / "object_images"
    paths = [folder / "drone_video.mp4"]
    if refs.is_dir():
        paths += sorted(p for p in refs.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    return {p.relative_to(samples_dir).as_posix(): sha256_file(p) for p in paths}


def infer_video(config, video_id, yolo, siam, transform, device):
    settings = config.inference
    samples_dir = config.paths.samples_dir
    video_path = samples_dir / video_id / "drone_video.mp4"
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        raise ValueError(f"Cannot decode video: {video_path}")
    expected_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    source_fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()

    _synchronize(device)
    started = time.perf_counter()
    ref_emb = load_reference_embeddings(samples_dir, video_id, siam, device,
                                        settings.use_multiscale_ref, settings.ref_scales)
    ref_hist = load_reference_histogram(samples_dir, video_id)
    predict_args = dict(stream=True, conf=settings.confidence_threshold, verbose=False, augment=settings.use_tta)
    if settings.device != "auto":
        predict_args["device"] = str(device)
    for name in ("imgsz", "batch"):
        value = getattr(settings, name)
        if value is not None:
            predict_args[name] = value
    results = yolo.predict(str(video_path), **predict_args)
    frame_best = {}
    max_frame_idx = -1
    candidate_count = 0
    for f_idx, res in enumerate(results):
        max_frame_idx = max(max_frame_idx, f_idx)
        if not res.boxes:
            continue
        boxes = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        orig_img = res.orig_img
        h, w = orig_img.shape[:2]
        pil_img = Image.fromarray(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB))
        cand_tensors = []
        cand_bgrs = []
        valid_indices = []
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 <= x1 + 5 or y2 <= y1 + 5:
                continue
            valid_indices.append(i)
            cand_tensors.append(transform(pil_img.crop((x1, y1, x2, y2))))
            cand_bgrs.append(orig_img[y1:y2, x1:x2])
        if not cand_tensors:
            continue
        candidate_count += len(cand_tensors)
        siam_scores = np.zeros(len(cand_tensors))
        color_scores = np.zeros(len(cand_tensors))
        if ref_emb is not None:
            with torch.no_grad():
                batch = torch.stack(cand_tensors).to(device)
                feats = siam(batch)
                dists = torch.cdist(ref_emb, feats).cpu().numpy()[0]
                siam_scores = np.maximum(0, 1.0 - dists / 2.0)
        if ref_hist is not None:
            for k, crop_bgr in enumerate(cand_bgrs):
                color_scores[k] = calculate_color_score(crop_bgr, ref_hist)
        best_box, best_score = select_best_candidate(
            boxes, confs, siam_scores, color_scores, valid_indices,
            ref_emb is not None, ref_hist is not None, settings,
        )
        thresh = settings.matching_threshold if ref_emb is not None else settings.yolo_only_threshold
        if best_box is not None and best_score >= thresh:
            frame_best[f_idx] = {"bbox": best_box.tolist(), "score": float(best_score)}

    count = max_frame_idx + 1
    if count == 0 or (expected_frames > 0 and count != expected_frames):
        raise ValueError(f"{video_id}: decoded {count} results, container reports {expected_frames} frames; check corruption/stride")
    if getattr(yolo.predictor.args, "vid_stride", 1) != 1:
        raise ValueError("vid_stride must remain 1 for absolute frame alignment")
    final_bboxes = temporal_smooth_detections(frame_best, max_frame_idx,
                                              settings.temporal_min_seg,
                                              settings.temporal_max_gap,
                                              settings.temporal_max_center_speed,
                                              settings.temporal_max_log_scale_speed) if frame_best else []
    _synchronize(device)
    elapsed = time.perf_counter() - started
    prediction = {"video_id": video_id, "detections": [{"bboxes": final_bboxes}] if final_bboxes else []}
    stats = {"video_id": video_id, "decoded_frames": count, "source_fps": source_fps,
             "pipeline_seconds": elapsed, "pipeline_fps": count / elapsed,
             "valid_candidates": candidate_count, "frames_before_temporal": len(frame_best),
             "frames_after_temporal": len(final_bboxes), "has_reference_embedding": ref_emb is not None,
             "has_reference_histogram": ref_hist is not None,
             "effective_predictor_args": _json_value(vars(yolo.predictor.args)),
             "predictor_imgsz": _json_value(yolo.predictor.imgsz)}
    return prediction, stats


def run(config, selection, gt_map):
    version = importlib.metadata.version("ultralytics")
    if version != "8.3.221":
        raise RuntimeError(f"Production requires ultralytics==8.3.221, found {version}; do not silently change runtime")
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if config.inference.device == "auto" else config.inference.device)
    output_dir = config.paths.output_dir
    if output_dir.exists():
        raise FileExistsError(f"Run directory already exists: {output_dir}; choose a new --output-dir")
    output_dir.mkdir(parents=True)
    manifest = {"status": "running", "config": config.to_dict(), "selection": selection,
                "frame_index_origin": 0, "source_hashes": package_hashes(),
                "runtime": {"python": platform.python_version(), "ultralytics": version,
                            "torch": torch.__version__, "torchvision": importlib.metadata.version("torchvision"),
                            "opencv": cv2.__version__, "numpy": np.__version__,
                            "pillow": importlib.metadata.version("Pillow"), "cuda": torch.version.cuda,
                            "cudnn": torch.backends.cudnn.version(), "device": str(device),
                            "visible_gpu_count": torch.cuda.device_count(),
                            "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]},
                "input_hashes": {}, "gpu_golden_parity_verified": False}
    manifest_path = output_dir / "run_manifest.json"
    write_json(manifest_path, manifest)
    try:
        # Hashing is provenance work, not model throughput. Record separately.
        hash_started = time.perf_counter()
        for video_id in selection["video_ids"]:
            manifest["input_hashes"].update(_video_inputs(config.paths.samples_dir, video_id))
        manifest["input_hash_seconds"] = time.perf_counter() - hash_started
        started = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        yolo = YOLO(str(config.paths.yolo_weights))
        siam = SiameseMobileNet().to(device)
        # Same loading policy as NB06; use only trusted, hash-verified checkpoints.
        siam.load_state_dict(torch.load(config.paths.siamese_weights, map_location=device))
        siam.eval()
        transform = get_inference_transforms(size=224)
        predictions, video_stats = [], []
        for video_id in selection["video_ids"]:
            prediction, stats = infer_video(config, video_id, yolo, siam, transform, device)
            if gt_map is not None and any(frame >= stats["decoded_frames"] for frame in gt_map[video_id]):
                raise ValueError(f"{video_id}: annotation frame is outside decoded video range")
            predictions.append(prediction)
            video_stats.append(stats)
            print(f"{video_id}: {stats['frames_after_temporal']} selected frames; {stats['pipeline_fps']:.2f} pipeline FPS")
        evaluation = evaluate_predictions(predictions, gt_map, selection["video_ids"]) if gt_map is not None else None
        write_json(output_dir / "predictions.json", predictions)
        total_frames = sum(v["decoded_frames"] for v in video_stats)
        pipeline_seconds = sum(v["pipeline_seconds"] for v in video_stats)
        cold_seconds = time.perf_counter() - started
        summary = {"evaluation": evaluation, "videos": video_stats, "decoded_frames": total_frames,
                   "pipeline_seconds": pipeline_seconds, "pipeline_fps": total_frames / pipeline_seconds,
                   "cold_run_seconds": cold_seconds, "cold_run_fps": total_frames / cold_seconds,
                   "timing_scope": "pipeline includes reference features, decode, detector, matcher, color and temporal; first video includes predictor initialization/warmup. Cold run also includes model load, evaluation and prediction serialization. Excludes input hashing, environment startup and final report writes.",
                   "fps_target": config.objective.target_fps,
                   "fps_target_met_on_this_worker": total_frames / pipeline_seconds >= config.objective.target_fps,
                   "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None}
        write_json(output_dir / "metrics.json", summary)
        manifest["status"] = "completed"
        manifest["predictions_sha256"] = sha256_file(output_dir / "predictions.json")
        write_json(manifest_path, manifest)
        if evaluation is not None:
            print(f"Local mean ST-IoU ({evaluation['video_count']} videos): {evaluation['mean_st_iou']:.6f}")
        else:
            print("No GT provided: evaluation is null, not a score of zero.")
        return summary
    except Exception as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_json(manifest_path, manifest)
        raise
