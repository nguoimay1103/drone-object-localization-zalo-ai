"""Temporal interpolation, motion gating and short-segment filtering."""

import math


def gap_motion_metrics(box1, box2, steps):
    """Return scale-normalized center speed and per-frame log-scale speed."""
    if steps <= 0:
        raise ValueError("steps must be positive")
    w1, h1 = max(1e-6, box1[2] - box1[0]), max(1e-6, box1[3] - box1[1])
    w2, h2 = max(1e-6, box2[2] - box2[0]), max(1e-6, box2[3] - box2[1])
    c1 = ((box1[0] + box1[2]) * 0.5, (box1[1] + box1[3]) * 0.5)
    c2 = ((box2[0] + box2[2]) * 0.5, (box2[1] + box2[3]) * 0.5)
    scale1, scale2 = math.sqrt(w1 * h1), math.sqrt(w2 * h2)
    mean_scale = max(1e-6, 0.5 * (scale1 + scale2))
    center_speed = math.hypot(c2[0] - c1[0], c2[1] - c1[1]) / (
        steps * mean_scale
    )
    log_scale_speed = abs(math.log(scale2 / scale1)) / steps
    return center_speed, log_scale_speed


def gap_is_plausible(box1, box2, steps, max_center_speed=None,
                     max_log_scale_speed=None):
    center_speed, log_scale_speed = gap_motion_metrics(box1, box2, steps)
    if max_center_speed is not None and center_speed > max_center_speed:
        return False
    if max_log_scale_speed is not None and log_scale_speed > max_log_scale_speed:
        return False
    return True


def temporal_smooth_detections(frame_best, max_frame_idx,
                               min_seg_len=3,
                               max_gap=5,
                               max_center_speed=None,
                               max_log_scale_speed=None):
    """
    Interpolate plausible gaps and remove short contiguous segments.

    Passing both motion thresholds as ``None`` preserves the historical
    interpolation behavior exactly. Coordinates are exported with ``round`` to
    match the canonical notebook.
    """
    T = max_frame_idx + 1
    det = [None] * T

    for f, info in frame_best.items():
        if 0 <= f < T:
            det[f] = info

    # Fill gaps with linear interpolation.
    for i in range(T):
        if det[i] is not None:
            next_idx = -1
            for k in range(1, max_gap + 2):
                if i + k < T and det[i + k] is not None:
                    next_idx = i + k
                    break
            gap_size = next_idx - i - 1 if next_idx != -1 else 999
            if 0 < gap_size <= max_gap:
                b1, s1 = det[i]['bbox'], det[i]['score']
                b2, s2 = det[next_idx]['bbox'], det[next_idx]['score']
                steps = next_idx - i
                if not gap_is_plausible(
                    b1, b2, steps,
                    max_center_speed=max_center_speed,
                    max_log_scale_speed=max_log_scale_speed,
                ):
                    continue
                for step in range(1, steps):
                    alpha = step / steps
                    ibox = [b1[j] * (1 - alpha) + b2[j] * alpha for j in range(4)]
                    iscore = s1 * (1 - alpha) + s2 * alpha
                    det[i + step] = {'bbox': ibox, 'score': iscore}

    # Remove short segments.
    i = 0
    while i < T:
        if det[i] is not None:
            j = i
            while j + 1 < T and det[j + 1] is not None:
                j += 1
            if (j - i + 1) < min_seg_len:
                for k in range(i, j + 1):
                    det[k] = None
            i = j + 1
        else:
            i += 1

    out = []
    for f in range(T):
        if det[f] is not None:
            x1, y1, x2, y2 = det[f]['bbox']
            out.append({
                "frame": f,
                "x1": int(round(x1)), "y1": int(round(y1)),
                "x2": int(round(x2)), "y2": int(round(y2))
            })
    return out
