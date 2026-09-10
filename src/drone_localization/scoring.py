"""NB06 weighted scoring. Keep first-candidate tie breaking and float operations."""


def select_best_candidate(boxes, confs, siam_scores, color_scores, valid_indices,
                          has_ref_embedding, has_ref_histogram, settings):
    best_score = -1.0
    best_box = None
    for k, idx in enumerate(valid_indices):
        yolo_s = float(confs[idx])
        siam_s = float(siam_scores[k])
        col_s = float(color_scores[k])
        w_y = settings.weight_yolo
        w_s = settings.weight_siamese if has_ref_embedding else 0.0
        w_c = settings.weight_color if has_ref_histogram else 0.0
        total_w = w_y + w_s + w_c
        final_score = (w_y * yolo_s + w_s * siam_s + w_c * col_s) / total_w if total_w > 0 else yolo_s
        if final_score > best_score:
            best_score = final_score
            best_box = boxes[idx]
    return best_box, best_score
