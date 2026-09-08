"""NB06 reference features and color scores with explicit path/device inputs."""
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from .models.siamese import get_inference_transforms


def load_reference_embeddings(samples_dir, video_folder, model, device,
                              use_multiscale_ref=True, ref_scales=(224, 112, 56)):
    """
    [SPRINT 1-C] Multi-scale reference embedding.
    Encode ref images tai 3 scales [224, 112, 56] roi average.
    Scale nho simulate DroneDegradation → bridge domain gap.
    """
    obj_dir = os.path.join(samples_dir, video_folder, "object_images")
    if not os.path.exists(obj_dir):
        return None

    img_paths = sorted([
        os.path.join(obj_dir, f) for f in os.listdir(obj_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])
    if not img_paths:
        return None

    all_tensors = []
    scales_to_use = ref_scales if use_multiscale_ref else [224]
    transform_224 = get_inference_transforms(size=224)

    for p in img_paths:
        img = cv2.imread(p)
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)

        for scale in scales_to_use:
            if scale != 224:
                # Downscale then NEAREST upsample = DroneDegradation effect
                small = pil_img.resize((scale, scale), Image.BILINEAR)
                pil_input = small.resize((224, 224), Image.NEAREST)
            else:
                pil_input = pil_img
            all_tensors.append(transform_224(pil_input))

    if not all_tensors:
        return None

    batch = torch.stack(all_tensors).to(device)
    with torch.no_grad():
        embs = model(batch)
        mean_emb = torch.mean(embs, dim=0, keepdim=True)
        mean_emb = F.normalize(mean_emb, p=2, dim=1)

    return mean_emb


def compute_hs_histogram(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, 30, 30]), np.array([180, 255, 255]))
    hist = cv2.calcHist([hsv], [0, 1], mask, [30, 32], [0, 180, 0, 256])
    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return hist


def load_reference_histogram(samples_dir, video_folder):
    obj_dir = os.path.join(samples_dir, video_folder, "object_images")
    if not os.path.exists(obj_dir):
        return None
    img_paths = [os.path.join(obj_dir, f) for f in os.listdir(obj_dir)
                 if f.endswith(('.jpg', '.png'))]
    agg_hist = None
    for p in img_paths:
        img = cv2.imread(p)
        if img is None:
            continue
        hist = compute_hs_histogram(img)
        agg_hist = hist if agg_hist is None else agg_hist + hist
    if agg_hist is not None:
        cv2.normalize(agg_hist, agg_hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return agg_hist


def calculate_color_score(crop_img, ref_hist):
    if ref_hist is None or crop_img is None or crop_img.size == 0:
        return 0.0
    crop_hist = compute_hs_histogram(crop_img)
    return max(0.0, cv2.compareHist(ref_hist, crop_hist, cv2.HISTCMP_CORREL))
