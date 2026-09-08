import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import mobilenet_v3_small

from ultralytics import YOLO
import streamlit as st

# ============================================================
# 1. CẤU HÌNH & LOAD MODEL
# ============================================================

# SỬA lại cho đúng với đường dẫn weight của bạn
YOLO_MODEL_PATH = "yolo_drone_best.pt"
SIAMESE_MODEL_PATH = "siamese_mobilenet_best.pth"

CONFIDENCE_DEFAULT = 0.05
MATCHING_THRESHOLD_DEFAULT = 0.45  # [SPRINT 1] aligned with NB06
IMGSZ = 640
# [SPRINT 1] Unified weights – matches 06_inference_main.ipynb
WEIGHT_YOLO    = 0.4
WEIGHT_SIAMESE = 0.3
WEIGHT_COLOR   = 0.3

# [SPRINT 1] Multi-scale ref embedding (same as NB06)
USE_MULTISCALE_REF = True
REF_SCALES = [224, 112, 56]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# 2. ĐỊNH NGHĨA SIAMESE MOBILENET (giống inference.py)
# ============================================================

class SiameseMobileNet(nn.Module):
    def __init__(self, embedding_dim=576):
        super().__init__()
        full_model = mobilenet_v3_small(weights=None)
        self.features = full_model.features
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
        )

    def forward(self, x):
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)
        x = self.projection(x)
        return F.normalize(x, p=2, dim=1)


def get_inference_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


@st.cache_resource
def load_models():
    yolo = YOLO(YOLO_MODEL_PATH)
    siamese = SiameseMobileNet().to(DEVICE)
    if os.path.exists(SIAMESE_MODEL_PATH):
        siamese.load_state_dict(
            torch.load(SIAMESE_MODEL_PATH, map_location=DEVICE)
        )
    siamese.eval()
    return yolo, siamese, get_inference_transforms()


yolo_model, siamese_model, siamese_transform = load_models()


# ============================================================
# 3. HÀM TÍNH EMBEDDING & MATCHING
# ============================================================

def encode_image_for_siamese(pil_img: Image.Image) -> torch.Tensor:
    t = siamese_transform(pil_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        emb = siamese_model(t)
    return emb  # (1, 128)


def build_ref_embedding(ref_imgs):
    """
    [SPRINT 1] Multi-scale reference embedding.
    ref_imgs: list cac anh numpy (RGB) hoac None
    Encode moi anh o 3 scales [224, 112, 56] roi average.
    Scale nho simulate DroneDegradation → bridge domain gap.
    """
    embs = []
    scales_to_use = REF_SCALES if USE_MULTISCALE_REF else [224]
    transform_224 = siamese_transform  # already 224

    for img in ref_imgs:
        if img is None:
            continue
        pil = Image.fromarray(img)
        for scale in scales_to_use:
            if scale != 224:
                small = pil.resize((scale, scale), Image.BILINEAR)
                pil_input = small.resize((224, 224), Image.NEAREST)
            else:
                pil_input = pil
            emb = encode_image_for_siamese(pil_input)
            embs.append(emb)

    if len(embs) == 0:
        return None

    embs_cat = torch.cat(embs, dim=0)      # (N, 128)
    mean_emb = embs_cat.mean(dim=0, keepdim=True)
    mean_emb = F.normalize(mean_emb, p=2, dim=1)
    return mean_emb


def compute_hs_histogram(img_bgr):
    """Tinh Histogram 2D Hue-Saturation cho color matching."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, 30, 30]), np.array([180, 255, 255]))
    hist = cv2.calcHist([hsv], [0, 1], mask, [30, 32], [0, 180, 0, 256])
    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return hist


def build_ref_histogram(ref_imgs):
    """[SPRINT 1] Tinh histogram mau trung binh cua reference images."""
    agg_hist = None
    for img_rgb in ref_imgs:
        if img_rgb is None:
            continue
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        hist = compute_hs_histogram(img_bgr)
        agg_hist = hist if agg_hist is None else agg_hist + hist
    if agg_hist is not None:
        cv2.normalize(agg_hist, agg_hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return agg_hist


def select_best_box_with_siamese(frame_rgb: np.ndarray,
                                 frame_bgr: np.ndarray,
                                 boxes_xyxy: np.ndarray,
                                 scores: np.ndarray,
                                 ref_emb: torch.Tensor,
                                 ref_hist,
                                 match_thresh: float):
    """
    [SPRINT 1] Updated scoring:
      final_score = (YOLO*yolo_conf + Siam*siam_score + Color*color_score) / total_w
      Aligned with NB06: YOLO=0.4, Siamese=0.3, Color=0.3
    """
    h_img, w_img = frame_rgb.shape[:2]
    pil_frame = Image.fromarray(frame_rgb)

    candidates = []
    best_box = None
    best_final = -1.0

    if ref_emb is None or len(boxes_xyxy) == 0:
        return None, []

    crops_pil = []
    crops_bgr = []
    valid_idx = []
    for i, box in enumerate(boxes_xyxy):
        x1, y1, x2, y2 = map(int, box)
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(w_img, x2); y2 = min(h_img, y2)
        if x2 <= x1 + 5 or y2 <= y1 + 5:
            continue
        crops_pil.append(siamese_transform(pil_frame.crop((x1, y1, x2, y2))))
        crops_bgr.append(frame_bgr[y1:y2, x1:x2])
        valid_idx.append(i)

    if len(crops_pil) == 0:
        return None, []

    # Siamese scores
    batch = torch.stack(crops_pil).to(DEVICE)
    with torch.no_grad():
        cand_embs = siamese_model(batch)
        dists = torch.cdist(ref_emb, cand_embs)[0].cpu().numpy()
    siam_scores = np.maximum(0, 1.0 - dists / 2.0)

    # Color scores
    color_scores = np.zeros(len(crops_pil))
    if ref_hist is not None:
        for k, crop_bgr in enumerate(crops_bgr):
            if crop_bgr.size > 0:
                crop_hist = compute_hs_histogram(crop_bgr)
                color_scores[k] = max(0.0, cv2.compareHist(ref_hist, crop_hist, cv2.HISTCMP_CORREL))

    # Fuse
    w_y = WEIGHT_YOLO
    w_s = WEIGHT_SIAMESE
    w_c = WEIGHT_COLOR if ref_hist is not None else 0.0
    total_w = w_y + w_s + w_c

    for k, idx_orig in enumerate(valid_idx):
        yolo_conf = float(scores[idx_orig])
        siam_score = float(siam_scores[k])
        color_score = float(color_scores[k])
        final_score = (w_y * yolo_conf + w_s * siam_score + w_c * color_score) / total_w

        box_xyxy = boxes_xyxy[idx_orig].tolist()
        candidates.append({
            "bbox": box_xyxy,
            "yolo_conf": yolo_conf,
            "siam_score": siam_score,
            "color_score": color_score,
            "final_score": final_score
        })

        if final_score > best_final:
            best_final = final_score
            best_box = box_xyxy

    if best_box is not None and best_final >= match_thresh:
        return best_box, candidates
    else:
        return None, candidates


# ============================================================
# 4. XỬ LÝ VIDEO → VIDEO
# ============================================================

def process_video_with_refs(video_path, ref_imgs,
                            conf_thres, match_thres):
    ref_emb = build_ref_embedding(ref_imgs)
    if ref_emb is None:
        return None, "Vui long upload it nhat 1 anh tham chieu."

    # [SPRINT 1] Build color histogram from reference images
    ref_hist = build_ref_histogram(ref_imgs)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, "Khong mo duoc video."

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    tmp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    out_path = tmp_out.name
    tmp_out.close()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    frame_idx = 0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_detected = 0

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        frame_idx += 1
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # YOLO detect
        results = yolo_model.predict(
            frame_bgr,
            conf=conf_thres,
            imgsz=IMGSZ,
            verbose=False
        )
        res = results[0]
        if res.boxes is not None and len(res.boxes) > 0:
            boxes = res.boxes.xyxy.cpu().numpy()
            scores = res.boxes.conf.cpu().numpy()
        else:
            boxes = np.zeros((0, 4))
            scores = np.zeros((0,))

        # [SPRINT 1] Pass frame_bgr and ref_hist for color scoring
        best_box, candidates = select_best_box_with_siamese(
            frame_rgb, frame_bgr, boxes, scores, ref_emb, ref_hist, match_thres
        )

        out_frame = frame_bgr.copy()

        # vẽ candidate mỏng (vàng)
        for cand in candidates:
            x1, y1, x2, y2 = map(int, cand["bbox"])
            cv2.rectangle(out_frame, (x1, y1), (x2, y2), (0, 255, 255), 1)

        # vẽ box được chọn (xanh lá)
        if best_box is not None:
            x1, y1, x2, y2 = map(int, best_box)
            cv2.rectangle(out_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            n_detected += 1

        txt = f"Frame {frame_idx}/{n_frames}"
        cv2.putText(out_frame, txt, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        writer.write(out_frame)

    cap.release()
    writer.release()

    msg = f"Xử lý xong {frame_idx} frame. Số frame có bbox được chọn: {n_detected}."
    return out_path, msg


# ============================================================
# 5. GIAO DIỆN WEB STREAMLIT
# ============================================================

st.title("Drone Target Demo – YOLOv11n + Siamese MobileNetV3")
st.write("Upload 1 video drone và 1–3 ảnh tham chiếu cùng object để xem hệ thống hoạt động.")

video_file = st.file_uploader("Video drone", type=["mp4", "avi", "mov"])
col1, col2, col3 = st.columns(3)
with col1:
    ref1_file = st.file_uploader("Ảnh tham chiếu 1", type=["jpg", "jpeg", "png"], key="ref1")
with col2:
    ref2_file = st.file_uploader("Ảnh tham chiếu 2 (tuỳ chọn)", type=["jpg", "jpeg", "png"], key="ref2")
with col3:
    ref3_file = st.file_uploader("Ảnh tham chiếu 3 (tuỳ chọn)", type=["jpg", "jpeg", "png"], key="ref3")

conf_thres = st.slider("YOLO confidence threshold", 0.01, 0.9, value=CONFIDENCE_DEFAULT, step=0.01)
match_thres = st.slider("Matching threshold (YOLO + Siamese)", 0.1, 1.0, value=MATCHING_THRESHOLD_DEFAULT, step=0.05)

if st.button("Chạy demo"):
    if video_file is None:
        st.warning("Vui lòng upload video drone.")
    elif ref1_file is None:
        st.warning("Vui lòng upload ít nhất 1 ảnh tham chiếu.")
    else:
        # lưu video tạm
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_vid:
            tmp_vid.write(video_file.read())
            video_path = tmp_vid.name

        # đọc ảnh tham chiếu
        ref_imgs = []
        for f in [ref1_file, ref2_file, ref3_file]:
            if f is None:
                ref_imgs.append(None)
            else:
                img = Image.open(f).convert("RGB")
                ref_imgs.append(np.array(img))

        with st.spinner("Đang xử lý video..."):
            out_path, msg = process_video_with_refs(
                video_path, ref_imgs, conf_thres, match_thres
            )

        if out_path is None:
            st.error(msg)
        else:
            st.success(msg)
            st.video(out_path)
