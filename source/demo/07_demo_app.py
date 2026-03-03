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
MATCHING_THRESHOLD_DEFAULT = 0.5
IMGSZ = 640
WEIGHT_YOLO = 0.7
WEIGHT_SIAMESE = 0.3

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
    ref_imgs: list các ảnh numpy (RGB) hoặc None
    Trả về embedding trung bình (1, 128) hoặc None nếu không có ảnh nào.
    """
    embs = []
    for img in ref_imgs:
        if img is None:
            continue
        pil = Image.fromarray(img)
        emb = encode_image_for_siamese(pil)
        embs.append(emb)

    if len(embs) == 0:
        return None

    embs_cat = torch.cat(embs, dim=0)      # (N, 128)
    mean_emb = embs_cat.mean(dim=0, keepdim=True)
    mean_emb = F.normalize(mean_emb, p=2, dim=1)
    return mean_emb


def select_best_box_with_siamese(frame_rgb: np.ndarray,
                                 boxes_xyxy: np.ndarray,
                                 scores: np.ndarray,
                                 ref_emb: torch.Tensor,
                                 match_thresh: float):
    """
    final_score = 0.7 * yolo_conf + 0.3 * siam_score
    siam_score = 1 - dist/2 (dist = L2(ref_emb, cand_emb))
    """
    h_img, w_img = frame_rgb.shape[:2]
    pil_frame = Image.fromarray(frame_rgb)

    candidates = []
    best_box = None
    best_final = -1.0

    if ref_emb is None or len(boxes_xyxy) == 0:
        return None, []

    crops = []
    valid_idx = []
    for i, box in enumerate(boxes_xyxy):
        x1, y1, x2, y2 = map(int, box)
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(w_img, x2); y2 = min(h_img, y2)
        if x2 <= x1 + 5 or y2 <= y1 + 5:
            continue
        crop = pil_frame.crop((x1, y1, x2, y2))
        crops.append(siamese_transform(crop))
        valid_idx.append(i)

    if len(crops) == 0:
        return None, []

    batch = torch.stack(crops).to(DEVICE)
    with torch.no_grad():
        cand_embs = siamese_model(batch)
        dists = torch.cdist(ref_emb, cand_embs)[0].cpu().numpy()

    for k, dist in enumerate(dists):
        idx_orig = valid_idx[k]
        yolo_conf = float(scores[idx_orig])
        siam_score = max(0.0, 1.0 - (dist / 2.0))
        final_score = WEIGHT_YOLO * yolo_conf + WEIGHT_SIAMESE * siam_score

        box_xyxy = boxes_xyxy[idx_orig].tolist()
        candidates.append(
            {"bbox": box_xyxy,
             "yolo_conf": yolo_conf,
             "siam_score": siam_score,
             "final_score": final_score}
        )

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
        return None, "Vui lòng upload ít nhất 1 ảnh tham chiếu."

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, "Không mở được video."

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

        best_box, candidates = select_best_box_with_siamese(
            frame_rgb, boxes, scores, ref_emb, match_thres
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
