# Phần 6 — Bracketed Temporal Threshold Hysteresis V1

Notebook Kaggle: [`06-inference-main.ipynb`](../06-inference-main.ipynb). Experiment dùng identity-v1 fusion `0.425/0.475/0.100`, strong threshold `0.540` và Temporal Gating production `{gap: 7, min_seg: 24, center_speed: 0.4, log_scale_speed: None}`. Control phải tái lập offline ST-IoU `0.740013`/8.890 frames.

## Động cơ

Candidate oracle diagnostic tìm thấy 822/9.505 GT frames bị `threshold_rejection`, trong đó CardboardBox chiếm 497. CardboardBox_0 có selected recall@IoU0.5 `0.8891`, nhưng accepted recall chỉ `0.5555`; hạ threshold toàn cục không an toàn vì production đã có 543 accepted background frames.

Hysteresis giữ mọi strong candidate có score ít nhất `0.54`. Candidate trong dải weak chỉ được nhận nếu có nearest strong anchor ở cả quá khứ và tương lai, nằm trong `max_anchor_span`, đồng thời ba chuyển động sau đều qua center-speed gate:

```text
past strong → weak candidate
weak candidate → future strong
past strong → future strong
```

Center speed dùng cùng định nghĩa đã kiểm tra của Temporal Gating V1:

```text
center_distance / (frame_steps * mean(sqrt(bbox_area)))
```

Weak candidate ở đầu/cuối track, chỉ có một anchor, vượt span hoặc nhảy bbox lớn không được nhận. Inference rule không đọc video ID, identity hoặc object class. Sau hysteresis, toàn bộ accepted frames tiếp tục đi qua Temporal Gating production hiện tại.

## Search space khóa trước kết quả

```text
strong_threshold:  0.54 (fixed)
weak_threshold:    [0.46, 0.48, 0.50, 0.52, 0.54]
max_anchor_span:   [8, 16, 32]
max_center_speed:  0.4 (fixed)
config_count:      15
```

Ba cấu hình `weak_threshold=0.54` là exact controls. Exact metric ties ưu tiên weak threshold cao hơn rồi span ngắn hơn.

## Identity-held-out protocol

Ba folds lần lượt giữ toàn bộ BlackBox, CardboardBox hoặc LifeJacket. Hai video `_0/_1` của cùng vật thể luôn nằm cùng held-out fold. Config recommendation là config nhận nhiều phiếu nhất từ ba fold winners; vote tie ưu tiên ít can thiệp hơn.

Chỉ cân nhắc promote khi:

1. Mean held-out delta ít nhất `+0.003`.
2. Ít nhất `2/3` held-out identities tăng.
3. Worst held-out identity delta không thấp hơn `-0.010`.
4. Recommendation có weak threshold thấp hơn strong threshold.

Notebook không tự promote kể cả khi criteria pass.

## Cách chạy

```python
RUN_CALIBRATION = False
RUN_TEMPORAL_CALIBRATION = False
RUN_RERANKING_EXPERIMENT = False
RUN_ORACLE_DIAGNOSTICS = False
RUN_HYSTERESIS_EXPERIMENT = True
USE_TEMPORAL_HYSTERESIS = False
```

Dùng identity-v1 checkpoint SHA-256 `4a1f438d1920e60129ceb89a153c047c27ce904214fc65e9a51f6d458b52839d` và cache signature tương ứng.

Artifacts:

- `temporal_hysteresis_v1_identity_cv_results.csv`
- `temporal_hysteresis_v1_identity_cv_folds.csv`
- `temporal_hysteresis_v1_identity_cv_summary.json`
- `predictions_temporal_hysteresis_v1_identity_cv.json`

Summary báo cả `rescued_frames` và `rescued_per_video`. Full-public metric chỉ mang tính mô tả; promote decision dựa trên identity-held-out folds. Chỉ có ba identity nên CV vẫn có phương sai lớn và không thay thế independent test set.

## Kết quả và quyết định

Ba exact controls tái lập `0.740013`. CV recommendation quay về `{weak_threshold: 0.54, max_anchor_span: 8}`; mean held-out delta `-0.019644`, improved identities `0/3`, worst delta `-0.036932`. Promotion criteria không đạt.

Mọi cấu hình bật hysteresis tăng CardboardBox_0 khoảng `+0.0461` đến `+0.0970`, nhưng BlackBox_1 giảm tới `-0.0830`, LifeJacket_1 giảm tới `-0.0593`, đồng thời final output tăng 336–672 frames. Motion bracketing cứu được các chuỗi weak candidate liên tục nhưng không phân biệt được target trajectory với background trajectory liên tục. Fold transfer xác nhận tradeoff không generalize qua identity. Nhánh bị reject; `RUN_HYSTERESIS_EXPERIMENT=False`, `USE_TEMPORAL_HYSTERESIS=False`, production giữ `0.740013`.

Ghi nhận máy đọc: [`audit/TEMPORAL_HYSTERESIS_V1_RESULT.json`](audit/TEMPORAL_HYSTERESIS_V1_RESULT.json).
