# Phần 7 — Candidate Margin & Component Diagnostics V1

Notebook Kaggle: [`06-inference-main.ipynb`](../06-inference-main.ipynb). Đây là diagnostic read-only trên identity-v1 candidate cache. Production prediction được tạo trước bằng fusion `0.425/0.475/0.100`, threshold `0.540` và Temporal Gating `{gap: 7, min_seg: 24, center_speed: 0.4}`; diagnostic không thay đổi ranking hoặc output.

## Mục tiêu

Absolute fused-score recovery đã bị reject vì weak-region positive prevalence chỉ `6.26%` và threshold transfer không generalize qua identity. Diagnostic này đo xem các tín hiệu sau có phân tách weak true candidate khỏi background tốt hơn hay không:

```text
selected_score
top1_score - top2_score
top1_score / top2_score
selected YOLO confidence
selected Siamese similarity
selected color similarity
negative candidate count
```

Mỗi frame có candidate được label positive khi GT tồn tại và selected top-1 bbox có IoU ít nhất `0.5`; các frame còn lại là negative. Weak region được khóa là `selected_score < 0.54`.

Frame output còn ghi top-2 components và oracle-candidate YOLO/Siamese/Color để drill-down sau khi đã chọn feature family; các oracle fields không được dùng trong inference.

## Metrics

Cho từng feature, runner báo trên toàn selected frames và riêng weak region:

- ROC-AUC với average-rank tie handling.
- Average precision.
- Positive/negative counts và prevalence.
- Median feature cho hai label.

Runner cũng báo weak-region AUC/AP theo video và identity.

## Leave-one-identity-out transfer

Với từng feature và held-out identity:

1. Chọn threshold tối đa F1 chỉ trên hai identity còn lại.
2. Áp nguyên threshold lên held-out identity.
3. Báo TP/FP/FN/TN, precision, recall, F1, FPR và balanced accuracy.

Đây vẫn chỉ là diagnostic. Threshold thắng không được tự động đưa vào production. Chỉ cân nhắc margin/component-gated recovery nếu held-out transfer ổn định trên ít nhất hai identity và không có identity suy giảm lớn.

## Cách chạy

```python
RUN_CALIBRATION = False
RUN_TEMPORAL_CALIBRATION = False
RUN_RERANKING_EXPERIMENT = False
RUN_ORACLE_DIAGNOSTICS = False
RUN_HYSTERESIS_EXPERIMENT = False
RUN_MARGIN_DIAGNOSTICS = False  # chỉ bật lại khi cần tái lập diagnostic
```

Dùng identity-v1 checkpoint SHA-256 `4a1f438d1920e60129ceb89a153c047c27ce904214fc65e9a51f6d458b52839d` và cache signature tương ứng. Control đầu run phải tái lập `0.740013`/8.890 frames.

Artifacts:

- `candidate_margin_diagnostics_v1_frames.csv`
- `candidate_margin_diagnostics_v1_videos.csv`
- `candidate_margin_diagnostics_v1_summary.json`

Summary có guard `production_predictions_unchanged` và `production_per_video_metric_exact_parity`. Nếu một guard sai, runner dừng trước khi ghi kết luận.

## Kết quả và quyết định

Run identity-v1 đạt exact production parity `0.7400130889`/8.890 frames. Trong weak region, fused `selected_score` đạt ROC-AUC `0.8951`, AP `0.3671`; top1-top2 margin chỉ đạt `0.5728/0.1359` và score ratio `0.6231/0.1690`. YOLO, Siamese và Color lần lượt đạt AUC `0.8244`, `0.7841`, `0.5701`.

Tín hiệu component không transfer đồng nhất: Siamese có AP `0.4774/0.5590` trên BlackBox/CardboardBox, trong khi Color có AP `0.4456` trên LifeJacket nhưng gần như không hữu ích cho CardboardBox. Leave-one-identity-out cho thấy threshold học từ hai identity có thể sụp trên identity còn lại; ví dụ threshold Siamese cho CardboardBox chỉ đạt held-out F1 `0.0080`, còn threshold Siamese cho LifeJacket đạt `0.0986` với FPR `0.4696`.

**Quyết định:** không promote margin, ratio hoặc component threshold vào inference. `RUN_MARGIN_DIAGNOSTICS` được tắt sau khi ghi audit; production vẫn dùng fusion/threshold/temporal đã khóa và giữ score `0.740013`. Bước kế tiếp phải tạo candidate mới ở detector thay vì tiếp tục recovery bằng scalar feature trên cache cũ.
