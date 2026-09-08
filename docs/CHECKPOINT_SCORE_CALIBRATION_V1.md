# Phần 9C — Checkpoint-Specific Score Calibration V1

Notebook: [`06-inference-main.ipynb`](../06-inference-main.ipynb).

## Mục tiêu

Kiểm tra liệu paired P3 hoặc P2 bị giảm ST-IoU vì dùng trực tiếp thang điểm fusion đã calibrate cho production detector hay không. Experiment chỉ replay candidate cache; không chạy YOLO/Siamese lại, không đổi temporal và không có tham số riêng theo object.

Ba detector nhận cùng một search budget:

- production;
- paired `p3_control`;
- `p2_stride4`.

## Cache cần upload

Dùng ba cache hoàn chỉnh từ các run trước:

```python
PRECOMPUTED_CANDIDATE_CACHE_FILE = \
    '/kaggle/input/.../candidate_scores.json.gz'

P2_AB_PRECOMPUTED_CACHES = {
    'p3_control': '/kaggle/input/.../candidate_scores_p3_control.json.gz',
    'p2_stride4': '/kaggle/input/.../candidate_scores_p2_stride4.json.gz',
}
```

`P3_CONTROL_MODEL_PATH` và `P2_STRIDE4_MODEL_PATH` có thể để trống trong experiment cache-only này. Loader kiểm tra toàn bộ cache signature và SHA checkpoint đã khóa. File gzip thật, plain JSON bị giữ nhầm đuôi `.gz`, và file `.gz.tmp` hoàn chỉnh đều được nhận diện theo nội dung.

Chỉ bật:

```python
RUN_CHECKPOINT_SCORE_CALIBRATION = True
RUN_P2_DETECTOR_AB = False
```

## Search space khóa trước

Mỗi checkpoint có đúng 1.100 cấu hình:

```text
yolo_power      = [0.75, 1.0, 1.25, 1.5]
yolo_weight     = [0.25, 0.35, 0.425, 0.50, 0.60]
color_weight    = [0.00, 0.05, 0.10, 0.15, 0.20]
siamese_weight  = 1 - yolo_weight - color_weight, tối thiểu 0.20
threshold       = 0.44 ... 0.64, bước 0.02
```

Detector confidence được biến đổi đơn điệu:

```text
calibrated_yolo = clip(yolo_conf, 0, 1) ** yolo_power
```

`yolo_power=1`, weights `0.425/0.475/0.100` và threshold `0.54` là exact control. Runner dừng nếu control không tái lập prediction của locked evaluation.

Temporal luôn giữ:

```text
max_gap=7, min_seg_len=24, max_center_speed=0.4,
max_log_scale_speed=None
```

## Identity-held-out protocol

Ba fold lần lượt giữ toàn bộ hai video của `BlackBox`, `CardboardBox` và `LifeJacket`. Mỗi fold chọn cấu hình bằng bốn video thuộc hai identity còn lại rồi mới chấm hai video held-out.

P3/P2 chỉ pass khi đồng thời:

1. mean identity-held-out delta so với production ít nhất `+0.003`;
2. ít nhất `2/3` held-out identities tăng;
3. worst held-out identity delta không thấp hơn `-0.010`;
4. ít nhất hai fold chọn đúng cùng một cấu hình global;
5. cấu hình được vote có full-public descriptive delta ít nhất `+0.003`;
6. detector đã đạt ít nhất `25 FPS` trong locked evaluation.

Full-public score không dùng thay identity CV. Runner chỉ ghi recommendation và không sửa production config tự động.

## Artifacts cần gửi lại

```text
checkpoint_score_calibration_v1_identity_cv_results.csv
checkpoint_score_calibration_v1_identity_cv_folds.csv
checkpoint_score_calibration_v1_identity_cv_summary.json
```

Gửi thêm log từ dòng `CHECKPOINT-SPECIFIC SCORE CALIBRATION V1 - IDENTITY CV`.

## Kết quả

Experiment đã hoàn tất và **không promote** checkpoint nào:

| Variant | Locked | Full-public best | Delta best vs production | Identity-CV | Config votes |
|---|---:|---:|---:|---:|---:|
| production | 0.740013 | 0.742907 | +0.002894 | 0.606315 | 1 |
| P3 control | 0.686636 | 0.690595 | -0.049418 | 0.515171 | 1 |
| P2 stride-4 | 0.653633 | 0.679114 | -0.060899 | 0.508691 | 1 |

P2 phục hồi `+0.025482` so với locked config và P3 phục hồi `+0.003958`, xác nhận checkpoint-specific score transfer có ảnh hưởng. Tuy nhiên, cả hai vẫn kém production rõ rệt. Mỗi fold chọn một cấu hình khác nhau; cấu hình chọn trên hai identity gây suy giảm lớn trên identity held-out. Production full-public best cũng chỉ tăng `+0.002894`, thấp hơn promotion gate `+0.003` và không transfer qua identity CV.

Runner đã tắt mặc định. Giữ production detector và không tune tiếp trên sáu public videos; xem [audit result](audit/CHECKPOINT_SCORE_CALIBRATION_V1_RESULT.json).
