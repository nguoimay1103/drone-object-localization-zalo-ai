# Phần 9B — YOLO P2 Detector Locked Evaluation V1

Notebook: [`06-inference-main.ipynb`](../06-inference-main.ipynb).

## Đường dẫn cần sửa

Upload hai checkpoint từ paired training rồi đặt:

```python
P3_CONTROL_MODEL_PATH = '/kaggle/input/.../p3_control_best.pt'
P2_STRIDE4_MODEL_PATH = '/kaggle/input/.../p2_stride4_best.pt'
```

Runner kiểm tra SHA-256 cố định:

```text
p3_control = d317a81a494113ddc789df116a7adb9ef90be362aa4ef7a63a06242f47843234
p2_stride4 = 8f8279f178dc7ec1f57a8b19a28ff7b470a4652398face895a00deef67d60e16
```

Production candidate cache có thể reuse qua `PRECOMPUTED_CANDIDATE_CACHE_FILE`. Lần đầu P3/P2 sẽ tạo hai cache riêng; các lần sau có thể đặt `P2_AB_PRECOMPUTED_CACHES`.

## Cấu hình khóa

```text
imgsz=640
confidence=0.05
TTA=True
fusion=0.425/0.475/0.100
matching threshold=0.54
temporal max_gap=7, min_seg_len=24, max_center_speed=0.4
```

Không chạy fusion calibration, temporal sweep hoặc rule riêng từng object.

## Promotion gates

Mỗi trained checkpoint được so độc lập với production:

- mean ST-IoU delta ≥ `+0.003`;
- ít nhất `2/3` identity tăng;
- worst identity delta ≥ `-0.010`;
- sub-16 oracle recall delta ≥ `+0.002`;
- aggregate candidate-pipeline FPS ≥ `25`.

Nếu cả P3 và P2 cùng pass, checkpoint có mean ST-IoU cao hơn được recommended. Runner chỉ ghi recommendation, không đổi production tự động.

## Artifacts cần gửi lại

```text
yolo_p2_detector_locked_eval_v1_results.csv
yolo_p2_detector_locked_eval_v1_videos.csv
yolo_p2_detector_locked_eval_v1_summary.json
```

Gửi thêm log bắt đầu từ `YOLO P2 DETECTOR LOCKED CHECKPOINT A/B V1`.

## Kết quả

Locked evaluation đã hoàn tất và **không promote** checkpoint mới:

| Variant | ST-IoU | Delta vs production | R@0.5 | Sub-16 R@0.5 | FPS |
|---|---:|---:|---:|---:|---:|
| production | 0.740013 | +0.000000 | 0.9481 | 0.9173 | n/a |
| P3 control | 0.686636 | -0.053377 | 0.9417 | 0.9346 | 30.59 |
| P2 stride-4 | 0.653633 | -0.086380 | 0.9345 | 0.9173 | 26.87 |

P3 tăng sub-16 recall nhưng giảm mạnh ở `BlackBox_1`. P2 không tăng sub-16 recall so với production và thấp hơn paired P3 `0.033004` ST-IoU. Cả hai đạt yêu cầu tốc độ nhưng thất bại các quality gates. Production checkpoint tiếp tục được giữ và runner P2 A/B đã tắt mặc định. Xem [audit result](audit/YOLO_P2_DETECTOR_LOCKED_EVAL_V1_RESULT.json).
