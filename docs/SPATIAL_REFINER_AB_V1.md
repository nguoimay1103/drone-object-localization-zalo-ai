# Spatial Refiner A/B V1

> Trạng thái: **đóng, không promote**. Partial identity-CV cho delta âm trên mọi
> fold quan sát được và kernel chết ở fold 2. Active successor là
> [Anchor-Preserving Spatial Refiner V2](SPATIAL_REFINER_V2.md).

Self-contained Kaggle entry point: [`07-train-spatial-refiner.ipynb`](../07-train-spatial-refiner.ipynb)
Maintained source copy: [`scripts/train_spatial_refiner.py`](../scripts/train_spatial_refiner.py)
Locked public A/B: [`06-inference-main.ipynb`](../06-inference-main.ipynb)

## Mục tiêu và phạm vi

Thí nghiệm sửa tọa độ bbox sau production temporal gating. Detector, candidate
fusion `0.425/0.475/0.100`, threshold `0.54`, temporal `max_gap=7`,
`min_seg_len=24`, `max_center_speed=0.4` và tập frame prediction đều bất biến.
Refiner không thêm/xóa frame và không nối qua camera cut.

## Model

- shared MobileNetV3-small encoder đến stride 8, khởi tạo từ Siamese identity-v1;
- pixel-wise reference/search correlation, lấy max và mean response;
- thêm coordinate channels để bảo toàn vị trí;
- hai corner distributions dự đoán top-left và bottom-right;
- loss gồm corner distribution, Smooth-L1 tọa độ và GIoU;
- không có mask head vì source chỉ có bbox GT;
- correction vượt global center/scale safety gate fallback về bbox production.

TorchScript được dùng làm contract giữa training và inference. NB06 kiểm tra SHA256,
preprocessing config, chứng nhận không dùng public test và identity-CV gate trước
khi chạy A/B.

## Data và protocol chống overfit

Trainer chỉ nhận train root có `samples/` và một annotation file trong
`annotations/`. Nó không có tham số public annotation. Frame được lấy mỗi 5 frame
có GT; reference lấy từ `object_images`. Coarse boxes được tạo bằng mixture bbox
jitter phù hợp residual diagnostic, với IoU trong `[0.35, 0.98]`.

Train annotation có thể chứa nhiều bbox phân biệt trên cùng `video_id/frame`.
Giống notebook 02/04, refiner lưu frame một lần và tạo một sample riêng cho mỗi
bbox; chỉ loại bbox trùng hoàn toàn. Không áp schema public evaluator một-box/frame
lên training data. Số multi-box frame, pair IoU và tối đa 20 ví dụ được lưu tại
`dataset_manifest.json -> annotation_diagnostics`.

Validation là group K-fold theo physical identity. Quy tắc đã xác nhận giữ toàn bộ
prefix và chỉ bỏ terminal `_0`/`_1`; tên ngoại lệ phải khai báo explicit override.
Final model dùng số epoch median từ best epoch của các fold và chỉ được export khi:

- sample-weighted mean IoU delta `>= +0.005`;
- ít nhất `ceil(2/3)` held-out identities cải thiện;
- worst identity delta `>= -0.005`.

Public-test GT chỉ được dùng một lần ở locked NB06 A/B, không chọn checkpoint hay
hyperparameter.

## Chạy training trên Kaggle T4 x2

Chỉ upload notebook, sửa hai input path trong cell config rồi chạy. Toàn bộ trainer
đã được nhúng trong notebook; không cần upload file `.py` riêng.
Artifacts cần gửi lại:

- `run_summary.json`;
- `cv_folds.csv`;
- `cv_identity_results.csv`;
- `training_history.csv`;
- `split_manifest.json`;
- `environment.json`;
- `weights/spatial_refiner_v1_final.ts` nếu CV pass.

## Locked public A/B

Trong NB06, sửa:

```python
SPATIAL_REFINER_MODEL_PATH = '/kaggle/input/.../spatial_refiner_v1_final.ts'
SPATIAL_REFINER_RUN_SUMMARY_PATH = '/kaggle/input/.../run_summary.json'
RUN_SPATIAL_REFINER_AB = True
```

Control vẫn được ghi vào `OUTPUT_FILE`. Refiner prediction được ghi riêng dưới
`spatial_refiner_ab_v1_locked_eval/`; không auto-promote. Quality gate yêu cầu
public delta `>=+0.003`, ít nhất 2 identity tốt lên và worst identity không giảm
quá `0.005`. T4x2 FPS được báo là estimate và vẫn cần integrated one-pass profile
trước khi production.
