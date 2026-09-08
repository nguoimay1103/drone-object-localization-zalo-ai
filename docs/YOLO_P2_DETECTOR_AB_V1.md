# Phần 9A — YOLO P2 Detector Training A/B V1

Notebook Kaggle: [`03_train_yolo.ipynb`](../03_train_yolo.ipynb).

## Mục tiêu

Detector profile cho thấy checkpoint production chỉ có strides `[8,16,32]`; 48,45% GT frames có projected minimum side dưới 16 px. Full-frame 960 làm candidate quality giảm, còn SAHI tăng oracle nhưng không transfer qua identity. Thí nghiệm này thay đổi representation lúc train bằng một nhánh P2/4, không tune policy trên public test.

Topology P2 theo cấu trúc chính thức [`yolov8-p2.yaml` tại Ultralytics v8.3.221](https://github.com/ultralytics/ultralytics/blob/v8.3.221/ultralytics/cfg/models/v8/yolov8-p2.yaml): upsample P3, concat backbone P2, tạo feature P2 và xây lại bottom-up path đến P5. Project giữ backbone YOLO11, C2f và GhostConv của checkpoint hiện tại.

## Paired design

| Variant | Detection strides | Khác biệt |
|---|---:|---|
| `p3_control` | 8, 16, 32 | GhostHead hiện tại được train lại trong cùng run |
| `p2_stride4` | 4, 8, 16, 32 | Thêm P2/4 head; các thành phần còn lại giữ nguyên |

Hai variant dùng chung dataset train/val và manifest, pretrained `yolo11n.pt`, seed `2026`, deterministic mode, 70 epochs, 640 px, 16 samples/GPU, optimizer, loss, augmentation và Ultralytics `8.3.221`.

Không dùng annotation public-test để train, early-stop, chọn epoch hoặc đổi hyperparameter.

## T4 x2

Notebook dùng cả hai GPU cho từng variant theo thứ tự:

```text
device=0,1
batch=32 = 16/GPU
p3_control -> giải phóng model/cache -> p2_stride4
```

Train tuần tự tránh hai job cùng đọc dataset và tranh CPU/RAM. Variant hoàn tất có summary và hash checkpoint; chạy lại notebook sẽ skip variant nếu signature và weight hash vẫn khớp.

Trước khi train P3, runner dựng cả hai graph, kiểm tra stride và thử load pretrained cho cả P3 lẫn P2. Vì vậy lỗi topology hoặc checkpoint P2 xuất hiện ngay, không xuất hiện sau khi đã tốn thời gian train control.

## Đường dẫn cần sửa

```python
DATA_ROOT = "/kaggle/input/datasets/phamnguyenanhtuan/data-train-yolo/yolo_dataset_doan_1"
PRETRAINED_WEIGHTS = "yolo11n.pt"  # hoặc đường dẫn Kaggle tuyệt đối
OUTPUT_ROOT = "/kaggle/working/yolo_p2_detector_ab_v1"
```

Dataset phải có `dataset.yaml`, `images/train`, `images/val`, `labels/train` và `labels/val`.

## Artifact cần gửi lại

```text
yolo_p2_detector_ab_v1/
├── run_summary.json
├── environment.json
├── dataset_manifest.json
├── training_ab_results.csv
├── p3_control_summary.json
├── p2_stride4_summary.json
└── weights/
    ├── p3_control_best.pt
    └── p2_stride4_best.pt
```

Giữ cả hai checkpoint. Val mAP chỉ kiểm tra training health, không quyết định promote.

## Locked evaluation sau training

Sau khi nhận artifacts, notebook 06 sẽ so sánh production checkpoint cũ, paired P3 control và P2 variant với:

```text
imgsz=640
confidence threshold=0.05
TTA=True
Siamese SHA=4a1f438d...
fusion=0.425/0.475/0.100
matching threshold=0.54
temporal=(max_gap=7, min_seg_len=24, max_center_speed=0.4)
```

Promotion yêu cầu:

1. ST-IoU tăng xuyên identity, không dựa vào một video.
2. Candidate recall/oracle của nhóm dưới 16 px tăng.
3. Worst identity delta không dưới `-0.01`.
4. End-to-end inference đạt ít nhất 25 FPS trên T4 x2.

Không recalibrate fusion/temporal trước khi đánh giá locked lần đầu.

## Kết quả training

| Variant | Params | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|
| `p3_control` | 2.697.555 | 0,97042 | 0,94515 | 0,97295 | 0,78842 |
| `p2_stride4` | 2.924.724 | 0,97645 | 0,94053 | 0,97631 | 0,79593 |

Hai checkpoints, dataset manifest, pretrained hash và stride preflight đều hợp lệ. Source of truth là dataset manifest `c1e80cc5...` với 18.965 train và 6.041 val images. P2 tăng precision `+0,00603`, mAP50 `+0,00336`, mAP50-95 `+0,00751`, nhưng recall giảm `-0,00462`; chưa đủ để promote. Artifact hiển thị best epoch 71 do parser cũ cộng thêm 1 vào epoch 1-based của Ultralytics; hàng thực tế là epoch 70 và parser đã được sửa.
