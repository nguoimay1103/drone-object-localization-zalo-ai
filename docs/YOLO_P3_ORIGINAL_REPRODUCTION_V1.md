# YOLO P3 Original Reproduction Control V1

## Mục tiêu

Thí nghiệm này kiểm tra nguyên nhân checkpoint P3 retrain kém production. Nó tái lập đường train của `03_train_yolo (1).ipynb` trước khi thử thêm P2 hoặc thay detector. Public-test GT không được dùng để train, chọn epoch hay thay đổi hyperparameter.

## Khác biệt quan trọng so với P2/P3 A/B trước

- Khởi tạo GhostHead P3/P4/P5 bằng YAML có `nc=80` và không có khóa `scale`, rồi mới load `yolo11n.pt`. Khi `train()` đọc dataset, Ultralytics override thành một class.
- Dùng một GPU T4 (`device=0`), batch 16 và seed mặc định gốc là 0.
- Chỉ truyền các hyperparameter đã xuất hiện rõ trong notebook gốc: 70 epochs, 640 px, `lr0=0.003`, `lrf=0.01`, workers 2, AMP, mixup 0.05, degrees 5 và shear 2. Các giá trị còn lại lấy từ Ultralytics 8.3.221.
- Không train P2 trong lần chạy này.

## Cách chạy trên Kaggle

1. Upload bản mới của `03_train_yolo.ipynb`.
2. Chỉ sửa `DATA_ROOT` hoặc `PRETRAINED_WEIGHTS` nếu Kaggle mount ở đường dẫn khác. Nội dung artifact phải giữ nguyên vì SHA256 và split manifest được khóa.
3. Giữ `RUN_ORIGINAL_P3_REPRODUCTION=True` và `RUN_PAIRED_TRAINING=False`.
4. Chọn một GPU T4. Notebook cố ý chỉ sử dụng `device=0` dù phiên Kaggle có hai GPU.
5. Run all. Nếu dữ liệu hoặc pretrained không khớp, notebook dừng trước khi train.

Output mặc định nằm ở `/kaggle/working/yolo_p3_original_reproduction_v1`:

- `run_summary.json`
- `environment.json`
- `dataset_manifest.json`
- `label_semantic_stats.json`
- `training_history.csv`
- `weights/p3_original_reproduction_best.pt`
- `weights/p3_original_reproduction_last.pt`
- `validation/best_full_val_single_gpu/`

## Điều kiện chuyển sang locked evaluation

`run_summary.json` phải có `status=complete`, `input_integrity_passed=true`, stride `[8,16,32]`, parameter count `2697555`, và SHA256 của best checkpoint. Sau đó mới pin checkpoint bằng SHA256 trong notebook 06 và chạy cùng fusion, threshold, temporal config cùng annotation đã khóa.

Gửi lại `run_summary.json`, `training_history.csv`, `environment.json`, `dataset_manifest.json`, `label_semantic_stats.json` và best checkpoint. Không cần hiệu chỉnh theo val mAP trước locked evaluation.

## Giới hạn tái lập

Split manifest hiện khóa đường dẫn tương đối và kích thước từng ảnh cùng toàn bộ bytes của label. Nó chưa phải hash toàn bộ bytes ảnh. Runtime mới cũng có thể khác Python 3.11.13, Torch 2.6.0+cu124 và CUDA 12.4 của lần train gốc, nên protocol được tái lập nhưng không cam kết checkpoint giống từng bit.
