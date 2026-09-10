# Phần 2 — Safe negative mining v2

Notebook Kaggle độc lập: [`04-data-prep-matching.ipynb`](../04-data-prep-matching.ipynb). Bản baseline chính xác nằm tại [`baselines/04-data-prep-matching-baseline.ipynb`](../baselines/04-data-prep-matching-baseline.ipynb).

Baseline offline hiện tại sau identity training và calibration là **0.722052**. Phần này chỉ thay hai thư mục negative; anchors/positives được sao chép nguyên byte từ dataset đã train identity-v1. Chưa đổi notebook 05 training, model architecture hoặc notebook 06 production config.

## Vì sao cần sửa

Notebook cũ dùng `yolov8n.pt` COCO để mining dù inference dùng YOLO-Drone. Hard negatives vì vậy không mô phỏng false positives của detector thật. Candidate chỉ bị loại khi IoU với GT không nhỏ hơn `0.05`; với drone small-object, bbox lớn có thể chứa trọn target nhưng vẫn có IoU dưới ngưỡng và bị gán nhãn negative. Background crop cũ có vấn đề tương tự với IoU `0.01`. Notebook cũng nuốt mọi lỗi YOLO bằng `except: pass` và tăng counter kể cả khi crop không được ghi.

## Thay đổi có chủ đích

- `MINING_MODEL_PATH` bắt buộc trỏ tới đúng YOLO-Drone checkpoint dùng trong pipeline. File hash được lưu trong config.
- `FROZEN_POSITIVE_DATASET_DIR` trỏ tới chính `dataset_finetune` đã train identity-v1. Notebook hash toàn bộ cây JPG anchors/positives và sao chép byte-for-byte vào output; annotation plan chỉ dùng để phát hiện chênh lệch và lọc negatives.
- Confidence mining giảm từ `0.20` xuống `0.05`, khớp candidate threshold NB06; vẫn chỉ giữ tối đa 2 hard negatives theo quota cũ.
- Negative chỉ hợp lệ khi không có giao với GT đã nới thêm `max(4 px, 15% kích thước GT)` mỗi chiều. Manifest vẫn ghi IoU và target coverage để audit.
- Random backgrounds lấy kích thước từ phân bố target của đúng video, jitter `0.75–1.50`, thay vì luôn chọn ngẫu nhiên `50–200 px`. Hai background cùng frame bị chống trùng ở IoU trên `0.50`.
- Giữ `SAMPLING_RATE=5`, `MAX_NEG_PER_POS=2`, `NUM_BG_PER_FRAME=2`, positive crop pixels, anchor copy và filename provenance mà NB05 đang đọc.
- Seed Python/NumPy; pin Ultralytics `8.3.221`; lỗi inference mặc định làm run thất bại có artifact thay vì bị bỏ qua.
- Output mới, không ghi đè nếu chưa đặt `ALLOW_OVERWRITE=True` rõ ràng.

## Hai bước chạy trên Kaggle

1. Sửa `DATASET_ROOT`, `MINING_MODEL_PATH`, `FROZEN_POSITIVE_DATASET_DIR`, `OUTPUT_DIR` và `DIAGNOSTICS_FILE`. Dùng đúng YOLO checkpoint và đúng dataset matching đã train identity-v1.
2. Giữ `RUN_MINING=False`, `REUSE_FROZEN_POSITIVES=True`, chạy tất cả cell. Gửi `mining_v2_diagnostics.json`; cần kiểm tra video → identity, frame count, annotation plan, frozen positive/anchor counts và `frozen_positive_tree_sha256`.
3. Khi diagnostics đúng, chọn `OUTPUT_DIR` mới và đặt `RUN_MINING=True`. Không cần đặt `ALLOW_OVERWRITE=True` nếu dùng thư mục mới.

Notebook tạo:

- `mining_config.json`: checkpoint hash, versions và toàn bộ policy.
- `mining_summary.json`: count theo video, detector candidates, quota hard chưa lấp đầy, rejection reasons và integrity result.
- `negative_manifest.jsonl`: provenance, bbox, detector confidence, IoU/GT coverage của từng negative.
- `mining_failed.json` nếu run lỗi sau khi output đã được tạo.

Integrity cuối run đối chiếu số file với counters/positive plan, kiểm tra mọi manifest path tồn tại và duy nhất, filename tương thích NB05, đồng thời yêu cầu mọi negative có IoU và GT coverage bằng 0.

## Cần gửi lại trước khi train

- `mining_v2_diagnostics.json` sau dry run.
- `mining_summary.json` sau mining.
- `mining_config.json` để xác nhận detector SHA/version.

Chưa train ngay nếu `inference_errors > 0`, integrity không `passed`, positive count lệch diagnostics hoặc `hard_quota_unfilled` quá lớn. Không tự nới exclusion margin để lấy đủ quota; cần xem detector candidate/rejection distribution trước.

## Protocol sau khi dataset đạt kiểm tra

1. Trong NB05 identity-v1, chỉ đổi `ROOT_FOLDER` sang dataset mining-v2 và dùng `SAVE_DIR` mới.
2. Giữ split, seeds, batch 128, LR `1e-4`, 15 epochs, margin `0.5` và sampling policy hiện tại.
3. Đánh giá checkpoint mới trong NB06 trước tiên bằng production config đã khóa `0.425/0.475/0.100 @ 0.540` để cô lập ảnh hưởng mining.
4. Chỉ calibration lại nếu checkpoint mới có tiềm năng; cache phải dùng thư mục mới vì Siamese SHA thay đổi.

## Giới hạn

Policy tin vào GT dense theo frame; người dùng đã xác nhận annotation phủ mọi frame target xuất hiện. Nó không phát hiện lỗi annotation bằng semantic model và chưa kiểm tra crop trùng byte. Strict expanded-GT exclusion có thể giảm lượng hard negatives gần target; summary quota/rejections được thêm để quyết định dựa trên số liệu.

## Kiểm chứng local

11 tests CPU thành công: backup/hash, config được giữ, detector task-specific, frozen positives được copy byte-for-byte, không silent exception, filename provenance, large-box false-negative case, exclusion margin, deterministic target-sized backgrounds, output overwrite guard và manifest/file/overlap integrity. Notebook parse thành công. Chưa chạy video, YOLO, GPU hoặc sinh dataset thật trong workspace.

Diagnostics đầu tiên của code trước frozen-copy cho thấy annotation plan có 4.026 positives, lệch 51 files so với dataset identity-v1 có 3.975. Mining đã được chặn trước khi chạy; notebook hiện tại sửa bằng frozen copy để giữ ablation chỉ thay negatives. Ghi nhận: [`audit/MINING_V2_DIAGNOSTICS_1.json`](audit/MINING_V2_DIAGNOSTICS_1.json).

Diagnostics schema 3 sau sửa đã được duyệt: 3.975 frozen positives, 42 frozen anchors, tree hash `693e9b6a...f21fc8`, detector hash `eb4b471a...274baa3`. Có thể bật mining với output directory mới; ghi nhận: [`audit/MINING_V2_DIAGNOSTICS_2_APPROVED.json`](audit/MINING_V2_DIAGNOSTICS_2_APPROVED.json).

Mining thật đã hoàn tất và integrity passed: 2.833 hard + 32.302 background = 35.135 negatives, 0 inference errors, 35.135 manifest paths duy nhất và mọi overlap metric bằng 0. Hard pool giảm 761 files (`-21,17%`) so với dataset cũ; 4.648/7.826 detector candidates bị loại vì giao expanded GT. Quota chỉ là trần nên tỷ lệ lấp `8,76%` không được xem là lỗi. Dataset được duyệt để chạy NB05 diagnostics, chưa train trước khi split/provenance audit của NB05 pass. Ghi nhận: [`audit/MINING_V2_RESULT_APPROVED.json`](audit/MINING_V2_RESULT_APPROVED.json).

NB05 diagnostics đã pass: train 3.093 samples với pool 25.599; validation 882 fixed triplets với pool 9.536; không identity/path overlap, không same-identity negative và không loại video/source. Training `identity_mining_v2` được duyệt với toàn bộ hyperparameters hiện tại giữ nguyên. Ghi nhận: [`audit/MINING_V2_NB05_DIAGNOSTICS_APPROVED.json`](audit/MINING_V2_NB05_DIAGNOSTICS_APPROVED.json).

Training `identity_mining_v2` đã hoàn tất đủ 15 epochs. Checkpoint tốt nhất được lưu đúng tại epoch 2 (`val_loss=0.246860`), SHA-256 tải về khớp manifest: `e8619e85...5d34e2d`. Từ epoch 3 trở đi train loss tiếp tục giảm trong khi validation xấu đi, nên không dùng checkpoint cuối. Run được duyệt để đánh giá NB06 có kiểm soát với production fusion giữ nguyên `0.425/0.475/0.100 @ 0.540`; chưa promote trước khi có ST-IoU. Ghi nhận: [`audit/SIAMESE_IDENTITY_MINING_V2_TRAINING_RESULT.json`](audit/SIAMESE_IDENTITY_MINING_V2_TRAINING_RESULT.json).

Đánh giá khóa trong NB06 đạt `0.699451`/8.588 frames, thấp hơn identity-v1 calibrated `0.722052`/9.295 frames. Delta `-0.022601` chủ yếu do CardboardBox_0 giảm `-0.2170`, trong khi BlackBox_1 và LifeJacket_1 tăng lần lượt `+0.0447` và `+0.0454`. Chưa loại checkpoint: chạy calibration riêng vì embedding mới có thể làm lệch score/threshold. Ghi nhận: [`audit/SIAMESE_IDENTITY_MINING_V2_LOCKED_EVAL_RESULT.json`](audit/SIAMESE_IDENTITY_MINING_V2_LOCKED_EVAL_RESULT.json).

Coarse calibration 1.820 configs phục hồi lên `0.717301`/9.246 frames tại weights `0.55/0.40/0.05`, threshold `0.58`. Kết quả cao hơn baseline checkpoint calibrated `0.707417`, nhưng vẫn thấp hơn identity-v1 `0.722052` khoảng `0.004751`. Có hai nghiệm trong `0.001` của best và tám nghiệm trong `0.002`; vùng top nằm tại threshold `0.55–0.60`, Siamese `0.30–0.50`, color `0–0.10`. Duyệt một fine search cuối trên cache, chưa promote. Ghi nhận: [`audit/SIAMESE_IDENTITY_MINING_V2_COARSE_CALIBRATION_RESULT.json`](audit/SIAMESE_IDENTITY_MINING_V2_COARSE_CALIBRATION_RESULT.json).

Fine calibration 6.105 configs đạt `0.719278`/9.274 frames tại weights `0.50/0.425/0.075`, threshold `0.555`. Optimum nằm trong interior của toàn bộ search dimensions, tăng `0.001978` so với coarse nhưng vẫn thấp hơn identity-v1 `0.002773`. Dừng calibration mining-v2 và không promote checkpoint; identity-v1 `0.722052` tiếp tục là production. Ghi nhận: [`audit/SIAMESE_IDENTITY_MINING_V2_FINE_CALIBRATION_RESULT.json`](audit/SIAMESE_IDENTITY_MINING_V2_FINE_CALIBRATION_RESULT.json).
