# Phần 4 — Top-K Temporal Candidate Reranking V1

Notebook Kaggle: [`06-inference-main.ipynb`](../06-inference-main.ipynb). Nhánh này dùng candidate cache identity-v1, fusion `0.425/0.475/0.100`, threshold `0.540` và Temporal Gating production `{gap: 7, min_seg: 24, center_speed: 0.4, log_scale_speed: None}`. Control bắt buộc phải tái lập offline ST-IoU `0.740013`/8.890 frames.

## Thay đổi thuật toán

Pipeline production hiện tại chọn top-1 fusion score độc lập ở mỗi frame. Reranking giữ top-K candidate rồi cộng một lượng temporal support chỉ để thay đổi thứ tự candidate trong cùng frame:

```text
center_speed = center_distance / (frame_steps * mean(sqrt(bbox_area)))
motion_affinity = exp(-0.5 * (center_speed / motion_scale)^2)
directional_support = max(motion_affinity * neighbor_original_score)
temporal_support = (past_support + future_support) / 2
contextual_score = original_score + rerank_weight * temporal_support
```

Candidate thắng vẫn lưu `score=original_score`; threshold `0.540` không đọc contextual score. Vì vậy temporal context không thể tự đẩy một candidate dưới threshold thành detection. Với `rerank_weight=0`, strict tie-break và candidate order được kiểm tra cho output top-1 giống legacy.

Scoring không đọc `video_id`, identity hay object class. Hàm tách identity chỉ được dùng để tạo evaluation folds.

## Search space đã khóa trước kết quả

```text
top_k:             [3, 5]
neighbor_radius:   [1, 3, 5]
rerank_weight:     [0.00, 0.03, 0.06, 0.10]
motion_scale:      0.4 (fixed)
config count:      24
```

Grid nhỏ có exact control `rerank_weight=0`. Không thêm grid sau khi xem metric từng object trong cùng run.

## Leave-one-identity-out

`BlackBox_0/1`, `CardboardBox_0/1` và `LifeJacket_0/1` lần lượt là ba identity groups. Mỗi fold giữ cả hai video của một identity để đánh giá và chỉ chọn config bằng bốn video của hai identity còn lại. Điều này tránh để cùng vật thể xuất hiện ở cả selection và held-out evaluation.

Config recommendation được lấy bằng phiếu của ba fold winners. Nếu số phiếu bằng nhau, ưu tiên rerank weight thấp hơn, rồi top-K và radius nhỏ hơn. Full-public result của recommendation chỉ là thống kê mô tả, không tham gia fold selection.

## Điều kiện promote đã khóa

Chỉ cân nhắc promote khi đồng thời:

1. Mean identity-held-out delta ít nhất `+0.003`.
2. Tối thiểu `2/3` held-out identities tăng.
3. Worst held-out identity delta không thấp hơn `-0.010`.
4. Recommendation có `rerank_weight > 0`.

Notebook luôn ghi `status=experiment_only_not_auto_promoted`; kể cả khi criteria pass vẫn cần audit kết quả trước khi đổi production flag.

## Cách chạy

1. Dùng đúng Siamese identity-v1 SHA-256 `4a1f438d1920e60129ceb89a153c047c27ce904214fc65e9a51f6d458b52839d` và candidate cache tương ứng.
2. Giữ `RUN_CALIBRATION=False`, `RUN_TEMPORAL_CALIBRATION=False`, `RUN_RERANKING_EXPERIMENT=True` và `USE_TEMPORAL_RERANKING=False`.
3. Dòng `PRODUCTION CONFIG` phải đạt khoảng `0.740013` trước khi đọc CV result.
4. Gửi lại summary, fold CSV và log cuối để audit.

Artifacts:

- `temporal_reranking_v1_identity_cv_results.csv`
- `temporal_reranking_v1_identity_cv_folds.csv`
- `temporal_reranking_v1_identity_cv_summary.json`
- `predictions_temporal_reranking_v1_identity_cv.json`

Chỉ có ba identity nên leave-one-identity-out vẫn có phương sai lớn và chưa thay thế independent test set. Mục tiêu của protocol là giảm leakage và ngăn rule riêng theo object, không biến public GT thành đánh giá hoàn toàn độc lập.

## Kết quả và quyết định

Grid 24 configs tái lập exact control `0.740013` cho cả sáu cấu hình weight-zero. Cả ba fold chọn `{top_k: 3, neighbor_radius: 1, rerank_weight: 0, motion_scale: 0.4}`; mean CV delta `0`, improved identities `0/3`, worst delta `0`. Promotion criteria không đạt.

Trong 18 cấu hình bật reranking, BlackBox_0/1, CardboardBox_0/1 và LifeJacket_0 không đổi. Chỉ LifeJacket_1 thay đổi và giảm khoảng `0.00348–0.00389`; mean giảm `0.00058–0.00065`, số frame tăng 7–8. Top-K và radius hầu như không ảnh hưởng: toàn grid chỉ tạo bốn metric outputs khác nhau. Kết luận: local motion reranking không xử lý bottleneck hiện tại và bị reject. `RUN_RERANKING_EXPERIMENT=False`, `USE_TEMPORAL_RERANKING=False`; production giữ Temporal Gating V1 `0.740013`.

Ghi nhận máy đọc: [`audit/TEMPORAL_RERANKING_V1_RESULT.json`](audit/TEMPORAL_RERANKING_V1_RESULT.json).
