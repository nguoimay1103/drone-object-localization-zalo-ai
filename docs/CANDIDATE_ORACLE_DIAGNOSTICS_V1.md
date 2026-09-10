# Phần 5 — Candidate-Cache Oracle Error Decomposition V1

Notebook Kaggle: [`06-inference-main.ipynb`](../06-inference-main.ipynb). Đây là diagnostic read-only trên candidate cache và GT gán tay. Nó chạy sau khi production prediction đã được tạo bằng identity-v1 fusion `0.425/0.475/0.100`, threshold `0.540` và Temporal Gating `{gap: 7, min_seg: 24, center_speed: 0.4, log_scale_speed: None}`. Diagnostic không thay đổi prediction JSON.

## Mục tiêu

Xác định điểm mất chất lượng đầu tiên trong chuỗi:

```text
raw detector candidates
  → query-conditioned matching
  → threshold acceptance
  → temporal filtering/interpolation
  → final prediction
```

Trước khi đọc diagnostic, dòng `PRODUCTION CONFIG` phải tái lập offline ST-IoU `0.740013`/8.890 frames.

## First-failure hierarchy

Mỗi GT frame được gán đúng một category theo thứ tự cố định:

1. `no_candidate`: cache không có candidate ở frame đó.
2. `detector_localization_miss`: có candidate nhưng best IoU dưới 0.5.
3. `matching_error`: có candidate IoU ít nhất 0.5 nhưng fusion top-1 dưới 0.5.
4. `threshold_rejection`: fusion top-1 có IoU ít nhất 0.5 nhưng score dưới acceptance threshold.
5. `temporal_removal`: candidate tốt đã accepted nhưng final temporal không giữ frame.
6. `temporal_localization_degradation`: final frame tồn tại nhưng IoU giảm xuống dưới 0.5.
7. `success`: final bbox có IoU ít nhất 0.5.

Các category không chồng lấp và tổng count bằng số GT frames. Threshold IoU 0.5 chỉ dùng để phân loại diagnostic; runner đồng thời báo recall tại IoU 0.3/0.5/0.7 để tránh kết luận dựa trên một ngưỡng duy nhất.

## Các upper bounds và stage metrics

- `oracle_st_iou_upper_bound`: ở mỗi GT frame chọn raw candidate có IoU cao nhất và không phát prediction ngoài GT. Đây là optimistic upper bound của candidate cache hiện tại, không phải thuật toán deploy được.
- `selected_prethreshold_st_iou`: fusion top-1 trước threshold và temporal.
- `accepted_pretemporal_st_iou`: fusion top-1 sau threshold, trước temporal.
- `final_production_st_iou`: output production hiện tại.
- `mean_matching_regret`: mean `max(0, oracle_iou - selected_iou)` trên GT frames.
- Temporal helped/harmed counts: so sánh IoU final với accepted trên từng GT frame.
- Accepted/final background frames: số output trên frame không có GT.

Các metric được xuất theo video, identity và toàn bộ benchmark. Frame CSV giữ dữ liệu chi tiết để drill-down nhưng không được dùng để viết rule riêng theo `video_id`.

## Cách chạy

```python
RUN_CALIBRATION = False
RUN_TEMPORAL_CALIBRATION = False
RUN_RERANKING_EXPERIMENT = False
RUN_ORACLE_DIAGNOSTICS = True
USE_TEMPORAL_RERANKING = False
```

Dùng đúng identity-v1 checkpoint SHA-256 `4a1f438d1920e60129ceb89a153c047c27ce904214fc65e9a51f6d458b52839d` và candidate cache có signature tương ứng.

Artifacts:

- `candidate_oracle_diagnostics_v1_frames.csv`
- `candidate_oracle_diagnostics_v1_videos.csv`
- `candidate_oracle_diagnostics_v1_summary.json`

## Cách ra quyết định

| Dấu hiệu chính | Bước thử tiếp theo |
|---|---|
| Oracle upper bound thấp hoặc `no_candidate`/`detector_localization_miss` lớn | Detector resolution, tiled/SAHI inference hoặc small-object detector training |
| Oracle cao nhưng matching regret và `matching_error` lớn | Query matching/Re-ID hoặc sequence association có appearance state |
| Accepted cao nhưng `temporal_removal` lớn | Segment confidence và adaptive temporal filtering chung cho mọi video |
| Final background frames lớn hơn accepted background frames | Hạn chế interpolation/track continuation vào target-absent regions |

Không chọn thay đổi theo video có metric thấp nhất. Chọn tầng pipeline dựa trên pattern toàn identity và chỉ dùng per-video table để giải thích variance.

GT public vẫn được dùng để diagnostic, vì vậy kết quả không trở thành independent test estimate. Kiểm soát overfit ở đây đến từ việc không tune hay thay prediction trong cùng run, dùng rule chung và khóa metric definitions trước khi xem output.

## Kết quả

Diagnostic tái lập exact production `0.740013` và xác nhận prediction JSON không đổi. Trên 9.505 GT frames, first-failure counts là: 118 `no_candidate`, 375 `detector_localization_miss`, 125 `matching_error`, 822 `threshold_rejection`, 106 `temporal_removal`, 2 `temporal_localization_degradation` và 7.957 `success`.

Mean-video candidate oracle upper bound là `0.843613`; accepted pre-temporal là `0.694876`, final production là `0.740013`. Temporal giúp giảm background frames từ 543 accepted xuống 409 final. CardboardBox chiếm 497 threshold rejections; riêng CardboardBox_0 có oracle/selected/accepted/final recall@IoU0.5 lần lượt `0.9221/0.8891/0.5555/0.7057`. Vì vậy không ưu tiên detector hoặc top-K reranking ở bước kế tiếp. Hướng thử tiếp là threshold hysteresis có strong-anchor temporal support, không hạ threshold toàn cục và không dùng object name.

Ghi nhận máy đọc: [`audit/CANDIDATE_ORACLE_DIAGNOSTICS_V1_RESULT.json`](audit/CANDIDATE_ORACLE_DIAGNOSTICS_V1_RESULT.json).
