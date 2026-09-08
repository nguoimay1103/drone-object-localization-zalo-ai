# Phần 3 — Temporal gating v1

Notebook Kaggle: [`06-inference-main.ipynb`](../06-inference-main.ipynb). Thí nghiệm dùng checkpoint identity-v1 production và giữ nguyên fusion `0.425/0.475/0.100`, matching threshold `0.540`, TTA, multi-scale reference và candidate ranking. Control cần tái lập **0.722052** trước khi đọc kết quả temporal.

## Mục tiêu

Temporal legacy nội suy mọi gap không quá 5 frame mà không kiểm tra hai bbox đầu cuối có thể thuộc cùng trajectory hay không. Temporal gating v1 chỉ nội suy khi chuyển động đầu cuối hợp lý theo hai đại lượng:

```text
center_speed = center_distance / (frame_steps * mean(sqrt(bbox_area)))
log_scale_speed = abs(log(scale2 / scale1)) / frame_steps
```

Chuẩn hóa theo kích thước bbox giúp một threshold dùng được cho nhiều scale. Đây chỉ là gate cho gap interpolation; candidate selection trên từng frame không đổi. Khi cả hai gate là `None`, hàm experimental được kiểm tra cho output giống temporal legacy chính xác.

## Search space

- `max_gap`: `2, 3, 4, 5, 6, 8, 10`
- `min_seg_len`: `1, 2, 3, 4, 5`
- `max_center_speed`: `None, 0.5, 1, 2, 4, 8`
- `max_log_scale_speed`: `None, 0.15, 0.30, 0.60, 1.20`

Tổng cộng 1.050 configs. Control legacy `{max_gap: 5, min_seg_len: 3, max_center_speed: None, max_log_scale_speed: None}` nằm trong grid. Exact metric ties ưu tiên cấu hình gần legacy nhất.

## Cách chạy

1. `SIAMESE_MODEL_PATH` phải trỏ tới identity-v1 checkpoint SHA-256 `4a1f438d1920e60129ceb89a153c047c27ce904214fc65e9a51f6d458b52839d`, không phải mining-v2.
2. Dùng candidate cache identity-v1 qua `CALIBRATION_DIR` hoặc `PRECOMPUTED_CANDIDATE_CACHE_FILE`. Cache mining-v2 sẽ bị signature guard từ chối.
3. Để tái chạy grid, giữ `RUN_CALIBRATION=False` và đặt `RUN_TEMPORAL_CALIBRATION=True`. Production đã khóa cả hai flag ở `False`.
4. Log `CACHE REPLAY - LEGACY TEMPORAL` phải đạt lại khoảng `0.722052`/9.295 frames trước khi chấp nhận grid result.

Artifacts:

- `temporal_gating_v1_results.csv`
- `temporal_gating_v1_summary.json`
- `predictions_temporal_gating_v1.json`

## Tiêu chí quyết định

Không promote chỉ vì một config thắng rất nhỏ trên cùng sáu video. Cần xem delta từng video, vùng top-k, config có nằm ở biên và gate có thật sự được bật. Nếu best chính là legacy hoặc delta không đủ ổn định, giữ production hiện tại và chuyển sang top-k candidate linking/tubelet.

## Kết quả coarse và fine follow-up

Coarse grid tái lập legacy `0.722052` và đạt `0.726446` tại `max_gap=3`, `min_seg_len=5`, center gate `0.5`, không scale gate. Delta là `+0.004395`, nhưng optimum chạm biên `min_seg` lớn nhất và center gate hữu hạn nhỏ nhất. Ablation cho thấy `gap=3/min_seg=5` không gate đã đạt `0.725585`; center gate trên temporal legacy đạt `0.723699`. Vì vậy chưa promote và chạy fine grid mở min segment tới 12, center gate xuống 0.1; log-scale được cố định `None`. Ghi nhận: [`audit/TEMPORAL_GATING_V1_COARSE_RESULT.json`](audit/TEMPORAL_GATING_V1_COARSE_RESULT.json).

Fine grid 352 configs đạt `0.735621`/8.850 frames tại gap 4/min segment 12/center gate 0.5. Delta so legacy là `+0.013569`. Center gate 0.5–2.0 tạo cùng optimum, nhưng mean tiếp tục tăng tại min-segment boundary 12. Chưa promote; chạy sparse segment sweep tới 50 frames để tìm điểm quay đầu. Ghi nhận: [`audit/TEMPORAL_GATING_V1_FINE_RESULT.json`](audit/TEMPORAL_GATING_V1_FINE_RESULT.json).

Segment sweep 204 configs đạt `0.737659`/8.744 frames tại gap 5/min segment 25/center gate 0.5. Min-segment đã được bracket: score giảm tại 30 và giảm mạnh từ 35. Tuy nhiên gap 5 và center gate 0.5 vẫn chạm biên, nên chạy một joint refinement cuối quanh min segment 22–30 với gap mở tới 20 và center gate tới 1.0. Ghi nhận: [`audit/TEMPORAL_GATING_V1_SEGMENT_SWEEP_RESULT.json`](audit/TEMPORAL_GATING_V1_SEGMENT_SWEEP_RESULT.json).

Joint refinement cuối gồm 630 configs đạt **`0.740013`/8.890 frames** tại gap 7/min segment 24/center gate 0.4, tăng `0.017961` so với legacy. Gap 6 và 8 đều thấp hơn; min segment 22 và 25 đều thấp hơn. Bốn trong sáu video tăng, CardboardBox_0 và CardboardBox_1 giảm lần lượt khoảng `0.00729` và `0.00391`. Cấu hình được promote thành offline production và dừng tuning temporal trên cùng tập sáu video. Artifact Kaggle vẫn mang tên `segment_sweep`, nhưng search space trong summary chính xác là joint-refinement grid; đây là sai khác tên file, không phải sai khác run. Ghi nhận cuối: [`audit/TEMPORAL_GATING_V1_FINAL_RESULT.json`](audit/TEMPORAL_GATING_V1_FINAL_RESULT.json).

## Kiểm chứng local

Legacy temporal function được giữ nguyên. Test kiểm tra disabled-gate parity, spatial jump lớn bị chặn, scale jump lớn bị chặn, normalization theo thời gian/kích thước, joint grid có đúng 630 cấu hình duy nhất, và production truyền explicit gated config trong khi calibration replay vẫn dùng legacy control.
