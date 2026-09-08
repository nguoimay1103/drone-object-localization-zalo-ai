# Phần 8B — Explicit Resolution Candidate A/B V1

Notebook Kaggle: [`06-inference-main.ipynb`](../06-inference-main.ipynb). Experiment này tạo candidate cache mới tại explicit `imgsz=640` và `imgsz=960`, sau đó chạy cùng Siamese, fusion, threshold và Temporal Gating production. Nó không tự thay production cache hoặc checkpoint.

## Giả thuyết

Detector Profile V1 cho thấy checkpoint chỉ có strides `[8,16,32]`; `48,45%` GT frames có cạnh ngắn projected dưới 16 px và bin `8–16 px` chứa 370/493 lỗi recall@0.5. Tăng full-frame resolution là phép thử ít thay đổi nhất trước SAHI hoặc retrain P2.

## Variants

| Variant | Candidate source | Vai trò |
|---|---|---|
| `legacy` | Cache production hiện tại, lời gọi YOLO không explicit `imgsz` | Production control |
| `explicit_640` | Chạy lại YOLO với `imgsz=640` | Xác nhận library default/control |
| `explicit_960` | Chạy lại YOLO với `imgsz=960` | Candidate-quality experiment |

Giữ cố định `conf=0.05`, TTA, checkpoint hashes, multi-scale reference, fusion `0.425/0.475/0.100`, threshold `0.54` và temporal `{gap:7, min_seg:24, center_speed:0.4}`. Không calibration lại trong vòng này.

## Cache và resume

Mỗi explicit variant có cache riêng:

```text
resolution_candidate_ab_v1/
├── candidate_scores_imgsz_640.json.gz
└── candidate_scores_imgsz_960.json.gz
```

Signature chứa `imgsz`, confidence, TTA, Ultralytics version và toàn bộ checkpoint/data hashes của base cache. Cache được atomic-save sau từng video; chạy lại notebook sẽ bỏ qua video đã hoàn tất. Không dùng chung cache giữa hai resolution.

## Control gate

`explicit_640` chỉ pass khi so với legacy trong tolerance `1e-6` cho:

- Mean final ST-IoU.
- Maximum absolute per-video ST-IoU delta.
- Tổng candidate count phải bằng nhau.
- Global oracle recall@0.5.
- Sub-16 oracle recall@0.5 trên cùng fixed cohort projection@640.

Nếu control fail, `explicit_960` không đủ điều kiện promotion. Khi đó cần xem version/default preprocessing trước khi diễn giải resolution gain.

## Promotion gate cho 960

Runner chỉ báo `passed`; không tự đổi production. Tất cả điều kiện phải thỏa:

```text
explicit-640 control passed
mean ST-IoU delta >= +0.003
improved identities >= 2/3
worst identity delta >= -0.010
sub-16 oracle recall@0.5 delta >= +0.002
```

Identity score là mean của `_0/_1`; không có cấu hình theo tên object/video.

## Runtime metrics

Mỗi video cache lưu processed frames, candidate count và extraction seconds. Summary báo:

- Aggregate FPS dựa trên tổng video processing time.
- Peak CUDA allocated memory của invocation.
- Total candidates và candidates trên GT frames.
- Candidate recall IoU `0.3/0.5/0.7`, sub-16 recall và final ST-IoU.

Legacy cache không có timing đồng nhất nên runtime fields có thể là `null`.

## Cách chạy

```python
RUN_CALIBRATION = False
RUN_TEMPORAL_CALIBRATION = False
RUN_RERANKING_EXPERIMENT = False
RUN_ORACLE_DIAGNOSTICS = False
RUN_HYSTERESIS_EXPERIMENT = False
RUN_MARGIN_DIAGNOSTICS = False
RUN_DETECTOR_PROFILE = False
RUN_RESOLUTION_AB = False  # chỉ bật lại để tái lập A/B
```

Nên trỏ `PRECOMPUTED_CANDIDATE_CACHE_FILE` tới cache identity-v1 production đã upload để tránh extract lại legacy. Hai explicit caches luôn được ghi trong Kaggle working output.

Artifacts:

- `resolution_candidate_ab_v1_results.csv`
- `resolution_candidate_ab_v1_videos.csv`
- `resolution_candidate_ab_v1_summary.json`
- `predictions_explicit_640.json`
- `predictions_explicit_960.json`

Sau run, chỉ thử 1280 nếu 960 tăng candidate recall xuyên identity nhưng chưa đủ final gain. Nếu 960 không giải quyết nhóm sub-16, chuyển sang hybrid full-frame+SAHI hoặc P2 training thay vì tiếp tục tăng resolution mù.

## Kết quả và quyết định

Explicit-640 tái lập legacy chính xác: ST-IoU `0,740013`, 42.237 candidates, global recall@0.5 `0,948133`, sub-16 recall `0,917264`; toàn bộ control gates pass. Runtime đo được `30,31 FPS`.

Explicit-960 không pass promotion:

- ST-IoU `0,686272`, delta `-0,053741`.
- `0/3` identity cải thiện; worst identity delta `-0,081381`.
- Global recall@0.5 giảm `0,948133 → 0,938348` và mean oracle IoU giảm `0,829718 → 0,815299`.
- No-candidate GT frames tăng `118 → 234`; tổng candidate giảm `42.237 → 37.281`.
- Sub-16 recall chỉ tăng `+0,001954`, thấp hơn gate; gain không transfer đồng đều: BlackBox giảm, CardboardBox tăng, LifeJacket gần như không đổi.
- FPS giảm nhẹ `30,31 → 29,46`; peak allocated CUDA memory tăng `81,15 → 121,96 MB`.

**Quyết định:** reject 960 và không chạy 1280. Oracle candidate quality đã giảm nên không dành thêm vòng calibration cho 960. Production giữ explicit-equivalent 640. Bước tiếp theo là hybrid legacy full-frame + native-scale SAHI tiles, bảo toàn candidate legacy và chỉ bổ sung tile candidates; P2 vẫn là nhánh retraining sau SAHI.
