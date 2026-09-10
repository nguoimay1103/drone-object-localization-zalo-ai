# Phần 8D — SAHI Source Admission V1

Notebook Kaggle: [`06-inference-main.ipynb`](../06-inference-main.ipynb). Experiment này chỉ replay legacy và SAHI tile caches; không chạy detector nếu `PRECOMPUTED_SAHI_TILE_CACHE_FILE` trỏ tới tile cache hoàn chỉnh.

## Mục tiêu

Equal-source hybrid tăng candidate oracle nhưng cho tile cạnh tranh trên quá nhiều frame, làm LifeJacket_1 giảm mạnh. V1 admission kiểm tra hai policy cố định trước khi thêm penalty hoặc threshold mới.

## Policies

### `legacy_only`

Production control, không thêm tile candidate.

### `missing_legacy_only`

Nếu frame có ít nhất một legacy candidate, chỉ giữ legacy. Nếu không có legacy candidate, dùng tile candidates. Policy này không thể thay top-1 legacy đã tồn tại.

### `weak_legacy_rescue`

Tính best legacy fused score bằng production weights. Nếu score `>=0.54`, khóa legacy; nếu `<0.54` hoặc không có legacy candidate, thêm tile candidates để cạnh tranh. `0.54` là production threshold đã khóa, không được tune lại trong experiment này.

Mọi policy giữ nguyên candidate legacy. Oracle IoU trên từng GT frame phải monotonic; bất kỳ regression nào làm runner dừng.

## Identity-held-out selection

Với mỗi held-out identity:

1. Tính mean ST-IoU của ba policy trên bốn video thuộc hai identity còn lại.
2. Chọn policy tốt nhất; khi hòa ưu tiên `legacy_only`, rồi `missing_legacy_only`, rồi `weak_legacy_rescue`.
3. Áp policy đã chọn lên hai video held-out.

Promotion chỉ pass khi:

```text
mean held-out delta >= +0.003
held-out identities cải thiện >= 2/3
worst held-out delta >= -0.010
recommended full-public policy là dominant policy ở >=2/3 folds
full-public delta/identity gates cũng pass
oracle monotonic cho mọi policy
```

Runner không auto-promote.

## Cache paths

```python
PRECOMPUTED_CANDIDATE_CACHE_FILE = '/kaggle/input/.../candidate_scores.json.gz'
PRECOMPUTED_SAHI_TILE_CACHE_FILE = '/kaggle/input/.../tile_candidate_scores_size640_overlap025.json.gz'
```

Ưu tiên upload file cuối cùng `.json.gz`. Loader nhận dạng theo nội dung thay vì phần mở rộng: hỗ trợ cả Gzip JSON có magic bytes `1f 8b` và plain UTF-8 JSON đã bị giải nén nhưng vẫn mang đuôi `.json.gz`/`.tmp`. `.tmp` vẫn có thể là dấu hiệu notebook trước bị dừng giữa atomic write, nên loader tiếp tục kiểm tra JSON, cờ `complete` và cache signature trước khi replay.

Nếu đường dẫn được sao chép dưới dạng `/kaggle/input/datasets/<owner>/<slug>/...`, loader thử cả mounted path `/kaggle/input/<slug>/...` và tìm filename duy nhất dưới `/kaggle/input`. Đường dẫn đã cấu hình nhưng không tồn tại sẽ dừng sớm thay vì âm thầm chạy lại detector.

Nếu tile path để trống, notebook dùng local tile cache trong `hybrid_sahi_candidate_v1/`; nếu file chưa tồn tại, tile extraction sẽ chạy lại.

## Cách chạy

```python
RUN_CALIBRATION = False
RUN_TEMPORAL_CALIBRATION = False
RUN_RERANKING_EXPERIMENT = False
RUN_ORACLE_DIAGNOSTICS = False
RUN_HYSTERESIS_EXPERIMENT = False
RUN_MARGIN_DIAGNOSTICS = False
RUN_DETECTOR_PROFILE = False
RUN_RESOLUTION_AB = False
RUN_HYBRID_SAHI = False
RUN_SAHI_ADMISSION = True
```

Artifacts:

- `sahi_source_admission_v1_results.csv`
- `sahi_source_admission_v1_videos.csv`
- `sahi_source_admission_v1_folds.csv`
- `sahi_source_admission_v1_summary.json`
- `predictions_<policy>.json`

Gửi lại ba CSV, file JSON summary và log `SAHI SOURCE ADMISSION V1`. Nếu cả hai policy không transfer, giữ production legacy và chuyển sang P2 training; không mở grid tile penalty trên public.

## Kết quả và quyết định

- `missing_legacy_only`: ST-IoU giữ nguyên `0,740013`; 3.776 tile candidates trên 2.658 frames chỉ tạo 14 accepted tile frames và không sống qua temporal output.
- `weak_legacy_rescue`: full-public `0,741842` (`+0,001828`), nhưng identity-held-out CV delta `-0,009150`, chỉ `1/3` held-out identity tăng và worst delta `-0,028640`.
- CardboardBox tăng `+0,032936`, còn LifeJacket giảm `-0,028640`; riêng `LifeJacket_1` giảm `-0,046713` dù candidate oracle gần như đã bão hòa.

**Quyết định:** không promote cả hai policy. `RUN_SAHI_ADMISSION=False`; production tiếp tục dùng legacy candidate pipeline `0,740013`. Dừng tune tile admission trên public và chuyển sang detector P2 A/B với stride 4.
