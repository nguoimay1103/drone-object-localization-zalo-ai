# Phần 8A — Detector Profile V1

Notebook Kaggle: [`06-inference-main.ipynb`](../06-inference-main.ipynb). Đây là diagnostic read-only dùng candidate cache identity-v1 và manual GT. Nó không đổi YOLO call, candidate, fusion, threshold, temporal hoặc `predictions.json`.

## Mục tiêu

Profile này được chạy trước Resolution/SAHI/P2 A/B để trả lời:

1. Checkpoint thật sự xuất detection tại stride nào; có P2 stride 4 hoặc P3 stride 8 hay không.
2. Target nhỏ đến mức nào sau khi quy đổi theo isotropic letterbox scale về reference `640`.
3. Candidate oracle recall tại IoU `0.3/0.5/0.7` giảm ở nhóm kích thước nào.
4. `no_candidate` và localization miss phân bố theo video, identity và target scale ra sao.

Runner tải checkpoint YOLO chỉ để đọc kiến trúc, `model.stride`, head type và metadata. Candidate vẫn lấy nguyên từ cache đã khóa.

## Định nghĩa kích thước

Primary size measure là cạnh ngắn nhất của GT bbox sau khi nhân với:

```python
scale = min(640 / frame_width, 640 / frame_height)
projected_min_side = min(bbox_width, bbox_height) * scale
```

Các bin:

| Bin | Projected minimum side |
|---|---:|
| `tiny_lt_8` | `< 8 px` |
| `very_small_8_16` | `[8, 16) px` |
| `small_16_32` | `[16, 32) px` |
| `medium_32_64` | `[32, 64) px` |
| `large_ge_64` | `>= 64 px` |

Reference `640` chỉ phục vụ profiling. Production hiện không truyền `imgsz` vào `yolo.predict`; summary ghi rõ điều này để không nhầm reference projection với runtime proof.

## Cách chạy

```python
RUN_CALIBRATION = False
RUN_TEMPORAL_CALIBRATION = False
RUN_RERANKING_EXPERIMENT = False
RUN_ORACLE_DIAGNOSTICS = False
RUN_HYSTERESIS_EXPERIMENT = False
RUN_MARGIN_DIAGNOSTICS = False
RUN_DETECTOR_PROFILE = False  # chỉ bật lại để tái lập profile
```

Giữ checkpoint/cache identity-v1 đã dùng cho production `0.740013`. Runner xuất:

- `detector_profile_v1_frames.csv`: một dòng cho mỗi GT frame, bbox scale, candidate count, oracle IoU và YOLO confidence.
- `detector_profile_v1_videos.csv`: video resolution/FPS/frame count, candidate density và recall.
- `detector_profile_v1_summary.json`: architecture, stride, global/by-size/by-identity aggregates và parity guards.

Run chỉ hợp lệ khi ba guard đều true:

- `production_predictions_unchanged`
- `production_per_video_metric_exact_parity`
- `gt_frame_coverage_exact`

## Cách dùng kết quả

- Nếu miss tập trung ở `tiny_lt_8`/`very_small_8_16` và checkpoint không có stride 4: ưu tiên P2 hoặc SAHI.
- Nếu recall tăng theo projected size nhưng nhiều target vẫn dưới 16 px: thử explicit `imgsz=960`, sau đó `1280`.
- Nếu `no_candidate` thấp nhưng IoU `0.5/0.7` thấp: ưu tiên localization, tile overlap/merge hoặc detector retraining thay vì threshold recovery.
- Không chọn cấu hình riêng theo tên object. Profile này không tự promote bất kỳ kỹ thuật nào.

## Kết quả

Run identity-v1 giữ exact production parity `0.7400130889`/8.890 frames và đủ 9.505 GT frames. Checkpoint có detection strides `[8, 16, 32]`, tức có P3 nhưng không có P2 stride 4.

- `48,45%` GT frames có projected minimum side dưới 16 px; `78,50%` dưới 32 px.
- Global oracle candidate recall đạt `0,9579/0,9481/0,8898` tại IoU `0,3/0,5/0,7`.
- Nhóm `8–16 px` chứa 4.439 GT frames và 370/493 lỗi recall@0.5.
- LifeJacket chứa 332/493 lỗi recall@0.5; toàn bộ 332 lỗi nằm ở `LifeJacket_0`.
- CardboardBox_0 có recall@0.3 `0,9898` nhưng recall@0.5 `0,9221`, cho thấy localization là vấn đề chính.
- LifeJacket_0 có recall@0.3 và @0.5 cùng `0,8818`, cho thấy phần lớn lỗi là miss/chọn vùng sai thay vì IoU chỉ hơi thấp.

**Quyết định:** profile được tắt sau khi hoàn tất. Bước tiếp theo là explicit-resolution candidate A/B với control 640 và variant 960; chỉ thử 1280 hoặc hybrid SAHI sau khi 960 chứng minh candidate recall tăng đủ và không gây suy giảm xuyên identity. P2 được giữ làm nhánh retraining nếu inference-scale experiment không giải quyết nhóm dưới 16 px.
