# Motion Distribution Diagnostics V1

Notebook Kaggle: [`06-inference-main.ipynb`](../06-inference-main.ipynb).

## Mục tiêu

Candidate Tracklet Graph V1 giảm mạnh điểm vì hard motion transition loại nhiều strong production detections. Diagnostic này trả lời trước khi thiết kế V2:

1. Chuyển động thật giữa hai GT frame liên tiếp có thường vượt `0.4/0.6` hay không?
2. Gate ảnh hưởng thế nào theo kích thước target, identity và video?
3. Bao nhiêu cặp production top-1 có IoU tốt ở cả hai đầu nhưng vẫn bị gate từ chối?
4. Motion của top-1, oracle candidate và final production lệch GT bao nhiêu?

Đây là runner read-only. GT chỉ được đọc sau khi production prediction đã đóng băng; không tìm tham số mới và không sửa prediction.

## Định nghĩa

Chỉ xét cặp có `current_frame = previous_frame + 1`. Vì annotation dùng absolute frame index bắt đầu từ 0, một đoạn bắt đầu ở frame 36 sẽ tạo cặp đầu tiên `36 → 37`; diagnostic không đánh lại index từ 0.

```text
scale             = sqrt(bbox_width * bbox_height)
mean_scale        = (scale_previous + scale_current) / 2
center_speed      = center_displacement_pixels / mean_scale
log_scale_speed   = abs(log(scale_current / scale_previous))
```

Các stage:

- `gt`: bbox nhãn ở hai frame liên tiếp.
- `selected_top1`: candidate có fusion score cao nhất, chưa threshold.
- `oracle_candidate`: candidate có IoU GT cao nhất ở mỗi frame; chỉ dùng diagnostic.
- `production_final`: bbox sau threshold, temporal interpolation và segment removal.

Một candidate pair được gọi là `good_endpoint_pair` khi bbox ở **cả hai đầu** đều có IoU GT `>=0.5`. `good_pair_exceeds_gate_04/06` đo trực tiếp cặp đúng mà hard graph transition tương ứng sẽ loại.

Size bin tái sử dụng Detector Profile V1, dựa trên trung bình min-side GT của hai frame sau khi project giữ aspect ratio về 640 px:

```text
tiny_lt_8, very_small_8_16, small_16_32, medium_32_64, large_ge_64
```

## Bảo toàn production

Runner kiểm tra:

- ST-IoU từng video tính lại từ prediction phải khớp production;
- prediction JSON trước và sau diagnostic phải giống nhau;
- số dòng phải bằng chính xác số cặp GT frame liên tiếp;
- chỉ một experiment flag được bật.

Production vẫn dùng ST-IoU `0.740013`; Tracklet Graph V1 vẫn tắt.

## Cách chạy

Giữ candidate cache, YOLO và Siamese checkpoint đúng hash của production. Cấu hình:

```python
RUN_MOTION_PROFILE = True
RUN_TRACKLET_GRAPH_EXPERIMENT = False
USE_TRACKLET_GRAPH = False
```

Runner dùng candidate cache production, không chạy graph grid và không cần SAHI cache.

## Artifact cần gửi lại

Thư mục:

```text
/kaggle/working/siamese_identity_v1_calibration/
  motion_distribution_diagnostics_v1/
```

Các file:

- `motion_distribution_diagnostics_v1_summary.json`
- `motion_distribution_diagnostics_v1_frames.csv`
- `motion_distribution_diagnostics_v1_groups.csv`
- `motion_distribution_diagnostics_v1_videos.csv`

`frames.csv` dùng để truy vết từng transition. `groups.csv` chứa global/size/identity/video × bốn stage. `videos.csv` là view dài theo video × stage.

## Cách quyết định V2

- Nếu GT/top-1 good pairs thường vượt 0.4/0.6, bỏ hard adjacent-frame gate; không tune gate theo object.
- Nếu lỗi tập trung ở target dưới 16 px, dùng uncertainty-aware/soft motion theo size thay vì cùng một cutoff.
- Nếu oracle motion gần GT nhưng top-1 motion lệch lớn, ưu tiên candidate selection/appearance.
- Nếu cả oracle motion cũng nhiễu, candidate bbox jitter là giới hạn detector; Kalman trên image coordinates có thể không phù hợp.
- Association V2 chỉ nên triển khai sau khi đọc phân phối này. Thiết kế mặc định phải anchor-preserving: strong production detections không được xóa, temporal evidence chỉ bổ sung hoặc thay thế khi có bằng chứng kiểm soát được.

## Kết quả

Production prediction và metric giữ exact parity. Diagnostic thu được 9.449 cặp GT frame liên tiếp. GT center-speed có P50 `0.0821`, P90 `0.2626`, P95 `0.3432`, P99 `0.7673`; `3.28%` cặp vượt gate `0.4` và `1.24%` vượt `0.6`. Trong các selected top-1 pairs có IoU tốt ở cả hai đầu, `2.62%` sẽ bị gate 0.4 loại và `0.66%` bị gate 0.6 loại.

Tác động không đồng đều theo kích thước. Với `very_small_8_16`, `5.22%` GT pairs vượt 0.4 và `5.44%` good top-1 pairs bị loại; production-final pair coverage chỉ `83.25%`. `tiny_lt_8` chỉ có 161 pairs nhưng coverage còn `67.70%`. Medium `32–64` có P95 `0.1584`, gần như không bị gate 0.4.

`CardboardBox_0` là bằng chứng trực tiếp cho lỗi Tracklet Graph V1: `8.81%` GT pairs vượt 0.4; `8.62%` good top-1 pairs bị gate đó loại. Ở gate 0.6 tỷ lệ này về 0, nhưng selected pair accuracy chỉ `83.05%` và oracle pair accuracy `88.30%`, nên nới gate không giải quyết candidate switching/localization. `LifeJacket_0` vẫn có `1.83%` good top-1 pairs vượt 0.6.

Có 62 GT transitions với normalized speed trên 2.0. Nhiều transition ở `LifeJacket_1` trong vùng frame 826–927 nhảy khoảng 765–795 px; `BlackBox_1` có transition 5458→5459 nhảy 1.023 px. Chủ dữ liệu đã xác nhận đây là các chuyển cảnh/camera lia ra khỏi target, không phải lỗi frame index. Vì vậy các transition này là biên reset temporal hợp lệ; tuyệt đối không nội suy hoặc nối tracklet qua chúng. Mean motion bị các outlier này chi phối, nên thiết kế tiếp theo dựa trên percentile và scene-discontinuity handling thay vì mean.

Kết luận tạm thời: hard adjacent transition phải bỏ; nới cùng graph grid chỉ quay về baseline. Candidate association tiếp theo, nếu triển khai, phải giữ strong production detections, dùng soft/size-aware motion chỉ để bổ sung evidence, và reset tại scene discontinuity. `RUN_MOTION_PROFILE=False` sau khi hoàn tất.

Ghi nhận máy đọc: [`audit/MOTION_DISTRIBUTION_DIAGNOSTICS_V1_RESULT.json`](audit/MOTION_DISTRIBUTION_DIAGNOSTICS_V1_RESULT.json).
