# Anchor-Preserving Residual Recovery V2

Notebook Kaggle: [`06-inference-main.ipynb`](../06-inference-main.ipynb).

## Lý do thiết kế

Candidate Tracklet Graph V1 thất bại vì hard transition được quyền xóa strong detections. Motion Diagnostics xác nhận gate `0.4` loại `8.62%` good top-1 pairs của CardboardBox_0 và đặc biệt bất lợi với target dưới 16 px. Người gán nhãn xác nhận các bước nhảy 700–1.023 px là lúc camera chuyển cảnh/lướt ra khỏi target, nên temporal state phải reset thay vì nối trajectory.

V2 là nhánh **add-only**. Production `0.740013` là lớp nền bất biến; thuật toán chỉ thêm candidate vào frame mà production không có bbox.

## Phạm vi recovery

1. Lấy top-K candidate bằng fusion production `0.425/0.475/0.100`.
2. Strong anchor là top-1 có score `>=0.54` và frame đó tồn tại trong final production prediction.
3. Chỉ xét khoảng giữa hai strong anchor liên tiếp có hơn 7 frame trống; gap `<=7` đã thuộc trách nhiệm của production interpolation.
4. Không xét gap dài hơn `max_anchor_gap`.
5. Nếu normalized endpoint motion vượt `max_center_speed`, đánh dấu `scene_discontinuity_reset` và không bridge.
6. Trong gap còn lại, dựng DAG từ weak top-K candidates. Hai node chỉ nối khi cách nhau không quá bốn frame và motion hợp lệ.
7. Chỉ nhận đường đi nối được cả hai anchor, có ít nhất ba candidate và coverage tối thiểu 50% gap.
8. Append candidate boxes vào các frame production đang thiếu. Existing production bbox không bao giờ bị thay thế hay xóa.

Camera pan/cut được xử lý bằng reset ở biên anchor. V2 không tuyên bố nhận dạng scene bằng RGB/histogram; đây là recovery guard dựa trên discontinuity của target anchors.

## Grid pre-registered

| Tham số | Giá trị |
|---|---|
| `top_k` | 5, cố định |
| `weak_threshold` | 0.50, 0.52, 0.54 |
| `max_anchor_gap` | 12, 20 |
| `max_center_speed` | 0.6, 1.0 |
| `max_candidate_gap` | 3 missing frames |
| `min_recovered_frames` | 3 |
| `min_gap_coverage` | 0.5 |

Tổng cộng 12 configurations. Bốn cấu hình `weak_threshold=0.54` được buộc thành exact no-op control. Tie-break ưu tiên threshold cao hơn, gap ngắn hơn và speed nhỏ hơn.

## Chống overfit

Leave-one-identity-out giữ trọn `_0/_1` trong cùng held-out fold. Một global config được vote nguyên bộ; không đọc video/object name trong recovery logic. Promotion yêu cầu đồng thời:

- mean identity-CV delta `>=+0.003`;
- ít nhất 2/3 identity cải thiện;
- worst identity delta không dưới `-0.010`;
- cùng config nhận ít nhất 2/3 vote;
- full-public descriptive delta `>=+0.003`;
- replay đạt ít nhất 25 FPS;
- config được chọn phải có `weak_threshold<0.54`.

Notebook không tự promote: `USE_RESIDUAL_RECOVERY=False`.

## Trạng thái sau khi chạy Kaggle

Thí nghiệm đã hoàn tất và **không được promote**. Cấu hình được identity-CV vote là:

```python
top_k = 5
weak_threshold = 0.50
max_anchor_gap = 12
max_center_speed = 0.60
max_candidate_gap = 3
min_recovered_frames = 3
min_gap_coverage = 0.50
```

Full-public descriptive tăng `0.740013 → 0.740424` (`+0.000411`), thấp hơn promotion gate `+0.003`. Chỉ `CardboardBox_0` thay đổi: thêm 5 frame trong một gap và tăng `+0.002468`; năm video còn lại giữ nguyên. Mean identity-held-out delta bằng `0`, `0/3` held-out identities cải thiện và worst delta bằng `0`.

Có 44 candidate gaps giữa strong anchors ở cấu hình được chọn. 43 gap bị từ chối vì dài hơn giới hạn 12 frame; chỉ một gap được recovery. Nâng `max_anchor_gap` từ 12 lên 20 và `max_center_speed` từ 0.6 lên 1.0 vẫn không recovery thêm frame. Artifact không phân tách được bao nhiêu gap thực sự dài hơn 20 với bao nhiêu gap 13–20 thiếu dense path, nên không suy diễn xa hơn. Không mở rộng gap quá 20 frame vì vùng đã thử không có thêm gain, còn các khoảng dài có thể chứa target rời khung/camera lia và làm tăng temporal union sai.

`scene_discontinuity_reset_count=0` trong artifact không chứng minh video không có camera cut: implementation kiểm tra `anchor_gap_too_long` trước motion discontinuity, nên các gap dài không đi tới bộ đếm reset. Prediction vẫn an toàn vì các gap đó bị từ chối.

Runner được lưu để tái lập nhưng đã tắt:

```python
RUN_RESIDUAL_RECOVERY_EXPERIMENT = False
USE_RESIDUAL_RECOVERY = False

RUN_MOTION_PROFILE = False
RUN_TRACKLET_GRAPH_EXPERIMENT = False
```

Dùng đúng production candidate cache/checkpoint. Đây là cache replay, không cần SAHI tile cache và không chạy lại detector nếu cache hợp lệ.

## Artifact cần gửi lại

```text
/kaggle/working/siamese_identity_v1_calibration/
  anchor_preserving_residual_v2_identity_cv/
```

- `anchor_preserving_residual_v2_identity_cv_summary.json`
- `anchor_preserving_residual_v2_identity_cv_results.csv`
- `anchor_preserving_residual_v2_identity_cv_folds.csv`
- `anchor_preserving_residual_v2_identity_cv_videos.csv`
- `predictions_anchor_preserving_residual_v2_identity_cv.json`

Cấu trúc kết quả đã được ghi tại [`audit/ANCHOR_PRESERVING_RESIDUAL_V2_RESULT.json`](audit/ANCHOR_PRESERVING_RESIDUAL_V2_RESULT.json). Production tiếp tục là `0.740013`.
