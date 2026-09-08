# Candidate Tracklet Graph V1

Notebook chạy Kaggle: [`06-inference-main.ipynb`](../06-inference-main.ipynb).

## Mục tiêu

Thí nghiệm này xử lý bottleneck còn lại sau Detector Profile/Oracle Diagnostics: candidate cache có recall tốt nhưng top-1 theo từng frame, threshold cứng và hậu xử lý đoạn chưa khai thác được tính nhất quán toàn video. Production control vẫn là fusion `0.425/0.475/0.100`, threshold `0.54` và Temporal Gating `{max_gap: 7, min_seg_len: 24, max_center_speed: 0.4}` với offline ST-IoU đã ghi nhận `0.740013`.

Để tái chạy, đặt `RUN_TRACKLET_GRAPH_EXPERIMENT = True`. Sau run đã kiểm tra, repository giữ cờ này ở `False`. `USE_TRACKLET_GRAPH = False` được giữ cố ý: thí nghiệm không tự thay prediction production dù promotion gate có qua.

## Thuật toán

Mỗi frame có tối đa top-K candidate theo fusion score và một state `ABSENT`. Viterbi tìm đường đi có tổng điểm lớn nhất trên toàn video:

```text
emission(candidate) = emission_scale * (fusion_score - locked_threshold)
emission(ABSENT)    = 0

candidate -> candidate = -motion_weight * (center_speed / max_center_speed)^2
candidate <-> ABSENT   = -state_change_penalty
ABSENT -> ABSENT       = 0
```

Transition candidate bị cấm nếu `center_speed > max_center_speed`. Candidate dưới threshold có emission âm, nhưng vẫn có thể được giữ khi nó nối hai đoạn có evidence mạnh và tránh hai lần chuyển state. Candidate cao điểm nhưng cô lập có thể thua đường `ABSENT`. Sau decode, cùng Temporal Gating production được áp dụng; fusion, checkpoint và detector candidate không đổi.

### Camera-motion proxy

Ablation `camera_compensation=True` dùng median displacement của các cặp candidate mutual-nearest giữa hai frame liên tiếp. Chỉ dùng khi có ít nhất ba cặp và bỏ displacement lớn hơn 256 px. Đây là proxy chỉ dựa trên cache, **không phải** optical flow, affine transform hay homography. Nếu identity-CV không chọn ổn định ablation này thì giữ `False`; camera-motion thật thuộc V2 và cần đọc frame video.

## Grid cố định

| Tham số | Giá trị |
|---|---|
| `top_k` | 3, 5 |
| `motion_weight` | 0.02, 0.05, 0.10 |
| `state_change_penalty` | 0.02, 0.05 |
| `max_center_speed` | 0.4, 0.6 |
| `camera_compensation` | false, true |
| `emission_scale` | 1.0 |

Tổng cộng 48 cấu hình global, không có tham số theo object/video.

## Chống overfit và promotion gate

Ba fold giữ trọn cặp `_0/_1` của một identity làm held-out. Mỗi fold chỉ chọn config bằng bốn video thuộc hai identity còn lại. Config báo cáo cuối được vote nguyên bộ tham số; tie ưu tiên can thiệp nhẹ và không dùng camera proxy.

Promotion chỉ đạt khi đồng thời:

- mean identity-CV delta `>= +0.003`;
- cải thiện ít nhất 2/3 identity;
- identity tệ nhất không giảm quá `0.010`;
- cùng một config nhận ít nhất 2/3 vote;
- full-public descriptive delta `>= +0.003`;
- cache-replay graph đạt ít nhất 25 FPS.

Control gọi lại chính pipeline production và yêu cầu prediction JSON khớp chính xác. Runner cũng chụp snapshot để phát hiện graph vô tình mutate control.

## Artifact cần gửi lại

Thư mục mặc định:

```text
/kaggle/working/siamese_identity_v1_calibration/
  candidate_tracklet_graph_v1_identity_cv/
```

Các file chính:

- `candidate_tracklet_graph_v1_identity_cv_summary.json`
- `candidate_tracklet_graph_v1_identity_cv_results.csv`
- `candidate_tracklet_graph_v1_identity_cv_folds.csv`
- `candidate_tracklet_graph_v1_identity_cv_videos.csv`
- `predictions_candidate_tracklet_graph_v1_identity_cv.json`

Khi trao đổi kết quả, gửi ít nhất summary, folds và videos. Cần xem `rescued_below_threshold`, `suppressed_above_threshold`, per-video delta, config vote và ablation camera proxy; mean public score một mình chưa đủ để promote.

## Giới hạn V1

V1 chỉ nối candidate đã có và không tạo bbox mới từ ảnh. `no_candidate`/detector localization miss chỉ có thể được bù một phần bởi interpolation production. Nếu graph chọn đúng candidate tốt hơn nhưng vẫn không qua identity-CV, dừng nhánh này. Nếu graph có lợi ổn định nhưng camera proxy không đủ, V2 mới thử GMC từ ORB/optical flow hoặc homography trên frame, với runtime A/B riêng.

## Kết quả và quyết định

Production control tái lập chính xác `0.740013`/8.890 frames. Không cấu hình nào trong 48 cấu hình vượt control: tốt nhất đạt `0.699135` (delta `-0.040878`). Config được vote mô tả `{top_k: 5, motion_weight: 0.02, state_change_penalty: 0.02, max_center_speed: 0.4, camera_compensation: true}` chỉ đạt `0.694336`/8.042 frames.

Identity-CV delta là `-0.057634`, cải thiện `0/3` identities, worst delta `-0.140015`, và ba fold chọn ba cấu hình khác nhau nên recommendation chỉ có một vote. `CardboardBox_0` giảm mạnh từ `0.604380` xuống `0.338139`; graph đã suppress 189 top-1 frame trên video này. Trên toàn bộ recommendation, graph chỉ cứu 81 candidate dưới threshold nhưng loại 475 top-1 candidate vốn trên threshold.

Camera translation proxy có ích tương đối trong phần lớn paired configs, nhưng không bù được lỗi chính. Grid tốt dần khi giảm motion/state penalty và tăng `max_center_speed`, tức là nghiệm đang đi về framewise production. Kết luận: hard motion transition trên mọi cặp frame làm vỡ trajectory thật, nhất là target nhỏ/chuyển động nhanh; V1 bị reject và không nên mở rộng grid chỉ để tiến gần lại control.

`RUN_TRACKLET_GRAPH_EXPERIMENT=False` và `USE_TRACKLET_GRAPH=False`. Production tiếp tục dùng Temporal Gating V1 `0.740013`. Ghi nhận máy đọc: [`audit/CANDIDATE_TRACKLET_GRAPH_V1_RESULT.json`](audit/CANDIDATE_TRACKLET_GRAPH_V1_RESULT.json).
