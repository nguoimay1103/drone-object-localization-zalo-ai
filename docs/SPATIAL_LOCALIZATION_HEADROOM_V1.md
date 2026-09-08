# Spatial Localization Headroom Diagnostic V1

Notebook Kaggle: [`06-inference-main.ipynb`](../06-inference-main.ipynb).

## Mục tiêu

Diagnostic xác định liệu một bbox refiner kiểu Alpha-Refine có đủ dư địa để chạy A/B hay không. Nó không train model, không chọn hyperparameter và không thay prediction production. Candidate cache, fusion `0.425/0.475/0.100`, matching threshold `0.54` và Temporal Gating production được giữ nguyên.

Đơn vị đánh giá vẫn là mean ST-IoU trung bình đều theo video. Mỗi counterfactual giữ nguyên chính xác tập frame dự đoán, do đó mọi false-positive frame và missed-GT frame vẫn nằm trong mẫu số.

## Các mức trần

| Scenario | Ý nghĩa | Dùng để quyết định refiner |
|---|---|---|
| `perfect_shared` | Bbox hoàn hảo trên mọi frame có cả GT và prediction | Trần tuyệt đối của spatial correction với temporal support cố định; có thể bao gồm sửa nhầm object |
| `perfect_good_overlap` | Chỉ bbox có production IoU `>=0.5` được làm hoàn hảo | Trần bảo thủ cho bbox refiner và là gate chính |
| `candidate_keep_best` | GT chọn bbox tốt nhất giữa production và toàn bộ cached candidates | Headroom của candidate selection; không phải refiner |
| `local_candidate_keep_best` | Như trên nhưng candidate phải overlap production bbox `>=0.5` | Proxy bảo thủ cho local proposal selection; proximity không đảm bảo identity |
| `center_only_good_overlap` | Dùng GT center, giữ width/height production trên good-overlap frames | Chẩn đoán lỗi tâm |
| `size_only_good_overlap` | Dùng GT width/height, giữ center production trên good-overlap frames | Chẩn đoán lỗi scale/aspect ratio |

Hai counterfactual center/size có thể tạo delta âm và không cộng được với nhau. Chúng mô tả thành phần hình học, không phải kết quả deployable. `frames.csv` còn ghi residual jitter của prediction error giữa hai good-overlap frame liên tiếp; phép đo này trừ chuyển động GT và không nối qua gap.

## Gate quyết định Alpha-Refine A/B

Gate được khóa trước khi xem kết quả:

- `perfect_good_overlap` macro delta tối thiểu `+0.010`;
- ít nhất 2/3 identity có ceiling delta tối thiểu `+0.005`;
- A/B sau đó phải train chỉ từ train-set GT với bbox jitter;
- inference A/B phải giữ nguyên temporal support của production để cô lập spatial localization.

Gate chỉ trả lời có đáng train refiner hay không. Nó không promote một refiner chưa tồn tại và không biến oracle thành claim về mức tăng thực tế.

## Integrity checks

Runner bắt buộc kiểm tra:

- cache, GT và production có cùng tập video;
- ST-IoU từng video và macro mean tái lập production đến tolerance `1e-12`;
- số prediction frame khớp production;
- tổng contribution theo frame khớp macro delta của từng scenario;
- prediction, cache và GT không bị sửa;
- frame index tuyệt đối bắt đầu từ 0 được giữ nguyên.

## Kết quả locked diagnostic

Kaggle run trên production control `0.740013` cho kết quả:

- `perfect_good_overlap = 0.839178`, macro headroom `+0.099165`;
- cả 3/3 identity vượt ceiling delta `+0.005`;
- center-only headroom `+0.022462`, size-only headroom `+0.019317`;
- cached-candidate oracle chỉ `+0.007541`, local-candidate oracle `+0.002866`.

Gate `run_alpha_refine_ab=True`. Kết quả này chỉ duyệt việc train một global
spatial refiner; nó không phải kết quả model và không được dùng làm mức tăng dự
kiến. Artifact tóm tắt đã được khóa tại
[`audit/SPATIAL_LOCALIZATION_HEADROOM_V1_RESULT.json`](audit/SPATIAL_LOCALIZATION_HEADROOM_V1_RESULT.json).

## Cách tái lập diagnostic đã lưu trữ

```python
RUN_SPATIAL_HEADROOM = True  # hiện mặc định False vì diagnostic đã hoàn tất
```

Các experiment runner khác phải để `False`. Dùng đúng candidate cache của Siamese identity-v1 production. Đây là cache replay nên không chạy lại detector nếu cache signature hợp lệ.

## Artifacts cần gửi lại

```text
/kaggle/working/siamese_identity_v1_calibration/
  spatial_localization_headroom_v1/
```

- `spatial_localization_headroom_v1_summary.json`
- `spatial_localization_headroom_v1_frames.csv`
- `spatial_localization_headroom_v1_groups.csv`
- `spatial_localization_headroom_v1_videos.csv`

Khi phân tích kết quả, ưu tiên `refiner_ab_decision`, `perfect_good_overlap`, center/size counterfactuals, residual jitter và phân rã theo identity/size bin. `perfect_shared` chỉ cho biết trần tổng quát của spatial correction. Bước active kế tiếp là [Spatial Refiner A/B V1](SPATIAL_REFINER_AB_V1.md).
