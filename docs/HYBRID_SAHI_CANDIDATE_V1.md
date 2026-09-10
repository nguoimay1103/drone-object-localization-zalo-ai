# Phần 8C — Hybrid Full-frame + SAHI Tiles V1

Notebook Kaggle: [`06-inference-main.ipynb`](../06-inference-main.ipynb). Experiment này giữ nguyên mọi candidate trong cache production và bổ sung candidate từ native-scale tiles. Không cài dependency `sahi`; slicing, ownership và coordinate mapping được triển khai trực tiếp bằng OpenCV + Ultralytics.

## Lý do

Explicit-960 làm global oracle recall giảm và tăng no-candidate frames. Scale toàn frame đã đẩy một số object ra khỏi distribution detector được train. Hybrid SAHI giữ nhánh full-frame 640 đã xác minh, đồng thời phóng đại object trong tile mà không thay thế candidate tốt hiện có.

## Tile geometry

```python
SAHI_TILE_SIZE = 640
SAHI_TILE_IMGSZ = 640
SAHI_OVERLAP_RATIO = 0.25
SAHI_FRAME_BATCH_SIZE = 8
SAHI_TILE_INFERENCE_BATCH_SIZE = 16
```

Với video `1024×576`, layout có hai tile ngang `[0,640]` và `[384,1024]`. Overlap thực tế là 256 px do tile cuối được ép sát biên phải. Vùng overlap được chia tại midpoint `x=512`:

- Candidate có center `<512` thuộc tile trái.
- Candidate có center `>=512` thuộc tile phải.

Mỗi center thuộc đúng một tile; box ở global boundary vẫn được giữ. Nếu frame nhỏ hơn hoặc bằng tile theo cả hai chiều, tile branch bỏ qua vì không tạo thêm scale information.

## Immutable union

Hybrid cache được tạo theo thứ tự:

```text
all legacy full-frame candidates
+ all ownership-filtered tile candidates
```

Không NMS hoặc suppression giữa tile và legacy. Tie ranking vẫn ưu tiên legacy vì candidate legacy đứng trước. Candidate có trường `source=legacy_full_frame|sahi_tile`; thuật toán fusion không dùng source này.

Runner kiểm tra oracle IoU trên từng GT frame. Vì legacy candidate được giữ nguyên, `oracle_regression_count` phải bằng 0; nếu không, run dừng ngay.

## Cache và resume

Tile-only cache:

```text
hybrid_sahi_candidate_v1/
└── tile_candidate_scores_size640_overlap025.json.gz
```

Signature khóa tile size, inference imgsz, overlap, ownership rule, batch sizes, confidence, TTA, Ultralytics version và checkpoint/data hashes. Cache atomic-save sau mỗi video và resume được.

## Promotion gate

Downstream được khóa ở fusion `0.425/0.475/0.100`, threshold `0.54` và Temporal Gating production. Hybrid chỉ pass khi:

```text
oracle monotonic trên mọi GT frame
global oracle recall@0.5 không giảm
mean ST-IoU delta >= +0.003
improved identities >= 2/3
worst identity delta >= -0.010
sub-16 recall@0.5 delta >= +0.002
```

Không auto-promote và không calibration tile score trong vòng đầu.

## Runtime

Tile inference dùng micro-batch 8 frames. Summary báo:

- Tile-branch FPS thực đo.
- Sequential hybrid FPS ước tính từ explicit-640 reference `30,3087 FPS` cộng thời gian tile branch.
- T4×2 parallel FPS ước tính bằng nhánh chậm hơn nếu full-frame và tile chạy độc lập.
- Peak CUDA allocated memory.

Hai chỉ số hybrid là estimate; prediction hiện tại được tạo bằng legacy cache + tile extraction, chưa phải benchmark production pipeline đồng thời hai GPU.

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
RUN_HYBRID_SAHI = False  # chỉ bật lại để tái lập tile extraction
```

Trỏ `PRECOMPUTED_CANDIDATE_CACHE_FILE` tới legacy identity-v1 cache để chỉ chạy tile branch.

Artifacts:

- `hybrid_sahi_candidate_v1_results.csv`
- `hybrid_sahi_candidate_v1_videos.csv`
- `hybrid_sahi_candidate_v1_summary.json`
- `predictions_hybrid_sahi_v1.json`

Nếu oracle tăng nhưng final ST-IoU giảm, bước sau mới đánh giá source-aware calibration/ranking bằng held-out identity. Nếu oracle gần như không tăng, dừng SAHI và chuyển sang P2 training.

## Kết quả và quyết định

Hybrid giữ oracle monotonic trên toàn bộ 9.505 GT frames, cải thiện oracle IoU ở 4.430 frames và tăng global recall@0.5 `0,9481326 → 0,9556023`. Mean oracle IoU tăng `0,829718 → 0,854581`; no-candidate GT frames giảm `118 → 89`; sub-16 recall tăng `+0,006080`.

Final ST-IoU gần như hòa nhưng không pass: `0,739846`, delta `-0,000167`. BlackBox/CardboardBox tăng theo identity `+0,000479/+0,022049`, LifeJacket giảm `-0,023030`. CardboardBox_0 tăng mạnh `+0,045864`, trong khi LifeJacket_1 giảm `-0,044604` dù baseline candidate recall đã là `1,0`.

Equal-source union làm 9.874 frame chọn tile top-1 và 3.737 tile frames vượt threshold; final frames tăng `8.890 → 9.419`. Tile candidates đã tạo headroom nhưng đang cạnh tranh quá tự do với legacy candidates.

Runtime tile branch đạt `42,26 FPS`; sequential hybrid ước tính `17,65 FPS`, còn T4×2 parallel estimate giữ `30,31 FPS`. Peak allocated CUDA memory `453,26 MB`.

**Quyết định:** không promote equal-source hybrid, nhưng giữ tile cache và tiếp tục bằng source-aware admission cache replay. Vòng kế tiếp ưu tiên hai policy ít tham số: tile chỉ cứu khi legacy không có candidate, hoặc tile chỉ cạnh tranh khi best legacy score dưới production threshold. Không chạy lại YOLO và chưa chuyển sang P2 vì candidate oracle đã tăng rõ rệt.
