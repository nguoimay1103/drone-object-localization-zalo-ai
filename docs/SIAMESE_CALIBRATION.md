# Phần 1B — Calibrate checkpoint Siamese identity-v1

Notebook chạy độc lập trên Kaggle: [`06-inference-main.ipynb`](../06-inference-main.ipynb). Bản NB06 trước calibration được giữ nguyên byte tại [`baselines/06-inference-main-baseline.ipynb`](../baselines/06-inference-main-baseline.ipynb).

## Mục tiêu và phạm vi

Checkpoint identity-v1 đạt mean ST-IoU `0.698426` khi dùng calibration cũ `YOLO/Siamese/Color = 0.4/0.3/0.3`, `MATCHING_THRESHOLD = 0.45`; baseline checkpoint đạt `0.700454`. Phần này kiểm tra liệu phân bố embedding mới cần trọng số hoặc threshold khác hay không.

Không đổi YOLO, cách tạo Siamese/reference embedding, color histogram, TTA, multiscale reference, bbox candidates, temporal interpolation/filtering hoặc metric. Notebook tách thành hai pha:

1. Chạy YOLO và Siamese một lần, lưu mọi bbox candidate cùng ba score thô vào `candidate_scores.json.gz`.
2. Replay candidates để thử 55 bộ trọng số × 31 threshold = 1.705 cấu hình mà không gọi model lại.

Mỗi bộ trọng số nằm trên simplex tổng bằng 1; Siamese từ `0.15–0.50`, Color từ `0.15–0.45`, YOLO còn lại và không nhỏ hơn `0.10`. Threshold chạy từ `0.30–0.60`, bước `0.01`. Cấu hình hiện tại `0.4/0.3/0.3 @ 0.45` luôn có trong grid.

## Cách chạy trên Kaggle

1. Upload notebook mới và gắn đúng dataset video, GT tự gán, YOLO checkpoint và checkpoint identity-v1.
2. Sửa bốn đường dẫn đầu notebook, đặc biệt `SIAMESE_MODEL_PATH`. Chọn `CALIBRATION_DIR` mới dành riêng cho checkpoint này.
3. Để chạy lại grid, đặt `RUN_CALIBRATION=True`; production mặc định hiện là `False`. Giữ `REUSE_CANDIDATE_CACHE=True`, TTA/reference và temporal như đã cung cấp.
4. Chạy toàn bộ notebook. Pha candidate extraction có thời gian gần một lần inference cũ. Cache được ghi sau từng video hoàn tất; nếu Kaggle ngắt, attach lại output cache đúng đường dẫn để tiếp tục.
5. Notebook production hiện replay cấu hình đã khóa; điểm phải gần `0.722052`, 9.295 frames. Log lịch sử trước khi khóa production đã tái hiện chính xác identity-v1 default `0.698426`, 9.254 frames.

Cache bị khóa bằng SHA-256 của YOLO, Siamese, GT; danh sách video; confidence; TTA và reference scales. Notebook từ chối cache khi một trong các thành phần thay đổi. Khi đổi checkpoint, dùng thư mục cache mới hoặc xóa đúng `candidate_scores.json.gz`; không sửa nội dung cache.

## Outputs cần gửi lại

- `calibration_summary.json`: cấu hình mặc định, cấu hình tốt nhất, delta, per-video và top 20.
- `calibration_results.csv`: đủ 1.705 cấu hình, đã sort giảm dần theo mean ST-IoU.
- Log của `CACHE REPLAY - DEFAULT CONFIG` và `BEST CALIBRATION CONFIG`.
- `predictions_calibrated.json` chỉ cần khi cần kiểm tra chi tiết frame.

Không cần gửi `candidate_scores.json.gz` nếu replay đúng và không có lỗi; giữ file để thử grid hẹp hơn mà không chạy model lại.

## Cách diễn giải

Cấu hình được chọn và báo điểm trên cùng sáu video GT tự gán. Đây là optimum offline theo mục tiêu hiện tại, không phải ước lượng tổng quát độc lập. Sau grid rộng, cần xem top configurations có tạo một vùng ổn định hay chỉ một điểm nhọn; chỉ khóa config mới khi replay mặc định đúng và kết quả tốt nhất vượt baseline `0.700454` với biên đủ rõ.

## Kiểm chứng local

8 tests CPU thành công: backup chính xác, notebook sạch output, model/preprocessing/metric/temporal giữ nguyên theo AST, fusion mặc định tương đương công thức cũ, tie candidate được giữ, replay synthetic qua temporal đạt đúng metric, grid chứa baseline và cache sai signature bị từ chối.

Chưa chạy YOLO/Siamese, GPU, video thật hoặc grid thật trong workspace local.

## Kết quả coarse grid do người dùng cung cấp

Cache replay khớp chính xác identity-v1 trước calibration: `0.698426`, 9.254 frames. Trong 1.705 cấu hình, optimum là `YOLO/Siamese/Color = 0.40/0.45/0.15`, threshold `0.51`, đạt mean ST-IoU `0.720274` với 9.358 frames. Mức tăng là `+0.021848` so với identity-v1 default và `+0.019820` so với baseline chưa calibrate `0.700454`.

Có 86 cấu hình vượt `0.700454`, nhưng chỉ 2 cấu hình nằm trong `0.001` của optimum. Với đúng weights tốt nhất, threshold `0.50/0.51/0.52` lần lượt đạt `0.715828/0.720274/0.707856`; kết quả nhạy với threshold. Color `0.15` cũng nằm tại biên dưới của grid.

Baseline checkpoint sau cùng coarse grid đạt `0.707417` tại weights `0.35/0.45/0.20`, threshold `0.48`. Identity-v1 calibrated cao hơn `+0.012857`, nên identity-v1 là checkpoint tốt nhất hiện tại sau so sánh calibrated-to-calibrated. Kết quả có cấu trúc: [`audit/SIAMESE_IDENTITY_V1_CALIBRATION_RESULT.json`](audit/SIAMESE_IDENTITY_V1_CALIBRATION_RESULT.json) và [`audit/SIAMESE_BASELINE_CALIBRATION_RESULT.json`](audit/SIAMESE_BASELINE_CALIBRATION_RESULT.json).

Notebook giữ profile `fine_identity_v1`: Siamese `0.250–0.575`, Color `0.000–0.225`, bước weight `0.025`; threshold `0.460–0.560`, bước `0.005`. Profile coarse vẫn được giữ để tái lập. Có thể điền `PRECOMPUTED_CANDIDATE_CACHE_FILE` tới cache identity-v1 hoàn chỉnh đã upload; nếu để trống, notebook dùng cache trong working directory hoặc tự extract lại.

## Kết quả fine grid và cấu hình đã khóa

Fine run lịch sử đánh giá 2.961 cấu hình (140 fine weight sets cộng cấu hình default cũ, nhân 21 thresholds). Optimum đạt `0.722052`, weights `0.425/0.475/0.100`, threshold `0.540`, 9.295 frames. Mức tăng là `+0.001778` so với coarse identity-v1, `+0.014635` so với baseline calibrated và `+0.021598` so với baseline gốc.

Ba cấu hình khác nhau nằm trong `0.00005` của optimum; optimum weight không chạm biên grid. Không tiếp tục micro-tune trên cùng sáu video vì temporal threshold tạo metric gián đoạn và nguy cơ fit riêng GT tăng cao. Notebook 06 đã khóa cấu hình này với `RUN_CALIBRATION=False`; bật lại cờ khi cần tái chạy grid. Chi tiết: [`audit/SIAMESE_IDENTITY_V1_FINE_CALIBRATION_RESULT.json`](audit/SIAMESE_IDENTITY_V1_FINE_CALIBRATION_RESULT.json).
