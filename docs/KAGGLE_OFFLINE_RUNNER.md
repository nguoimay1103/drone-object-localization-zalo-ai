# Chạy baseline offline trên Kaggle T4 ×2

## Trạng thái và mục tiêu

- Frame đầu tiên của video là **0**, frame index tuyệt đối; không reset khi target xuất hiện.
- Ưu tiên **độ chính xác offline**. Mục tiêu **25 FPS** đang được coi là mục tiêu phụ, chờ xác nhận nếu đó là điều kiện bắt buộc.
- Không giảm TTA, không skip frame, không đổi FP16/INT8, loss, sampling hay checkpoint để đạt FPS trong runner này.
- Đã tách inference thành package, thêm YAML/CLI, preflight, metric/provenance và ghép các shard video. Notebook, demo và weights gốc được giữ nguyên.
- Kiểm tra CPU/source có thể xác nhận logic; **chưa chạy GPU/video thật, chưa tái lập 0.700454 và chưa đo 25 FPS**. Cần qua bước B0 bên dưới trước khi gọi refactor tương đương toàn pipeline.

## 1. Chuẩn bị repo và môi trường

Đưa repo vào một thư mục ghi được, ví dụ `/kaggle/working/drone-object-localization-zalo-ai`. Không chạy ngay notebook 01–05: baseline chỉ cần hai checkpoint hiện có và video/reference.

Trong cell notebook Kaggle:

```python
%cd /kaggle/working/drone-object-localization-zalo-ai
%pip install -e ".[inference]"
```

Ultralytics được khóa ở **8.3.221**. PyTorch/torchvision/CUDA và các thư viện khác chưa có lock của run cũ; ưu tiên giữ môi trường Kaggle đã chạy được thay vì force-upgrade. Nếu pip làm đổi thư viện đã import, restart kernel rồi kiểm tra lại. Đây chưa phải environment lock chứng minh tái lập lịch sử.

CLI cũng chạy trực tiếp từ đường dẫn script mà không cần editable install, nếu dependencies đã có. `--help` không import torch; `--dry-run` cần PyYAML nhưng không cần GPU libraries.

Chỉ dùng checkpoint đáng tin cậy. Runner yêu cầu file local tồn tại và khớp hash baseline; không tải weight ngầm. Khi thử weight mới, tạo config thí nghiệm với hash tương ứng, không sửa config baseline.

## 2. Điền đường dẫn ở một nơi

Sửa `configs/baseline_0_700454.yaml`:

```yaml
paths:
  samples_dir: /kaggle/input/YOUR_DATASET/samples
  annotations: /kaggle/input/YOUR_DATASET/annotations/drone_annotations.json
  yolo_weights: /kaggle/input/YOUR_WEIGHTS/yolo_drone_700.pt
  siamese_weights: /kaggle/input/YOUR_WEIGHTS/siamese_mobilenet_best_1.pth
  output_dir: /kaggle/working/b0_single_gpu
```

Đường dẫn tương đối trong YAML tính từ thư mục chứa YAML; đường dẫn CLI tương đối tính từ working directory. Có thể ghi đè bằng `--samples-dir`, `--annotations`, `--yolo-weights`, `--siamese-weights`, `--output-dir`. CLI không viết lại YAML.

Dataset layout:

```text
samples/
  BlackBox_0/
    drone_video.mp4
    object_images/
      ref1.jpg
      ref2.jpg
  ...
```

Annotation giữ schema notebook: list record `video_id`, `annotations`, `bboxes` với `frame,x1,y1,x2,y2`. Mỗi frame tối đa một bbox cho target. Record target-absent được phép có annotations rỗng. Runner báo lỗi nếu thiếu record GT, duplicate frame hoặc bbox không hợp lệ; không silently tính điểm 0 cho GT bị thiếu.

Kiểm tra paths/hash/GT trước:

```bash
python scripts/run_inference.py --config configs/baseline_0_700454.yaml --dry-run
```

Dry-run **không giải mã video và không thử GPU**. Nếu chưa muốn đánh giá, dùng `--no-eval`; output evaluation sẽ là `null`, không phải số 0.

## 3. Tái lập B0 trên một GPU trước

```bash
python scripts/run_inference.py --config configs/baseline_0_700454.yaml --device cuda:0 --output-dir /kaggle/working/b0_single_gpu
```

Thư mục output phải chưa tồn tại để tránh ghi đè kết quả. Có thể chạy nhanh một video bằng `--video-id BlackBox_0`; số điểm lúc đó chỉ tính trên subset và không được so với mean sáu video.

Giá trị baseline: TTA on, confidence 0.05, matching 0.45, fallback threshold 0.2, weights 0.4/0.3/0.3, reference scales 224/112/56, temporal gap 5/min segment 3. `imgsz` và `batch` để null vì notebook 06 không truyền trực tiếp; runner ghi effective predictor args để xác nhận giá trị thực. Không áp mặc định của phiên bản Ultralytics mới hơn.

Artifacts mỗi run:

| File | Nội dung |
|---|---|
| `predictions.json` | Schema đầu ra lịch sử, frame bắt đầu từ 0 |
| `metrics.json` | Mean/per-video ST-IoU nếu có GT, số frame, candidate count, FPS, thời gian và peak GPU allocated memory |
| `run_manifest.json` | Resolved config, source/weight/video/reference/GT hashes, versions, GPU, trạng thái completed/failed |

Nếu có `predictions.json` của notebook gốc, so sánh **JSON sau khi parse**, không yêu cầu byte hash trùng vì indentation khác:

```python
import json
with open("/kaggle/input/GOLDEN/predictions.json") as f:
    golden = json.load(f)
with open("/kaggle/working/b0_single_gpu/predictions.json") as f:
    current = json.load(f)
assert current == golden, "Cần điều tra khác biệt trước khi chạy thí nghiệm cải tiến"
```

Nếu không khớp, so effective runtime/config, checkpoint, frame count, reference preprocessing và bbox tại frame lệch. Không chỉ chấp nhận cùng mean tới sáu chữ số nếu predictions khác.

Đánh giá lại JSON độc lập:

```bash
python scripts/evaluate.py --predictions /kaggle/working/b0_single_gpu/predictions.json --annotations /kaggle/input/YOUR_DATASET/annotations/drone_annotations.json
```

Evaluator mặc định yêu cầu phủ đúng toàn bộ video có trong GT. Dùng `--video-id` lặp lại nếu chủ động đánh giá subset.

## 4. Dùng hai T4: chia video, không chia một video giữa GPU

Sau khi single-GPU B0 đã đối chiếu, có thể chạy hai process song song. Mỗi process có cả YOLO và Siamese trên GPU của nó, xử lý các video khác nhau; frame index và temporal processing của mỗi video giữ nguyên.

Cell Python Kaggle dưới đây chờ cả hai process và đo thời gian chung. Hai GPU phải đang được bật/hiển thị. Không chạy hai lệnh tuần tự rồi gọi đó là benchmark song song.

```python
from contextlib import ExitStack
from pathlib import Path
import json
import subprocess
import sys
import time

repo = Path("/kaggle/working/drone-object-localization-zalo-ai")
outputs = [Path("/kaggle/working/b0_gpu0"), Path("/kaggle/working/b0_gpu1")]
assert all(not p.exists() for p in outputs), "Chọn output mới, không ghi đè run cũ"
started = time.perf_counter()
with ExitStack() as stack:
    processes = []
    for i, output in enumerate(outputs):
        log = stack.enter_context(Path(f"/kaggle/working/b0_worker_{i}.log").open("w"))
        command = [sys.executable, str(repo / "scripts/run_inference.py"),
                   "--config", str(repo / "configs/baseline_0_700454.yaml"),
                   "--device", f"cuda:{i}", "--num-shards", "2", "--shard-index", str(i),
                   "--output-dir", str(output)]
        processes.append(subprocess.Popen(command, cwd=repo, stdout=log, stderr=subprocess.STDOUT))
    returncodes = [process.wait() for process in processes]
elapsed = time.perf_counter() - started
assert returncodes == [0, 0], "Xem worker logs; không merge run bị lỗi"
total_frames = 0
for output in outputs:
    with (output / "metrics.json").open() as f:
        total_frames += json.load(f)["decoded_frames"]
print({"joint_cold_wall_seconds": elapsed, "total_frames": total_frames,
       "joint_cold_fps": total_frames / elapsed})
```

Ghép và tính lại mean trên toàn bộ video:

```bash
python scripts/merge_predictions.py --config configs/baseline_0_700454.yaml --runs /kaggle/working/b0_gpu0 /kaggle/working/b0_gpu1 --output-dir /kaggle/working/b0_merged
```

Merge kiểm tra trạng thái completed, hash predictions, cấu hình inference (trừ device), checkpoints, source, runtime, GT và đủ/không trùng video. Không trung bình hai worker mean nếu số video khác nhau. Không cộng FPS hai worker; `parallel_pipeline_fps` để null vì merger không biết thời gian chạy chung. Dùng số đo wall-clock của cell trên, ghi rõ nó gồm process startup/hash/model load và report writes, không gồm bước merge sau đó.

Hai T4 có bộ nhớ độc lập; runner này không coi chúng là một GPU có VRAM gộp. Thông lượng nhiều video trên hai GPU không chứng minh đạt 25 FPS cho một video trên một GPU. CPU decode, I/O và chia tải lệch theo thời lượng video có thể hạn chế hiệu quả; chưa có bằng chứng tăng tốc 2 lần.

## 5. Đọc FPS đúng phạm vi

- `pipeline_fps`: tổng frame / tổng thời gian reference features + decode + detector + crop/matcher + histogram + temporal. Video đầu bao gồm predictor initialization/warmup. Không phải microbenchmark YOLO riêng.
- `cold_run_fps`: thêm model load, evaluation và serialization predictions; không gồm hashing đầu vào, thời gian khởi động process/imports hoặc ghi báo cáo cuối.
- Cell hai GPU đo cold wall-clock chung với phạm vi rộng hơn; không so trực tiếp với steady-state model FPS.
- Peak memory là torch allocated memory trên GPU của worker, không phải tổng VRAM theo nvidia-smi.
- `fps_target_met_on_this_worker` chỉ báo kết quả đo, không tự đổi cấu hình, không tự chọn model hay bác bỏ run điểm cao. Objective ưu tiên chất lượng; chưa có cơ chế search tự động.

Khi cần benchmark ổn định, chạy lặp và báo cả cold/steady-state với cùng video, thời lượng, model/runtime và settings. B0 hiện không cài thêm warmup nhân tạo làm đổi đường đi trước khi so golden.

## 6. Điểm khác về vận hành so với notebook

Numerical path giữ nguyên cho input đầy đủ: model, transformations, L2 distances, histogram, fusion/tie-break, crop integer truncation, temporal interpolation và rounding đầu ra.

Khác có chủ đích: strict preflight; bắt buộc hai checkpoint đúng hash thay vì fallback khi thiếu weight; không vẽ random debug frames/giữ toàn bộ candidates trong RAM; không overwrite output; báo lỗi decode/annotation mismatch; cho phép tắt evaluation rõ ràng. Reference bị thiếu vẫn dùng nhánh giảm trọng số/ngưỡng như notebook, được ghi validity flags.

Notebook 05 hiện có thí nghiệm standalone `identity_v1`, xem [hướng dẫn](SIAMESE_IDENTITY_V1.md); notebook 04 và negative mining chưa đổi. Runner baseline vẫn giữ nguyên. Khi đánh giá checkpoint mới, tạo config riêng và cập nhật hash Siamese. Chưa triển khai cache, FP16/export hoặc batching tự động; `batch` chỉ là tùy chọn thí nghiệm có kiểm tra parity riêng.

## 7. Test cục bộ

Kết quả kiểm tra runner trước bước `identity_v1`: **12 tests chạy thành công** trên Python 3.12 với PyYAML 6.0.3; 18 file Python mới parse thành công. Khi đó SHA-256 của 14 file gốc vẫn không đổi. Hiện notebook 05 đã được sửa riêng; bản gốc đúng hash được bảo toàn trong `baselines/`. PyYAML chỉ được cài vào thư mục tạm phục vụ kiểm tra, không thay môi trường training/model.

```bash
python -m pip install -e .
python -B -m unittest discover -s tests -v
```

Tests đối chiếu trực tiếp AST/hàm notebook hiện tại: model/reference, crop/embedding blocks, metric, weighted scoring/ties/missing features, temporal gap/rounding. Thêm kiểm tra paths/hash/GT, shard coverage, merge global mean và dry-run CLI.

Tests không giải mã video, không load tensor checkpoint, không chạy torch/CUDA. Dùng chúng để phát hiện drift logic; dùng golden video trên Kaggle để chứng nhận parity và đo chất lượng/tốc độ.
