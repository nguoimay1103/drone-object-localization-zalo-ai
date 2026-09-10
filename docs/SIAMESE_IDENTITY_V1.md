# Phần 1 — Identity sampling và validation Siamese

Notebook chạy độc lập trên Kaggle: [05-train-siamese.ipynb](../05-train-siamese.ipynb).
Bản trước thay đổi: [05-train-siamese-baseline.ipynb](../baselines/05-train-siamese-baseline.ipynb), giữ nguyên byte và output lịch sử.

## Phạm vi thay đổi

- Nhận diện vật thể bằng physical identity, không dùng video ID như một class độc lập. Quy tắc đã được người dùng xác nhận: cặp hậu tố `_0`/`_1` là cùng vật thể. Giữ nguyên toàn bộ tiền tố, kể cả dấu gạch dưới; ngoại lệ cần `IDENTITY_OVERRIDES`.
- Train/val tách identity. Giữ tỷ lệ validation cấu hình 0.2, nhưng đây là 20% identities, không phải 20% videos/frames. Split tự động dùng danh sách identity đã sort, Python `Random(42).shuffle` và làm tròn lên số identity validation; không tái sử dụng split sklearn theo video cũ. Có thể khóa `VAL_IDENTITY_IDS` rõ ràng cho các thí nghiệm sau.
- Negative pool được lọc theo video nguồn thuộc split; lấy provenance từ đúng định dạng filename notebook 04. File không rõ nguồn phải được khai báo override hoặc sửa dữ liệu, không trộn âm thầm.
- Nhánh lấy positive crop làm negative chỉ được chọn vật thể khác identity. Train vẫn lấy query video ngẫu nhiên như cũ; positive vẫn cùng video. Xác suất chọn pool vẫn là 0.7; nếu một nguồn không khả dụng, fallback sang nguồn hợp lệ và có cảnh báo. Không còn fallback dùng anchor làm negative.
- Validation cố định một triplet cho mỗi positive crop, seed riêng, không bốc lại mỗi epoch. Validation loss/accuracy được tính trung bình theo số mẫu, tránh batch cuối nhỏ có trọng số bằng batch đầy.
- Seed Python/NumPy/torch và DataLoader workers/generators; thêm diagnostics, split/triplet manifests, CSV/JSON history và checksum checkpoint.

## Những phần không đổi

Đã đối chiếu AST với bản baseline: `DroneDegradation`, `get_transforms`, `SiameseMobileNet`, hàm đọc ảnh, trainer constructor, `train_epoch` và `calc_accuracy` giữ nguyên.

Batch 128, 15 epochs, learning rate 1e-4, margin 0.5, AdamW weight decay 1e-4, cosine schedule và AMP giữ nguyên. Không thêm DataParallel: training vẫn dùng GPU mặc định như notebook gốc; không tự dùng GPU thứ hai để thay đổi BatchNorm/training behavior.

Chưa sửa notebook 04 hoặc 06, chưa đổi detector mining hay xử lý IoU/GT coverage. Pool negatives vẫn có thể chứa target do cách lọc cũ — phần 2 mới xử lý. Hash data index chỉ bao gồm paths/provenance, không chứng thực nội dung ảnh; chưa kiểm tra ảnh trùng byte hoặc gán nhãn sai giữa các identity.

## Cách chạy

1. Upload notebook 05 lên Kaggle, bật GPU và gắn dataset matching sampling 5 đang có. Không cần chạy lại notebook 04 ở bước này; không cần repo package.
2. Sửa `ROOT_FOLDER`; chọn `SAVE_DIR` mới, ví dụ `/kaggle/working/siamese_identity_v1`.
3. Giữ `RUN_TRAINING=False`, chạy tất cả cell. Chưa tải pretrained model hoặc train; bước này tạo index/split/validation triplets và in diagnostics (vẫn cần các thư viện torch/torchvision/OpenCV của notebook).
4. Kiểm tra bảng video → identity → split, số positive/negative và cảnh báo. Nếu tên khác quy tắc, bổ sung `IDENTITY_OVERRIDES`. Nếu filename negative khác định dạng NB04, bổ sung `NEGATIVE_VIDEO_OVERRIDES` sau khi xác định video nguồn chính xác.
5. Nếu cần đổi split/mapping sau khi artifacts đã lưu, dùng `SAVE_DIR` mới. Notebook từ chối trộn manifest/triplets cũ với cấu hình mới.
6. Khi bảng đúng, đặt `RUN_TRAINING=True` và chạy lại các cell. Không đổi hyperparameter khác. Nếu output đã có checkpoint/history, chọn thư mục mới để không ghi đè.

Một guard kiểm tra batch train cuối có một mẫu vì projection dùng BatchNorm. Notebook sẽ báo lỗi thay vì tự đổi `drop_last` hoặc batch size; gửi `data_summary.json` nếu gặp trường hợp này để quyết định điều chỉnh riêng.

## Cần trao đổi gì ở bước này?

**Sau diagnostics:** gửi bảng mapping/split hoặc `data_summary.json`, kèm cảnh báo nếu có. Cần xác nhận mọi video cùng vật thể được gộp đúng và cả hai split đủ dữ liệu. Nếu một split chỉ có một identity, validation không có other-identity negatives, nên kết quả không đo đủ khả năng phân biệt vật thể.

**Sau training:** gửi:

- `training_history.csv`: loss/accuracy từng epoch, LR đã dùng và LR epoch kế tiếp.
- `run_summary.json`: best epoch, best val loss/accuracy, đường dẫn và SHA-256 checkpoint.
- `split_manifest.json`: mapping, split, seeds, index/triplet hashes.
- `environment.json`: phiên bản thư viện và GPU.

Giữ `data_index.json` và `val_triplets.json` để tái lập; chưa cần gửi toàn bộ nếu file lớn và không có lỗi.

**Đánh giá end-to-end:** trong notebook 06 chỉ thay `SIAMESE_MODEL_PATH` sang checkpoint mới `siamese_mobilenet_best.pth`, giữ nguyên detector, TTA, reference scales, thresholds, fusion và temporal. Gửi mean ST-IoU cùng điểm sáu video. Nếu dùng runner YAML, tạo config riêng với hash Siamese mới từ summary, không sửa cấu hình baseline.

Val loss hiện tại không so trực tiếp với val loss cũ vì split/triplets và cách lấy trung bình đã thay đổi. So score end-to-end trên cùng GT tự gán; đây là thí nghiệm sửa data/validation nền tảng, không phải ablation đơn tham số chứng minh riêng đóng góp của identity sampling.

## Kiểm chứng đã thực hiện

13 tests CPU thành công, dùng trực tiếp hàm/class trích từ notebook. Bao gồm split không giao identity, ngăn negative cùng identity, pool không giao split, validation cố định và phủ mỗi positive một lần, không tiêu thụ RNG training, từ chối provenance không rõ, bảo vệ artifacts cũ, validation mean theo sample và logging/chọn best epoch bằng mock trainer.

Source parse thành công; notebook không còn outputs/execution metadata cũ. Model/augmentation/train step đã đối chiếu nguyên vẹn. **Chưa chạy tensor training, DataLoader multiprocessing thực, GPU hoặc đo ST-IoU mới.** Chờ kết quả Kaggle để quyết định phần 2.

## Kết quả Kaggle do người dùng cung cấp

Training hoàn tất 15 epoch; checkpoint được chọn tại epoch 2 với validation loss `0.208124` và validation accuracy `0.927438`. Checksum checkpoint là `4a1f438d1920e60129ceb89a153c047c27ce904214fc65e9a51f6d458b52839d`.

Khi thay duy nhất checkpoint Siamese trong NB06, giữ TTA, multiscale reference, fusion `0.4/0.3/0.3` và temporal `max_gap=5`, `min_seg=3`, mean ST-IoU đạt `0.698426`; baseline là `0.700454`. Chênh lệch báo cáo là `-0.002028` (`-0.29%` tương đối). Identity-v1 cải thiện `BlackBox_0` `+0.0345`, `LifeJacket_0` `+0.0096` và `LifeJacket_1` `+0.0869`, nhưng giảm mạnh nhất ở `CardboardBox_0` `-0.0935`.

**Quyết định:** không thay checkpoint baseline mặc định. Kết quả xác nhận sửa leakage/validation về mặt phương pháp nhưng chưa cải thiện objective end-to-end trên sáu video tự gán nhãn. Chi tiết có cấu trúc: [`audit/SIAMESE_IDENTITY_V1_RESULT.json`](audit/SIAMESE_IDENTITY_V1_RESULT.json). Runtime thấp hơn khoảng 4% chỉ là một quan sát từ một lần chạy, không được coi là cải thiện tốc độ.
