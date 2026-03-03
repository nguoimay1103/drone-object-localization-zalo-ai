```markdown
# Object Detection and Localization System from Drone Videos 🚁

**Đồ án môn học:** Thị giác máy tính nâng cao (CS331.Q11)  
**Tên đề tài:** Object Detection and Localization System from Drone Videos - ZALO_AI_CHALLENGE

## 👥 Thông tin nhóm thực hiện
- **Phạm Nguyễn Anh Tuấn** (MSSV: 22521610)

---

## 📂 Cấu trúc thư mục

```text
.
├── 01_data_gen_yolo.ipynb           # Tạo dữ liệu giả lập
├── 02_data_merge_yolo.ipynb         # Gộp dữ liệu training
├── 03_train_yolo.ipynb              # Train model YOLO nhận diện
├── 04_data_prep_matching.ipynb      # Chuẩn bị dữ liệu so khớp
├── 05_train_siamese.ipynb           # Train model so khớp
├── 06_inference_main.ipynb          # Chạy suy luận ra kết quả
├── demo/
│   ├── data_test_demo/              # Dữ liệu chạy thử demo
│   ├── 07_demo_app.py               # Ứng dụng Demo Streamlit
│   ├── siamese_mobilenet_best.pth   # Model so khớp
│   ├── yolo_drone_best.pt           # Model nhận diện
│   └── requirements.txt             # Danh sách thư viện cần thiết
├── Slide.pdf                        # Slide báo cáo
├── Report.docx                      # File báo cáo chi tiết
└── readme.txt                       # File hướng dẫn gốc

```

---

## 💻 Yêu cầu hệ thống & Cài đặt

* **GPU:** Khuyến nghị sử dụng NVIDIA GPU (Tesla T4, RTX 3060 trở lên) để training.
* **RAM:** Tối thiểu 16GB.
* **Môi trường:** Code đã được tối ưu và kiểm thử tốt nhất trên nền tảng Kaggle / Google Colab.

> **⚠️ LƯU Ý QUAN TRỌNG VỀ DỮ LIỆU (DATASET):**
> Do quy định về BẢO MẬT DỮ LIỆU CỦA CUỘC THI, nhóm KHÔNG nộp kèm tập dataset gốc (video/ảnh) trong gói source code này.

---

## 🚀 Quy trình thực thi (Pipeline)

Để tái hiện kết quả, vui lòng chạy các file Jupyter Notebook theo thứ tự sau:

### [BƯỚC 1] Chuẩn bị dữ liệu

1. `01_data_gen_yolo.ipynb`: Sinh dữ liệu tổng hợp (synthetic) để tăng cường tập train.
2. `02_data_merge_yolo.ipynb`: Gộp dữ liệu gốc và dữ liệu tổng hợp thành định dạng chuẩn cho YOLO.

### [BƯỚC 2] Huấn luyện model nhận diện

3. `03_train_yolo.ipynb`: Fine-tune YOLO-Drone trên tập dữ liệu đã merge. Có thể thay đổi hoặc load dữ liệu từ file JSON có sẵn.
* **Output quan trọng:** file trọng số `best.pt`.
* **Mục đích:** Đánh giá mAP và tốc độ suy luận của mô hình YOLO.



### [BƯỚC 3] Huấn luyện model so khớp (Matching)

4. `04_data_prep_matching.ipynb`: Sử dụng model YOLO (từ bước 3) để cắt vật thể, tạo bộ dữ liệu Triplet (Anchor - Positive - Negative).
5. `05_train_siamese.ipynb`: Train mạng Siamese Network (sử dụng Backbone MobileNetV3).
* **Output quan trọng:** file trọng số `siamese_mobilenet_best.pth`.



### [BƯỚC 4] Suy luận tổng hợp

6. `06_inference_main.ipynb`: Kết hợp Detection + Re-Identification để chạy trên tập Test. Xuất file kết quả cuối cùng dưới dạng JSON/CSV.

---

## 🌐 Hướng dẫn chạy Demo App (Streamlit)

Nhóm đã xây dựng một giao diện Web App để demo nhanh kết quả trực quan.

### Bước 1: Chuẩn bị Model

Đảm bảo 2 file trọng số sau đây đang nằm trong thư mục `demo/`:

* `yolo_drone_best.pt` *(Lấy từ output folder sau khi chạy xong file 03)*
* `siamese_mobilenet_best.pth` *(Lấy từ output folder sau khi chạy xong file 05)*

### Bước 2: Chạy lệnh khởi động

Mở terminal, di chuyển vào thư mục `demo/` và chạy các lệnh sau:

```bash
pip install -r requirements.txt
streamlit run 07_demo_app.py

```

### Bước 3: Sử dụng Demo

* Truy cập vào link hiển thị trên Terminal (thường là `http://localhost:8501`).
* Upload **1 Video Drone** và **1 đến 3 Ảnh đối tượng** cần tìm (có thể lấy dữ liệu mẫu từ thư mục `demo/data_test_demo`).
* Nhấn nút **"Chạy Demo"** và xem kết quả theo dõi đối tượng trên video.

---

*Trân trọng cảm ơn Quý Thầy/Cô đã xem xét đồ án của nhóm!*

```

Làm xong bước này, bạn gõ 3 lệnh Git để đẩy lên là dự án của bạn trên GitHub sẽ cực kỳ chuẩn chỉnh! Có bước nào bạn thấy vướng mắc nữa không?

```
