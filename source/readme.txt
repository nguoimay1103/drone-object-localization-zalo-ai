================================================================================
                       THÔNG TIN ĐỒ ÁN MÔN HỌC
================================================================================

TÊN ĐỀ TÀI: Object Detection and Localization System from Drone Videos
MÔN HỌC:    THỊ GIÁC MÁY TÍNH NÂNG CAO
LỚP:        CS331.Q11
--------------------------------------------------------------------------------
THÔNG TIN NHÓM THỰC HIỆN:
1. Phạm Nguyễn Anh Tuấn MSSV: 22521610

================================================================================
                       CẤU TRÚC THƯ MỤC NỘP BÀI
================================================================================

├── source/                          <-- Chứa toàn bộ mã nguồn
│   ├── readme.txt                       <-- File hướng dẫn
│   ├── 01_data_gen_yolo.ipynb       (Tạo dữ liệu giả lập)
│   ├── 02_data_merge_yolo.ipynb     (Gộp dữ liệu training)
│   ├── 03_train_yolo.ipynb          (Train model YOLO nhận diện)
│   ├── 04_data_prep_matching.ipynb  (Chuẩn bị dữ liệu so khớp)
│   ├── 05_train_siamese.ipynb       (Train model so khớp)
│   ├── 06_inference_main.ipynb      (Chạy suy luận ra kết quả)
│   ├── demo
	└── data_test_demo			(dữ lieu chạy thử demo)
	└── 07_demo_app.py           		(Ứng dụng Demo Streamlit)
	└── siamese_mobilenet_best.pth 		(model so khớp)
	└── yolo_drone_best.pt			(model nhận diện)
   	└── requirements.txt             	(Danh sách thư viện cần thiết)
│   └── Slide.pdf
│   └── Report.docx


================================================================================
                       YÊU CẦU HỆ THỐNG & CÀI ĐẶT
================================================================================

1. YÊU CẦU PHẦN CỨNG:
   - GPU: Khuyến nghị NVIDIA GPU (Tesla T4, RTX 3060 trở lên) để training.
   - RAM: Tối thiểu 16GB.
   - Code đã được tối ưu và kiểm thử tốt nhất trên môi trường Kaggle/Google Colab.

================================================================================
                       HƯỚNG DẪN VỀ DỮ LIỆU (DATASET)
================================================================================

⚠️ LƯU Ý QUAN TRỌNG:
Do quy định về BẢO MẬT DỮ LIỆU CỦA CUỘC THI, nhóm KHÔNG nộp kèm tập dataset gốc (video/ảnh) trong gói source code này.
================================================================================
                       QUY TRÌNH THỰC THI (PIPELINE)
================================================================================

Để tái hiện kết quả, vui lòng chạy các file trong thư mục 'source/' theo thứ tự đánh số:

[BƯỚC 1: CHUẨN BỊ DỮ LIỆU]
   1. 01_data_gen_yolo.ipynb
      - Sinh dữ liệu tổng hợp (synthetic) để tăng cường tập train.
   2. 02_data_merge_yolo.ipynb
      - Gộp dữ liệu gốc và dữ liệu tổng hợp thành định dạng chuẩn YOLO.

[BƯỚC 2: HUẤN LUYỆN MODEL NHẬN DIỆN]
   3. 03_train_yolo.ipynb
      - Fine-tune YOLO-Drone trên tập dữ liệu đã merge.
      - Output quan trọng: file trọng số `best.pt`.
đổi hoặc load dữ liệu từ file JSON có sẵn.
       - Mục đích: So sánh mAP và tốc độ suy luận với YOLO.
[BƯỚC 3: HUẤN LUYỆN MODEL SO KHỚP (MATCHING)]
   4. 04_data_prep_matching.ipynb
      - Sử dụng model YOLO (từ bước 3) để cắt vật thể, tạo bộ dữ liệu Triplet (Anchor-Positive-Negative).
   5. 05_train_siamese.ipynb
      - Train mạng Siamese Network (Backbone MobileNetV3).
      - Output quan trọng: file trọng số `siamese_mobilenet_best.pth`.

[BƯỚC 4: SUY LUẬN TỔNG HỢP]
   6. 06_inference_main.ipynb
      - Kết hợp Detection + Re-Identification để chạy trên tập Test.
      - Xuất file kết quả JSON/CSV.

================================================================================
                       HƯỚNG DẪN CHẠY DEMO APP
================================================================================

Nhóm đã xây dựng một giao diện Web App (Streamlit) để demo nhanh kết quả.

BƯỚC 1: CHUẨN BỊ MODEL
   Đảm bảo 2 file trọng số sau đây đang nằm trong thư mục 'source/demo':
   1. `yolo_drone_best.pt` (Lấy từ output folder sau khi chạy file 03)
   2. `siamese_mobilenet_best.pth` (Lấy từ output folder sau khi chạy file 05)

BƯỚC 2: CHẠY LỆNH
   Mở terminal tại thư mục 'source/demo' và gõ lệnh:
   pip install -r requirements.txt
   streamlit run 07_demo_app.py

BƯỚC 3: SỬ DỤNG
   - Truy cập link hiển thị (thường là http://localhost:8501).
   - Upload 1 Video Drone + 1 đến 3 Ảnh đối tượng cần tìm (từ đường dẫn Source/demo/data_test_demo)
   - Nhấn "Chạy Demo" và xem kết quả trực quan.

================================================================================
Trân trọng cảm ơn Quý Thầy/Cô đã xem xét đồ án của nhóm!