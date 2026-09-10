# Anchor-Preserving Spatial Refiner V2 (offline)

Training notebook: [`08-train-spatial-refiner-v2.ipynb`](../08-train-spatial-refiner-v2.ipynb)
Maintained source: [`scripts/train_spatial_refiner_v2.py`](../scripts/train_spatial_refiner_v2.py)
Locked offline A/B: [`06-inference-main.ipynb`](../06-inference-main.ipynb)

## Phạm vi

V2 chỉ sửa tọa độ của bbox đã được production chấp nhận. Nó không thêm/xóa
frame, đổi candidate, đổi matching/fusion, hoặc nối qua camera cut. Đây là offline
A/B; tốc độ được ghi mô tả nhưng không nằm trong promotion gate.

## Khác V1

V1 dự đoán hai góc tuyệt đối và đã cho delta validation âm từ `-0.1358` đến
`-0.2253` trước khi kernel chết. V2 nhận trực tiếp coarse bbox và dự đoán residual
`dx,dy,dlogw,dlogh`. Layer cuối được zero-initialize; epoch 0 phải tái tạo coarse
bbox trong tolerance `1e-7` và trở thành checkpoint baseline. Epoch xấu không thể
thay epoch 0.

## Dữ liệu coarse

Production YOLO chạy một lần trên train frames sampled stride 5. Trong mỗi frame,
candidate và các GT bbox được greedy one-to-one matching theo IoU. Training trộn
70% real coarse proposal với 30% synthetic bbox jitter; held-out validation chỉ
dùng real proposal có IoU ít nhất `0.5`. Public GT không phải input của trainer.
Cache `real_coarse_proposals_v2.json` khóa YOLO SHA256, image size, confidence và
dataset record hash.

## Kiến trúc và loss

- MobileNetV3-small P2/P3 khởi tạo từ Siamese identity-v1;
- cosine reference/search response tại P2 và P3;
- coarse mask, coordinate và signed-edge geometry channels;
- bounded residual: center tối đa `0.5` coarse size, log-scale tối đa `0.7`;
- residual Smooth-L1, GIoU, corner Smooth-L1 và anchor regularization;
- TorchScript contract: `model(reference, search, coarse_box)`.

## Validation và an toàn

Group K-fold theo physical identity; `_0/_1` luôn cùng fold. Gate:

- sample-weighted IoU delta `>= +0.005`;
- ít nhất `ceil(2/3)` held-out identities cải thiện;
- worst identity delta `>= -0.005`;
- checkpoint tốt nhất phải vượt epoch 0 để pass.

Nếu CV fail, không export final TorchScript và NB06 chặn public A/B. Nếu pass,
NB06 vẫn giữ frame support tuyệt đối và fallback về bbox production nếu geometry
không hợp lệ.

## Tài nguyên

Training mặc định batch 32, hai workers không persistent, AMP API mới và explicit
GC/CUDA-cache cleanup sau mỗi fold. History/progress và fold summary được ghi ngay
trong thư mục fold để một kernel failure không làm mất các fold đã hoàn tất.
