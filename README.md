# OmniSense — Hệ Thống Nhận Diện Khuôn Mặt (Web Application)

Ứng dụng Web nhận diện khuôn mặt theo thời gian thực tích hợp camera, sử dụng **InsightFace**, **ONNX Runtime (CPU)** và **Flask Backend**.

---

## Yêu cầu hệ thống

| Thành phần | Yêu cầu tối thiểu |
|---|---|
| **Python** | 3.9 trở lên (khuyên dùng Python 3.10) |
| **RAM** | 4 GB (khuyến nghị 8 GB) |
| **CPU** | 4 nhân trở lên |
| **Webcam** | Bất kỳ webcam USB hoặc tích hợp |
| **OS** | Windows 10/11, Ubuntu 20.04+, macOS 12+ |

> Hệ thống chạy mặc định trên **CPU** — không bắt buộc có GPU.

---

## Cấu trúc thư mục

```
OmniSense/
├── face_recognition_pipeline.py   ← Core engine nhận diện khuôn mặt
├── benchmark.py                   ← Công cụ kiểm tra hiệu suất
├── requirements.txt               ← Thư viện pipeline chính
├── environment.yml                ← File cấu hình môi trường Conda
├── README.md                      ← File hướng dẫn này
├── backend/                       ← Backend Server (Flask API)
│   ├── app.py                     ← Server chính
│   ├── settings.json              ← File lưu cài đặt hệ thống
│   └── requirements.txt           ← Dependencies backend (Flask, CORS)
├── web/                           ← Giao diện Web (Frontend)
│   ├── index.html                 ← HTML giao diện
│   ├── index.css                  ← Stylesheet
│   └── app.js                     ← Logic điều khiển UI
└── face_recognition/
    ├── database/                  ← Dữ liệu thư mục khuôn mặt đã đăng ký
    └── metadata.json              ← Thông tin thông tin người dùng
```

---

## Hướng dẫn cài đặt & chạy ứng dụng Web

### Bước 1 — Tạo môi trường Conda

Mở **Anaconda Prompt** (hoặc Terminal đã kích hoạt Conda) và chạy:

```bash
# Tạo môi trường mới tên "smartcam" với Python 3.10
conda create -n smartcam python=3.10 -y

# Kích hoạt môi trường
conda activate smartcam
```

> Từ đây, mọi lệnh `pip` đều chạy trong môi trường `smartcam` độc lập.

### Bước 2 — Cài đặt thư viện

Di chuyển vào thư mục dự án và cài đặt các thư viện cần thiết:

```bash
# Di chuyển vào thư mục dự án OmniSense
cd d:\Project\OmniSense

# Cài đặt thư viện chính (bao gồm InsightFace, OpenCV, ONNX Runtime, Flask, Flask-CORS...)
pip install -r requirements.txt

# Cài đặt / cập nhật các thư viện Backend Flask (nếu cài riêng)
pip install -r backend/requirements.txt
```


> ⏳ **Lưu ý:** Lần đầu khởi chạy, ứng dụng sẽ tự động tải model InsightFace (`buffalo_sc` ~100 MB).

### Bước 3 — Chạy Backend Server

Chạy file `app.py` để khởi động máy chủ Flask Backend:

```bash
python backend/app.py
```

Khi máy chủ khởi chạy thành công, log sẽ hiển thị:
```
[INFO] ============================================================
[INFO]   OMNISENSE BACKEND API
[INFO]   Web UI: http://localhost:5000
[INFO]   Database: d:\Coding\OmniSense\face_recognition\database
[INFO] ============================================================
[INFO] Model sẵn sàng! Database: X người
```

### Bước 4 — Mở giao diện Web

Mở trình duyệt web bất kỳ (Chrome, Edge, Firefox, Brave...) và truy cập:

👉 **http://localhost:5000** (hoặc **http://127.0.0.1:5000**)

> 💡 Backend sẽ tự động:
> - Serve giao diện Web tại trang chủ (`/`).
> - Khởi tạo mô hình InsightFace và nạp dữ liệu khuôn mặt trong nền.
> - Xử lý stream webcam và nhận diện theo thời gian thực.

---

## 🌐 Tính năng chính trên Giao diện Web

OmniSense cung cấp giao diện Web trực quan với 4 nhóm tính năng chính:

| Chức năng | Mô tả chi tiết |
|---|---|
| 📷 **Camera nhận diện** | Stream webcam hiển thị khung nhận diện & tên người dùng thời gian thực (MJPEG Stream) |
| 📝 **Đăng ký khuôn mặt** | Thêm người mới trực tiếp trên Web với họ tên, chức vụ, giới tính và tải lên ảnh khuôn mặt |
| 📋 **Danh sách đã đăng ký** | Quản lý danh sách người dùng, xem ảnh cá nhân, tìm kiếm, chỉnh sửa thông tin hoặc xóa |
| ⚙️ **Cài đặt model** | Tinh chỉnh trực tiếp Model (`buffalo_sc` / `buffalo_l`), Threshold, số CPU threads, Camera Index |

---

## Kiến trúc hệ thống & API Endpoints

### Sơ đồ hoạt động

```
Trình duyệt Web (Port 5000)  ←→  Flask Backend Server  ←→  InsightFace + ONNX Runtime Engine
         │                              │                                 │
         ├─ Đăng ký ─────────→  POST /api/users    ──→  Lưu ảnh & metadata vào database/
         ├─ Danh sách ───────→  GET  /api/users    ──→  Đọc danh sách từ database/
         ├─ Camera Stream ───→  GET  /api/video_feed ─→ MJPEG Stream khung hình nhận diện
         └─ Cài đặt ─────────→  POST /api/settings ──→  Cập nhật cấu hình settings.json
```

### API Endpoints

| Method | Endpoint | Mô tả |
|---|---|---|
| `GET` | `/api/status` | Kiểm tra trạng thái model, database, camera, FPS |
| `GET` | `/api/users` | Lấy danh sách người đã đăng ký |
| `POST` | `/api/users` | Đăng ký người dùng mới (gửi `multipart/form-data`) |
| `PUT` | `/api/users/<id>` | Cập nhật thông tin người dùng |
| `DELETE` | `/api/users/<id>` | Xóa người dùng và dữ liệu ảnh |
| `POST` | `/api/users/<user_id>/photos` | Thêm ảnh mới cho người dùng |
| `DELETE` | `/api/users/<user_id>/photos/<photo_name>` | Xóa 1 ảnh của người dùng |
| `GET` | `/api/settings` | Đọc cấu hình hiện tại |
| `POST` | `/api/settings` | Cập nhật cấu hình hệ thống |
| `POST` | `/api/recognize` | Nhận diện khuôn mặt từ ảnh gửi lên |
| `POST` | `/api/camera/start` | Khởi động camera nhận diện |
| `POST` | `/api/camera/stop` | Dừng camera nhận diện |
| `GET` | `/api/video_feed` | Stream video MJPEG thời gian thực |

---

## Tinh chỉnh cấu hình

Bạn có thể chỉnh sửa cấu hình trực tiếp trong phần **Cài đặt** trên Web UI hoặc thay đổi file `backend/settings.json`:

```json
{
  "model": "buffalo_sc",
  "threshold": 0.40,
  "threads": 8,
  "det_size": 640,
  "camera_index": 0,
  "cuda": false
}
```

### Gợi ý Ngưỡng nhận diện (Cosine Threshold)

| Threshold | Đặc điểm |
|---|---|
| `0.35` | Rất chặt — hạn chế tối đa nhận nhầm, nhưng có thể bỏ sót khi góc mặt khó |
| `0.40` | **Khuyến nghị Mặc định** cho model `buffalo_sc` |
| `0.45` | Nhận dễ hơn — phù hợp khi điều kiện ánh sáng yếu |
| `0.50` | Khuyên dùng khi chuyển sang model `buffalo_l` |

### So sánh các Model

| Model | Tốc độ | Độ chính xác | Kích thước model |
|---|---|---|---|
| `buffalo_sc` | ⚡⚡⚡ Rất nhanh | ⭐⭐⭐ Tốt | ~100 MB |
| `buffalo_l`  | ⚡⚡ Nhanh | ⭐⭐⭐⭐ Rất tốt | ~300 MB |

---

## Xử lý sự cố thường gặp

### Lỗi `No module named 'flask'` hoặc `No module named 'insightface'`
Đảm bảo bạn đã kích hoạt môi trường Conda và cài đặt đủ requirements:
```bash
conda activate smartcam
pip install -r requirements.txt
pip install -r backend/requirements.txt
```

### `Could not open camera index 0`
- Kiểm tra webcam có bị ứng dụng khác (Zoom, Teams, Skype...) chiếm dụng hay không.
- Thử đổi `camera_index` thành `1` hoặc `2` trong phần Cài đặt trên Web UI.

### Tốc độ nhận diện chậm / FPS thấp
1. Mở phần Cài đặt trên Web UI, chỉnh số **CPU Threads** bằng với số nhân thực tế của CPU.
2. Sử dụng model nhẹ `buffalo_sc`.
3. Giảm kích thước phát hiện `det_size` xuống `320` hoặc `480`.

---

## Nâng cấp tăng tốc với GPU (Tùy chọn)

Nếu hệ thống có GPU NVIDIA:

```bash
conda activate smartcam

# Cài CUDA toolkit & cuDNN tự động qua Conda
conda install -c conda-forge cudatoolkit=11.8 cudnn -y

# Chuyển đổi onnxruntime từ CPU sang GPU
pip uninstall onnxruntime -y
pip install onnxruntime-gpu
```

Sau đó trong phần Cài đặt trên Web UI, bật tùy chọn **CUDA Acceleration**.

---

## Giấy phép & Thư viện sử dụng

- [InsightFace](https://github.com/deepinsight/insightface) — MIT License
- [Flask](https://flask.palletsprojects.com/) — BSD License
- [ONNX Runtime](https://github.com/microsoft/onnxruntime) — MIT License
- [OpenCV](https://opencv.org/) — Apache 2.0 License
