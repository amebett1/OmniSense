# 🎯 Face Recognition Pipeline

Pipeline nhận diện khuôn mặt theo thời gian thực sử dụng **InsightFace** + **ONNX Runtime (CPU)**.

---

## 📋 Yêu cầu hệ thống

| Thành phần | Yêu cầu tối thiểu |
|---|---|
| **Python** | 3.9 trở lên |
| **RAM** | 4 GB (khuyến nghị 8 GB) |
| **CPU** | 4 nhân trở lên (tối ưu nhất với 8 nhân) |
| **Webcam** | Bất kỳ webcam USB hoặc tích hợp |
| **OS** | Windows 10/11, Ubuntu 20.04+, macOS 12+ |

> ⚠️ Pipeline này chạy hoàn toàn trên **CPU** — không cần GPU.

---

## 📁 Cấu trúc thư mục

```
smartCam/
├── face_recognition_pipeline.py   ← Pipeline chính
├── benchmark.py                   ← Công cụ kiểm tra hiệu suất
├── requirements.txt               ← Danh sách thư viện
├── README.md                      ← File này
└── face_recognition/
    └── database/
        ├── README.md              ← Hướng dẫn thêm ảnh
        ├── nguyen_van_a/          ← Mỗi thư mục = 1 người
        │   ├── anh1.jpg
        │   └── anh2.jpg
        └── tran_thi_b/
            └── anh1.png
```

---

## 🚀 Hướng dẫn cài đặt & chạy

### Bước 1 — Tạo môi trường Conda

Mở **Anaconda Prompt** (hoặc terminal đã kích hoạt conda) và chạy:

```bash
# Tạo môi trường mới tên "smartcam" với Python 3.10
conda create -n smartcam python=3.10 -y

# Kích hoạt môi trường
conda activate smartcam
```

> 💡 Từ đây, mọi lệnh `pip` đều chạy trong môi trường `smartcam` — không ảnh hưởng hệ thống.

### Bước 2 — Cài đặt thư viện

```bash
# Di chuyển vào thư mục dự án
cd d:\Project\smartCam

# Cài tất cả dependencies
pip install -r requirements.txt
```

> ⏳ Lần đầu chạy sẽ mất vài phút để tải model InsightFace (~300 MB).

### Bước 3 — Thêm ảnh vào database

Tạo **một thư mục con cho mỗi người** bên trong `face_recognition/database/`:

```
face_recognition/database/
├── nguyen_van_a/       ← Tên thư mục = tên hiển thị trên màn hình
│   ├── anh1.jpg        ← Nên có 2–5 ảnh/người
│   └── anh2.jpg
├── tran_thi_b/
│   └── photo.jpg
└── le_van_c/
    ├── front.jpg
    └── side.jpg
```

**Lưu ý khi chụp ảnh đăng ký:**
- ✅ Ảnh rõ nét, đủ sáng, khuôn mặt nhìn thẳng vào camera
- ✅ Mỗi người nên có **2–5 ảnh** từ các góc độ / ánh sáng khác nhau
- ✅ Định dạng hỗ trợ: `.jpg`, `.jpeg`, `.png`, `.bmp`
- ❌ Tránh ảnh mờ, che mặt, đeo khẩu trang

### Bước 4 — (Tuỳ chọn) Kiểm tra cấu hình

```bash
python benchmark.py
```

Kết quả mẫu:
```
[21:00:01] ONNX Runtime Execution Providers:
[21:00:01]   - CPUExecutionProvider
[21:00:01] ONNX Runtime version: 1.18.0
[21:00:02] Batch  (matrix @):   0.0021 ms/query
[21:00:02] Loop   (python for): 0.2847 ms/query
[21:00:02] Speedup:             135.6x
```

### Bước 5 — Chạy pipeline

```bash
python face_recognition_pipeline.py
```

Lần đầu chạy, InsightFace sẽ tự động tải model về `C:\Users\<tên_user>\.insightface\models\buffalo_sc\` (Windows).

---

## 🎮 Phím điều khiển

| Phím | Chức năng |
|---|---|
| `Q` hoặc `ESC` | Thoát chương trình |
| `S` | Chụp screenshot (lưu vào thư mục `screenshots/`) |

---

## 🖥️ Giao diện hiển thị

```
┌─────────────────────────────────────┐
│  Faces: 2                           │
│                                     │
│  ┌──────────────┐                   │
│  │ nguyen_van_a │  (bounding box    │
│  │    (0.87)    │   màu xanh lá)    │
│  └──────────────┘                   │
│                                     │
│  ┌──────────────┐                   │
│  │   Unknown    │  (bounding box    │
│  │    (0.31)    │   màu đỏ)         │
│  └──────────────┘                   │
│                                     │
│  FPS: 24.3                          │
│  Inference: 41.2ms                  │
│  DB: 3 nguoi                        │
│  Threshold: 0.4                     │
│  Model: buffalo_sc                  │
└─────────────────────────────────────┘
```

- 🟢 **Xanh lá** = Nhận diện thành công
- 🔴 **Đỏ** = Không nhận ra (Unknown)
- Số trong ngoặc = điểm Cosine Similarity (0.0 – 1.0)

---

## ⚙️ Tinh chỉnh cấu hình

Mở file `face_recognition_pipeline.py`, chỉnh các biến ở **SECTION 1**:

```python
# Chọn model: "buffalo_sc" (nhanh) hoặc "buffalo_l" (chính xác hơn)
MODEL_NAME = "buffalo_sc"

# Số CPU threads cho ONNX Runtime (nên = số nhân logic của CPU)
INTRA_OP_NUM_THREADS = 8

# Ngưỡng nhận diện: tăng → chặt hơn, giảm → lỏng hơn
COSINE_THRESHOLD = 0.4

# Index webcam: 0 = mặc định, 1 = webcam thứ 2, ...
CAMERA_INDEX = 0
```

### Chọn threshold phù hợp

| Threshold | Hành vi |
|---|---|
| `0.35` | Rất chặt — ít nhận nhầm, dễ bỏ sót |
| `0.40` | **Khuyến nghị** cho `buffalo_sc` |
| `0.45` | Lỏng hơn — nhận dễ hơn, có thể nhầm |
| `0.50` | **Khuyến nghị** cho `buffalo_l` |

### So sánh model

| Model | Tốc độ | Độ chính xác | Kích thước |
|---|---|---|---|
| `buffalo_sc` | ⚡⚡⚡ Rất nhanh | ⭐⭐⭐ Tốt | ~100 MB |
| `buffalo_l`  | ⚡⚡ Nhanh | ⭐⭐⭐⭐ Rất tốt | ~300 MB |

---

## 🔧 Xử lý sự cố

### ❌ `No module named 'insightface'`
```bash
pip install insightface
```

### ❌ `Could not open camera index 0`
- Kiểm tra webcam có đang được dùng bởi ứng dụng khác không (Zoom, Teams, …)
- Thử đổi `CAMERA_INDEX = 1` hoặc `2`

### ❌ Model tải về chậm / lỗi mạng
InsightFace tải model về `~/.insightface/models/`. Nếu mạng chậm, thử:
```bash
# Đặt timeout lớn hơn (Linux/macOS)
export INSIGHTFACE_DOWNLOAD_TIMEOUT=300
python face_recognition_pipeline.py
```

### ❌ Nhận diện sai hoặc luôn ra "Unknown"
1. Kiểm tra ảnh trong database có rõ nét, đủ sáng không
2. Tăng số ảnh mỗi người lên 3–5 ảnh
3. Điều chỉnh `COSINE_THRESHOLD` xuống thấp hơn (ví dụ `0.35`)
4. Thử dùng `buffalo_l` thay vì `buffalo_sc`

### ❌ FPS thấp (dưới 10 FPS)
1. Tăng `INTRA_OP_NUM_THREADS` bằng số nhân CPU thực tế
2. Chuyển sang model `buffalo_sc` nếu đang dùng `buffalo_l`
3. Giảm `DET_SIZE` từ `(640, 640)` xuống `(320, 320)`

---

## 📦 Nâng cấp lên GPU (tuỳ chọn)

Nếu có GPU NVIDIA, dùng Conda để cài CUDA toolkit dễ dàng hơn:

```bash
# Kích hoạt môi trường
conda activate smartcam

# Cài CUDA toolkit + cuDNN qua conda (không cần cài CUDA thủ công)
conda install -c conda-forge cudatoolkit=11.8 cudnn -y

# Gỡ onnxruntime CPU, cài bản GPU
pip uninstall onnxruntime -y
pip install onnxruntime-gpu
```

Sau đó sửa trong `face_recognition_pipeline.py`:
```python
# Dòng providers= trong init_model()
providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
```

> ✅ Ưu điểm của Conda: tự động cài đúng phiên bản CUDA/cuDNN tương thích — không cần cài thủ công từ NVIDIA website.

---

## 📄 Giấy phép

Dự án sử dụng:
- [InsightFace](https://github.com/deepinsight/insightface) — MIT License
- [ONNX Runtime](https://github.com/microsoft/onnxruntime) — MIT License
- [OpenCV](https://opencv.org/) — Apache 2.0 License
