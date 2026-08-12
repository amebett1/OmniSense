# OmniSense

OmniSense là ứng dụng web đa năng kết hợp nhận diện khuôn mặt theo thời gian thực (chạy bằng Flask, InsightFace, ONNX Runtime) và Trợ lý AI hỏi đáp tri thức RAG (Retrieval-Augmented Generation) thông minh hỗ trợ tài liệu Tiếng Việt. 

Ứng dụng phục vụ trực tiếp giao diện web từ backend, hỗ trợ camera livestream, đăng ký phân quyền khuôn mặt (Sinh viên, Giảng viên, Cựu sinh viên, Học viên cao học, Nghiên cứu sinh,...), quản lý tài liệu RAG và trợ lý AI giao tiếp giọng nói/văn bản.

---

## Tính năng chính

| Chức năng | Mô tả |
|---|---|
| **Camera nhận diện** | Stream webcam MJPEG thời gian thực, hiển thị FPS, số khuôn mặt và nhận diện vai trò (Học viên cao học, Nghiên cứu sinh, Sinh viên, Giảng viên,...) |
| **Đăng ký khuôn mặt** | Thêm người dùng mới với Họ tên, Vai trò, Giới tính (xử lý xưng hô Anh/Chị/Em tự động) và nhiều ảnh đại diện |
| **Trợ lý AI RAG thông minh** | Trả lời tự nhiên hoặc trích xuất tri thức từ tài liệu. Tự động hiển thị nhãn **`[RAG]`** khi tra cứu được tài liệu hoặc **`[LLM]`** khi giao tiếp xã giao |
| **Quản lý Tài liệu RAG** | Cho phép người dùng upload/xóa tài liệu (`.pdf`, `.docx`, `.txt`) ngay trong mục Cài đặt. Tự động chia nhỏ (Chunking) và lưu vào Vector Database |
| **Bóc chữ OCR Thuần Python** | Tích hợp `RapidOCR` (ONNX Engine) bóc chữ Tiếng Việt từ file PDF ảnh scan **mà không cần cài ứng dụng desktop bên ngoài (Không cần Tesseract `.exe` hay Poppler)** |
| **Tự động nhận diện phần cứng** | Tự động phát hiện GPU (NVIDIA CUDA) để tăng tốc 10x hoặc tự động chuyển sang CPU mượt mà trên mọi dòng máy |
| **Cài đặt & Danh sách** | Quản lý người dùng, tùy chỉnh mô hình, ngưỡng nhận diện, số CPU threads và xem log trực tiếp |

---

## Yêu cầu hệ thống

| Thành phần | Khuyến nghị |
|---|---|
| **Python** | 3.9 trở lên, khuyên dùng 3.10 hoặc 3.11 |
| **RAM** | Tối thiểu 4 GB, khuyên dùng 8 GB |
| **CPU** | 4 nhân trở lên |
| **GPU** | (Tùy chọn) NVIDIA GPU hỗ trợ CUDA (Hệ thống tự động phát hiện & tối ưu) |
| **Webcam** | Webcam USB hoặc webcam tích hợp |
| **OS** | Windows 10/11, Ubuntu 20.04+, macOS 12+ |

---

## Cấu trúc thư mục

```text
OmniSense/
├── benchmark.py
├── requirements.txt
├── backend/
│   ├── app.py                     # Flask Web API Server chính
│   ├── rag_pipeline.py            # Component-based RAG Pipeline (Embedding, OCR, ChromaDB)
│   ├── requirements.txt           # Dependencies cho backend
│   └── settings.json              # Cấu hình hệ thống
├── data/
│   ├── chroma_db/                 # Cơ sở dữ liệu Vector (ChromaDB Persistent Store)
│   └── rag_docs/                  # Thư mục chứa tài liệu tri thức RAG
├── face_recognition/
│   ├── database/                  # Ảnh khuôn mặt đã đăng ký
│   └── metadata.json              # Metadata phân quyền & giới tính người dùng
└── web/
    ├── app.js                     # Script điều khiển UI & Trợ lý giọng nói
    ├── index.css                  # Thiết kế giao diện Modern UI
    └── index.html                 # Giao diện điều khiển chính
```

---

## Cài đặt

### 1. Tạo môi trường Python

```bash
conda create -n omnisense python=3.10 -y
conda activate omnisense
```

### 2. Cài đặt các thư viện phụ thuộc

```bash
cd OmniSense
pip install -r backend/requirements.txt
```

> **Lưu ý:** Lần chạy đầu tiên, hệ thống sẽ tự động tải mô hình nhận diện khuôn mặt (`buffalo_sc`) và mô hình RAG Embedding (`all-MiniLM-L6-v2`).

---

## Chạy ứng dụng

Khởi động backend từ thư mục gốc dự án:

```bash
python backend/app.py
```

Sau khi khởi động thành công, mở trình duyệt tại:

```text
http://localhost:5000
```

---

## Danh sách API chính

### 1. Nhận diện Khuôn mặt & Người dùng

| Method | Endpoint | Mô tả |
|---|---|---|
| `GET` | `/api/status` | Trạng thái mô hình, database khuôn mặt, camera và FPS |
| `GET` | `/api/users` | Lấy danh sách người dùng đã đăng ký |
| `POST` | `/api/users` | Đăng ký người dùng mới |
| `PUT` | `/api/users/<id>` | Cập nhật thông tin người dùng |
| `DELETE` | `/api/users/<id>` | Xoá người dùng |
| `GET` | `/api/video_feed` | Stream MJPEG thời gian thực từ webcam |

### 2. Trợ lý AI & RAG Tài liệu

| Method | Endpoint | Mô tả |
|---|---|---|
| `POST` | `/api/chat` | Gửi câu hỏi cho Trợ lý AI (Trả về phản hồi + Thẻ nhãn `[RAG]` hoặc `[LLM]`) |
| `POST/GET` | `/api/greet` | Sinh câu chào cá nhân hóa theo vai trò & giới tính người trước camera |
| `GET` | `/api/documents` | Lấy danh sách tài liệu RAG hiện có |
| `POST` | `/api/documents/upload` | Upload tài liệu mới (`.pdf`, `.docx`, `.txt`) và tự động tạo Vector |
| `DELETE` | `/api/documents/<filename>` | Xóa tài liệu khỏi đĩa vật lý và xóa Vector trong ChromaDB |

---

## Cấu hình Hệ thống (`backend/settings.json`)

```json
{
  "model": "buffalo_sc",
  "threshold": 0.4,
  "threads": 8,
  "det_size": 640,
  "camera_index": 0,
  "cuda": false
}
```

---

## Giấy phép và Thư viện phát triển

- [InsightFace](https://github.com/deepinsight/insightface) — Mô hình nhận diện khuôn mặt
- [ChromaDB](https://www.trychroma.com/) — Cơ sở dữ liệu Vector lưu trữ RAG Chunks
- [SentenceTransformers](https://www.sbert.net/) — Mô hình nhúng ngôn ngữ (Embedding)
- [RapidOCR](https://github.com/RapidAI/RapidOCR) — Mô hình bóc chữ OCR thuần Python (ONNX Engine)
- [Flask Framework](https://flask.palletsprojects.com/) — Web Backend Service
