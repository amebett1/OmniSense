# OmniSense

OmniSense là ứng dụng web nhận diện khuôn mặt theo thời gian thực, chạy bằng Flask backend, InsightFace và ONNX Runtime. Ứng dụng phục vụ trực tiếp giao diện web từ backend, hỗ trợ camera livestream, đăng ký khuôn mặt, quản lý danh sách người dùng, cấu hình model và trợ lý AI bằng chat/giọng nói.

## Tính năng chính

| Chức năng | Mô tả |
|---|---|
| Camera nhận diện | Stream webcam MJPEG theo thời gian thực, hiển thị FPS, số khuôn mặt và kết quả nhận diện |
| Đăng ký khuôn mặt | Thêm người dùng mới với họ tên, chức vụ, giới tính và nhiều ảnh khuôn mặt |
| Danh sách đã đăng ký | Xem, tìm kiếm, cập nhật và xoá người dùng cùng ảnh của họ |
| Cài đặt hệ thống | Tùy chỉnh model, threshold, số CPU threads, kích thước phát hiện và camera index |
| Trợ lý AI | Chat bằng văn bản hoặc giọng nói trong giao diện camera; backend có endpoint tạo câu chào và trả lời hội thoại |

## Yêu cầu hệ thống

| Thành phần | Khuyến nghị |
|---|---|
| Python | 3.9 trở lên, khuyên dùng 3.10 |
| RAM | Tối thiểu 4 GB, khuyên dùng 8 GB |
| CPU | 4 nhân trở lên |
| Webcam | Webcam USB hoặc tích hợp |
| OS | Windows 10/11, Ubuntu 20.04+, macOS 12+ |

Hệ thống mặc định chạy trên CPU, không bắt buộc GPU.

## Cấu trúc thư mục

```text
OmniSense/
├── benchmark.py
├── environment.yml
├── face_recognition_pipeline.py
├── requirements.txt
├── backend/
│   ├── app.py
│   ├── requirements.txt
│   └── settings.json
├── face_recognition/
│   ├── database/
│   └── metadata.json
└── web/
    ├── app.js
    ├── index.css
    └── index.html
```

## Cài đặt

### 1. Tạo môi trường Python

```bash
conda create -n smartcam python=3.10 -y
conda activate smartcam
```

### 2. Cài dependencies

```bash
cd d:\Coding\OmniSense
pip install -r requirements.txt
pip install -r backend/requirements.txt
```

Lần chạy đầu tiên, InsightFace sẽ tự tải model mặc định (`buffalo_sc`) nếu chưa có cache cục bộ.

## Chạy ứng dụng

Khởi động backend từ thư mục gốc dự án:

```bash
python backend/app.py
```

Sau khi chạy thành công, mở trình duyệt tại:

```text
http://localhost:5000
```

Backend sẽ tự phục vụ toàn bộ giao diện web từ thư mục `web/`, khởi tạo model trong nền và mở các API cho camera, đăng ký khuôn mặt, cài đặt và trợ lý AI.

## API chính

| Method | Endpoint | Mô tả |
|---|---|---|
| GET | `/api/status` | Trạng thái model, database, camera và FPS |
| GET | `/api/users` | Lấy danh sách người đã đăng ký |
| POST | `/api/users` | Đăng ký người dùng mới |
| PUT | `/api/users/<id>` | Cập nhật thông tin người dùng |
| DELETE | `/api/users/<id>` | Xoá người dùng |
| GET | `/api/users/<user_id>/photo/<photo_name>` | Lấy ảnh của người dùng |
| POST | `/api/users/<user_id>/photos` | Thêm ảnh cho người dùng |
| DELETE | `/api/users/<user_id>/photos/<photo_name>` | Xoá một ảnh của người dùng |
| GET | `/api/settings` | Đọc cấu hình hiện tại |
| POST | `/api/settings` | Cập nhật cấu hình |
| POST | `/api/recognize` | Nhận diện khuôn mặt từ ảnh |
| POST | `/api/camera/start` | Khởi động camera |
| POST | `/api/camera/stop` | Dừng camera |
| GET | `/api/camera/status` | Trạng thái camera và kết quả nhận diện gần nhất |
| GET | `/api/video_feed` | Stream MJPEG thời gian thực |
| POST | `/api/chat` | Trả lời hội thoại từ trợ lý AI |
| POST/GET | `/api/greet` | Sinh câu chào theo người được nhận diện |

## Cấu hình

Cấu hình được lưu trong `backend/settings.json` và cũng có thể chỉnh trực tiếp trong giao diện web:

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

Gợi ý nhanh:

| Giá trị | Gợi ý |
|---|---|
| `threshold = 0.40` | Mức mặc định phù hợp với `buffalo_sc` |
| `threshold = 0.45 - 0.50` | Dễ nhận hơn khi ảnh khó hoặc ánh sáng yếu |
| `det_size = 320 - 480` | Giảm tải khi FPS thấp |
| `threads` | Nên để gần với số nhân CPU thực tế |

## Khắc phục sự cố

Nếu gặp lỗi thiếu thư viện như `flask`, `insightface` hoặc `cv2`, hãy kiểm tra lại môi trường đang kích hoạt và cài lại dependencies.

Nếu camera không mở được, thử đổi `camera_index` trong phần Cài đặt hoặc kiểm tra ứng dụng khác đang chiếm webcam.

Nếu FPS thấp, hãy dùng model nhẹ `buffalo_sc`, giảm `det_size` và chỉnh `threads` phù hợp CPU.

## Ghi chú

- Giao diện web chạy cùng backend, không cần server frontend riêng.
- Tính năng trợ lý AI trong trình duyệt phụ thuộc vào hỗ trợ Web Speech API của trình duyệt và cấu hình backend tương ứng.

## Giấy phép và thư viện

- [InsightFace](https://github.com/deepinsight/insightface)
- [Flask](https://flask.palletsprojects.com/)
- [ONNX Runtime](https://github.com/microsoft/onnxruntime)
- [OpenCV](https://opencv.org/)
