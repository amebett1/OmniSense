# Thư mục ảnh đăng ký (Registered Faces)

Đặt ảnh của từng người vào đây theo cấu trúc:

```
database/
├── nguyen_van_a/
│   ├── photo1.jpg
│   └── photo2.png
├── tran_thi_b/
│   └── photo1.jpg
└── le_van_c/
    ├── img1.jpg
    └── img2.jpg
```

- Tên thư mục con sẽ được dùng làm **ID / Tên người** hiển thị trên màn hình.
- Hỗ trợ định dạng: `.jpg`, `.jpeg`, `.png`, `.bmp`.
- Nên dùng ít nhất **2-3 ảnh** mỗi người để tăng độ chính xác (embedding trung bình).
