face_recognition_project/
├── src/                          # Mã nguồn chính
│   ├── preprocessing/            # 🧼 Dev 1: Tiền xử lý ảnh với YOLOv8
│   ├── embeddings/               # 🧠 Dev 2: Trích xuất & so sánh embeddings (DeepFace)
│   ├── detection/                # 👁️ Dev 3: Phát hiện khuôn mặt (YOLOv8)
│   ├── recognition/              # 🧾 Dev 4: Nhận diện khuôn mặt (DeepFace)
│   ├── video_processing/         # 🎥 Dev 5: Xử lý video & phân tích (real-time)
│   ├── gui/                      # 🖥️ Dev 6: Giao diện người dùng (PyQt5)
│   └── config/                   # ⚙️ Cấu hình chung của hệ thống
├── data/                         # 📂 Dữ liệu (không commit lên Git)
│   ├── raw/                      # Ảnh gốc theo từng người
│   ├── processed/                # Ảnh đã cắt & resize (160x160)
│   ├── embeddings/               # Embeddings đã trích xuất (.npz)
│   ├── videos/                   # Video đầu vào
│   └── results/                  # Kết quả: ảnh, video, file CSV phân tích
├── docs/                         # 📄 Tài liệu dự án
│   ├── README.md                 # Hướng dẫn sử dụng
│   ├── workflow.md               # Chi tiết quy trình
│   └── report.pdf                # Báo cáo cuối kỳ
├── tests/                        # 🧪 Test cho từng module (tùy chọn)
├── requirements.txt              # 🧷 Thư viện cần thiết
├── main.py                       # 🚀 Chạy toàn bộ hệ thống
├── .gitignore                    # 🛑 Bỏ qua dữ liệu lớn/không cần thiết
└── LICENSE                       # 📜 Giấy phép sử dụng (nếu có)
