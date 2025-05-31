face_recognition_project/
├── src/                          # Source code
│   ├── preprocessing/            # Module Dev 1: Tiền xử lý dữ liệu
│   │   ├── preprocess.py         # Script tiền xử lý ảnh với YOLOv8
│   │   └── utils.py              # Hàm hỗ trợ (resize, kiểm tra ảnh)
│   ├── embeddings/               # Module Dev 2: Trích xuất và quản lý embeddings
│   │   ├── extract_embeddings.py # Script trích xuất embeddings với DeepFace
│   │   └── compare_embeddings.py # Hàm so sánh embeddings
│   ├── detection/                # Module Dev 3: Phát hiện khuôn mặt
│   │   ├── detect_faces.py       # Script phát hiện khuôn mặt với YOLOv8
│   │   └── utils.py              # Hàm hỗ trợ (xử lý tọa độ, cắt ảnh)
│   ├── recognition/              # Module Dev 4: Nhận diện danh tính
│   │   ├── recognize_faces.py    # Script nhận diện với DeepFace
│   │   └── utils.py              # Hàm hỗ trợ (xử lý lỗi nhận diện)
│   ├── video_processing/         # Module Dev 5: Xử lý video và phân tích
│   │   ├── process_video.py      # Script xử lý video real-time
│   │   ├── analyze_results.py    # Script phân tích (đếm, theo dõi)
│   │   └── utils.py              # Hàm hỗ trợ (lưu CSV, theo dõi tọa độ)
│   ├── gui/                      # Module Dev 6: GUI và tích hợp
│   │   ├── main_app.py           # Script chính chạy GUI với PyQt5
│   │   ├── interface.py          # Định nghĩa giao diện PyQt5
│   │   └── utils.py              # Hàm hỗ trợ (hiển thị video, vẽ khung)
│   └── config/                   # Cấu hình chung
│       ├── config.py             # Tham số chung (đường dẫn, ngưỡng,...)
│       └── __init__.py           # File khởi tạo Python package
├── data/                         # Dữ liệu (không commit lên Git)
│   ├── raw/                      # Ảnh gốc
│   │   └── person1/              # 500 ảnh khuôn mặt của Person 1
│   ├── processed/                # Ảnh đã tiền xử lý
│   │   └── person1/              # Ảnh khuôn mặt cắt (160x160)
│   ├── embeddings/               # Embeddings đã trích xuất
│   │   └── person1_embeddings.npz # File embeddings
│   ├── videos/                   # Video đầu vào
│   │   └── input_video.mp4       # Video mẫu
│   └── results/                  # Kết quả đầu ra
│       ├── images/               # Ảnh khuôn mặt Person 1
│       ├── videos/               # Video với khung/nhãn
│       └── analysis.csv          # File CSV kết quả phân tích
├── docs/                         # Tài liệu
│   ├── README.md                 # Hướng dẫn dự án
│   ├── workflow.md               # Workflow chi tiết
│   └── report.pdf                # Báo cáo bài tập lớn
├── tests/                        # Test cases (tùy chọn)
│   ├── test_preprocessing.py     # Test module Dev 1
│   ├── test_embeddings.py        # Test module Dev 2
│   ├── test_detection.py         # Test module Dev 3
│   ├── test_recognition.py       # Test module Dev 4
│   ├── test_video_processing.py  # Test module Dev 5
│   └── test_gui.py               # Test module Dev 6
├── requirements.txt              # File yêu cầu thư viện
├── main.py                       # Script chính chạy toàn bộ hệ thống
├── .gitignore                    # File bỏ qua dữ liệu lớn
└── LICENSE                       # Giấy phép (tùy chọn)
