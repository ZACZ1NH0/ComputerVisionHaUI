# Giao diện 
### Demo với hình ảnh
<img width="596" height="383" alt="image" src="https://github.com/user-attachments/assets/4aba101b-7236-466d-9e77-df6fd8175626" />
<img width="574" height="348" alt="image" src="https://github.com/user-attachments/assets/7f965f16-677a-4d61-83c9-4196a797779b" />

### Demo với video 
<img width="518" height="335" alt="image" src="https://github.com/user-attachments/assets/9134257c-613e-445c-a626-555382767309" />

### Demo với webcam
<img width="452" height="292" alt="image" src="https://github.com/user-attachments/assets/7d60f8ab-430c-482b-8a51-7aaa77aa03a7" />

## 📂 Project Structure

```text
face_recognition_project/
├── src/                          # Source code
│   ├── preprocessing/            # Module Dev 1: Tiền xử lý dữ liệu (YOLOv8)
│   │   ├── preprocess.py         # Tiền xử lý ảnh
│   │   └── utils.py              # Hàm hỗ trợ (resize, kiểm tra ảnh)
│   ├── embeddings/               # Module Dev 2: Trích xuất & so sánh embeddings
│   │   ├── extract_embeddings.py # Trích xuất embeddings (DeepFace)
│   │   └── compare_embeddings.py # So sánh embeddings
│   ├── detection/                # Module Dev 3: Phát hiện khuôn mặt
│   │   ├── detect_faces.py       # Phát hiện khuôn mặt (YOLOv8)
│   │   └── utils.py              # Hàm hỗ trợ (tọa độ, cắt ảnh)
│   ├── recognition/              # Module Dev 4: Nhận diện khuôn mặt
│   │   ├── recognize_faces.py    # Nhận diện danh tính (DeepFace)
│   │   └── utils.py              # Xử lý lỗi nhận diện
│   ├── video_processing/         # Module Dev 5: Xử lý & phân tích video
│   │   ├── process_video.py      # Xử lý video real-time
│   │   ├── analyze_results.py    # Phân tích kết quả (đếm, theo dõi)
│   │   └── utils.py              # Hàm hỗ trợ (lưu CSV, theo dõi tọa độ)
│   ├── gui/                      # Module Dev 6: Giao diện người dùng
│   │   ├── main_app.py           # Giao diện PyQt5 chính
│   │   ├── interface.py          # Định nghĩa giao diện
│   │   └── utils.py              # Hỗ trợ hiển thị, vẽ khung, video
│   └── config/                   # Cấu hình chung
│       ├── config.py             # Tham số toàn cục
│       └── __init__.py           # File khởi tạo package
├── data/                         # Dữ liệu gốc và xử lý (KHÔNG commit)
│   ├── raw/                      # Ảnh gốc theo người
│   ├── processed/                # Ảnh đã cắt (160x160)
│   ├── embeddings/               # File .npz chứa embeddings
│   ├── videos/                   # Video đầu vào
│   └── results/                  # Kết quả: ảnh, video, CSV phân tích
├── docs/                         # Tài liệu đi kèm
│   ├── README.md                 # Hướng dẫn sử dụng
│   ├── workflow.md               # Mô tả quy trình hệ thống
│   └── report.pdf                # Báo cáo bài tập lớn
├── tests/                        # Các file kiểm thử (unit test)
├── requirements.txt              # Thư viện cần cài đặt
├── main.py                       # File chạy chính cho toàn hệ thống
├── .gitignore                    # Loại trừ dữ liệu không cần track
└── LICENSE                       # Giấy phép sử dụng (tùy chọn)


```







### 1. 📥 Clone dự án

```bash
git clone https://github.com/ZACZ1NH0/ComputerVisionHaUI.git
```
### 2. 🐍 Tạo môi trường ảo với Python 3.9 (nếu chưa tạo)

> ⚠️ Đảm bảo bạn đã cài Python 3.9 trước đó.

```bash
py -3.9 -m venv venv
```

Kích hoạt môi trường ảo:

- **Windows (CMD):**

```cmd
.\venv\Scripts\activate
```

### 3. 📦 Cài đặt các thư viện phụ thuộc

```bash
pip install -r requirements.txt
```
