# 🧠 KẾ HOẠCH PHÂN CÔNG NHÓM – DỰ ÁN NHẬN DIỆN KHUÔN MẶT REAL-TIME

## 🎯 Mục tiêu dự án
Xây dựng hệ thống nhận diện khuôn mặt real-time từ video, sử dụng YOLOv8 để phát hiện và DeepFace để nhận diện, có giao diện GUI trực quan bằng PyQt5.

---

## 🧩 Phân công theo module

| Dev | Module | Nhiệm vụ chính |
|-----|--------|----------------|
| **Dev 1** | Tiền xử lý (Preprocessing) | - Phát hiện & cắt khuôn mặt từ 500 ảnh bằng YOLOv8<br> - Chuẩn hóa kích thước 160x160 pixel<br> - Lưu vào `dataset/person1/`<br> - Xử lý lỗi ảnh không hợp lệ |
| **Dev 2** | Trích xuất Embeddings | - Dùng DeepFace để trích xuất embeddings từ ảnh<br> - Lưu vào `person1_embeddings.npz`<br> - Viết hàm so sánh embeddings<br> - Test độ chính xác trên mẫu nhỏ |
| **Dev 3** | Phát hiện khuôn mặt (Detection) | - Viết hàm phát hiện mặt từ ảnh/video bằng YOLOv8<br> - Xử lý trường hợp khó: mặt nhỏ, nghiêng, mờ<br> - Trả về ảnh mặt và tọa độ |
| **Dev 4** | Nhận diện (Recognition) | - So sánh embedding đầu vào với Person 1<br> - Xử lý lỗi embedding không hợp lệ<br> - Trả nhãn (Person 1/Khác) và tọa độ |
| **Dev 5** | Xử lý video (Video Processing) | - Xử lý video real-time bằng OpenCV<br> - Tích hợp Detection & Recognition<br> - Đếm số lần xuất hiện, theo dõi tọa độ<br> - Lưu kết quả CSV |
| **Dev 6** | Giao diện (GUI) | - Xây dựng GUI bằng PyQt5<br> - Tích hợp video, nhãn, nút chức năng<br> - Hiển thị kết quả nhận diện trong thời gian thực |

---

## 🔗 Phối hợp giữa các Dev

| Mối quan hệ | Nội dung phối hợp |
|-------------|--------------------|
| Dev 1 ➡ Dev 2 | Cung cấp ảnh đã cắt khuôn mặt để trích xuất |
| Dev 2 ➡ Dev 4 | Cung cấp embeddings để so sánh |
| Dev 3 ➡ Dev 4,5 | Cung cấp hàm phát hiện khuôn mặt |
| Dev 4 ➡ Dev 5,6 | Cung cấp nhãn & tọa độ để xử lý video và GUI |
| Dev 5 ↔ Dev 6 | Tích hợp GUI với luồng video và phân tích kết quả |

---

## 📅 Tiến độ dự kiến

| Mốc thời gian | Nội dung |
|---------------|----------|
| **Ngày 1** | Phân công nhóm, thiết lập repo, chia module, mỗi Dev setup môi trường riêng |
| **Ngày 2** | - Hoàn thành từng module cơ bản<br> - Chuyển dữ liệu giữa các Dev<br> - Mock dữ liệu nếu cần thiết<br> - Họp nhóm cuối ngày 2 để cập nhật tiến độ & fix lỗi phối hợp |
| **Ngày 3** | Tích hợp toàn bộ hệ thống, test trên video, fix lỗi cuối cùng, hoàn thiện GUI & báo cáo |

---

## ✅ Kết quả đầu ra mỗi Dev (cuối Ngày 2)

| Dev | Kết quả |
|-----|----------|
| Dev 1 | Thư mục `dataset/person1/` với ảnh 160x160 |
| Dev 2 | File `person1_embeddings.npz` và hàm so sánh |
| Dev 3 | Hàm phát hiện khuôn mặt |
| Dev 4 | Hàm nhận diện danh tính |
| Dev 5 | Module xử lý video sơ bộ |
| Dev 6 | Giao diện cơ bản hiển thị video, khung, nhãn |

---

## 🛠️ Công nghệ sử dụng

- YOLOv8 (phát hiện khuôn mặt)
- DeepFace (embedding & nhận diện)
- OpenCV (xử lý ảnh/video)
- PyQt5 (giao diện)
- Python 3.10+
