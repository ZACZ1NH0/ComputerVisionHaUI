import os
import numpy as np
import pickle
from tqdm import tqdm
import sys
import cv2
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image

# Thêm đường dẫn đến thư mục chứa config
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(f'{base_dir}/src')
try:
    from config.config import PROCESSED_PATH, EMBEDDINGS_PATH
except ImportError as e:
    print(f"Lỗi khi import config: {e}")
    sys.exit(1)

# Khởi tạo MTCNN (phát hiện khuôn mặt) và InceptionResnetV1 (trích xuất embedding)
try:
    mtcnn = MTCNN(image_size=160, margin=0, min_face_size=20, device='cpu')
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to('cpu')
except Exception as e:
    print(f"Lỗi khi khởi tạo mô hình: {e}")
    sys.exit(1)


def extract_all_embeddings(processed_root=PROCESSED_PATH, save_path=EMBEDDINGS_PATH):
    all_embeddings = []
    labels = []

    # Danh sách thư mục người dùng
    persons = [p for p in os.listdir(processed_root) if os.path.isdir(os.path.join(processed_root, p))]
    for person in tqdm(persons, desc="🔍 Đang xử lý người"):
        person_path = os.path.join(processed_root, person)
        image_files = [f for f in os.listdir(person_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        for img_file in tqdm(image_files, desc=f"📸 {person}", leave=False):
            img_path = os.path.join(person_path, img_file)
            try:
                # Đọc và chuyển ảnh sang định dạng PIL
                img = Image.open(img_path).convert('RGB')

                # Phát hiện khuôn mặt bằng MTCNN
                boxes, probs = mtcnn.detect(img)
                if boxes is not None and probs[0] > 0.9:  # Chỉ lấy khuôn mặt có độ tin cậy cao
                    # Trích xuất khuôn mặt và embedding
                    face = mtcnn(img)  # Tự động căn chỉnh khuôn mặt
                    if face is not None:
                        embedding = resnet(face.unsqueeze(0)).detach().cpu().numpy()[0]
                        all_embeddings.append(embedding)
                        labels.append(person)
                    else:
                        print(f"⚠️ Không trích xuất được khuôn mặt từ {img_file}")
                else:
                    print(f"⚠️ Không tìm thấy khuôn mặt trong {img_file}")
            except Exception as e:
                print(f"⚠️ Lỗi khi xử lý ảnh {img_file} của {person}: {e}")

    # Lưu kết quả vào file .pkl
    with open(save_path, "wb") as f:
        pickle.dump({"embeddings": all_embeddings, "labels": labels}, f)
    print(f"\n✅ Đã lưu {len(all_embeddings)} embeddings vào '{save_path}'")


# Chạy trực tiếp
if __name__ == "__main__":
    extract_all_embeddings()
