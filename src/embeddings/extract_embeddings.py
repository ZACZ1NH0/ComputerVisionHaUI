import os
import numpy as np
import pickle
from deepface import DeepFace
from tqdm import tqdm
import sys
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(f'{base_dir}/src')  # Thêm đúng thư mục chứa module
from config.config import PROCESSED_PATH, EMBEDDINGS_PATH 

def extract_all_embeddings(processed_root=PROCESSED_PATH, save_path=EMBEDDINGS_PATH, model_name="Facenet"):
    """
    Duyệt qua tất cả thư mục con (mỗi người), trích xuất embedding từ từng ảnh,
    gắn nhãn tương ứng và lưu vào file pickle.
    """
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
                embedding = DeepFace.represent(img_path=img_path, model_name=model_name)[0]["embedding"]
                all_embeddings.append(embedding)
                labels.append(person)
            except Exception as e:
                print(f"⚠️ Lỗi khi xử lý ảnh {img_file} của {person}: {e}")

    # Lưu kết quả
    with open(save_path, "wb") as f:
        pickle.dump({"embeddings": all_embeddings, "labels": labels}, f)
    print(f"\n✅ Đã lưu {len(all_embeddings)} embeddings vào '{save_path}'")

# Cho phép chạy trực tiếp
if __name__ == "__main__":
    extract_all_embeddings()

