import os
import pickle
from tqdm import tqdm
from PIL import Image

import numpy as np
import cv2

# Nếu cần dùng thiết bị cụ thể
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1

# Load config
try:
    from config.config import PROCESSED_PATH, EMBEDDINGS_PATH
except ImportError as e:
    raise ImportError(f"Lỗi khi import config: {e}")

# Khởi tạo model dùng CPU
device = torch.device("cpu")
try:
    mtcnn = MTCNN(image_size=160, margin=0, min_face_size=20, device=device)
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
except Exception as e:
    raise RuntimeError(f"Lỗi khi khởi tạo mô hình: {e}")

def extract_all_embeddings(processed_root=PROCESSED_PATH, save_path=EMBEDDINGS_PATH, parent_widget=None):
    """
    Trích xuất embeddings từ thư mục ảnh đã xử lý và lưu vào file pickle.
    Nếu có `parent_widget`, lỗi sẽ được báo qua QMessageBox.
    """
    all_embeddings = []
    labels = []

    try:
        persons = [p for p in os.listdir(processed_root) if os.path.isdir(os.path.join(processed_root, p))]
    except Exception as e:
        raise RuntimeError(f"Không thể đọc thư mục '{processed_root}': {e}")

    for person in tqdm(persons, desc="🔍 Đang xử lý người"):
        person_path = os.path.join(processed_root, person)
        image_files = [f for f in os.listdir(person_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        for img_file in tqdm(image_files, desc=f"📸 {person}", leave=False):
            img_path = os.path.join(person_path, img_file)
            try:
                img = Image.open(img_path).convert('RGB')
                boxes, probs = mtcnn.detect(img)

                if boxes is not None and probs[0] > 0.9:
                    face = mtcnn(img)
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

    try:
        with open(save_path, "wb") as f:
            pickle.dump({"embeddings": all_embeddings, "labels": labels}, f)
        print(f"\n✅ Đã lưu {len(all_embeddings)} embeddings vào '{save_path}'")
        return len(all_embeddings)
    except Exception as e:
        raise IOError(f"Lỗi khi lưu embeddings vào '{save_path}': {e}")
