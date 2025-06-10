import numpy as np
from deepface import DeepFace
import os


# Đường dẫn tuyệt đối tới thư mục embeddings
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
EMBEDDINGS_DIR = os.path.normpath(os.path.join(CURRENT_DIR, "../../data/embeddings/"))

# EMBEDDINGS_DIR = "data/embeddings/"
MODEL_NAME = "Facenet"
SIMILARITY_THRESHOLD = 0.6  # Điều chỉnh theo độ chính xác mong muốn


def load_reference_embeddings():
    embeddings = {}
    for file in os.listdir(EMBEDDINGS_DIR):
        if file.endswith(".pkl"):
            data = np.load(os.path.join(EMBEDDINGS_DIR, file))
            for key in data.files:
                person_name = os.path.splitext(file)[0] + "_" + key  # ví dụ: caonam_0.jpg
                embeddings[person_name] = data[key]
    return embeddings


def extract_embedding(face_image):
    #Trích xuất embedding từ ảnh đầu vào (ảnh mặt đã cắt)
    try:
        result = DeepFace.represent(face_image, model_name=MODEL_NAME, enforce_detection=False)
        if result:
            return np.array(result[0]['embedding'])
        else:
            return None
    except Exception as e:
        print(f"[RECOGNITION] Lỗi khi trích xuất embedding: {e}")
        return None


def recognize_faces(cropped_faces_dict, reference_embeddings):
    #So sánh các khuôn mặt đã cắt với embeddings mẫu. Trả về dict kết quả: {person_1: (name, % giống nhau), ...}
    results = {}

    for face_id, face_image in cropped_faces_dict.items():
        embedding = extract_embedding(face_image)
        if embedding is None:
            results[face_id] = ("unknown", 0.0)
            continue

        best_match = "unknown"
        best_score = float("inf")

        for name, ref_emb in reference_embeddings.items():
            distance = np.linalg.norm(embedding - ref_emb)
            if distance < SIMILARITY_THRESHOLD and distance < best_score:
                best_match = name
                best_score = distance

        similarity = max(0, 100 * (1 - best_score)) if best_match != "unknown" else 0.0
        results[face_id] = (best_match, round(similarity, 1))

    return results
