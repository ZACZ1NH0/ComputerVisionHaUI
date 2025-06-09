import os
import numpy as np
from deepface import DeepFace

def load_known_embeddings(embedding_path):
    #Tai file npz chua cac embedding da luu
    if not os.path.exists(embedding_path):
        raise FileNotFoundError(f"Không tìm thấy file embeddings: {embedding_path}")
    data = np.load(embedding_path)
    return {k: data[k] for k in data.files}

def compute_embedding(image):
    #Trịch xuat embedding tu anh khuon mat
    try:
        result = DeepFace.represent(img_path=image, model_name='Facenet', enforce_detection=False)
        return np.array(result[0]["embedding"])
    except Exception as e:
        print(f"[!] Lỗi khi trích xuất embedding: {e}")
        return None

def cosine_distance(vec1, vec2):
    #Tinh khoang cach cosine giua 2 vecto
    dot = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return 1 - dot / (norm1 * norm2)

