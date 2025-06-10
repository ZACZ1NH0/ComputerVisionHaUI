import numpy as np
from deepface import DeepFace
import os
import pickle
import logging
import cv2


# Đường dẫn tuyệt đối tới thư mục embeddings
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
EMBEDDINGS_DIR = os.path.normpath(os.path.join(CURRENT_DIR, "../../data/embeddings/"))

# EMBEDDINGS_DIR = "data/embeddings/"
MODEL_NAME = "Facenet"
SIMILARITY_THRESHOLD = 0.75  # Tăng ngưỡng vì khoảng cách lớn hơn 0.8


def load_reference_embeddings():
    """
    Load embeddings và labels từ file pickle
    Returns:
        tuple: (embeddings, labels) hoặc (None, None) nếu có lỗi
    """
    try:
        # Đường dẫn tới file embeddings
        embeddings_file = os.path.join(EMBEDDINGS_DIR, "all_embeddings.pkl")
        
        if not os.path.exists(embeddings_file):
            logging.error(f"Không tìm thấy file embeddings tại: {embeddings_file}")
            return None, None
            
        with open(embeddings_file, 'rb') as f:
            data = pickle.load(f)
            
        if 'embeddings' not in data or 'labels' not in data:
            logging.error("File embeddings không đúng định dạng")
            return None, None
            
        # Chuẩn hóa tất cả embeddings
        embeddings = np.array(data['embeddings'])
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        logging.info(f"Đã load {len(embeddings)} embeddings mẫu")
        return embeddings, data['labels']
        
    except Exception as e:
        logging.error(f"Lỗi khi load embeddings: {str(e)}")
        return None, None


def preprocess_face(face_img):
    """
    Tiền xử lý ảnh khuôn mặt trước khi trích xuất embedding
    """
    try:
        # Chuyển về grayscale nếu là ảnh màu
        if len(face_img.shape) == 3:
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            
        # Resize về kích thước chuẩn
        face_img = cv2.resize(face_img, (160, 160))
        
        # Chuẩn hóa pixel values về [-1, 1]
        face_img = face_img.astype(np.float32) / 255.0
        face_img = (face_img - 0.5) * 2
        
        return face_img
    except Exception as e:
        logging.error(f"Lỗi khi tiền xử lý ảnh: {e}")
        return None


# def extract_embedding(face_image):
#     """
#     Trích xuất embedding từ ảnh đầu vào (ảnh mặt đã cắt)
#     """
#     try:
#         # Tiền xử lý ảnh
#         processed_face = preprocess_face(face_image)
#         if processed_face is None:
#             return None
            
#         # Trích xuất embedding
#         result = DeepFace.represent(
#             processed_face,
#             model_name=MODEL_NAME,
#             enforce_detection=False,
#             detector_backend='skip'  # Bỏ qua bước detect vì ảnh đã được cắt
#         )
        
#         if result and isinstance(result, list) and len(result) > 0:
#             embedding = np.array(result[0]['embedding'])
            
#             # Chuẩn hóa embedding
#             embedding = embedding / np.linalg.norm(embedding)
            
#             logging.info(f"Trích xuất embedding thành công, shape: {embedding.shape}")
#             return embedding
#         else:
#             logging.warning("Không thể trích xuất embedding từ ảnh")
#             return None
            
#     except Exception as e:
#         logging.error(f"Lỗi khi trích xuất embedding: {e}")
#         return None
MODEL_NAME = "Facenet512"  # Hoặc "ArcFace" nếu bạn dùng model khác

def extract_embedding(face_image):
    """
    Trích xuất embedding từ ảnh đầu vào (ảnh mặt đã cắt)
    """
    try:
        processed_face = preprocess_face(face_image)
        if processed_face is None:
            return None
            
        result = DeepFace.represent(
            processed_face,
            model_name=MODEL_NAME,
            enforce_detection=False,
            detector_backend='skip'
        )
        
        if result and isinstance(result, list) and len(result) > 0:
            embedding = np.array(result[0]['embedding'])
            
            # Chuẩn hóa embedding
            embedding = embedding / np.linalg.norm(embedding)
            
            logging.info(f"Trích xuất embedding thành công, shape: {embedding.shape}")
            return embedding
        else:
            logging.warning("Không thể trích xuất embedding từ ảnh")
            return None
            
    except Exception as e:
        logging.error(f"Lỗi khi trích xuất embedding: {e}")
        return None


def cosine_similarity(a, b):
    """
    Tính độ tương đồng cosine giữa hai vector
    """
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def recognize_faces(embedding, reference_embeddings, reference_labels):
    """
    So sánh embedding với các embeddings mẫu
    Args:
        embedding: embedding cần so sánh
        reference_embeddings: list các embeddings mẫu
        reference_labels: list các nhãn tương ứng
    Returns:
        tuple: (tên người, độ tương đồng)
    """
    if embedding is None or reference_embeddings is None or reference_labels is None:
        return "Unknown", 0.0

    # Tính độ tương đồng cosine với tất cả embeddings mẫu
    similarities = np.dot(reference_embeddings, embedding)
    
    # Tìm độ tương đồng lớn nhất và index tương ứng
    max_similarity_idx = np.argmax(similarities)
    max_similarity = similarities[max_similarity_idx]
    
    # Chuyển đổi độ tương đồng thành phần trăm
    similarity_percent = max_similarity * 100
    
    # Log thông tin debug
    logging.info(f"Độ tương đồng lớn nhất: {max_similarity:.4f}, Ngưỡng: {SIMILARITY_THRESHOLD}")
    logging.info(f"Label tương ứng: {reference_labels[max_similarity_idx]}")
    
    # Nếu độ tương đồng lớn hơn ngưỡng
    if max_similarity > SIMILARITY_THRESHOLD:
        return reference_labels[max_similarity_idx], round(similarity_percent, 1)
    
    # Nếu không nhận diện được, vẫn trả về Unknown với độ tương đồng
    return "Unknown", round(similarity_percent, 1)
