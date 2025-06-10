import os
import sys
import cv2
import numpy as np
import time
from datetime import datetime
import logging
from collections import defaultdict

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(f'{base_dir}/src')

from embeddings.compare_embeddings import compare_embeddings
from recognition.recognize_faces import load_reference_embeddings, extract_embedding, recognize_faces
from detection.detect_faces import detect_and_process_faces
from config.config import THRESHOLD, FACE_DETECTION_CONFIDENCE, VIDEO_PATH, RESULTS_PATH

# Thiết lập logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_webcam():
    """Kết hợp phát hiện + nhận diện khuôn mặt realtime từ webcam"""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Lỗi: Không thể kết nối với webcam.")
        return

    logger.info("Webcam đang chạy... Nhấn 'q' để thoát.")

    reference_embeddings, reference_labels = load_reference_embeddings()
    if reference_embeddings is None or reference_labels is None:
        logger.error("Không thể load reference embeddings")
        return

    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            logger.error("Lỗi: Mất kết nối với webcam.")
            break

        frame = cv2.flip(frame, 1)

        result = detect_and_process_faces(frame, flag=4)
        if result is None:
            continue

        coords_dict, frame_with_boxes, faces_dict = result
        logger.info(f"Phát hiện được {len(faces_dict)} khuôn mặt")
        
        # Xử lý từng khuôn mặt riêng biệt
        recog_results = {}
        for face_id, face_img in faces_dict.items():
            embedding = extract_embedding(face_img)
            if embedding is not None:
                name, score = recognize_faces(embedding, reference_embeddings, reference_labels)
                logger.info(f"Khuôn mặt {face_id}: {name} ({score:.1f}%)")
            else:
                name, score = "Unknown", 0.0
                logger.warning(f"Không thể trích xuất embedding cho khuôn mặt {face_id}")
            recog_results[face_id] = (name, score)

        for face_id, (name, score) in recog_results.items():
            if face_id not in coords_dict:
                continue

            x1, y1, x2, y2 = coords_dict[face_id]
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            label = f"{name} ({score:.1f}%)" if name != "Unknown" else "Unknown"

            cv2.rectangle(frame_with_boxes, (x1, y1), (x2, y2), color, 2)
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            cv2.rectangle(frame_with_boxes, (x1, y1 - text_size[1] - 10), 
                         (x1 + text_size[0], y1), color, -1)
            cv2.putText(frame_with_boxes, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        # Tính FPS
        processing_time = time.time() - start_time
        fps = 1 / processing_time if processing_time > 0 else 0
        
        # Vẽ FPS
        fps_text = f"FPS: {int(fps)}"
        cv2.putText(frame_with_boxes, fps_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Hiển thị frame
        cv2.imshow("Webcam Face Recognition", frame_with_boxes)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    process_webcam()