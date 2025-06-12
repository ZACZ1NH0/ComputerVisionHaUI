from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QImage
import cv2
import numpy as np
import os
import sys
import logging

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(f'{base_dir}/src')

from recognition.recognize_faces import load_reference_embeddings, extract_embedding, recognize_faces
from detection.detect_faces import detect_and_process_faces

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class ImageRecognitionThread(QThread):
    image_processed = pyqtSignal(QImage)
    recognition_result_signal = pyqtSignal(str)

    def __init__(self, image_path):
        super().__init__()
        self.image_path = image_path

    def run(self):
        reference_embeddings, reference_labels = load_reference_embeddings()

        frame = cv2.imread(self.image_path)
        if frame is None:
            self.recognition_result_signal.emit(f"Không thể đọc ảnh: {self.image_path}")
            return

        result = detect_and_process_faces(frame, flag=4)
        if result is None:
            self.recognition_result_signal.emit("Không phát hiện khuôn mặt nào.")
            return

        coords_dict, frame_with_boxes, faces_dict = result

        recog_results = {}
        for face_id, face_img in faces_dict.items():
            embedding = extract_embedding(face_img)
            if embedding is not None:
                name, score = recognize_faces(embedding, reference_embeddings, reference_labels)
            else:
                name, score = "Unknown", 0.0
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
            self.recognition_result_signal.emit(label)
        # Convert frame_with_boxes to QImage
        rgb_image = cv2.cvtColor(frame_with_boxes, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        scaled_img = qt_img.scaled(640, 480)  # Resize nếu muốn

        self.image_processed.emit(scaled_img)
