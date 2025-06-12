# video_recognition_thread.py
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QImage
import cv2
import numpy as np
import os
import sys
import time
import logging

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(f'{base_dir}/src')

from recognition.recognize_faces import load_reference_embeddings, extract_embedding, recognize_faces
from detection.detect_faces import detect_and_process_faces

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class VideoRecognitionThread(QThread):
    frame_updated = pyqtSignal(QImage)
    recognition_result_signal = pyqtSignal(str)

    def __init__(self, video_path):
        super().__init__()
        self.video_path = video_path
        self._is_running = True
        self._run_flag = True

    def run(self):
        reference_embeddings, reference_labels = load_reference_embeddings()
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            logger.error(f"Không thể mở file video: {self.video_path}")
            return

        reference_embeddings, reference_labels = load_reference_embeddings()
        if reference_embeddings is None or reference_labels is None:
            self.recognition_result_signal.emit("❌ Không thể load embeddings")
            return

        while self._run_flag:
            ret, frame = cap.read()
            if not ret:
                self.recognition_result_signal.emit("❌ Mất kết nối ")
                break

            frame = cv2.flip(frame, 1)
            result = detect_and_process_faces(frame, flag=4)
            if result is None:
                continue

            coords_dict, frame_with_boxes, faces_dict = result
            for face_id, face_img in faces_dict.items():
                embedding = extract_embedding(face_img)
                if embedding is not None:
                    name, score = recognize_faces(embedding, reference_embeddings, reference_labels)
                    label = f"{name} ({score:.1f}%)" if name != "Unknown" else "Unknown"
                else:
                    label = "Unknown"

                # Vẽ kết quả
                x1, y1, x2, y2 = coords_dict[face_id]
                color = (0, 255, 0) if label != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame_with_boxes, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame_with_boxes, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                self.recognition_result_signal.emit(label)

            # Hiển thị frame
            rgb_image = cv2.cvtColor(frame_with_boxes, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.frame_updated.emit(qt_image)

            self.msleep(30)

        cap.release()

    def stop(self):
        self._is_running = False
