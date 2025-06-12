import cv2
import os
import sys
import time
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QImage

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(f'{base_dir}/src')

from recognition.recognize_faces import load_reference_embeddings, extract_embedding, recognize_faces
from detection.detect_faces import detect_and_process_faces


class CameraRecognitionThread(QThread):
    change_pixmap_signal = pyqtSignal(QImage)
    recognition_result_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._run_flag = True

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.recognition_result_signal.emit("❌ Không mở được webcam")
            return

        reference_embeddings, reference_labels = load_reference_embeddings()
        if reference_embeddings is None or reference_labels is None:
            self.recognition_result_signal.emit("❌ Không thể load embeddings")
            return

        while self._run_flag:
            ret, frame = cap.read()
            if not ret:
                self.recognition_result_signal.emit("❌ Mất kết nối webcam")
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
            self.change_pixmap_signal.emit(qt_image)

            self.msleep(30)

        cap.release()

    def stop(self):
        self._run_flag = False
        self.wait()
