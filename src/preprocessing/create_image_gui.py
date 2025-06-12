# file: capture_module.py
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QImage
import cv2
import os
import time

class CaptureThread(QThread):
    image_update = pyqtSignal(QImage)
    finished_signal = pyqtSignal(bool, str)

    def __init__(self, person_name, num_images, delay, save_folder):
        super().__init__()
        self.person_name = person_name
        self.num_images = num_images
        self.delay = delay
        self.save_folder = save_folder
        self._is_running = True

    def run(self):
        os.makedirs(self.save_folder, exist_ok=True)
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            self.finished_signal.emit(False, "Không thể mở webcam.")
            return

        count = len([f for f in os.listdir(self.save_folder) if f.endswith(('.jpg', '.png'))])

        while count < self.num_images and self._is_running:
            ret, frame = cap.read()
            if not ret:
                self.finished_signal.emit(False, "Không thể đọc frame từ webcam.")
                break

            # Hiển thị frame lên GUI
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.image_update.emit(qt_image)

            # Lưu ảnh
            save_path = os.path.join(self.save_folder, f"{count}.jpg")
            cv2.imwrite(save_path, frame)
            count += 1

            time.sleep(self.delay)

        cap.release()
        self.finished_signal.emit(True, f"Đã lưu {count} ảnh.")
    
    def stop(self):
        self._is_running = False
