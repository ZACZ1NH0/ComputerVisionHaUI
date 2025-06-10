import sys
import os
import cv2
import numpy as np
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(f'{base_dir}/src')
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QFileDialog, QHBoxLayout
from video_processing.process_video import process_webcam, process_video_file

class VideoProcessingApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face Recognition GUI")
        self.setGeometry(200, 200, 400, 200)

        self.label = QLabel("Chọn chế độ xử lý video", self)

        self.btn_webcam = QPushButton("Xử lý từ Webcam", self)
        self.btn_video = QPushButton("Xử lý từ File Video", self)

        self.btn_webcam.clicked.connect(self.run_webcam)
        self.btn_video.clicked.connect(self.select_video_file)

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.btn_webcam)
        layout.addWidget(self.btn_video)

        self.setLayout(layout)

    def run_webcam(self):
        self.label.setText("Đang chạy webcam...")
        process_webcam()
        self.label.setText("Xong webcam.")

    def select_video_file(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Chọn file video", "", "Video Files (*.mp4 *.avi)")
        if file_name:
            self.label.setText(f"Đang xử lý: {file_name}")
            process_video_file(file_name)
            self.label.setText("Xong video.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoProcessingApp()
    window.show()
    sys.exit(app.exec_())
