import sys
import os
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel, QVBoxLayout,
    QHBoxLayout, QStackedWidget, QFileDialog, QTextEdit, QMessageBox, QLineEdit
)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QInputDialog

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(base_dir, 'src'))
from preprocessing.preprocess_gui import preprocess_images
from preprocessing.create_image_gui import CaptureThread 
from embeddings.extract_embeddings_gui import extract_all_embeddings
from video_processing.webcam_recognition_gui import CameraRecognitionThread
from video_processing.video_recognition_gui import VideoRecognitionThread
from video_processing.picture_recognition_gui import ImageRecognitionThread
import cv2
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QThread, pyqtSignal

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(QImage)
    def __init__(self, video_path, process_func=None):
        super().__init__()
        self.video_path = video_path
        self.process_func = process_func
        self._run_flag = True

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        while self._run_flag:
            ret, frame = cap.read()
            if not ret:
                break
            # Xử lý frame nếu cần
            if self.process_func:
                frame = self.process_func(frame)
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.change_pixmap_signal.emit(qt_image)
            self.msleep(30)  # ~30 FPS
        cap.release()

    def stop(self):
        self._run_flag = False
        self.wait()


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.label_video = QLabel(self)
        self.label_video.setFixedSize(640, 480)
        self.label_video.setStyleSheet("background-color: black")
        # self.label_image = QLabel(self)
        # self.label_image.setFixedSize(640, 480)
        # self.label_image.setStyleSheet("background-color: black")
        self.setWindowTitle("Face Recognition Tool")
        self.setStyleSheet("background-color: #f0f0f0;")
        self.setMinimumSize(1000, 600)
        self.cam_thread = None
        self.video_thread = None
        self.image_thread = None

        self.init_ui()

    def init_ui(self):
        # Sidebar buttons
        self.sidebar = QVBoxLayout()
        self.sidebar.setSpacing(15)

        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Tên người dùng (trong thư mục raw)")
        self.sidebar.addWidget(self.name_input)

        self.create_btn("Create Image", self.on_create_image)
        self.create_btn("Preprocess", self.on_preprocess)
        self.create_btn("Extract Embedding", self.on_extract)
        # self.create_btn("Add Image", self.on_add_image)
        # self.create_btn("Add Video", self.on_add_video)
        self.create_btn("Webcam Recognition", self.on_webcam)
        self.create_btn("Video Recognition", self.on_video_recognition)
        self.create_btn("Image Recognition", self.on_image_recognition)
        self.sidebar.addStretch()

        # Central widget with stacked layout
        self.stack = QStackedWidget()
        self.video_capture_widget = QWidget()
        video_layout = QVBoxLayout()
        video_layout.addWidget(self.label_video)
        self.video_capture_widget.setLayout(video_layout)
        self.stack.addWidget(self.video_capture_widget)

        self.default_widget = QLabel("Welcome! Please select a function from the sidebar.")
        self.default_widget.setAlignment(Qt.AlignCenter)
        self.stack.addWidget(self.default_widget)

        self.display_label = QLabel("Video/Image Display")
        self.display_label.setAlignment(Qt.AlignCenter)
        self.stack.addWidget(self.display_label)

        # Right panel: prediction text
        self.prediction_label = QLabel("Prediction:")
        self.prediction_label.setStyleSheet("font-size: 16px; color: #333;")
        self.prediction_text = QTextEdit()
        self.prediction_text.setReadOnly(True)

        prediction_layout = QVBoxLayout()
        prediction_layout.addWidget(self.prediction_label)
        prediction_layout.addWidget(self.prediction_text)

        # Main layout
        main_layout = QHBoxLayout()
        sidebar_widget = QWidget()
        sidebar_widget.setLayout(self.sidebar)
        sidebar_widget.setStyleSheet("background-color: #e0e0e0; padding: 10px;")

        main_layout.addWidget(sidebar_widget, 1)
        main_layout.addWidget(self.stack, 3)
        main_layout.addLayout(prediction_layout, 1)

        self.setLayout(main_layout)

    def create_btn(self, text, slot):
        btn = QPushButton(text)
        btn.setMinimumHeight(40)
        btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                border-radius: 8px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        btn.clicked.connect(slot)
        self.sidebar.addWidget(btn)

    # Button slots (chức năng chưa thực thi, chỉ placeholder)
    def on_create_image(self):
        person_name, ok1 = QInputDialog.getText(self, "Tên người dùng", "Nhập tên (vd: person_thanh):")
        if not ok1 or not person_name.strip():
            return

        num_images, ok2 = QInputDialog.getInt(self, "Số lượng ảnh", "Nhập số lượng ảnh cần chụp:", min=1)
        if not ok2:
            return

        delay, ok3 = QInputDialog.getDouble(self, "Độ trễ", "Nhập độ trễ giữa ảnh (giây):", min=0.1)
        if not ok3:
            return

        # Tạo đường dẫn lưu
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        save_folder = os.path.join(project_root, "data", "raw", person_name)

        QMessageBox.information(self, "Bắt đầu", f"Ảnh sẽ lưu tại:\n{save_folder}")

        # Tạo và khởi động thread
        self.stack.setCurrentWidget(self.video_capture_widget)
        self.capture_thread = CaptureThread(person_name, num_images, delay, save_folder)
        self.capture_thread.image_update.connect(self.update_image)
        self.capture_thread.finished_signal.connect(self.on_capture_finished)
        self.capture_thread.start()

    def update_image(self, qt_image):
        self.label_video.setPixmap(QPixmap.fromImage(qt_image))

    def on_capture_finished(self, success, message):
        if success:
            QMessageBox.information(self, "Thành công", message)
        else:
            QMessageBox.critical(self, "Lỗi", message)

    def on_preprocess(self):
        person_name = self.name_input.text()  # Lấy từ QLineEdit hoặc input bất kỳ
        preprocess_images(person_name, parent_widget=self)

    def on_extract(self):
        try:
            count = extract_all_embeddings()
            QMessageBox.information(self, "Hoàn tất", f"Đã trích xuất {count} embeddings.")
        except Exception as e:
            QMessageBox.critical(self, "Lỗi", f"Lỗi khi trích xuất embeddings:\n{str(e)}")

    # def on_add_image(self):
    #     file_path, _ = QFileDialog.getOpenFileName(self, "Select an Image", "", "Images (*.png *.jpg *.jpeg)")
    #     if file_path:
    #         self.display_label.setPixmap(QPixmap(file_path).scaled(640, 480, Qt.KeepAspectRatio))
    #         self.stack.setCurrentIndex(1)

    def on_webcam(self):
        # Dừng camera cũ nếu có
        if self.cam_thread:
            self.cam_thread.stop()
            self.cam_thread = None

        self.stack.setCurrentWidget(self.video_capture_widget)
        self.prediction_text.clear()

        self.cam_thread = CameraRecognitionThread()
        self.cam_thread.change_pixmap_signal.connect(self.update_video_frame_1)
        self.cam_thread.recognition_result_signal.connect(self.update_prediction_text_1)
        self.cam_thread.start()

    def update_video_frame_1(self, qt_image):
        self.label_video.setPixmap(QPixmap.fromImage(qt_image))

    def update_prediction_text_1(self, text):
        current = self.prediction_text.toPlainText()
        if text not in current:
            self.prediction_text.append(text)

    def on_video_recognition(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Chọn file video", "", "Video Files (*.mp4 *.avi *.mov)")
        if file_path:
            if self.video_thread and self.video_thread.isRunning():
                self.video_thread.stop()
                self.video_thread.wait()
            self.stack.setCurrentWidget(self.video_capture_widget)
            self.prediction_text.clear()

            self.video_thread = VideoRecognitionThread(file_path)
            self.video_thread.frame_updated.connect(self.update_video_frame_2)
            self.video_thread.recognition_result_signal.connect(self.update_prediction_text_2)
            self.video_thread.start()
    
    def update_video_frame_2(self, qt_image):
        self.label_video.setPixmap(QPixmap.fromImage(qt_image))

    def update_prediction_text_2(self, text):
        current = self.prediction_text.toPlainText()
        if text not in current:
            self.prediction_text.append(text)

    def closeEvent(self, event):
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.stop()
            self.video_thread.wait()
        if self.cam_thread and self.cam_thread.isRunning():
            self.cam_thread.stop()
            self.cam_thread.wait()
        if self.image_thread and self.image_thread.isRunning():
            self.image_thread.quit()
            self.image_thread.wait()
        event.accept()

    def on_image_recognition(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Chọn ảnh", "", "Image Files (*.png *.jpg *.jpeg)")
        if file_path:
            self.image_thread = ImageRecognitionThread(file_path)
            self.image_thread.image_processed.connect(self.update_image_result_3)
            self.image_thread.recognition_result_signal.connect(self.update_prediction_text_3)
            self.image_thread.start()

    def update_image_result_3(self, qt_image):
        self.label_video.setPixmap(QPixmap.fromImage(qt_image))

    def update_prediction_text_3(self, text):
        current = self.prediction_text.toPlainText()
        if text not in current:
            self.prediction_text.append(text)

    # def on_add_video(self):
    #     file_path, _ = QFileDialog.getOpenFileName(self, "Select a Video", "", "Videos (*.mp4 *.avi *.mov)")
    #     if file_path:
    #         # Nếu đã có thread cũ thì dừng lại
    #         if hasattr(self, 'video_thread') and self.video_thread.isRunning():
    #             self.video_thread.stop()
    #         self.video_thread = VideoThread(file_path)
    #         self.video_thread.change_pixmap_signal.connect(self.update_video_frame)
    #         self.video_thread.start()
    #         self.stack.setCurrentIndex(1)

    # def update_video_frame(self, qt_image):
    #     self.display_label.setPixmap(QPixmap.fromImage(qt_image).scaled(640, 480, Qt.KeepAspectRatio))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
