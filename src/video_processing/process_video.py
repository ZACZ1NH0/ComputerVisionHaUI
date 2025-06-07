import cv2
import numpy as np
from typing import Tuple, List, Dict, Optional
import time
from .utils import draw_face_box, draw_fps, calculate_iou, update_tracking
from .analyze_results import ResultAnalyzer

class VideoProcessor:
    def __init__(self):
        """Khởi tạo VideoProcessor với các thành phần cần thiết"""
        self.face_detector = None  # Sẽ được khởi tạo từ module detection
        self.face_recognizer = None  # Sẽ được khởi tạo từ module recognition
        self.tracking_results = {}  # Lưu kết quả theo dõi
        self.is_processing = False  # Flag để kiểm soát quá trình xử lý
        self.frame_count = 0  # Đếm số frame đã xử lý
        self.fps = 0  # FPS hiện tại
        self.next_face_id = 0  # ID cho khuôn mặt mới
        self.tracking_threshold = 0.5  # Ngưỡng IoU để tracking
        self.result_analyzer = ResultAnalyzer()  # Phân tích kết quả
        
    def start_processing(self):
        """Bắt đầu xử lý video"""
        self.is_processing = True
        self.frame_count = 0
        self.fps = 0
        self.tracking_results = {}  # Reset tracking results
        self.next_face_id = 0
        self.result_analyzer.start_session()  # Bắt đầu phiên phân tích
        
    def stop_processing(self):
        """Dừng xử lý video"""
        self.is_processing = False
        self.result_analyzer.end_session()  # Kết thúc phiên phân tích
        
    def process_camera(self, camera_id: int = 0, callback=None):
        """
        Xử lý video từ camera
        
        Args:
            camera_id: ID của camera (mặc định là 0 - webcam)
            callback: Hàm callback để cập nhật UI
        """
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            raise ValueError(f"Không thể mở camera với ID {camera_id}")
            
        self.start_processing()
        start_time = time.time()
        
        try:
            while self.is_processing:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Xử lý frame
                processed_frame, results = self.process_frame(frame)
                
                # Cập nhật FPS
                self.frame_count += 1
                elapsed_time = time.time() - start_time
                if elapsed_time >= 1.0:
                    self.fps = self.frame_count / elapsed_time
                    self.frame_count = 0
                    start_time = time.time()
                
                # Gọi callback nếu có
                if callback:
                    callback(processed_frame, results, self.fps)
                    
        finally:
            cap.release()
            self.stop_processing()
            
    def process_video_file(self, video_path: str, callback=None):
        """
        Xử lý video từ file
        
        Args:
            video_path: Đường dẫn đến file video
            callback: Hàm callback để cập nhật UI
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Không thể mở video file: {video_path}")
            
        self.start_processing()
        start_time = time.time()
        
        try:
            while self.is_processing:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Xử lý frame
                processed_frame, results = self.process_frame(frame)
                
                # Cập nhật FPS
                self.frame_count += 1
                elapsed_time = time.time() - start_time
                if elapsed_time >= 1.0:
                    self.fps = self.frame_count / elapsed_time
                    self.frame_count = 0
                    start_time = time.time()
                
                # Gọi callback nếu có
                if callback:
                    callback(processed_frame, results, self.fps)
                    
        finally:
            cap.release()
            self.stop_processing()
            
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Xử lý một frame video
        
        Args:
            frame: Frame video cần xử lý
            
        Returns:
            Tuple chứa frame đã xử lý và kết quả nhận diện
        """
        if self.face_detector is None or self.face_recognizer is None:
            raise ValueError("Face detector hoặc face recognizer chưa được khởi tạo")
            
        # Lưu frame gốc để vẽ kết quả
        processed_frame = frame.copy()
        
        # Phát hiện khuôn mặt
        face_boxes = self.face_detector.detect(frame)
        
        # Lưu kết quả nhận diện
        results = {
            'faces': [],
            'timestamp': time.time()
        }
        
        # Xử lý từng khuôn mặt
        for box in face_boxes:
            x1, y1, x2, y2 = box
            
            # Cắt khuôn mặt
            face_img = frame[y1:y2, x1:x2]
            
            # Nhận diện khuôn mặt
            label, confidence = self.face_recognizer.recognize(face_img)
            
            # Tracking khuôn mặt
            face_id = self._track_face(box, label, confidence)
            
            # Vẽ kết quả
            processed_frame = draw_face_box(processed_frame, box, label, confidence)
            
            # Lưu kết quả
            results['faces'].append({
                'id': face_id,
                'box': box,
                'label': label,
                'confidence': confidence
            })
        
        # Vẽ FPS
        processed_frame = draw_fps(processed_frame, self.fps)
        
        # Cập nhật kết quả phân tích
        self.result_analyzer.update_frame(results)
        
        return processed_frame, results
        
    def _track_face(self, box: Tuple[int, int, int, int], 
                   label: str, confidence: float) -> str:
        """
        Tracking khuôn mặt dựa trên IoU
        
        Args:
            box: Tọa độ khung
            label: Nhãn của khuôn mặt
            confidence: Độ tin cậy
            
        Returns:
            ID của khuôn mặt
        """
        timestamp = time.time()
        
        # Tìm khuôn mặt có IoU cao nhất
        max_iou = 0
        best_match = None
        
        for face_id, data in self.tracking_results.items():
            last_box = data['boxes'][-1]
            iou = calculate_iou(box, last_box)
            
            if iou > max_iou and iou > self.tracking_threshold:
                max_iou = iou
                best_match = face_id
        
        # Nếu tìm thấy match
        if best_match is not None:
            face_id = best_match
        else:
            # Tạo ID mới
            face_id = f"face_{self.next_face_id}"
            self.next_face_id += 1
        
        # Cập nhật tracking
        self.tracking_results = update_tracking(
            self.tracking_results,
            face_id,
            box,
            label,
            confidence,
            timestamp
        )
        
        return face_id
        
    def set_face_detector(self, detector):
        """Set face detector từ module detection"""
        self.face_detector = detector
        
    def set_face_recognizer(self, recognizer):
        """Set face recognizer từ module recognition"""
        self.face_recognizer = recognizer
        
    def get_analysis_report(self) -> str:
        """
        Lấy báo cáo phân tích
        
        Returns:
            String chứa nội dung báo cáo
        """
        return self.result_analyzer.generate_report()
