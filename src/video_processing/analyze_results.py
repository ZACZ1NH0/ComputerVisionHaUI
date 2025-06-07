import csv
import json
import time
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime
import os

class ResultAnalyzer:
    def __init__(self, output_dir: str = "data/results"):
        """
        Khởi tạo ResultAnalyzer
        
        Args:
            output_dir: Thư mục lưu kết quả
        """
        self.output_dir = output_dir
        self.current_session = None
        self.session_start_time = None
        self.session_data = {
            'faces': {},  # Lưu thông tin theo dõi khuôn mặt
            'frames': [],  # Lưu kết quả từng frame
            'statistics': {  # Thống kê tổng hợp
                'total_frames': 0,
                'total_faces_detected': 0,
                'unique_faces': 0,
                'start_time': None,
                'end_time': None,
                'duration': 0
            }
        }
        
        # Tạo thư mục output nếu chưa tồn tại
        os.makedirs(output_dir, exist_ok=True)
        
    def start_session(self):
        """Bắt đầu một phiên phân tích mới"""
        self.current_session = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_start_time = time.time()
        self.session_data = {
            'faces': {},
            'frames': [],
            'statistics': {
                'total_frames': 0,
                'total_faces_detected': 0,
                'unique_faces': 0,
                'start_time': self.session_start_time,
                'end_time': None,
                'duration': 0
            }
        }
        
    def end_session(self):
        """Kết thúc phiên phân tích và lưu kết quả"""
        if not self.current_session:
            return
            
        # Cập nhật thống kê
        self.session_data['statistics']['end_time'] = time.time()
        self.session_data['statistics']['duration'] = (
            self.session_data['statistics']['end_time'] - 
            self.session_data['statistics']['start_time']
        )
        self.session_data['statistics']['unique_faces'] = len(self.session_data['faces'])
        
        # Lưu kết quả
        self._save_results()
        
        # Reset session
        self.current_session = None
        self.session_start_time = None
        
    def update_frame(self, frame_results: Dict):
        """
        Cập nhật kết quả từ một frame
        
        Args:
            frame_results: Kết quả từ VideoProcessor.process_frame
        """
        if not self.current_session:
            self.start_session()
            
        # Cập nhật thống kê
        self.session_data['statistics']['total_frames'] += 1
        self.session_data['statistics']['total_faces_detected'] += len(frame_results['faces'])
        
        # Lưu kết quả frame
        frame_data = {
            'timestamp': frame_results['timestamp'],
            'faces': frame_results['faces']
        }
        self.session_data['frames'].append(frame_data)
        
        # Cập nhật thông tin khuôn mặt
        for face in frame_results['faces']:
            face_id = face['id']
            if face_id not in self.session_data['faces']:
                self.session_data['faces'][face_id] = {
                    'id': face_id,
                    'label': face['label'],
                    'first_seen': frame_results['timestamp'],
                    'last_seen': frame_results['timestamp'],
                    'appearances': 1,
                    'boxes': [face['box']],
                    'confidences': [face['confidence']]
                }
            else:
                face_data = self.session_data['faces'][face_id]
                face_data['last_seen'] = frame_results['timestamp']
                face_data['appearances'] += 1
                face_data['boxes'].append(face['box'])
                face_data['confidences'].append(face['confidence'])
                
    def get_statistics(self) -> Dict:
        """
        Lấy thống kê hiện tại
        
        Returns:
            Dict chứa thống kê
        """
        return self.session_data['statistics']
        
    def get_face_statistics(self) -> Dict:
        """
        Lấy thống kê về các khuôn mặt
        
        Returns:
            Dict chứa thống kê khuôn mặt
        """
        return {
            face_id: {
                'label': data['label'],
                'appearances': data['appearances'],
                'duration': data['last_seen'] - data['first_seen'],
                'avg_confidence': np.mean(data['confidences'])
            }
            for face_id, data in self.session_data['faces'].items()
        }
        
    def _save_results(self):
        """Lưu kết quả phân tích"""
        if not self.current_session:
            return
            
        # Tạo thư mục cho session
        session_dir = os.path.join(self.output_dir, self.current_session)
        os.makedirs(session_dir, exist_ok=True)
        
        # Lưu thống kê
        with open(os.path.join(session_dir, 'statistics.json'), 'w') as f:
            json.dump(self.session_data['statistics'], f, indent=4)
            
        # Lưu thông tin khuôn mặt
        with open(os.path.join(session_dir, 'faces.json'), 'w') as f:
            json.dump(self.session_data['faces'], f, indent=4)
            
        # Lưu kết quả frame dưới dạng CSV
        frames_df = pd.DataFrame([
            {
                'timestamp': frame['timestamp'],
                'face_id': face['id'],
                'label': face['label'],
                'confidence': face['confidence'],
                'box_x1': face['box'][0],
                'box_y1': face['box'][1],
                'box_x2': face['box'][2],
                'box_y2': face['box'][3]
            }
            for frame in self.session_data['frames']
            for face in frame['faces']
        ])
        frames_df.to_csv(os.path.join(session_dir, 'frames.csv'), index=False)
        
    def generate_report(self) -> str:
        """
        Tạo báo cáo tổng hợp
        
        Returns:
            String chứa nội dung báo cáo
        """
        if not self.current_session:
            return "No active session"
            
        stats = self.session_data['statistics']
        face_stats = self.get_face_statistics()
        
        report = [
            "=== Face Recognition Analysis Report ===",
            f"Session: {self.current_session}",
            f"Duration: {stats['duration']:.2f} seconds",
            f"Total Frames: {stats['total_frames']}",
            f"Total Faces Detected: {stats['total_faces_detected']}",
            f"Unique Faces: {stats['unique_faces']}",
            "\nFace Statistics:",
        ]
        
        for face_id, data in face_stats.items():
            report.extend([
                f"\nFace {face_id}:",
                f"  Label: {data['label']}",
                f"  Appearances: {data['appearances']}",
                f"  Duration: {data['duration']:.2f} seconds",
                f"  Average Confidence: {data['avg_confidence']:.2f}"
            ])
            
        return "\n".join(report)
