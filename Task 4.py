import sys
import cv2
from ultralytics import YOLO
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                            QPushButton, QLabel, QComboBox, QSlider, QFileDialog,
                            QWidget, QGroupBox, QCheckBox)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap


class ObjectDetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Object Detection and Tracking System")
        self.setGeometry(100, 100, 1200, 800)
        
        # Initialize YOLO model
        self.model = None
        self.track_history = {}
        self.class_names = []
        self.is_running = False
        self.cap = None
        self.video_writer = None
        
        # UI Elements
        self.init_ui()
        
        # Default model
        self.load_model("yolov8n.pt")
        
    def init_ui(self):
        """Initialize the user interface"""
        main_widget = QWidget()
        main_layout = QHBoxLayout()
        
        # Left panel - controls
        control_panel = QGroupBox("Controls")
        control_layout = QVBoxLayout()
        
        # Model selection
        self.model_combo = QComboBox()
        self.model_combo.addItems(["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"])
        self.model_combo.currentTextChanged.connect(self.model_changed)
        
        # Confidence threshold
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(0, 100)
        self.conf_slider.setValue(50)
        self.conf_label = QLabel("Confidence Threshold: 0.5")
        
        # Tracking toggle
        self.tracking_check = QCheckBox("Enable Tracking")
        self.tracking_check.setChecked(True)
        
        # Path history toggle
        self.path_check = QCheckBox("Show Tracking Path")
        self.path_check.setChecked(True)
        
        # Buttons
        self.webcam_btn = QPushButton("Start Webcam")
        self.webcam_btn.clicked.connect(self.toggle_webcam)
        
        self.video_btn = QPushButton("Open Video File")
        self.video_btn.clicked.connect(self.open_video_file)
        
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop_processing)
        self.stop_btn.setEnabled(False)
        
        # Add widgets to control panel
        control_layout.addWidget(QLabel("Model:"))
        control_layout.addWidget(self.model_combo)
        control_layout.addSpacing(10)
        control_layout.addWidget(self.conf_label)
        control_layout.addWidget(self.conf_slider)
        control_layout.addSpacing(10)
        control_layout.addWidget(self.tracking_check)
        control_layout.addWidget(self.path_check)
        control_layout.addSpacing(20)
        control_layout.addWidget(self.webcam_btn)
        control_layout.addWidget(self.video_btn)
        control_layout.addWidget(self.stop_btn)
        control_layout.addStretch()
        
        control_panel.setLayout(control_layout)
        
        # Right panel - video display
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(800, 600)
        self.video_label.setText("Video feed will appear here")
        
        # Add panels to main layout
        main_layout.addWidget(control_panel, 1)
        main_layout.addWidget(self.video_label, 3)
        
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
        
        # Timer for video updates
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        
    def load_model(self, model_path):
        """Load the YOLO model"""
        try:
            self.model = YOLO(model_path)
            self.class_names = self.model.names
            self.track_history = {}  # Reset tracking history
            print(f"Model {model_path} loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
    
    def model_changed(self, model_name):
        """Handle model selection change"""
        self.load_model(model_name)
    
    def toggle_webcam(self):
        """Toggle webcam on/off"""
        if not self.is_running:
            self.start_webcam()
        else:
            self.stop_processing()
    
    def start_webcam(self):
        """Start webcam capture"""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        self.is_running = True
        self.webcam_btn.setText("Stop Webcam")
        self.video_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.timer.start(30)  # Update every 30ms
    
    def open_video_file(self):
        """Open and process a video file"""
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Open Video File", "", 
            "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)", 
            options=options)
        
        if file_name:
            self.cap = cv2.VideoCapture(file_name)
            if not self.cap.isOpened():
                print("Error: Could not open video file")
                return
            
            # Prepare video writer for output
            frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            
            output_file = file_name.rsplit('.', 1)[0] + '_output.mp4'
            self.video_writer = cv2.VideoWriter(
                output_file, 
                cv2.VideoWriter_fourcc(*'mp4v'), 
                fps, 
                (frame_width, frame_height))
            
            self.is_running = True
            self.video_btn.setEnabled(False)
            self.webcam_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.timer.start(30)  # Update every 30ms
    
    def stop_processing(self):
        """Stop the current video processing"""
        self.timer.stop()
        if self.cap:
            self.cap.release()
        if self.video_writer:
            self.video_writer.release()
        
        self.is_running = False
        self.webcam_btn.setText("Start Webcam")
        self.video_btn.setEnabled(True)
        self.webcam_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.video_label.setText("Video feed will appear here")
    
    def update_frame(self):
        """Process and update the current frame"""
        if not self.cap or not self.cap.isOpened():
            return
        
        ret, frame = self.cap.read()
        if not ret:
            self.stop_processing()
            return
        
        # Process frame with YOLO
        processed_frame = self.process_frame(frame)
        
        # Write to output video if processing file
        if self.video_writer:
            self.video_writer.write(processed_frame)
        
        # Convert to QImage and display
        rgb_image = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qt_image).scaled(
            self.video_label.width(), self.video_label.height(), 
            Qt.KeepAspectRatio, Qt.SmoothTransformation))
    
    def process_frame(self, frame):
        """Process a single frame for object detection and tracking"""
        conf_threshold = self.conf_slider.value() / 100
        self.conf_label.setText(f"Confidence Threshold: {conf_threshold:.2f}")
        
        # Run YOLO detection
        if self.tracking_check.isChecked():
            results = self.model.track(frame, persist=True, verbose=False, conf=conf_threshold)
        else:
            results = self.model.predict(frame, verbose=False, conf=conf_threshold)
        
        # Get the boxes and track IDs
        if len(results) > 0 and results[0].boxes is not None:
            if self.tracking_check.isChecked() and results[0].boxes.id is not None:
                boxes = results[0].boxes.xywh.cpu()
                track_ids = results[0].boxes.id.int().cpu().tolist()
                confidences = results[0].boxes.conf.cpu()
                class_ids = results[0].boxes.cls.int().cpu().tolist()
                
                annotated_frame = results[0].plot()
                
                # Update track history if showing paths
                if self.path_check.isChecked():
                    for box, track_id, confidence, class_id in zip(boxes, track_ids, confidences, class_ids):
                        if confidence < conf_threshold:
                            continue
                            
                        x, y, w, h = box
                        center = (float(x), float(y))
                        
                        if track_id not in self.track_history:
                            self.track_history[track_id] = []
                        
                        self.track_history[track_id].append(center)
                        
                        # Draw tracking lines (last 30 points)
                        points = np.array(self.track_history[track_id][-30:], np.int32)
                        cv2.polylines(annotated_frame, [points], isClosed=False, 
                                     color=self._get_color(track_id), thickness=2)
            else:
                annotated_frame = results[0].plot()
        else:
            annotated_frame = frame
            
        return annotated_frame
    
    def _get_color(self, track_id):
        """Generate a consistent color for each track ID"""
        np.random.seed(track_id)
        return (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
    
    def closeEvent(self, event):
        """Clean up when closing the application"""
        self.stop_processing()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ObjectDetectionApp()
    window.show()
    sys.exit(app.exec_())