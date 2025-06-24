#The BehaviorTagger module provides a GUI application for tagging behaviors in video frames.
#It allows users to select tracks, view video frames, and annotate behaviors with start and end frames,
#movement types, maneuver types, and lane changes for each tracked vehicle in the video.

import os
import json
import sys
import cv2
import numpy as np
from PySide6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
                               QWidget, QPushButton, QLabel, QLineEdit, QListWidget,
                               QSpinBox, QFileDialog, QMessageBox, QGroupBox,
                               QTableWidget, QTableWidgetItem, QHeaderView, QSlider,
                               QSplitter, QComboBox, QFormLayout)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QPixmap, QImage, QKeySequence, QShortcut

def tag_behaviors(video_path, output_path, load=None):
    app = QApplication.instance()
    if not app:
        app = QApplication(sys.argv)
    window = BehaviorTagger(video_path=video_path, output_path=output_path)
    window.show()
    sys.exit(app.exec())

class VideoFrameWidget(QLabel):
    objectClicked = Signal(str)

    def __init__(self, width=1280, height=720):
        super().__init__()
        self.setMinimumSize(width, height)
        self.setStyleSheet("QLabel { background-color: black; }")
        self.setAlignment(Qt.AlignCenter)
        self.tracks = {}
        self.behaviors = []
        self.selected_track_id = None
        self.scale_factor = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.original_frame_width = width
        self.original_frame_height = height

    def set_tracks(self, tracks):
        self.tracks = tracks

    def set_behaviors(self, behaviors):
        self.behaviors = behaviors

    def display_frame(self, frame, frame_number):
        if frame is None:
            blank_frame = np.zeros((self.height(), self.width(), 3), dtype=np.uint8)
            q_image = QImage(blank_frame.data, self.width(), self.height(), 3 * self.width(), QImage.Format_RGB888)
            self.setPixmap(QPixmap.fromImage(q_image))
            return

        annotated_frame = frame.copy()
        self.original_frame_height, self.original_frame_width, _ = annotated_frame.shape

        for track_id, track_data in self.tracks.items():
            detection = None
            
            if "detections" in track_data:
                for det in track_data["detections"]:
                    if det["frame"] == frame_number:
                        detection = det
                        break
            elif "frames" in track_data and str(frame_number) in track_data["frames"]:
                detection = track_data["frames"][str(frame_number)]

            if detection:
                color = (0, 255, 255) if track_id == self.selected_track_id else (0, 255, 0)
                thickness = 3 if track_id == self.selected_track_id else 2
                
                if "obb" in detection:
                    corners = detection["obb"]["corners"]
                    pts = np.array(corners, np.int32).reshape((-1, 1, 2))
                    cv2.polylines(annotated_frame, [pts], True, color, thickness)
                    
                    x_coords = [pt[0] for pt in corners]
                    y_coords = [pt[1] for pt in corners]
                    x, y = int(min(x_coords)), int(min(y_coords))
                elif "bbox" in detection:
                    x, y, w, h = map(int, detection["bbox"])
                    cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), color, thickness)
                else:
                    continue
                
                class_id = track_data.get("class", "N/A")
                behavior_text = ""
                for behavior in self.behaviors:
                    if (str(behavior["track_id"]) == track_id and
                        behavior["start_frame"] <= frame_number <= behavior["end_frame"]):
                        b_data = behavior.get("behavior")
                        if isinstance(b_data, dict):
                            mov = b_data.get("movement", "-")
                            man = b_data.get("maneuver_type", "-")
                            lc = b_data.get("lane_change", "-")
                            behavior_text = f"Mov: {mov}, Type: {man}, LC: {lc}"
                        elif b_data:
                            behavior_text = f"Behavior: {b_data}"
                        break
                
                label = f"ID: {track_id} | Class: {class_id}"
                if behavior_text:
                    label += f" | {behavior_text}"

                label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                label_y = y - 10
                if label_y < 0: 
                    label_y = y + 30

                cv2.rectangle(annotated_frame, (x, label_y - label_size[1] - 5),
                              (x + label_size[0], label_y + baseline), (0,0,0), -1)
                cv2.putText(annotated_frame, label, (x, label_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        height, width, channel = annotated_frame.shape
        bytes_per_line = 3 * width
        q_image = QImage(annotated_frame.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        
        label_size = self.size()
        scaled_pixmap = QPixmap.fromImage(q_image).scaled(
            label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        
        self.setPixmap(scaled_pixmap)
        
        if width > 0 and height > 0:
            self.scale_factor = min(label_size.width() / width, label_size.height() / height) if width > 0 and height > 0 else 1.0
        else:
            self.scale_factor = 1.0

        self.offset_x = (label_size.width() - scaled_pixmap.width()) // 2
        self.offset_y = (label_size.height() - scaled_pixmap.height()) // 2

    def mousePressEvent(self, event):
        if not self.tracks:
            return

        click_x = (event.pos().x() - self.offset_x) / self.scale_factor
        click_y = (event.pos().y() - self.offset_y) / self.scale_factor
        
        self.objectClicked.emit(f"{click_x},{click_y}")


class BehaviorTagger(QMainWindow):
    def __init__(self, video_path, output_path):
        super().__init__()
        self.video_path = video_path
        self.output_path = output_path
        self.tracks_path = os.path.splitext(video_path)[0] + '.json'
        
        self.data = {}
        self.tracks = {}
        self.behaviors = []
        self.video_capture = None
        self.total_frames = 0
        self.current_frame_number = 1
        
        self.init_ui()
        self.setup_shortcuts()
        self.initial_load()

    def init_ui(self):
        self.setWindowTitle("Behavior Tagger")
        self.setGeometry(100, 100, 1600, 900)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        main_splitter = QSplitter(Qt.Horizontal)
        
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        self.frame_display = VideoFrameWidget()
        self.frame_display.objectClicked.connect(self.handle_object_click)
        left_layout.addWidget(self.frame_display)
        
        frame_controls_group = QGroupBox("Frame Controls")
        frame_controls_layout = QVBoxLayout(frame_controls_group)
        
        slider_layout = QHBoxLayout()
        self.prev_frame_btn = QPushButton("◀")
        self.prev_frame_btn.clicked.connect(self.prev_frame)
        self.prev_frame_btn.setEnabled(False)
        slider_layout.addWidget(self.prev_frame_btn)

        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setMinimum(1)
        self.frame_slider.valueChanged.connect(self.seek_frame)
        self.frame_slider.setEnabled(False)
        slider_layout.addWidget(self.frame_slider)

        self.next_frame_btn = QPushButton("▶")
        self.next_frame_btn.clicked.connect(self.next_frame)
        self.next_frame_btn.setEnabled(False)
        slider_layout.addWidget(self.next_frame_btn)
        
        self.frame_label = QLabel("1 / 1")
        slider_layout.addWidget(self.frame_label)
        frame_controls_layout.addLayout(slider_layout)
        
        left_layout.addWidget(frame_controls_group)
        
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        track_group = QGroupBox("Available Tracks")
        track_layout = QVBoxLayout(track_group)
        self.track_list = QListWidget()
        self.track_list.itemClicked.connect(self.on_track_selected)
        track_layout.addWidget(self.track_list)
        right_layout.addWidget(track_group)
        
        tag_group = QGroupBox("Tag Behavior")
        tag_layout = QVBoxLayout(tag_group)
        
        self.selected_track_label = QLabel("Selected Track: None")
        self.selected_track_label.setFont(QFont("Arial", 10, QFont.Bold))
        tag_layout.addWidget(self.selected_track_label)
        
        frame_layout = QHBoxLayout()
        frame_layout.addWidget(QLabel("Start Frame:"))
        self.start_frame_spin = QSpinBox()
        self.start_frame_spin.setMinimum(1)
        self.start_frame_spin.setMaximum(999999)
        frame_layout.addWidget(self.start_frame_spin)
        
        frame_layout.addWidget(QLabel("End Frame:"))
        self.end_frame_spin = QSpinBox()
        self.end_frame_spin.setMinimum(1)
        self.end_frame_spin.setMaximum(999999)
        frame_layout.addWidget(self.end_frame_spin)
        tag_layout.addLayout(frame_layout)
        
        frame_buttons_layout = QHBoxLayout()
        self.set_start_btn = QPushButton("Set Start to Current")
        self.set_start_btn.clicked.connect(self.set_start_frame)
        self.set_start_btn.setEnabled(False)
        frame_buttons_layout.addWidget(self.set_start_btn)
        
        self.set_end_btn = QPushButton("Set End to Current")
        self.set_end_btn.clicked.connect(self.set_end_frame)
        self.set_end_btn.setEnabled(False)
        frame_buttons_layout.addWidget(self.set_end_btn)
        tag_layout.addLayout(frame_buttons_layout)
        
        behavior_form_layout = QFormLayout()
        self.movement_combo = QComboBox()
        self.movement_combo.addItems(["moving", "stopped"])
        behavior_form_layout.addRow("Movement:", self.movement_combo)
        self.maneuver_input = QLineEdit()
        self.maneuver_input.setPlaceholderText("Enter maneuver type")
        behavior_form_layout.addRow("Maneuver Type:", self.maneuver_input)
        self.lane_change_combo = QComboBox()
        self.lane_change_combo.addItems(["no_change", "left", "right"])
        behavior_form_layout.addRow("Lane Change:", self.lane_change_combo)
        tag_layout.addLayout(behavior_form_layout)
        
        self.add_behavior_btn = QPushButton("Add Behavior")
        self.add_behavior_btn.clicked.connect(self.add_behavior)
        self.add_behavior_btn.setEnabled(False)
        tag_layout.addWidget(self.add_behavior_btn)
        
        right_layout.addWidget(tag_group)
        
        behaviors_group = QGroupBox("Tagged Behaviors")
        behaviors_layout = QVBoxLayout(behaviors_group)
        
        self.behaviors_table = QTableWidget()
        self.behaviors_table.setColumnCount(6)
        self.behaviors_table.setHorizontalHeaderLabels(["Track ID", "Start Frame", "End Frame", "Movement", "Maneuver Type", "Lane Change"])
        self.behaviors_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        behaviors_layout.addWidget(self.behaviors_table)
        
        self.remove_behavior_btn = QPushButton("Remove Selected")
        self.remove_behavior_btn.clicked.connect(self.remove_behavior)
        behaviors_layout.addWidget(self.remove_behavior_btn)
        
        right_layout.addWidget(behaviors_group)
        
        self.save_btn = QPushButton("Save Behaviors")
        self.save_btn.clicked.connect(self.save_behaviors)
        self.save_btn.setEnabled(False)
        right_layout.addWidget(self.save_btn)
        
        left_widget.setMinimumWidth(800)
        right_widget.setMinimumWidth(400)
        
        main_splitter.addWidget(left_widget)
        main_splitter.addWidget(right_widget)
        main_splitter.setStretchFactor(0, 3)
        main_splitter.setStretchFactor(1, 1)
        
        layout.addWidget(main_splitter)

    def setup_shortcuts(self):
        self.left_shortcut = QShortcut(QKeySequence(Qt.Key_Left), self)
        self.left_shortcut.activated.connect(self.prev_frame)
        self.right_shortcut = QShortcut(QKeySequence(Qt.Key_Right), self)
        self.right_shortcut.activated.connect(self.next_frame)
        self.a_shortcut = QShortcut(QKeySequence(Qt.Key_A), self)
        self.a_shortcut.activated.connect(self.set_start_frame)
        self.s_shortcut = QShortcut(QKeySequence(Qt.Key_S), self)
        self.s_shortcut.activated.connect(self.set_end_frame)
        self.enter_shortcut = QShortcut(QKeySequence(Qt.Key_Return), self)
        self.enter_shortcut.activated.connect(self.add_behavior)
    
    def initial_load(self):
        if not os.path.exists(self.video_path):
            QMessageBox.critical(self, "Error", f"Video file not found:\n{self.video_path}")
            self.close()
            return
            
        if not os.path.exists(self.tracks_path):
            QMessageBox.critical(self, "Error", f"Tracks JSON file not found:\n{self.tracks_path}")
            self.close()
            return

        self.video_capture = cv2.VideoCapture(self.video_path)
        if not self.video_capture.isOpened():
            QMessageBox.critical(self, "Error", "Failed to open video file.")
            self.close()
            return
            
        self.total_frames = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_slider.setMaximum(self.total_frames)
        self.frame_slider.setEnabled(True)
        self.prev_frame_btn.setEnabled(True)
        self.next_frame_btn.setEnabled(True)
        self.set_start_btn.setEnabled(True)
        self.set_end_btn.setEnabled(True)
        self.save_btn.setEnabled(True)

        try:
            with open(self.tracks_path, "r") as f:
                self.data = json.load(f)
                self.tracks = self.data.get("tracks", {})
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load tracks file: {e}")
            self.close()
            return

        if os.path.exists(self.output_path):
            try:
                with open(self.output_path, "r") as f:
                    output_data = json.load(f)
                    self.behaviors = output_data.get("behaviors", [])
            except Exception as e:
                QMessageBox.warning(self, "Warning", f"Could not load existing behaviors file: {e}")

        self.frame_display.set_tracks(self.tracks)
        self.frame_display.set_behaviors(self.behaviors)
        self.populate_track_list()
        self.update_behaviors_table()
        self.seek_to_frame(1)

    def populate_track_list(self):
        self.track_list.clear()
        for track_id, track_data in self.tracks.items():
            first = track_data.get("first_frame", "N/A")
            last = track_data.get("last_frame", "N/A")
            class_id = track_data.get("class", "Unknown")
            
            if "detections" in track_data and track_data["detections"]:
                if first == "N/A":
                    first = min(det["frame"] for det in track_data["detections"])
                if last == "N/A":
                    last = max(det["frame"] for det in track_data["detections"])
            
            self.track_list.addItem(f"Track {track_id} (Class: {class_id}, Frames: {first}-{last})")

    def update_behaviors_table(self):
        self.behaviors_table.setRowCount(len(self.behaviors))
        for i, behavior in enumerate(self.behaviors):
            self.behaviors_table.setItem(i, 0, QTableWidgetItem(str(behavior["track_id"])))
            self.behaviors_table.setItem(i, 1, QTableWidgetItem(str(behavior["start_frame"])))
            self.behaviors_table.setItem(i, 2, QTableWidgetItem(str(behavior["end_frame"])))
            
            b_data = behavior.get("behavior", {})
            if isinstance(b_data, dict):
                self.behaviors_table.setItem(i, 3, QTableWidgetItem(b_data.get("movement", "")))
                self.behaviors_table.setItem(i, 4, QTableWidgetItem(b_data.get("maneuver_type", "")))
                self.behaviors_table.setItem(i, 5, QTableWidgetItem(b_data.get("lane_change", "")))
            else:
                self.behaviors_table.setItem(i, 3, QTableWidgetItem(""))
                self.behaviors_table.setItem(i, 4, QTableWidgetItem(str(b_data)))
                self.behaviors_table.setItem(i, 5, QTableWidgetItem(""))

    def on_track_selected(self, item):
        track_id = item.text().split()[1]
        self.select_track(track_id)

    def handle_object_click(self, pos_str):
        x_click, y_click = map(float, pos_str.split(','))
        
        selected_track = None
        min_area = float('inf')

        for track_id, track_data in self.tracks.items():
            detection = None
            
            if "detections" in track_data:
                for det in track_data["detections"]:
                    if det["frame"] == self.current_frame_number:
                        detection = det
                        break
            elif "frames" in track_data and str(self.current_frame_number) in track_data["frames"]:
                detection = track_data["frames"][str(self.current_frame_number)]

            if detection:
                if "obb" in detection:
                    corners = detection["obb"]["corners"]
                    path = cv2.pointPolygonTest(np.array(corners, np.float32), (x_click, y_click), False)
                    if path >= 0:
                        center = detection["obb"]["center"]
                        size = detection["obb"]["size"]
                        area = size[0] * size[1]
                        if area < min_area:
                            min_area = area
                            selected_track = track_id
                elif "bbox" in detection:
                    x, y, w, h = detection["bbox"]
                    if x <= x_click <= x + w and y <= y_click <= y + h:
                        area = w * h
                        if area < min_area:
                            min_area = area
                            selected_track = track_id
        
        if selected_track:
            self.select_track(selected_track)

    def select_track(self, track_id):
        if track_id in self.tracks:
            self.frame_display.selected_track_id = track_id
            track_data = self.tracks[track_id]
            first = track_data.get("first_frame", 1)
            last = track_data.get("last_frame", self.total_frames)
            
            self.selected_track_label.setText(f"Selected Track: {track_id}")
            self.start_frame_spin.setMinimum(first)
            self.start_frame_spin.setMaximum(last)
            self.start_frame_spin.setValue(self.current_frame_number)
            
            self.end_frame_spin.setMinimum(first)
            self.end_frame_spin.setMaximum(last)
            self.end_frame_spin.setValue(self.current_frame_number)
            
            self.add_behavior_btn.setEnabled(True)
            self.seek_to_frame(self.current_frame_number)
            
            for i in range(self.track_list.count()):
                item = self.track_list.item(i)
                if f"Track {track_id}" in item.text():
                    self.track_list.setCurrentItem(item)
                    break

    def prev_frame(self):
        if self.current_frame_number > 1:
            self.seek_to_frame(self.current_frame_number - 1)

    def next_frame(self):
        if self.current_frame_number < self.total_frames:
            self.seek_to_frame(self.current_frame_number + 1)

    def seek_frame(self, frame_number):
        if frame_number != self.current_frame_number:
            self.seek_to_frame(frame_number)

    def seek_to_frame(self, frame_number):
        if not self.video_capture or not self.video_capture.isOpened():
            return
        
        self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
        ret, frame = self.video_capture.read()
        
        self.current_frame_number = frame_number
        if ret:
            self.frame_slider.blockSignals(True)
            self.frame_slider.setValue(frame_number)
            self.frame_slider.blockSignals(False)
            self.frame_label.setText(f"{frame_number} / {self.total_frames}")
            self.frame_display.display_frame(frame, frame_number)
        else:
            self.frame_display.display_frame(None, frame_number)


    def set_start_frame(self):
        if self.frame_display.selected_track_id:
            self.start_frame_spin.setValue(self.current_frame_number)

    def set_end_frame(self):
        if self.frame_display.selected_track_id:
            self.end_frame_spin.setValue(self.current_frame_number)

    def add_behavior(self):
        track_text = self.selected_track_label.text()
        if "None" in track_text:
            QMessageBox.warning(self, "Error", "Please select a track first.")
            return
        
        track_id = self.frame_display.selected_track_id
        start = self.start_frame_spin.value()
        end = self.end_frame_spin.value()
        
        movement = self.movement_combo.currentText()
        maneuver_type = self.maneuver_input.text().strip()
        lane_change = self.lane_change_combo.currentText()
        
        if not maneuver_type:
            QMessageBox.warning(self, "Error", "Please enter a maneuver type.")
            return
        
        if start > end:
            QMessageBox.warning(self, "Error", "Start frame must be <= end frame.")
            return
        
        behavior_data = {
            "movement": movement,
            "maneuver_type": maneuver_type,
            "lane_change": lane_change
        }
        
        self.behaviors.append({
            "track_id": track_id, "start_frame": start, "end_frame": end, "behavior": behavior_data
        })
        self.update_behaviors_table()
        self.frame_display.set_behaviors(self.behaviors)
        self.seek_to_frame(self.current_frame_number)
        self.maneuver_input.clear()
        self.movement_combo.setCurrentIndex(0)
        self.lane_change_combo.setCurrentIndex(0)

    def remove_behavior(self):
        current_row = self.behaviors_table.currentRow()
        if current_row >= 0:
            del self.behaviors[current_row]
            self.update_behaviors_table()
            self.frame_display.set_behaviors(self.behaviors)
            self.seek_to_frame(self.current_frame_number)

    def save_behaviors(self):
        try:
            self.data["behaviors"] = self.behaviors
            os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
            
            # Use the loaded data as base if it exists to preserve other top-level keys
            save_data = self.data
            if os.path.exists(self.tracks_path):
                 with open(self.tracks_path, "r") as f:
                    # Don't overwrite tracks, just add/update behaviors
                    save_data = json.load(f)
            save_data["behaviors"] = self.behaviors

            with open(self.output_path, "w") as f:
                json.dump(save_data, f, indent=2)
            QMessageBox.information(self, "Success", f"Saved {len(self.behaviors)} behaviors to\n{self.output_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save behaviors: {e}")
            
    def closeEvent(self, event):
        reply = QMessageBox.question(self, 'Confirm Exit',
                                     "Do you want to save your changes before exiting?",
                                     QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
                                     QMessageBox.Save)

        if reply == QMessageBox.Save:
            self.save_behaviors()
            if self.video_capture:
                self.video_capture.release()
            event.accept()
        elif reply == QMessageBox.Discard:
            if self.video_capture:
                self.video_capture.release()
            event.accept()
        else:
            event.ignore()