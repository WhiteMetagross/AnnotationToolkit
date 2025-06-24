#Vehicle Annotation Toolbox
#This code provides a GUI for annotating vehicle images and videos using YOLO models.
#It allows users to perform image detection, video tracking, manual annotation, behavior tagging, merging annotations, and batch processing.
#It uses PySide6 for the GUI and supports various functionalities through separate modules.

import sys
import os
import subprocess
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QFileDialog, QLabel, QLineEdit, QTabWidget, QDoubleSpinBox, QMessageBox
)
from PySide6.QtCore import QThread, Signal

from modules import detect_image, track_video, merge_annotations
from config import DEFAULT_MODEL_PATH, DEFAULT_OUTPUT_DIR
from utils import validate_file_path, validate_model_path, create_output_directory
from batch_process import batch_detect_images, batch_track_videos

class Worker(QThread):
    finished = Signal(str)
    error = Signal(str)

    def __init__(self, target, *args, **kwargs):
        super().__init__()
        self.target = target
        self.args = args
        self.kwargs = kwargs

    def run(self):
        try:
            result = self.target(*self.args, **self.kwargs)
            self.finished.emit(str(result) if result is not None else "Process finished successfully.")
        except Exception as e:
            self.error.emit(str(e))

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Vehicle Annotation Toolbox")
        self.setGeometry(100, 100, 700, 400)

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self.create_detect_tab()
        self.create_track_tab()
        self.create_manual_tab()
        self.create_behavior_tab()
        self.create_merge_tab()
        self.create_batch_tab()
        
        self.external_processes = []

    def create_detect_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        self.detect_input_path = self.create_file_input("Input Image:", "Select Image")
        self.detect_model_path = self.create_file_input("YOLO Model:", "Select Model", DEFAULT_MODEL_PATH)
        self.detect_output_dir = self.create_folder_input("Output Directory:", "Select Directory", DEFAULT_OUTPUT_DIR)

        conf_layout = QHBoxLayout()
        conf_layout.addWidget(QLabel("Confidence Threshold:"))
        self.detect_conf = QDoubleSpinBox()
        self.detect_conf.setRange(0.0, 1.0)
        self.detect_conf.setSingleStep(0.05)
        self.detect_conf.setValue(0.2)
        conf_layout.addWidget(self.detect_conf)

        run_button = QPushButton("Run Detection")
        run_button.clicked.connect(self.run_detection)

        layout.addLayout(self.detect_input_path)
        layout.addLayout(self.detect_model_path)
        layout.addLayout(self.detect_output_dir)
        layout.addLayout(conf_layout)
        layout.addWidget(run_button)
        layout.addStretch()
        self.tabs.addTab(tab, "Detect")

    def create_track_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        self.track_input_path = self.create_file_input("Input Video:", "Select Video")
        self.track_model_path = self.create_file_input("YOLO Model:", "Select Model", DEFAULT_MODEL_PATH)
        self.track_output_dir = self.create_folder_input("Output Directory:", "Select Directory", DEFAULT_OUTPUT_DIR)

        conf_layout = QHBoxLayout()
        conf_layout.addWidget(QLabel("Confidence Threshold:"))
        self.track_conf = QDoubleSpinBox()
        self.track_conf.setRange(0.0, 1.0)
        self.track_conf.setSingleStep(0.05)
        self.track_conf.setValue(0.15)
        conf_layout.addWidget(self.track_conf)

        run_button = QPushButton("Run Tracking")
        run_button.clicked.connect(self.run_tracking)

        layout.addLayout(self.track_input_path)
        layout.addLayout(self.track_model_path)
        layout.addLayout(self.track_output_dir)
        layout.addLayout(conf_layout)
        layout.addWidget(run_button)
        layout.addStretch()
        self.tabs.addTab(tab, "Track")

    def create_manual_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        self.manual_input_path = self.create_file_input("Input Video/Image:", "Select File")
        self.manual_output_file = self.create_file_input("Output Annotation File:", "Save Annotation File", file_mode=QFileDialog.FileMode.AnyFile, save=True)
        self.manual_load_file = self.create_file_input("Load Existing Annotations (Optional):", "Select Annotation File")

        run_button = QPushButton("Start Manual Annotation")
        run_button.clicked.connect(self.run_manual_annotation)

        layout.addLayout(self.manual_input_path)
        layout.addLayout(self.manual_output_file)
        layout.addLayout(self.manual_load_file)
        layout.addWidget(run_button)
        layout.addStretch()
        self.tabs.addTab(tab, "Manual Annotate")

    def create_behavior_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        self.behavior_input_file = self.create_file_input("Input Video File:", "Select Video")
        self.behavior_output_file = self.create_file_input("Output Behaviors File:", "Save Behaviors File", file_mode=QFileDialog.FileMode.AnyFile, save=True)
        self.behavior_load_file = self.create_file_input("Load Existing Behaviors (Optional):", "Select Behaviors File")

        run_button = QPushButton("Tag Behaviors")
        run_button.clicked.connect(self.run_behavior_tagging)

        layout.addLayout(self.behavior_input_file)
        layout.addLayout(self.behavior_output_file)
        layout.addLayout(self.behavior_load_file)
        layout.addWidget(run_button)
        layout.addStretch()
        self.tabs.addTab(tab, "Tag Behavior")

    def create_merge_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        self.merge_manual_file = self.create_file_input("Manual Annotations (Optional):", "Select File")
        self.merge_tracks_file = self.create_file_input("Tracks File (Optional):", "Select File")
        self.merge_behaviors_file = self.create_file_input("Behaviors File (Optional):", "Select File")
        self.merge_output_file = self.create_file_input("Merged Output File:", "Save Merged File", file_mode=QFileDialog.FileMode.AnyFile, save=True)

        run_button = QPushButton("Merge Annotations")
        run_button.clicked.connect(self.run_merge)

        layout.addLayout(self.merge_manual_file)
        layout.addLayout(self.merge_tracks_file)
        layout.addLayout(self.merge_behaviors_file)
        layout.addLayout(self.merge_output_file)
        layout.addWidget(run_button)
        layout.addStretch()
        self.tabs.addTab(tab, "Merge")

    def create_batch_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        self.batch_input_dir = self.create_folder_input("Input Directory:", "Select Directory")
        self.batch_model_path = self.create_file_input("YOLO Model:", "Select Model", DEFAULT_MODEL_PATH)
        self.batch_output_dir = self.create_folder_input("Output Directory:", "Select Directory")

        run_detect_button = QPushButton("Run Batch Detection")
        run_detect_button.clicked.connect(lambda: self.run_batch('detect'))

        run_track_button = QPushButton("Run Batch Tracking")
        run_track_button.clicked.connect(lambda: self.run_batch('track'))

        layout.addLayout(self.batch_input_dir)
        layout.addLayout(self.batch_model_path)
        layout.addLayout(self.batch_output_dir)
        layout.addWidget(run_detect_button)
        layout.addWidget(run_track_button)
        layout.addStretch()
        self.tabs.addTab(tab, "Batch Process")

    def create_file_input(self, label_text, dialog_title, default_path="", file_mode=QFileDialog.FileMode.ExistingFile, save=False):
        layout = QHBoxLayout()
        label = QLabel(label_text)
        line_edit = QLineEdit(default_path)
        button = QPushButton("Browse")

        def browse():
            if save:
                path, _ = QFileDialog.getSaveFileName(self, dialog_title, filter="JSON files (*.json);;All files (*)")
            else:
                dialog = QFileDialog(self, dialog_title)
                dialog.setFileMode(file_mode)
                if dialog.exec():
                    path = dialog.selectedFiles()[0]
                else:
                    path = ""
            if path:
                line_edit.setText(path)

        button.clicked.connect(browse)
        layout.addWidget(label)
        layout.addWidget(line_edit)
        layout.addWidget(button)
        return layout

    def create_folder_input(self, label_text, dialog_title, default_path=""):
        layout = QHBoxLayout()
        label = QLabel(label_text)
        line_edit = QLineEdit(default_path)
        button = QPushButton("Browse")

        def browse():
            path = QFileDialog.getExistingDirectory(self, dialog_title)
            if path:
                line_edit.setText(path)

        button.clicked.connect(browse)
        layout.addWidget(label)
        layout.addWidget(line_edit)
        layout.addWidget(button)
        return layout

    def get_path_from_layout(self, layout):
        return layout.itemAt(1).widget().text()

    def run_detection(self):
        input_path = self.get_path_from_layout(self.detect_input_path)
        model_path = self.get_path_from_layout(self.detect_model_path)
        output_dir = self.get_path_from_layout(self.detect_output_dir)
        conf = self.detect_conf.value()

        if not all([input_path, model_path, output_dir]):
            self.show_error("All fields are required.")
            return

        validate_file_path(input_path)
        validate_model_path(model_path)
        create_output_directory(output_dir)

        self.run_task(detect_image, input_path, model_path, output_dir, conf)

    def run_tracking(self):
        input_path = self.get_path_from_layout(self.track_input_path)
        model_path = self.get_path_from_layout(self.track_model_path)
        output_dir = self.get_path_from_layout(self.track_output_dir)
        conf = self.track_conf.value()

        if not all([input_path, model_path, output_dir]):
            self.show_error("All fields are required.")
            return

        validate_file_path(input_path)
        validate_model_path(model_path)
        create_output_directory(output_dir)

        self.run_task(track_video, input_path, model_path, output_dir, conf)

    def run_manual_annotation(self):
        input_path = self.get_path_from_layout(self.manual_input_path)
        output_file = self.get_path_from_layout(self.manual_output_file)
        load_file = self.get_path_from_layout(self.manual_load_file)

        if not all([input_path, output_file]):
            self.show_error("Input path and output file are required.")
            return
        
        command = [sys.executable, 'modules/annotator.py', input_path, output_file]
        if load_file and os.path.exists(load_file):
            command.append(load_file)
        
        process = subprocess.Popen(command)
        self.external_processes.append(process)

    def run_behavior_tagging(self):
        input_file = self.get_path_from_layout(self.behavior_input_file)
        output_file = self.get_path_from_layout(self.behavior_output_file)
        load_file = self.get_path_from_layout(self.behavior_load_file)

        if not all([input_file, output_file]):
            self.show_error("Input and output files are required.")
            return

        function_call = f"tag_behaviors(video_path={repr(input_file)}, output_path={repr(output_file)}"
        if load_file and os.path.exists(load_file):
            function_call += f", load={repr(load_file)}"
        function_call += ")"
        
        py_command = f"from modules.behavior import tag_behaviors; {function_call}"
        command = [sys.executable, "-c", py_command]
            
        process = subprocess.Popen(command)
        self.external_processes.append(process)

    def run_merge(self):
        manual_file = self.get_path_from_layout(self.merge_manual_file)
        tracks_file = self.get_path_from_layout(self.merge_tracks_file)
        behaviors_file = self.get_path_from_layout(self.merge_behaviors_file)
        output_file = self.get_path_from_layout(self.merge_output_file)

        if not output_file:
            self.show_error("Output file is required.")
            return

        create_output_directory(os.path.dirname(output_file))
        self.run_task(merge_annotations, manual_file or None, tracks_file or None, behaviors_file or None, output_file)

    def run_batch(self, mode):
        input_dir = self.get_path_from_layout(self.batch_input_dir)
        model_path = self.get_path_from_layout(self.batch_model_path)
        output_dir = self.get_path_from_layout(self.batch_output_dir)

        if not all([input_dir, model_path, output_dir]):
            self.show_error("All fields are required.")
            return

        validate_model_path(model_path)
        create_output_directory(output_dir)

        if mode == 'detect':
            self.run_task(batch_detect_images, input_dir, model_path, output_dir)
        elif mode == 'track':
            self.run_task(batch_track_videos, input_dir, model_path, output_dir)

    def run_task(self, target, *args, **kwargs):
        self.worker = Worker(target, *args, **kwargs)
        self.worker.finished.connect(self.task_finished)
        self.worker.error.connect(self.task_error)
        self.worker.start()
        self.show_message("Processing...", "Task is running in the background.")

    def task_finished(self, message):
        self.show_message("Success", message)

    def task_error(self, error_message):
        self.show_error(f"An error occurred: {error_message}")

    def show_message(self, title, message):
        QMessageBox.information(self, title, message)

    def show_error(self, message):
        QMessageBox.critical(self, "Error", message)

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()