#Configuration file for the manual annotator tool
#This file contains default settings and paths used by the tool.

import os

DEFAULT_MODEL_PATH = "models/yolo11m-obb.pt"
DEFAULT_OUTPUT_DIR = "output"

DETECTION_CONFIG = {
    "confidence_threshold": 0.2,
    "iou_threshold": 0.3,
    "target_classes": [9, 10],
    "image_extensions": [".jpg", ".jpeg", ".png", ".bmp", ".tiff"],
}

TRACKING_CONFIG = {
    "confidence_threshold": 0.15,
    "target_classes": [9, 10],
    "tracker": "botsort.yaml",
    "miss_threshold": 100,
    "max_history": 400,
    "video_extensions": [".mp4", ".avi", ".mov", ".mkv", ".wmv"],
}

GUI_CONFIG = {
    "window_name": "Manual Annotator",
    "status_font": 0,
    "status_scale": 0.7,
    "status_color": (255, 255, 255),
    "status_thickness": 2,
    "box_color": (0, 255, 0),
    "box_thickness": 2,
    "temp_box_color": (0, 0, 255),
    "temp_box_thickness": 1,
}

OUTPUT_STRUCTURE = {
    "detections": "detections",
    "tracking": "tracking", 
    "manual": "manual",
    "behaviors": "behaviors",
    "merged": "merged",
}

def get_output_path(base_dir, category):
    return os.path.join(base_dir, OUTPUT_STRUCTURE[category])

def ensure_directories(base_dir):
    for category in OUTPUT_STRUCTURE.values():
        os.makedirs(os.path.join(base_dir, category), exist_ok=True)