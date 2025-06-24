#Utility functions for file handling, JSON operations, and progress tracking.
#This module provides functions to validate file paths, create directories,
#load and save JSON files, format time, and display progress bars.
#It also includes a Timer class for measuring elapsed time.

import os
import json
import time
from datetime import datetime

def validate_file_path(file_path, extensions=None):
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if extensions:
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext not in extensions:
            raise ValueError(f"Invalid file extension. Expected: {extensions}")
    
    return True

def validate_model_path(model_path):
    validate_file_path(model_path, ['.pt'])
    return True

def create_output_directory(output_path):
    os.makedirs(output_path, exist_ok=True)
    return True

def load_json_safe(file_path):
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError as e:
        print(f"Error reading JSON file {file_path}: {e}")
        return {}

def save_json_safe(data, file_path):
    try:
        create_output_directory(os.path.dirname(file_path))
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving JSON file {file_path}: {e}")
        return False

def get_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"

def print_progress(current, total, prefix="Progress", suffix="Complete", bar_length=50):
    progress = float(current) / float(total)
    arrow = '-' * int(round(progress * bar_length) - 1) + '>'
    spaces = ' ' * (bar_length - len(arrow))
    
    print(f'\r{prefix}: [{arrow}{spaces}] {int(progress * 100)}% {suffix}', end='', flush=True)
    
    if current == total:
        print()

class Timer:
    def __init__(self):
        self.start_time = None
        self.end_time = None
    
    def start(self):
        self.start_time = time.time()
    
    def stop(self):
        self.end_time = time.time()
        return self.elapsed()
    
    def elapsed(self):
        if self.start_time is None:
            return 0
        end = self.end_time or time.time()
        return end - self.start_time