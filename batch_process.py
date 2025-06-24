#Batch processing code for image detection and video tracking for multiple images, and videos
#for many different formats in a directory.
#It uses the YOLO model for object detection in images and video tracking.

import os
import glob
from modules import detect_image, track_video

def batch_detect_images(input_dir, model_path, output_dir, conf_threshold=0.2):
    os.makedirs(output_dir, exist_ok=True)

    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    image_files = []

    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_dir, ext)))
        image_files.extend(glob.glob(os.path.join(input_dir, ext.upper())))

    if not image_files:
        return "No image files found in {}".format(input_dir)

    log_message = f"Found {len(image_files)} images to process.\n"

    for i, image_path in enumerate(image_files, 1):
        log_message += f"Processing {i}/{len(image_files)}: {os.path.basename(image_path)}\n"
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        image_output_dir = os.path.join(output_dir, image_name)

        try:
            detect_image(image_path, model_path, image_output_dir, conf_threshold)
        except Exception as e:
            log_message += f"Error processing {image_path}: {e}\n"
    return log_message

def batch_track_videos(input_dir, model_path, output_dir, conf_threshold=0.15):
    os.makedirs(output_dir, exist_ok=True)

    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.wmv']
    video_files = []

    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(input_dir, ext)))
        video_files.extend(glob.glob(os.path.join(input_dir, ext.upper())))

    if not video_files:
        return "No video files found in {}".format(input_dir)

    log_message = f"Found {len(video_files)} videos to process.\n"

    for i, video_path in enumerate(video_files, 1):
        log_message += f"Processing {i}/{len(video_files)}: {os.path.basename(video_path)}\n"
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        video_output_dir = os.path.join(output_dir, video_name)

        try:
            track_video(video_path, model_path, video_output_dir, conf_threshold)
        except Exception as e:
            log_message += f"Error processing {video_path}: {e}\n"
    return log_message