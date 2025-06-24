#This code is a Python script that uses the YOLO object detection model to detect objects in an image.
#It reads an image, applies the YOLO model to detect objects, and saves the results in a JSON file and an annotated image.
#It supports both Oriented Bounding Boxes (OBB) and standard bounding boxes, depending on the model's capabilities.
#It also handles the case where the image cannot be loaded, and it creates necessary directories for saving results.
#The script uses the Ultralytics YOLO library for object detection and OpenCV for image processing.
#It also uses the Supervision library for handling bounding boxes and annotations.

import os
import cv2
import json
import numpy as np
from ultralytics import YOLO
import supervision as sv

def detect_image(image_path, model_path, output_dir, conf_threshold=0.2):
    os.makedirs(output_dir, exist_ok=True)
    
    model = YOLO(model_path)
    img = cv2.imread(image_path)
    
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    results = model(img, conf=conf_threshold, iou=0.3, classes=[9, 10])
    result = results[0]
    
    tracks = {}
    annotated_img = img.copy()
    
    if hasattr(result, 'obb') and result.obb is not None:
        for i, (obb, conf, cls) in enumerate(zip(result.obb.xyxyxyxy, result.obb.conf, result.obb.cls)):
            points = obb.cpu().numpy()
            confidence = float(conf.cpu().numpy())
            class_id = int(cls.cpu().numpy())
            
            center_x = float(np.mean(points[:, 0]))
            center_y = float(np.mean(points[:, 1]))
            
            width = float(np.linalg.norm(points[1] - points[0]))
            height = float(np.linalg.norm(points[3] - points[0]))
            
            v1 = points[1] - points[0]
            rotation = float(np.arctan2(v1[1], v1[0]))
            
            track_id = i + 1
            tracks[str(track_id)] = {
                "track_id": track_id,
                "class": class_id,
                "first_frame": 1,
                "last_frame": 1,
                "detections": [{
                    "frame": 1,
                    "obb": {
                        "center": [center_x, center_y],
                        "size": [width, height],
                        "rotation": rotation,
                        "corners": points.tolist()
                    },
                    "confidence": confidence
                }]
            }
            
            points_int = points.astype(int)
            cv2.polylines(annotated_img, [points_int.reshape(-1, 1, 2)], True, (0, 255, 0), 2)
            cv2.circle(annotated_img, (int(center_x), int(center_y)), 3, (0, 0, 255), -1)
    
    elif hasattr(result, 'boxes') and result.boxes is not None:
        for i, (box, conf, cls) in enumerate(zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls)):
            x1, y1, x2, y2 = box.cpu().numpy()
            confidence = float(conf.cpu().numpy())
            class_id = int(cls.cpu().numpy())
            cx = float((x1 + x2) / 2)
            cy = float((y1 + y2) / 2)
            
            track_id = i + 1
            tracks[str(track_id)] = {
                "track_id": track_id,
                "class": class_id,
                "first_frame": 1,
                "last_frame": 1,
                "detections": [{
                    "frame": 1,
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "center": [cx, cy],
                    "confidence": confidence
                }]
            }
            
            cv2.rectangle(annotated_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.circle(annotated_img, (int(cx), int(cy)), 3, (0, 0, 255), -1)
    
    output_json = os.path.join(output_dir, "detections.json")
    with open(output_json, "w") as f:
        json.dump({"tracks": tracks}, f, indent=2)
    
    output_img = os.path.join(output_dir, "annotated_image.jpg")
    cv2.imwrite(output_img, annotated_img)
    
    print(f"Detection complete. Found {len(tracks)} objects.")
    print(f"Results saved to: {output_dir}")