#This code does automated tracking of objects in a video using the YOLO model, drawing bounding boxes 
#or oriented bounding boxes (OBBs) around detected objects, and saving the results in both video and JSON formats. 
#It also handles tracking history and missed frames for each object.

import os
import cv2
import json
import numpy as np
import colorsys
from ultralytics import YOLO
from config import TRACKING_CONFIG
from utils import save_json_safe, get_timestamp, Timer, print_progress

def get_color_for_track(trackID):
    trackID = int(trackID)
    hue = (trackID * 0.618033988749895) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 1, 1)
    return (int(r * 255), int(g * 255), int(b * 255))

def _calculate_obb_corners(x_center, y_center, width, height, rotation):
    cos_r = np.cos(rotation)
    sin_r = np.sin(rotation)
    dx = width / 2
    dy = height / 2
    corners = np.array([
        [-dx, -dy],
        [dx, -dy],
        [dx, dy],
        [-dx, dy]
    ])
    rotation_matrix = np.array([
        [cos_r, -sin_r],
        [sin_r, cos_r]
    ])
    rotated_corners = corners @ rotation_matrix.T
    rotated_corners[:, 0] += x_center
    rotated_corners[:, 1] += y_center
    return rotated_corners

def draw_obb(img, obb_corners, color, thickness=2):
    corners = np.array(obb_corners, dtype=np.int32)
    cv2.polylines(img, [corners], True, color, thickness)

def draw_bbox(img, bbox, color, thickness=2):
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

def track_video(input_path, model_path, output_dir, conf_threshold=None):
    if conf_threshold is None:
        conf_threshold = TRACKING_CONFIG["confidence_threshold"]

    timer = Timer()
    timer.start()

    model = YOLO(model_path)

    cap = cv2.VideoCapture(input_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    input_filename = os.path.splitext(os.path.basename(input_path))[0]
    
    os.makedirs(output_dir, exist_ok=True)
    output_video_path = os.path.join(output_dir, f"{input_filename}_tracks.mp4")
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        print("Failed to open video writer with XVID, falling back to mp4v...")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    tracks_data = {
        "tracks": {},
        "frames": {}
    }

    trackingHistory = {}
    missedFrames = {}
    MISS_THRESHOLD = 100

    results = model.track(
        source=input_path,
        conf=conf_threshold,
        iou=TRACKING_CONFIG.get("iou_threshold", 0.3),
        classes=TRACKING_CONFIG["target_classes"],
        tracker="botsort.yaml",
        persist=True,
        verbose=False,
        stream=True
    )

    frame_num = 0
    for result in results:
        frame_num += 1
        print_progress(frame_num, total_frames, "Tracking")
        
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num - 1)
            ret, frame = cap.read()
        
        if not ret or frame is None:
            continue
            
        if frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height))
        
        plottedImg = frame.copy()
        currentRedCenters = []
        currentTrackIDs = set()
        frame_detections = []

        if hasattr(result, 'obb') and result.obb is not None and result.obb.id is not None:
            obb_data = result.obb.xywhr.cpu().numpy()
            track_ids = result.obb.id.cpu().numpy().astype(int)
            confidences = result.obb.conf.cpu().numpy()
            classes = result.obb.cls.cpu().numpy().astype(int)
            for i, track_id in enumerate(track_ids):
                if classes[i] in TRACKING_CONFIG["target_classes"]:
                    x_center, y_center, width_box, height_box, rotation = obb_data[i]
                    obb_corners = _calculate_obb_corners(x_center, y_center, width_box, height_box, rotation)
                    cx = int(x_center)
                    cy = int(y_center)
                    currentRedCenters.append((cx, cy))
                    currentTrackIDs.add(track_id)
                    
                    color = get_color_for_track(track_id)
                    draw_obb(plottedImg, obb_corners, color)
                    cv2.putText(plottedImg, str(track_id), (cx-10, cy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    detection = {
                        "track_id": int(track_id),
                        "obb": {
                            "center": [float(x_center), float(y_center)],
                            "size": [float(width_box), float(height_box)],
                            "rotation": float(rotation),
                            "corners": obb_corners.tolist()
                        },
                        "confidence": float(confidences[i]),
                        "class": int(classes[i]),
                        "frame": frame_num
                    }
                    frame_detections.append(detection)
                    if str(track_id) not in tracks_data["tracks"]:
                        tracks_data["tracks"][str(track_id)] = {
                            "track_id": int(track_id),
                            "class": int(classes[i]),
                            "first_frame": frame_num,
                            "last_frame": frame_num,
                            "detections": []
                        }
                    tracks_data["tracks"][str(track_id)]["last_frame"] = frame_num
                    tracks_data["tracks"][str(track_id)]["detections"].append({
                        "frame": frame_num,
                        "obb": {
                            "center": [float(x_center), float(y_center)],
                            "size": [float(width_box), float(height_box)],
                            "rotation": float(rotation),
                            "corners": obb_corners.tolist()
                        },
                        "confidence": float(confidences[i])
                    })
                    if track_id not in trackingHistory:
                        trackingHistory[track_id] = []
                    trackingHistory[track_id].append((cx, cy))
                    if len(trackingHistory[track_id]) > 400:
                        trackingHistory[track_id] = trackingHistory[track_id][-400:]
                    missedFrames[track_id] = 0

        elif hasattr(result, 'boxes') and result.boxes is not None and result.boxes.id is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            track_ids = result.boxes.id.cpu().numpy().astype(int)
            confidences = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)
            for i, track_id in enumerate(track_ids):
                if classes[i] in TRACKING_CONFIG["target_classes"]:
                    x1, y1, x2, y2 = boxes[i]
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    currentRedCenters.append((cx, cy))
                    currentTrackIDs.add(track_id)
                    
                    color = get_color_for_track(track_id)
                    draw_bbox(plottedImg, [x1, y1, x2, y2], color)
                    cv2.putText(plottedImg, str(track_id), (cx-10, cy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    detection = {
                        "track_id": int(track_id),
                        "bbox": [float(x1), float(y1), float(x2), float(y2)],
                        "confidence": float(confidences[i]),
                        "class": int(classes[i]),
                        "frame": frame_num
                    }
                    frame_detections.append(detection)
                    if str(track_id) not in tracks_data["tracks"]:
                        tracks_data["tracks"][str(track_id)] = {
                            "track_id": int(track_id),
                            "class": int(classes[i]),
                            "first_frame": frame_num,
                            "last_frame": frame_num,
                            "detections": []
                        }
                    tracks_data["tracks"][str(track_id)]["last_frame"] = frame_num
                    tracks_data["tracks"][str(track_id)]["detections"].append({
                        "frame": frame_num,
                        "bbox": [float(x1), float(y1), float(x2), float(y2)],
                        "confidence": float(confidences[i])
                    })
                    if track_id not in trackingHistory:
                        trackingHistory[track_id] = []
                    trackingHistory[track_id].append((cx, cy))
                    if len(trackingHistory[track_id]) > 400:
                        trackingHistory[track_id] = trackingHistory[track_id][-400:]
                    missedFrames[track_id] = 0

        for trackID in list(trackingHistory.keys()):
            if trackID not in currentTrackIDs:
                missedFrames[trackID] = missedFrames.get(trackID, 0) + 1
                if missedFrames[trackID] > MISS_THRESHOLD:
                    del trackingHistory[trackID]
                    del missedFrames[trackID]

        for trackID, pts in trackingHistory.items():
            color = get_color_for_track(trackID)
            if len(pts) >= 2:
                pts_np = np.array(pts, dtype=np.int32)
                cv2.polylines(plottedImg, [pts_np], False, color, 2)
            elif len(pts) == 1:
                cv2.circle(plottedImg, pts[0], 3, color, -1)

        for (cx, cy) in currentRedCenters:
            cv2.circle(plottedImg, (cx, cy), 3, (0, 0, 255), -1)

        tracks_data["frames"][str(frame_num)] = {
            "frame_number": frame_num,
            "detections": frame_detections
        }

        if out.isOpened() and plottedImg is not None:
            out.write(plottedImg)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    elapsed = timer.stop()

    tracks_data["processing_info"] = {
        "processing_time": elapsed,
        "model_path": model_path,
        "confidence_threshold": conf_threshold,
        "target_classes": TRACKING_CONFIG["target_classes"],
        "tracker": "botsort.yaml",
        "reid_enabled": True,
        "reid_model": "veriwild_bot_R50-ibn.pth",
        "timestamp": get_timestamp()
    }

    output_file = os.path.join(output_dir, f"{input_filename}_tracks.json")
    save_json_safe(tracks_data, output_file)

    summary_message = (
        f"Tracking completed in {elapsed:.2f}s\n"
        f"Total tracks: {len(tracks_data['tracks'])}\n"
        f"Output video saved to: {output_video_path}\n"
        f"Output JSON saved to: {output_file}"
    )
    print(f"\n{summary_message}")

    return summary_message