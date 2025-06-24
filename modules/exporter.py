#This code merges manual annotations, tracked centers, and behaviors into a single JSON file.
#It reads data from three JSON files: manual annotations, tracked centers, and behaviors.
#It combines the data into a structured format, ensuring that each frame contains both manual boxes and tracked centers.
#Finally, it saves the merged data into a specified output JSON file.

import os
import json

def load_json_file(path):
    if path and os.path.isfile(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}

def merge_annotations(manual_path, tracks_path, behaviors_path, output_path):
    merged_data = {
        "frames": {},
        "tracks": {},
        "behaviors": []
    }
    
    manual_data = {}
    if manual_path:
        manual_data = load_json_file(manual_path)
        manual_data = {int(k): v for k, v in manual_data.items()}
    
    tracks_data = {}
    if tracks_path:
        tracks_file = load_json_file(tracks_path)
        tracks_data = tracks_file.get("tracks", {})
    
    behaviors_data = []
    if behaviors_path:
        behaviors_file = load_json_file(behaviors_path)
        behaviors_data = behaviors_file.get("behaviors", [])
    
    for frame_idx, box_list in manual_data.items():
        merged_data["frames"][str(frame_idx)] = {
            "manual_boxes": box_list,
            "tracked_centers": []
        }
    
    merged_data["tracks"] = tracks_data
    
    for track_id, sequence in tracks_data.items():
        for entry in sequence:
            frame_str = str(entry["frame"])
            center = entry["center"]
            
            if frame_str not in merged_data["frames"]:
                merged_data["frames"][frame_str] = {
                    "manual_boxes": [],
                    "tracked_centers": []
                }
            
            merged_data["frames"][frame_str]["tracked_centers"].append({
                "track_id": track_id,
                "center": center
            })
    
    merged_data["behaviors"] = behaviors_data
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(merged_data, f, indent=2)
    
    print(f"Merged annotations saved to {output_path}")
    print(f"Frames: {len(merged_data['frames'])}")
    print(f"Tracks: {len(merged_data['tracks'])}")
    print(f"Behaviors: {len(merged_data['behaviors'])}")