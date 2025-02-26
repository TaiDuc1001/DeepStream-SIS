import os
import json
import csv
import numpy as np
from tqdm import tqdm
from datetime import datetime, timedelta


ROI_CONSTRAINT = {
    'source0': 'DongKhoi_MacThiBuoi',
    'source1': 'RachBungBinh_NguyenThong_2',
    'source2': 'TranHungDao_NguyenVanCu',
    'source3': 'TranKhacChan_TranQuangKhai',
}

def load_rois():
    rois = {}
    for file in os.listdir('../rois'):
        key = file.split('.')[0]
        rois[key] = {}
        
        with open(f'../rois/{file}', 'r') as f:
            current_section = None
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if ':' not in line:
                    current_section = line
                    rois[key][current_section] = []
                else:
                    if current_section:
                        x, y = map(int, line.split(':')[1].replace('(', '').replace(')', '').split(','))
                        rois[key][current_section].append((x, y))
    
    regions = {}
    for region_name, region_data in rois.items():
        lines = sorted(filter(lambda x: x.startswith('line'), region_data.keys()))
        regions[region_name] = {
            **{f"lane{i}": [region_data[line], region_data[next_line]] 
               for i, (line, next_line) in enumerate(zip(lines, lines[1:]))},
            **{key: region_data[key] for key in region_data if not key.startswith('line')}
        }
    
    return regions

FPS = 30
BASE_TIME = datetime(1900, 1, 1, 12, 0, 0)
VEHICLE_TYPE = {
    'Two-Wheeler': ['bicycle', 'motorbike'],
    'Small': ['car', 'van'],
    'Large': ['bus', 'truck'],
    'Pedestrian': ['pedestrian'],
}

def format_time(frame_count):
    seconds = frame_count // FPS
    t = BASE_TIME + timedelta(seconds=seconds)
    return t.strftime("%H:%M:%S")

def load_json_files(directory, rois):
    json_files = [f for f in os.listdir(directory) if f.endswith('.json')]
    trajectory_data = {}
    obj_id_map = {}

    # First pass: collect all trajectories and check if they ever enter a lane
    for json_file in json_files:
        stream_name = json_file.split('.')[0]
        stream_id = next((i for i, v in ROI_CONSTRAINT.items() if v == stream_name), None)
        
        if stream_id is None:
            print(f"Warning: Stream name '{stream_name}' not found in ROI_CONSTRAINT.")
            continue
        
        with open(os.path.join(directory, json_file), 'r') as f:
            data = json.load(f)
            
            for annotation_group in tqdm(data, desc=f"Processing {json_file}", unit="annotation"):
                for annotation in annotation_group.get('annotations', []):
                    for result in annotation.get('result', []):
                        if 'value' in result and 'sequence' in result['value']:
                            sequence_id = result['id']
                            obj_id = obj_id_map.setdefault((stream_id, sequence_id), len(obj_id_map) + 1)
                            
                            vehicle_category = result.get('value', {}).get('labels', ['unknown'])[0]
                            vehicle_type = next((category for category, types in VEHICLE_TYPE.items() 
                                                  if vehicle_category in types), 'Unknown')
                            
                            # Process all frames first to check if object ever enters a lane
                            frames = []
                            ever_in_lane = False
                            for frame_idx, bbox in enumerate(result['value']['sequence']):
                                original_bbox = bbox['x'], bbox['y'], bbox['width'], bbox['height']
                                normalized_bbox = normalize_bbox(original_bbox)
                                position = check_bbox_position(normalized_bbox, rois[stream_name])
                                
                                if position['lane'] is not None:
                                    ever_in_lane = True
                                
                                frames.append({
                                    'frame_idx': frame_idx,
                                    'bbox': normalized_bbox,
                                    'position': position
                                })
                            
                            # Only store trajectory if object ever enters a lane
                            if ever_in_lane:
                                # Track lane status from beginning
                                current_lane = None
                                lane_history = []
                                
                                for frame in frames:
                                    if frame['position']['lane'] is not None:
                                        current_lane = frame['position']['lane']
                                    frame['position']['current_lane'] = current_lane
                                    lane_history.append(current_lane)
                                
                                trajectory_data[obj_id] = {
                                    'stream_id': stream_id,
                                    'stream_name': stream_name,
                                    'frames': frames,
                                    'first_appear_frame': frames[0]['frame_idx'],
                                    'last_appear_frame': frames[-1]['frame_idx'],
                                    'lane_frame': None,
                                    'stopline_frame': None,
                                    'vehicle_type': vehicle_type,
                                    'lane_history': lane_history
                                }
                                
                                # Find the first frame where object enters a lane
                                for frame in frames:
                                    if frame['position']['lane'] is not None:
                                        trajectory_data[obj_id]['lane_frame'] = frame['frame_idx']
                                        break
                                
                                # Find the frame where object reaches stopline (if applicable)
                                for frame in frames:
                                    if frame['position'].get('stopline'):
                                        trajectory_data[obj_id]['stopline_frame'] = frame['frame_idx']
                                        break

    summary_rows = []
    frame_rows = []

    for obj_id, trajectory in trajectory_data.items():
        directions = [frame['position']['direction'] for frame in trajectory['frames'] 
                     if frame['position']['direction'] is not None]
        
        direction = max(set(directions), key=directions.count) if directions else 'straight'
        
        # Use the most common lane from lane_history
        lanes = [lane for lane in trajectory['lane_history'] if lane is not None]
        lane = max(set(lanes), key=lanes.count) if lanes else 'N/A'
        
        first_appear_time = format_time(trajectory['first_appear_frame'])
        last_appear_time = format_time(trajectory['last_appear_frame'])
        lane_time = format_time(trajectory['lane_frame']) if trajectory['lane_frame'] is not None else 'N/A'
        stopline_time = format_time(trajectory['stopline_frame']) if trajectory['stopline_frame'] is not None else 'N/A'
        
        summary_rows.append([
            obj_id,
            trajectory['stream_id'],
            first_appear_time,
            lane_time,
            stopline_time,
            last_appear_time,
            lane.split('lane')[-1] if lane != 'N/A' else 'N/A',
            trajectory['vehicle_type'],
            direction,
        ])
        
        for frame in trajectory['frames']:
            bbox = frame['bbox']
            current_lane = frame['position']['current_lane']
            frame_rows.append([
                obj_id,
                trajectory['stream_id'],
                trajectory['stream_name'],
                frame['frame_idx'],
                bbox[0], bbox[1], bbox[2], bbox[3],
                current_lane.split('lane')[-1] if current_lane else 'N/A'
            ])

    with open('trajectory_summary.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Vehicle ID", "Stream ID", "First-Appeared Time", "In-Lane Time", 
                        "Reach-StopLine Time", "Last-Appear Time", "Lane ID",
                        "Vehicle Type", "Direction"])
        writer.writerows(summary_rows)

    with open('trajectory_frames.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Vehicle ID", "Stream ID", "Stream Name", "Frame Index", 
                        "X", "Y", "Width", "Height", "Lane ID"])
        writer.writerows(frame_rows)

    print("Files written: trajectory_summary.csv, trajectory_frames.csv")


def check_bbox_position(bbox, regions):
    x, y, width, height = bbox
    bottom_middle = (x + width/2, y + height)
    px, py = bottom_middle

    position = {'lane': None, 'direction': None}

    # Check lanes
    for lane in sorted([k for k in regions if k.startswith('lane')], key=lambda x: int(x[4:])):
        if is_point_between_lines(px, py, *regions[lane]):
            position['lane'] = lane
            break

    # Check stopline
    if 'stopline' in regions:
        stopline = regions['stopline']
        stopline_y = max(y for _, y in stopline)
        min_stopline_x = min(x for x, _ in stopline)
        max_stopline_x = max(x for x, _ in stopline)
        if py <= stopline_y and min_stopline_x <= px <= max_stopline_x:
            position['stopline'] = True

    # Check direction
    if 'left' in regions:
        left_line = regions['left']
        if is_between_y_bounds(py, left_line):
            p1, p2 = left_line[0], left_line[-1]
            if get_cross_product(p1, p2, (px, py)) > 0:
                position['direction'] = 'left'

    if 'right' in regions and not position['direction']:
        right_line = regions['right']
        if is_between_y_bounds(py, right_line):
            p1, p2 = right_line[0], right_line[-1]
            if get_cross_product(p1, p2, (px, py)) < 0:
                position['direction'] = 'right'

    return position

def is_point_between_lines(px, py, line0, line1):
    return min(p[1] for p in line0) <= py <= max(p[1] for p in line1)

def get_cross_product(p1, p2, p):
    return (p2[0] - p1[0]) * (p[1] - p1[1]) - (p2[1] - p1[1]) * (p[0] - p1[0])

def is_between_y_bounds(py, line):
    return min(p[1] for p in line) <= py <= max(p[1] for p in line)

def normalize_bbox(bbox, image_width=1920, image_height=1080):
    x, y, width, height = bbox
    return (x / 100) * image_width, (y / 100) * image_height, (width / 100) * image_width, (height / 100) * image_height

if __name__ == "__main__":
    json_dir = "."
    rois = load_rois()
    load_json_files(json_dir, rois)
