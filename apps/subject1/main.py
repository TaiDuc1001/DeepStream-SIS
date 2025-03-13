#!/usr/bin/env python3
import sys
sys.path.append('../')
from pathlib import Path
import gi
import configparser
import argparse
gi.require_version('Gst', '1.0')
from gi.repository import GLib, Gst
from ctypes import *
import time
import sys
import math
import platform
from common.is_aarch_64 import is_aarch64
from common.bus_call import bus_call
from common.FPS import PERF_DATA
import numpy as np
import pandas as pd
import pyds
import os
from collections import OrderedDict
import datetime
import math
from collections import defaultdict



no_display = False
silent = False
file_loop = False
perf_data = None
MAX_DISPLAY_LEN = 64
MUXER_OUTPUT_WIDTH = 1920
MUXER_OUTPUT_HEIGHT = 1080
MUXER_BATCH_TIMEOUT_USEC = 4000000
TILED_OUTPUT_WIDTH = 1280
TILED_OUTPUT_HEIGHT = 720
GST_CAPS_FEATURES_NVMM = "memory:NVMM"
OSD_PROCESS_MODE = 2
OSD_DISPLAY_TEXT = 0
SOURCES_NAMES = []


with open('settings/labels.txt', 'r') as f:
    labels = f.readlines()

ROIS = {}
for i, file in enumerate(os.listdir('settings/rois')):
    current_section = None
    key = file.split('.')[0]
    ROIS[key] = {}
    with open(f'settings/rois/{file}', 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if ':' not in line:
                current_section = line
                ROIS[key][current_section] = []
            else:
                if current_section:
                    coord_str = line.split(':')[1].strip()
                    x, y = map(int, coord_str.strip('()').split(','))
                    ROIS[key][current_section].append((x, y))

REGIONS = {
    region_name: {
        **{f"lane{i}": [region_data[line], region_data[next_line]] for i, (line, next_line) in enumerate(zip(lines, lines[1:]))},
        **{key: region_data[key] for key in region_data if not key.startswith('line')}
    }
    for region_name, region_data in ROIS.items()
    for lines in [list(filter(lambda x: x.startswith('line'), region_data.keys()))]
}
print(REGIONS)

REGION_COLORS = {
    'lane0': (0.0, 1.0, 0.0, 1.0), # green
    'lane1': (0.0, 1.0, 0.0, 1.0), # green
    'lane2': (0.0, 1.0, 0.0, 1.0), # green
    'left': (1.0, 0.5, 0.0, 1.0), # orange
    'right': (1.0, 0.0, 0.5, 1.0), # pink
    'stopline': (1.0, 0.0, 0.0, 1.0), # red
    'default': (0.0, 1.0, 0.0, 1.0) # green
}

def is_point_between_lines(px, py, line0, line1):
    def nearest_point_on_line_segment(p, line):
        min_dist, idx = float('inf'), 0
        for i in range(len(line) - 1):
            p1, p2 = line[i], line[i+1]
            line_vec, point_vec = (p2[0] - p1[0], p2[1] - p1[1]), (p[0] - p1[0], p[1] - p1[1])
            line_len_sq = line_vec[0]**2 + line_vec[1]**2
            t = max(0, min(1, (point_vec[0] * line_vec[0] + point_vec[1] * line_vec[1]) / line_len_sq)) if line_len_sq != 0 else 0
            proj_x, proj_y = p1[0] + t * line_vec[0], p1[1] + t * line_vec[1]
            dist = math.sqrt((p[0] - proj_x) ** 2 + (p[1] - proj_y) ** 2)
            if dist < min_dist: min_dist, idx = dist, i
        return idx

    idx0, idx1 = nearest_point_on_line_segment((px, py), line0), nearest_point_on_line_segment((px, py), line1)
    if idx0 >= len(line0) - 1: idx0 = len(line0) - 2
    if idx1 >= len(line1) - 1: idx1 = len(line1) - 2
    p1_0, p2_0, p1_1, p2_1 = line0[idx0], line0[idx0 + 1], line1[idx1], line1[idx1 + 1]
    lane_corners = [p1_0, p2_0, p2_1, p1_1]

    def is_point_inside_polygon(px, py, poly):
        crossings = 0
        for i in range(len(poly)):
            x1, y1, x2, y2 = *poly[i], *poly[(i + 1) % len(poly)]
            if ((y1 > py) != (y2 > py)) and (px < (x2 - x1) * (py - y1) / (y2 - y1) + x1): crossings += 1
        return crossings % 2 == 1

    return is_point_inside_polygon(px, py, lane_corners)

def is_point_reach_stopline(px, py, stopline):
    max_stopline_y = max(y for _, y in stopline)
    min_stopline_x = min(x for x, _ in stopline)
    max_stopline_x = max(x for x, _ in stopline)
    return py > max_stopline_y and min_stopline_x <= px <= max_stopline_x

obj_history = OrderedDict()
global_direction_history = OrderedDict()
frame_history = OrderedDict()
global_lane_history = OrderedDict()
global_init_time = {}
global_lane_time = {}
global_vehicle_type = {}
global_frame_number = 0
global_stopline_time = {}
global_last_appear = {}
base_time = datetime.datetime(1900, 1, 1, 12, 0, 0)
FPS = 30
FRAME_HISTORY_THRESHOLD = 30
FRAME_DISAPPEAR_THRESHOLD = 10
VEHICLE_TYPE = {
    'Two-Wheeler': ['bicycle', 'motorbike'],
    'Small': ['car', 'van'],
    'Large': ['bus', 'truck'],
    'Pedestrian': ['pedestrian'],
}

def create_time_lookup(max_frames=100000):
    lookup = {}
    for frame_count in range(0, max_frames, FPS):
        seconds = frame_count // FPS
        t = base_time + datetime.timedelta(seconds=seconds)
        lookup[frame_count] = t.strftime("%H:%M:%S")
    return lookup
TIME_LOOKUP = create_time_lookup()


def format_time(frame_count):
    # Round to nearest FPS multiple for lookup
    key = (frame_count // FPS) * FPS
    return TIME_LOOKUP.get(key, "00:00:00")

def write_analysis(inner_func):
    def wrapper(*args, **kwargs):
        inner_func(*args, **kwargs)
        with open('output/directions.csv', 'w') as f:
            f.write("Vehicle ID,Stream ID,First-Appeared Time,In-Lane Time,Reach-StopLine Time,Last-Appear Time,Lane ID,Vehicle Type,Direction\n")
            sorted_keys = sorted(
                global_direction_history.keys(),
                key=lambda x: x[1]
            )
            for key in sorted_keys:
                source, obj_id = key
                source_id = SOURCES_NAMES.index(source)
                lane_id = global_lane_history.get(key, "")
                init_time_str = format_time(global_init_time.get(key, 0)) if key in global_init_time else ""
                lane_time_str = format_time(global_lane_time.get(key, 0)) if key in global_lane_time else ""
                stopline_time_str = format_time(global_stopline_time.get(key, 0)) if key in global_stopline_time else ""
                last_appear_str = format_time(global_last_appear.get(key, 0)) if key in global_last_appear else ""
                direction = global_direction_history.get(key, "")
                vehicle_type = global_vehicle_type.get(key, "").strip()
                vehicle_type = next((k for k, v in VEHICLE_TYPE.items() if vehicle_type in v), "")
                # if lane_id and vehicle_type and stopline_time_str:
                lane_id = lane_id.split('lane')[-1]
                f.write(f"{obj_id},{source_id},{init_time_str},{lane_time_str},{stopline_time_str},{last_appear_str},{lane_id},{vehicle_type},{direction}\n")
    return wrapper

# Replace global frame counter with per-source frame counters
source_frame_numbers = defaultdict(int)

@write_analysis
def analytics(l_obj, frame_meta, REGIONS, REGION_COLORS, labels, obj_counter, source_name):
    global obj_history, global_direction_history, frame_history, global_lane_history
    global global_init_time, global_lane_time, global_frame_number, global_vehicle_type
    global global_stopline_time, global_last_appear
    global source_frame_numbers
    
    # Increment frame number for this source
    source_frame_numbers[source_name] += 1
    current_frame = source_frame_numbers[source_name]
    
    if source_name not in REGIONS:
        return global_direction_history
        
    region_data = REGIONS[source_name]
    lane_keys = [key for key in region_data if 'lane' in key]
    
    # Pre-calculate tiler parameters once
    tiler_rows = int(math.sqrt(number_sources))
    tiler_columns = int(math.ceil((1.0 * number_sources) / tiler_rows))
    tile_width = TILED_OUTPUT_WIDTH / tiler_columns
    tile_height = TILED_OUTPUT_HEIGHT / tiler_rows
    source_index = SOURCES_NAMES.index(source_name)
    source_x = (source_index % tiler_columns) * tile_width
    source_y = (source_index // tiler_columns) * tile_height

    # Process objects
    current_obj = l_obj
    while current_obj:
        obj_meta = pyds.NvDsObjectMeta.cast(current_obj.data)
        obj_id = obj_meta.object_id
        key = (source_name, obj_id)
        
        rect = obj_meta.rect_params
        bottom_middle_x = rect.left + rect.width / 2
        bottom_middle_y = rect.top + rect.height

        # Initialize object data if needed
        if key not in obj_history:
            obj_history[key] = False
            global_direction_history[key] = "unknown"
            frame_history[key] = 0
            global_lane_history[key] = ""
            global_init_time[key] = current_frame
            global_vehicle_type[key] = labels[obj_meta.class_id]

        # Check lanes
        if not obj_history[key]:
            for lane in lane_keys:
                line_a, line_b = region_data[lane]
                if is_point_between_lines(bottom_middle_x, bottom_middle_y, line_a, line_b):
                    obj_history[key] = True
                    global_lane_history[key] = lane
                    if key not in global_lane_time:
                        global_lane_time[key] = current_frame
                    obj_meta.rect_params.border_color.set(*REGION_COLORS[lane])
                    obj_counter[labels[obj_meta.class_id]] += 1
                    break

        # Process tracked objects
        if obj_history[key]:
            if 'stopline' in region_data:
                stopline = region_data['stopline']
                if is_point_reach_stopline(bottom_middle_x, bottom_middle_y, stopline):
                    if key not in global_stopline_time:
                        global_stopline_time[key] = current_frame
                    obj_meta.rect_params.border_color.set(*REGION_COLORS['stopline'])
            
            elif 'left' in region_data and global_direction_history[key] == "unknown":
                line = region_data['left']
                if line[0][1] <= bottom_middle_y <= line[-1][1]:
                    cp = get_cross_product(line, bottom_middle_x, bottom_middle_y)
                    if cp > 0:
                        obj_meta.rect_params.border_color.set(*REGION_COLORS['left'])
                        global_direction_history[key] = 'left'
            
            elif 'right' in region_data and global_direction_history[key] == "unknown":
                line = region_data['right']
                if line[0][1] <= bottom_middle_y <= line[-1][1]:
                    cp = get_cross_product(line, bottom_middle_x, bottom_middle_y)
                    if cp < 0:
                        obj_meta.rect_params.border_color.set(*REGION_COLORS['right'])
                        global_direction_history[key] = 'right'

        # Track object position
        if (bottom_middle_x < source_x or bottom_middle_x > source_x + tile_width or 
            bottom_middle_y < source_y or bottom_middle_y > source_y + tile_height):
            frame_history[key] = frame_history.get(key, 0) + 1
            if frame_history[key] >= FRAME_DISAPPEAR_THRESHOLD:
                global_last_appear[key] = current_frame
            if frame_history[key] >= FRAME_HISTORY_THRESHOLD and global_direction_history[key] == "unknown":
                global_direction_history[key] = "straight"
        else:
            frame_history[key] = 0

        current_obj = current_obj.next

    return global_direction_history

def osd_sink_pad_buffer_probe(pad, info, u_data):
    buffer = info.get_buffer()
    if not buffer:
        return Gst.PadProbeReturn.OK
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buffer))
    l_frame = batch_meta.frame_meta_list
    if l_frame is not None:
        frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        tiler_rows = int(math.sqrt(number_sources))
        tiler_columns = int(math.ceil((1.0 * number_sources) / tiler_rows))
        tile_width = TILED_OUTPUT_WIDTH / tiler_columns
        tile_height = TILED_OUTPUT_HEIGHT / tiler_rows
        scale_x = tile_width / MUXER_OUTPUT_WIDTH
        scale_y = tile_height / MUXER_OUTPUT_HEIGHT
        for i, source_name in zip(range(number_sources), SOURCES_NAMES):
            if source_name not in ROIS:
                continue
            offset_x = int((i % tiler_columns) * tile_width)
            offset_y = int((i // tiler_columns) * tile_height)
            for section, points in ROIS[source_name].items():
                num_lines = len(points) - 1
                if num_lines <= 0:
                    continue
                display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
                display_meta.num_lines = num_lines
                for j in range(num_lines):
                    line = display_meta.line_params[j]
                    x1, y1 = points[j]
                    x2, y2 = points[j + 1]
                    line.x1 = int(x1 * scale_x) + offset_x
                    line.y1 = int(y1 * scale_y) + offset_y
                    line.x2 = int(x2 * scale_x) + offset_x
                    line.y2 = int(y2 * scale_y) + offset_y
                    line.line_width = 2
                    line.line_color.set(*REGION_COLORS['default'])
                pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)
    return Gst.PadProbeReturn.OK



def get_cross_product(line, obj_x, obj_y):
    x1, y1 = line[0]
    x2, y2 = line[-1]
    return ((x2 - x1) * (obj_y - y1) - (y2 - y1) * (obj_x - x1))

def tracker_src_pad_buffer_probe(pad, info, u_data):
    frame_number = 0
    num_rects = 0
    got_fps = False
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer")
        return
    
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break
            
        source_id = frame_meta.pad_index
        if source_id < len(SOURCES_NAMES):
            source_name = SOURCES_NAMES[source_id]
        else:
            source_name = None
        frame_number = frame_meta.frame_num
        l_obj = frame_meta.obj_meta_list
        num_rects = frame_meta.num_obj_meta
        obj_counter = { label: 0 for label in labels }
        
        analytics(l_obj, frame_meta, REGIONS, REGION_COLORS, labels, obj_counter, source_name)
        stream_index = "stream{0}".format(frame_meta.pad_index)
        global perf_data
        perf_data.update_fps(stream_index)
        try:
            l_frame = l_frame.next
        except StopIteration:
            break
    return Gst.PadProbeReturn.OK

def cb_newpad(decodebin, decoder_src_pad, data):
    caps = decoder_src_pad.get_current_caps()
    if not caps:
        caps = decoder_src_pad.query_caps()
    gststruct = caps.get_structure(0)
    gstname = gststruct.get_name()
    source_bin = data
    features = caps.get_features(0)
    print("gstname=", gstname)
    if gstname.find("video") != -1:
        print("features=", features)
        if features.contains("memory:NVMM"):
            bin_ghost_pad = source_bin.get_static_pad("src")
            if not bin_ghost_pad.set_target(decoder_src_pad):
                sys.stderr.write("Failed to link decoder src pad to source bin ghost pad\n")
        else:
            sys.stderr.write("Error: Decodebin did not pick nvidia decoder plugin.\n")

def decodebin_child_added(child_proxy, Object, name, user_data):
    print("Decodebin child added:", name, "\n")
    if name.find("decodebin") != -1:
        Object.connect("child-added", decodebin_child_added, user_data)
    if "source" in name:
        source_element = child_proxy.get_by_name("source")
        if source_element.find_property('drop-on-latency') is not None:
            Object.set_property("drop-on-latency", True)

def create_source_bin(index, uri):
    bin_name = "source-bin-%02d" % index
    nbin = Gst.Bin.new(bin_name)
    if not nbin:
        sys.stderr.write("Unable to create source bin\n")
    if file_loop:
        uri_decode_bin = Gst.ElementFactory.make("nvurisrcbin", "uri-decode-bin")
        uri_decode_bin.set_property("file-loop", 1)
        uri_decode_bin.set_property("cudadec-memtype", 0)
    else:
        uri_decode_bin = Gst.ElementFactory.make("uridecodebin", "uri-decode-bin")
    if not uri_decode_bin:
        sys.stderr.write("Unable to create uri decode bin\n")
    uri_decode_bin.set_property("uri", uri)
    uri_decode_bin.connect("pad-added", cb_newpad, nbin)
    uri_decode_bin.connect("child-added", decodebin_child_added, nbin)
    Gst.Bin.add(nbin, uri_decode_bin)
    bin_pad = nbin.add_pad(Gst.GhostPad.new_no_target("src", Gst.PadDirection.SRC))
    if not bin_pad:
        sys.stderr.write("Failed to add ghost pad in source bin\n")
        return None
    return nbin

def main(args, requested_pgie=None, config=None, disable_probe=False):
    global perf_data
    perf_data = PERF_DATA(len(args))
    global number_sources
    global SOURCES_NAMES
    for i, source in enumerate(args):
        SOURCES_NAMES.append(source.split('/')[-1].split('.')[0])
    number_sources = len(args)
    Gst.init(None)
    print("Creating Pipeline")
    pipeline = Gst.Pipeline()
    is_live = False
    if not pipeline:
        sys.stderr.write("Unable to create Pipeline\n")
    print("Creating streammux")
    streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
    if not streammux:
        sys.stderr.write("Unable to create NvStreamMux\n")
    pipeline.add(streammux)
    for i in range(number_sources):
        print("Creating source_bin", i)
        uri_name = args[i]
        if uri_name.find("rtsp://") == 0:
            is_live = True
        source_bin = create_source_bin(i, uri_name)
        if not source_bin:
            sys.stderr.write("Unable to create source bin\n")
        pipeline.add(source_bin)
        padname = "sink_%u" % i
        sinkpad = streammux.get_request_pad(padname)
        if not sinkpad:
            sys.stderr.write("Unable to create sink pad bin\n")
        srcpad = source_bin.get_static_pad("src")
        if not srcpad:
            sys.stderr.write("Unable to create src pad bin\n")
        srcpad.link(sinkpad)
    print("Creating nvvideoconvert and capsfilter for RGBA")
    vidconv = Gst.ElementFactory.make("nvvideoconvert", "nvvideoconvert-for-rgba")
    if not vidconv:
        sys.stderr.write("Unable to create nvvideoconvert\n")
    caps_filter = Gst.ElementFactory.make("capsfilter", "capsfilter-rgba")
    if not caps_filter:
        sys.stderr.write("Unable to create capsfilter\n")
    rgba_caps = Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA")
    caps_filter.set_property("caps", rgba_caps)
    pipeline.add(vidconv)
    pipeline.add(caps_filter)
    print("Linking streammux -> vidconv -> capsfilter -> queue1")
    streammux.link(vidconv)
    vidconv.link(caps_filter)
    queue1 = Gst.ElementFactory.make("queue", "queue1")
    queue2 = Gst.ElementFactory.make("queue", "queue2")
    queue3 = Gst.ElementFactory.make("queue", "queue3")
    queue4 = Gst.ElementFactory.make("queue", "queue4")
    queue5 = Gst.ElementFactory.make("queue", "queue5")
    queue6 = Gst.ElementFactory.make("queue", "queue6")
    pipeline.add(queue1)
    pipeline.add(queue2)
    pipeline.add(queue3)
    pipeline.add(queue4)
    pipeline.add(queue5)
    pipeline.add(queue6)
    nvdslogger = None
    print("Creating Pgie")
    if requested_pgie is not None and (requested_pgie == 'nvinferserver' or requested_pgie == 'nvinferserver-grpc'):
        pgie = Gst.ElementFactory.make("nvinferserver", "primary-inference")
    elif requested_pgie is not None and requested_pgie == 'nvinfer':
        pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    else:
        pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    if not pgie:
        sys.stderr.write("Unable to create pgie: %s\n" % requested_pgie)
    print("Create Tracker")
    tracker = Gst.ElementFactory.make("nvtracker", "tracker")
    if not tracker:
        sys.stderr.write("Unable to create tracker\n")
    else:
        tracker.set_property('tracker-width', 640)
        tracker.set_property('tracker-height', 480)
        tracker.set_property('gpu-id', 0)
        tracker.set_property('ll-lib-file', "/opt/nvidia/deepstream/deepstream-6.2/lib/libnvds_nvmultiobjecttracker.so")
        tracker.set_property('enable-batch-process', True)
    if disable_probe:
        print("Creating nvdslogger")
        nvdslogger = Gst.ElementFactory.make("nvdslogger", "nvdslogger")
    print("Creating tiler")
    tiler = Gst.ElementFactory.make("nvmultistreamtiler", "nvtiler")
    if not tiler:
        sys.stderr.write("Unable to create tiler\n")
    print("Creating nvvidconv")
    nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
    if not nvvidconv:
        sys.stderr.write("Unable to create nvvidconv\n")
    print("Creating nvosd")
    nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
    osd_sink_pad = nvosd.get_static_pad("sink")
    if not osd_sink_pad:
        sys.stderr.write("Unable to get osd sink pad\n")
    osd_sink_pad.add_probe(Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe, 0)
    if not nvosd:
        sys.stderr.write("Unable to create nvosd\n")
    nvosd.set_property('process-mode', OSD_PROCESS_MODE)
    nvosd.set_property('display-text', OSD_DISPLAY_TEXT)
    # if file_loop:
    #     if is_aarch64():
    #         streammux.set_property('nvbuf-memory-type', 4)
    #     else:
    #         streammux.set_property('nvbuf-memory-type', 2)
    if no_display:
        print("Creating Fakesink")
        sink = Gst.ElementFactory.make("fakesink", "fakesink")
        sink.set_property('enable-last-sample', 0)
        sink.set_property('sync', 0)
    else:
        if is_aarch64():
            print("Creating nv3dsink")
            sink = Gst.ElementFactory.make("nv3dsink", "nv3d-sink")
            if not sink:
                sys.stderr.write("Unable to create nv3dsink\n")
        else:
            print("Creating EGLSink")
            sink = Gst.ElementFactory.make("nveglglessink", "nvvideo-renderer")
            if not sink:
                sys.stderr.write("Unable to create egl sink\n")
    if not sink:
        sys.stderr.write("Unable to create sink element\n")
    if is_live:
        print("At least one of the sources is live")
        streammux.set_property('live-source', 1)
    streammux.set_property('width', 1920)
    streammux.set_property('height', 1080)
    streammux.set_property('batch-size', number_sources)
    streammux.set_property('batched-push-timeout', 40000)
    streammux.set_property('buffer-pool-size', 8)
    # streammux.set_property('nvbuf-memory-type', 4 if is_aarch64() else 2)
    if requested_pgie == "nvinferserver" and config is not None:
        pgie.set_property('config-file-path', config)
    elif requested_pgie == "nvinferserver-grpc" and config is not None:
        pgie.set_property('config-file-path', config)
    elif requested_pgie == "nvinfer" and config is not None:
        pgie.set_property('config-file-path', config)
    else:
        pgie.set_property('config-file-path', "settings/config.txt")
    pgie_batch_size = pgie.get_property("batch-size")
    if pgie_batch_size != number_sources:
        print("WARNING: Overriding infer-config batch-size", pgie_batch_size, "with number of sources", number_sources)
        pgie.set_property("batch-size", number_sources)
    tiler_rows = int(math.sqrt(number_sources))
    tiler_columns = int(math.ceil((1.0 * number_sources) / tiler_rows))
    tiler.set_property("rows", tiler_rows)
    tiler.set_property("columns", tiler_columns)
    tiler.set_property("width", TILED_OUTPUT_WIDTH)
    tiler.set_property("height", TILED_OUTPUT_HEIGHT)
    # tiler.set_property('nvbuf-memory-type', 4 if is_aarch64() else 2)
    tiler.set_property('gpu-id', 0)
    sink.set_property("qos", 0)
    print("Adding elements to Pipeline")
    pipeline.add(pgie)
    pipeline.add(tracker)
    if nvdslogger:
        pipeline.add(nvdslogger)
    pipeline.add(tiler)
    pipeline.add(nvvidconv)
    pipeline.add(nvosd)
    pipeline.add(sink)
    print("Linking elements in the Pipeline")
    caps_filter.link(queue1)
    queue1.link(pgie)
    pgie.link(queue2)
    queue2.link(tracker)
    tracker.link(queue6)
    if nvdslogger:
        queue6.link(nvdslogger)
        nvdslogger.link(tiler)
    else:
        queue6.link(tiler)
    tiler.link(queue3)
    queue3.link(nvvidconv)
    nvvidconv.link(queue4)
    queue4.link(nvosd)
    nvosd.link(queue5)
    queue5.link(sink)
    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)
    Gst.debug_set_threshold_for_name('h265parse', Gst.DebugLevel.NONE)
    tracker_src_pad = tracker.get_static_pad("src")
    if not tracker_src_pad:
        sys.stderr.write("Unable to get tracker src pad\n")
    else:
        if not disable_probe:
            tracker_src_pad.add_probe(Gst.PadProbeType.BUFFER, tracker_src_pad_buffer_probe, 0)
            GLib.timeout_add(5000, perf_data.perf_print_callback)
    print("Now playing...")
    for i, source in enumerate(args):
        print(i, ":", source)
    print("Starting pipeline")
    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    except:
        pass
    print("Exiting app")
    pipeline.set_state(Gst.State.NULL)

    # Add to vidconv, nvvidconv
    # vidconv.set_property('nvbuf-memory-type', 4 if is_aarch64() else 2)
    # nvvidconv.set_property('nvbuf-memory-type', 4 if is_aarch64() else 2)

    # Add queue properties
    for queue in [queue1, queue2, queue3, queue4, queue5, queue6]:
        queue.set_property('max-size-buffers', 4)
        queue.set_property('max-size-bytes', 0)
        queue.set_property('max-size-time', 0)
        queue.set_property('leaky', 2)  # Downstream leaky

    # For nvosd
    nvosd.set_property('gpu-id', 0)
    # nvosd.set_property('nvbuf-memory-type', 4 if is_aarch64() else 2)

    # For sink (if using nveglglessink)
    if not is_aarch64():
        sink.set_property('sync', False)  # Disable sync
        sink.set_property('max-lateness', -1)
        sink.set_property('async', True)
        sink.set_property('throttle-time', 0)

def parse_args():
    parser = argparse.ArgumentParser(prog="deepstream_test_3", description="deepstream-test3 multi stream, multi model inference reference app")
    parser.add_argument("-i", "--input", help="Path to input streams", nargs="+", metavar="URIs", default=["a"], required=True)
    parser.add_argument("-c", "--configfile", metavar="config_location.txt", default=None, help="Choose the config-file to be used with specified pgie")
    parser.add_argument("-g", "--pgie", default=None, help="Choose Primary GPU Inference Engine", choices=["nvinfer", "nvinferserver", "nvinferserver-grpc"])
    parser.add_argument("--no-display", action="store_true", default=False, dest='no_display', help="Disable display of video output")
    parser.add_argument("--file-loop", action="store_true", default=False, dest='file_loop', help="Loop the input file sources after EOS")
    parser.add_argument("--disable-probe", action="store_true", default=False, dest='disable_probe', help="Disable the probe function and use nvdslogger for FPS")
    parser.add_argument("-s", "--silent", action="store_true", default=False, dest='silent', help="Disable verbose output")
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    args = parser.parse_args()
    stream_paths = args.input
    pgie = args.pgie
    config = args.configfile
    disable_probe = args.disable_probe
    global no_display
    global silent
    global file_loop
    no_display = args.no_display
    silent = args.silent
    file_loop = args.file_loop
    if config and not pgie or pgie and not config:
        sys.stderr.write("\nEither pgie or configfile is missing. Please specify both! Exiting...\n\n\n\n")
        parser.print_help()
        sys.exit(1)
    if config:
        config_path = Path(config)
        if not config_path.is_file():
            sys.stderr.write("Specified config-file: %s doesn't exist. Exiting...\n\n" % config)
            sys.exit(1)
    print(vars(args))
    return stream_paths, pgie, config, disable_probe

if __name__ == '__main__':
    stream_paths, pgie, config, disable_probe = parse_args()
    sys.exit(main(stream_paths, pgie, config, disable_probe))
