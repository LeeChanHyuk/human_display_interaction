import os

import pyrealsense2 as rs
import cv2
import numpy as np


def realsense_initialization():
    align_to = rs.stream.color
    align = rs.align(align_to)
    pipeline = rs.pipeline()
    # Configure streams
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    pipeline.start(config)
    return pipeline, align

def video_loader_initialization(path):
    files = os.listdir(path)
    rgb_videos = []
    depth_videos = []
    for file in files:
        if 'rgb' in file:
            rgb_videos.append(file)
        elif 'depth' in file:
            depth_videos.append(file)
    rgb_videos.sort()
    depth_videos.sort()
    rgb_caps = []
    depth_caps = []
    for rgb_video, depth_video in zip(rgb_videos, depth_videos):
        rgb_cap = cv2.VideoCapture(os.path.join(path, rgb_video))
        depth_cap = cv2.VideoCapture(os.path.join(path, depth_video))
        rgb_caps.append(rgb_cap)
        depth_caps.append(depth_cap)
    return rgb_caps, depth_caps, len(rgb_caps)

def get_input(pipeline=None, align=None, rgb_cap=None, depth_cap=None, video_path=None):
    # Get input
    if not video_path:
        frames = pipeline.wait_for_frames()
        align_frames = align.process(frames)
        frame = align_frames.get_color_frame()
        depth = align_frames.get_depth_frame()
        depth = np.array(depth.get_data())
        frame = np.array(frame.get_data())
    else:
        ret, frame = rgb_cap.read()
        ret, depth = depth_cap.read()
    return frame, depth