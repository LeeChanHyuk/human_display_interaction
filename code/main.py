##############################################################
################## 2022.06.02.ChanHyukLee ####################
##############################################################
# Facial & Body landmark is from mediaPipe (Google)
# Head pose estimation module is from 1996scarlet (https://github.com/1996scarlet/Dense-Head-Pose-Estimation)
# Gaze estimation module is from david-wb (https://github.com/david-wb/gaze-estimation)

from lib2to3.pytree import BasePattern
from pathlib import WindowsPath
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import math
import cv2
import time
import mediapipe as mp
import numpy as np
import pyrealsense2 as rs
import re
import random
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Data collection
from user_information.data_collector import realsense_initialization, video_loader_initialization, get_input

# Visualization
from visualization.draw_utils import draw_axis
import visualization.visualization_tool as visualization_tool
from visualization.visualization_tool import visualization

# Estimators
from estimators.action_recognizer import inference
from estimators.gaze_estimation_module.util.gaze import draw_gaze
from estimators.body_pose_estimator import body_keypoint_extractor, upside_body_pose_calculator
import estimators.head_pose_estimation_module.service as service
from estimators.face_detector import face_box_visualization, face_detection, resnet_face_detection
from estimators.face_detector import calibration
from estimators.head_pose_estimator import head_pose_estimation
from estimators.main_user_classifier import main_user_classification
from estimators.body_pose_estimator import body_pose_estimation, body_keypoint_extractor
from estimators.action_recognizer import action_recognition
from zmq_router import networking
from estimators.face_detection_module.yolov5.detect import yolo_face_detection, yolo_initialization

# User class
from user_information.human import HumanInfo
from utils import preprocessing

# Mediapipe visualization objects
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh

# Mode (If you want to use multiple functions, then make that you use true)
mode = 3
precise_value_visualization = True

# Visualization settings
text_visualization = False
flip_mode = True
depth_face_tracker = False

# Landmark / action names
actions = [ 'left', 'left-up', 'up',
'right-up', 'right', 'right-down', 'down', 'left-down', 'zoom-in', 'zoom-out','standard']

def load_mode(base_path) -> int:
    communication_file = open(os.path.join(base_path, 'communication.txt'), 'r')
    mode = communication_file.readline().strip()
    communication_file.close()
    return int(mode)

def main(video_folder_path=None) -> None:
    base_path = os.path.dirname(os.path.abspath(__file__))
    fps = 21
    iteration = 0
    human_infos = None
    draw_frame = None
    ROOT = os.path.dirname(os.path.abspath(__file__))

    # Initialize face detection module
    fa = service.DepthFacialLandmarks(os.path.join(base_path, "estimators/head_pose_estimation_module/weights/sparse_face.tflite"))
    net = cv2.dnn.readNetFromCaffe(
        os.path.join(base_path, "estimators/deploy.prototxt.txt"), 
        os.path.join(base_path, "estimators/res10_300x300_ssd_iter_140000.caffemodel"))
    if depth_face_tracker:
        model, dt, device = yolo_initialization(
            frame_shape = (480, 640, 3),
            weights= WindowsPath(os.path.join(ROOT, 'estimators', 'face_detection_module', 'yolov5', 'weights', 'depth', 'best.pt')),
            data = WindowsPath(os.path.join(ROOT, 'estimators', 'face_detection_module', 'yolov5', 'data', 'coco128.yaml')),
            depth_face_tracker=depth_face_tracker
        )
    else:
        model, dt, device = yolo_initialization(
            frame_shape = (480, 640, 3),
            weights= WindowsPath(os.path.join(ROOT, 'estimators', 'face_detection_module', 'yolov5', 'weights', 'brightness_augmentation_best.pt')),
            data = WindowsPath(os.path.join(ROOT, 'estimators', 'face_detection_module', 'yolov5', 'data', 'coco128.yaml')),
            depth_face_tracker=depth_face_tracker
        )

    print('Face detection module is initialized')

    # Initialize head pose estimation module
    handler = getattr(service, 'pose')
    print('Head pose estimation module is initialized')

    if not video_folder_path:
        pipeline, align = realsense_initialization()
    # Video
    else:
        rgb_caps, depth_caps, total_video_num = video_loader_initialization(video_folder_path)
        current_video_index = 0
        rgb_cap, depth_cap = rgb_caps[current_video_index], depth_caps[current_video_index]
    # Define pose estimation & face detection thresholds
    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1) as pose:
        with mp_face_mesh.FaceMesh(
            max_num_faces=3,
            min_detection_confidence=0.5) as face_mesh:
            print("Initialization step is done. Please turn on the super multiview renderer")
            time_stamp1, time_stamp2, time_stamp3, time_stamp4, time_stamp5 = 0, 0, 0, 0,0 
            while True:
                # Load mode (0: No tracking / 1: Eye tracking / 2: Eye tracking + Head pose estimation / 3: Eye tracking + Head pose estimation + Action recongition)
                mode = load_mode(base_path=base_path) # mode
                if mode == 0:
                    break

                # Get input
                start_time = time.time()
                if not video_folder_path:
                    frame, depth = get_input(pipeline=pipeline, align=align, video_path=video_folder_path) # 5ms
                else: # Load next video automatically.
                    (rgb_ret, frame), (depth_ret, depth) = rgb_cap.read(), depth_cap.read()
                    if depth_ret:
                        depth = depth[: ,:, 0]
                    else:
                        if current_video_index + 1 < total_video_num:
                            current_video_index += 1
                            rgb_cap, depth_cap = rgb_caps[current_video_index], depth_caps[current_video_index]
                            continue
                        else:
                            break

                if not frame.any() or not depth.any():
                    continue
                if depth.shape != frame.shape:
                    frame = cv2.resize(frame, dsize=(depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_AREA)

                # Get frame information
                height, width = frame.shape[:2]
                
                # face detection
                if flip_mode:
                    draw_frame = cv2.flip(frame.copy(), 1)

                #human_infos, face_num = resnet_face_detection(frame, depth, net, human_infos)
                human_infos, face_num = yolo_face_detection( # 17ms ~ 21ms
                    im=frame, 
                    depth = depth, 
                    dt=dt, 
                    device = device, 
                    model = model, 
                    draw_frame=draw_frame, 
                    view_img = True, 
                    frame_shape = (3, 480, 640), 
                    human_infos= human_infos,
                    depth_face_tracker=depth_face_tracker)

                if face_num > 0:

                    # Head pose estimation
                    if mode > 1:
                        human_infos = head_pose_estimation(frame, human_infos, fa, handler)
                            
                        # Main user classification
                        main_user_index, draw_frame = main_user_classification(draw_frame, human_infos, flip_mode)

                    if mode > 2:
                        # Gaze estimation
                        # frame = gaze_estimation(frame_copy, frame, human_infos[main_user_index], visualization)

                        # Body pose estimation
                        draw_frame = body_pose_estimation(pose, frame, draw_frame, depth, human_infos[0])

                        # Action recognition
                        draw_frame = action_recognition(frame, draw_frame, human_infos[0], fps)

                    if mode > 1:
                        human_infos = [human_infos[main_user_index]]
                    else:
                        human_infos = [human_infos[0]]

                    # Visualization
                    draw_frame = face_box_visualization(draw_frame, human_infos, flip_mode) # 0.5ms
                    draw_frame = visualization(draw_frame, depth, human_infos[0], text_visualization, flip_mode) # 0.5ms

                    # Calibration
                    calibration(human_infos[0]) # 0.5ms

                    # Networking with renderer
                    networking(human_infos[0], mode, base_path) # 0.5ms
                if draw_frame is not None:
                    cv2.imshow('MediaPipe Pose1', draw_frame)
                else:
                    cv2.imshow('MediaPipe Pose1', cv2.flip(frame, 1))
                cv2.waitKey(1)


                fps = 1 / (time.time() - start_time)
                print(fps)

def main_function():
    #main(video_folder_path='C:/Users/user/Desktop/test')
    main()

if __name__ == "__main__":
    main_function()