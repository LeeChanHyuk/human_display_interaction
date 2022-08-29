#!/usr/bin/python3
# -*- coding:utf-8 -*-

import argparse
import estimators.head_pose_estimation_module.service as service
import cv2
import time
import os
import numpy as np

def box_extraction(face_landmarks, width, height):
    """Params
    face_landmarks: face landmarks from mediapipe's face detection results
    width: image_width
    height: image_height
    """
    face_boxes = []
    left_eye_boxes = []
    right_eye_boxes = []

    face_x1 = face_landmarks.landmark[234].x * width
    face_y1 = face_landmarks.landmark[10].y * height
    face_x2 = face_landmarks.landmark[454].x * width
    face_y2 = face_landmarks.landmark[152].y * height

    left_eye_inner_x1 = face_landmarks.landmark[161].x * width
    left_eye_inner_y1 = face_landmarks.landmark[161].y * height
    left_eye_inner_x2 = face_landmarks.landmark[154].x * width
    left_eye_inner_y2 = face_landmarks.landmark[154].y * height
    right_eye_inner_x1 = face_landmarks.landmark[398].x * width
    right_eye_inner_y1 = face_landmarks.landmark[398].y * height
    right_eye_inner_x2 = face_landmarks.landmark[390].x * width
    right_eye_inner_y2 = face_landmarks.landmark[390].y * height
    face_boxes.append([face_x1-10, face_y1-10, face_x2+10, face_y2+10])
    eye_detection_margin = 5
    left_eye_boxes.append([left_eye_inner_x1-eye_detection_margin, left_eye_inner_y1-eye_detection_margin, left_eye_inner_x2+eye_detection_margin, left_eye_inner_y2+eye_detection_margin])
    right_eye_boxes.append([right_eye_inner_x1-eye_detection_margin, right_eye_inner_y1-eye_detection_margin, right_eye_inner_x2+eye_detection_margin, right_eye_inner_y2+eye_detection_margin])
    return face_boxes, left_eye_boxes, right_eye_boxes

def head_pose_estimation(frame, human_infos, fa, handler):
    feed = frame.copy()
    # Estimate head pose
    for index, human_info in enumerate(human_infos):
        face_box  = human_info.face_box
        for results in fa.get_landmarks(feed, face_box):
            pitch, yaw, roll = handler(frame, results, color=(125, 125, 125))
            human_info._put_data([yaw, pitch, roll], 'head_poses')
    return human_infos