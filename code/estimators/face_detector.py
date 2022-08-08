import math
import cv2
import numpy as np

from user_information.human import HumanInfo
from estimators.head_pose_estimator import box_extraction


def calibration(human_info, real_sense_calibration = True):
    center_eyes = human_info.center_eyes[-1].copy()
    calib_parameter = [0.9245, -0.004, 0.0584, -0.0242, 0.9475, -0.0083, 0.0208, 0.1013, 0.8956, -32.2596, 121.3725, 26.666 + 200 + 350]
    # y = 240 - y
    # x = x - 320
    # for D435
    camera_horizontal_angle = 87 # RGB = 60
    camera_vertical_angle = 58 # RGB = 42

    i_width = 640
    i_height = 480
    
    # before calibration
    eye_x = center_eyes[0] - (i_width/2)
    eye_y = (i_height/2) - center_eyes[1]
    eye_z = center_eyes[2]

    detected_x_angle = (camera_horizontal_angle / 2) * (eye_x / (i_width/2))
    detected_y_angle = (camera_vertical_angle / 2) * (eye_y / (i_height/2))

    new_x = eye_z * math.sin(math.radians(detected_x_angle))
    new_y = eye_z * math.sin(math.radians(detected_y_angle))
    new_z = eye_z

    new_x, new_y, new_z = new_x * -1.0, new_y * 1.0, new_z * 1.0
    new_x = calib_parameter[0] * new_x + calib_parameter[3] * new_y + calib_parameter[6] * new_z + (calib_parameter[9])
    new_y = calib_parameter[1] * new_x + calib_parameter[4] * new_y + calib_parameter[7] * new_z + (calib_parameter[10])
    new_z = calib_parameter[2] * new_x + calib_parameter[5] * new_y + calib_parameter[8] * new_z + (calib_parameter[11])

    human_info.calib_center_eyes = [new_x, new_y, new_z]

    # Old calib
    #new_x = eye_z * math.sin(math.radians(detected_x_angle)) * 7 / 9
    #new_y = eye_z * math.sin(math.radians(detected_y_angle))
    #y_offset = eye_z * math.sin(math.radians(camera_vertical_angle/2))

def face_detection(frame, depth, face_mesh, human_infos = None):
    height, width = frame.shape[:2]
    bgr_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_results = face_mesh.process(bgr_image)
    if face_results.multi_face_landmarks:
        if not human_infos:
            human_infos = []
        for index, face_landmarks in enumerate(face_results.multi_face_landmarks):
            if index >= len(human_infos):
                if len(human_infos)>0:
                    human_info = human_infos[-1]
                else:
                    human_info = HumanInfo()
            else:
                human_info = human_infos[index]
            face_boxes, right_eye_box, left_eye_box = box_extraction(
                face_landmarks=face_landmarks,
                width = width,
                height = height)
            face_box = np.array(face_boxes)
            left_eye_box = np.array(left_eye_box)
            right_eye_box = np.array(right_eye_box)
            human_info.face_box = face_box # face box is not used for action recognition. Thus, face_box is not list.
            human_info.left_eye_box = left_eye_box
            human_info.right_eye_box = right_eye_box

            center_eyes_x = (left_eye_box[0][0] + left_eye_box[0][2]) / 2
            center_eyes_y = (left_eye_box[0][1] + left_eye_box[0][3]) / 2
            center_eyes_z = depth[int(center_eyes_y), int(center_eyes_x)]
            human_info._put_data([center_eyes_x, center_eyes_y, center_eyes_z], 'center_eyes')
            if index >= len(human_infos):
                human_infos.append(human_info)
    if face_results.multi_face_landmarks:
        return human_infos, len(face_results.multi_face_landmarks)
    else:
        return human_infos, 0