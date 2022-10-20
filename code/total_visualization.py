import cv2
import numpy as np
from math import *

def draw_axis(img, yaw, pitch, roll, visualization_point, size = 50, color1=(255,0,0), color2=(0,255,0), color3=(0,0,255)):
    pitch = (pitch * np.pi / 180)
    yaw = -(yaw * np.pi / 180)
    roll = (roll * np.pi / 180)

    # X-Axis pointing to right. drawn in red
    x1 = size * (cos(yaw) * cos(roll))
    y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw))

    # Y-Axis | drawn in green
    #        v
    x2 = size * (-cos(yaw) * sin(roll))
    y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll))

    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (sin(yaw))
    y3 = size * (-cos(yaw) * sin(pitch))

    cv2.line(img, (int(visualization_point[0]), int(visualization_point[1])), (int(visualization_point[0] + x1),int(visualization_point[1] + y1)),color1,3)
    cv2.line(img, (int(visualization_point[0]), int(visualization_point[1])), (int(visualization_point[0] + x2),int(visualization_point[1] + y2)),color2,3)
    cv2.line(img, (int(visualization_point[0]), int(visualization_point[1])), (int(visualization_point[0] + x3),int(visualization_point[1] + y3)),color3,2)

    return img

def visualization(draw_frame, human_info, mode, flip_mode):
	height, width = draw_frame.shape[:2]
	flip_val = 1
	if flip_mode:
		flip_val = -1
	# Visualize face box

	if flip_mode:
		height, width = draw_frame.shape[:2]
		x1, y1, x2, y2 = int(width - human_info.face_box[0]), int(human_info.face_box[1]), int(width - human_info.face_box[0][2]), int(human_info.face_box[0][3])
	else:
		x1, y1, x2, y2 = int(human_info.face_box[0]), int(human_info.face_box[1]), int(human_info.face_box[2]), int(human_info.face_box[3])
	cv2.rectangle(draw_frame, 
					(x1, y1), 
					(x2, y2), 
					(0, 0, 255), 3)
	text = "{:.2f}%".format(human_info.face_detection_confidence * 100)
	cv2.putText(draw_frame, text, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)

	if mode > 1:
		# Visualize head pose
		draw_frame = draw_axis(draw_frame, flip_val * human_info.head_poses[-1][0], human_info.head_poses[-1][1], flip_val * human_info.head_poses[-1][2], 
		[int((human_info.face_box[0] + human_info.face_box[2])/2), int(human_info.face_box[1] - 30)])

		# Visualize body pose
		#draw_frame = draw_axis(draw_frame, flip_val * human_info.body_poses[-1][1], human_info.body_poses[-1][0], flip_val * human_info.body_poses[-1][2], 
		#					[int((human_info.left_shoulders[-1][0] + human_info.right_shoulders[-1][0])/2), int(human_info.left_shoulders[-1][1])], 
		#					color1=(255,255,0), color2=(255,0,255), color3=(0,255,255))
	if mode > 2:
		draw_frame = draw_body_keypoints(draw_frame, 
		[human_info.center_eyes[-1],
		human_info.center_mouths[-1],
		human_info.left_shoulders[-1],
		human_info.right_shoulders[-1],
		human_info.center_stomachs[-1]], False)
	return draw_frame

def draw_body_keypoints(frame, keypoints, flip_mode):
    colors = [(255, 0, 0),
	(0, 255, 0),
	(0, 0, 255),
	(255, 255, 0),
	(0, 255, 255)]
    for index, keypoint in enumerate(keypoints):
        if flip_mode:
            cv2.circle(frame, (int(-1 * keypoint[0]), int(keypoint[1])), 3, colors[index], 3)
        else:
            cv2.circle(frame, (int(keypoint[0]), int(keypoint[1])), 3, colors[index], 3)
    return frame