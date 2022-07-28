from cv2 import flip
import numpy as np
from visualization.draw_utils import draw_axis
import cv2

def draw_gaze(image_in, eye_pos, pitchyaw, length=40.0, thickness=2, color=(0, 0, 255)):
    """Draw gaze angle on given image with a given eye positions."""
    image_out = image_in
    if len(image_out.shape) == 2 or image_out.shape[2] == 1:
        image_out = cv2.cvtColor(image_out, cv2.COLOR_GRAY2BGR)
    dx = -length * np.sin(pitchyaw[1])
    dy = length * np.sin(pitchyaw[0])
    cv2.arrowedLine(image_out, tuple(np.round(eye_pos).astype(np.int32)),
                   tuple(np.round([eye_pos[0] + dx, eye_pos[1] + dy]).astype(int)), color,
                   thickness, cv2.LINE_AA, tipLength=0.2)
    return image_out

def draw_body_information(zero_array, width, height, x, y, z, yaw, pitch, roll): # [width, height, x, y, z, yaw, pitch, roll]
	draw_axis(zero_array, yaw, pitch, roll, [(width / 40) * 4, (height/20) * 1.5], color1=(255,255,0), color2=(255,0,255), color3=(0,255,255))
	cv2.putText(zero_array, '[Body position]:', (int((width/40) * 1), int((height/20) * 6)), 1, 2, (255, 0, 0), 3)
	cv2.putText(zero_array, ' X :' + str(x), (int((width/40) * 1), int((height/20) * 7.5)), 1, 1.8, (255, 255, 0), 3)
	cv2.putText(zero_array, ' Y :' + str(y), (int((width/40) * 1), int((height/20) * 9)), 1, 1.8, (255, 255, 0), 3)
	cv2.putText(zero_array, ' Z :' + str(z), (int((width/40) * 1), int((height/20) * 10.5)), 1, 1.8, (255, 255, 0), 3)
	cv2.putText(zero_array, '[Body pose]:', (int((width/40) * 1), int((height/20) * 13)), 1, 2, (255, 0, 0), 3)
	cv2.putText(zero_array, ' Yaw :' + str(yaw), (int((width/40) * 1), int((height/20) * 14.5)), 1, 1.8, (255, 255, 0), 3)
	cv2.putText(zero_array, ' Pitch :' + str(pitch), (int((width/40) * 1), int((height/20) * 16)), 1, 1.8, (255, 255, 0), 3)
	cv2.putText(zero_array, ' Roll :' + str(roll), (int((width/40) * 1), int((height/20) * 17.5)), 1, 1.8, (255, 255, 0), 3)
	return zero_array

def draw_face_information(zero_array, width, height, x, y, z, yaw, pitch, roll): # [width, height, x, y, z, yaw, pitch, roll]
	draw_axis(zero_array, yaw, pitch, roll, [(width / 40) * 18.5, (height/20) * 1.5], color1=(255,0,0), color2=(0,255,0), color3=(0,0,255))
	cv2.putText(zero_array, '[Head position]:', (int((width/40) * 15), int((height/20) * 6)), 1, 2, (0, 255, 0), 3)
	cv2.putText(zero_array, ' X :' + str(x), (int((width/40) * 15), int((height/20) * 7.5)), 1, 1.8, (255, 255, 0), 3)
	cv2.putText(zero_array, ' Y :' + str(y), (int((width/40) * 15), int((height/20) * 9)), 1, 1.8, (255, 255, 0), 3)
	cv2.putText(zero_array, ' Z :' + str(z), (int((width/40) * 15), int((height/20) * 10.5)), 1, 1.8, (255, 255, 0), 3)
	cv2.putText(zero_array, '[Head pose]:', (int((width/40) * 15), int((height/20) * 13)), 1, 2, (0, 255, 0), 3)
	cv2.putText(zero_array, ' Yaw :' + str(yaw), (int((width/40) * 15), int((height/20) * 14.5)), 1, 1.8, (255, 255, 0), 3)
	cv2.putText(zero_array, ' Pitch :' + str(pitch), (int((width/40) * 15), int((height/20) * 16)), 1, 1.8, (255, 255, 0), 3)
	cv2.putText(zero_array, ' Roll :' + str(roll), (int((width/40) * 15), int((height/20) * 17.5)), 1, 1.8, (255, 255, 0), 3)
	return zero_array

def draw_gaze_information(zero_array, width, height, x, y, z, gazes): # [width, height, x, y, z, x, y]
	left_gaze = gazes[1]
	right_gaze = gazes[0]
	draw_gaze(zero_array, ((width/40) * 32, (height/20) * 1.5), left_gaze, length=60.0, thickness=2)
	draw_gaze(zero_array, ((width/40) * 34, (height/20) * 1.5), right_gaze, length=60.0, thickness=2)
	cv2.putText(zero_array, '[Eye position]:', (int((width/40) * 29), int((height/20) * 6)), 1, 2, (0, 0, 255), 3)
	cv2.putText(zero_array, ' X :' + str(x), (int((width/40) * 29), int((height/20) * 7.5)), 1, 1.8, (255, 255, 0), 3)
	cv2.putText(zero_array, ' Y :' + str(y), (int((width/40) * 29), int((height/20) * 9)), 1, 1.8, (255, 255, 0), 3)
	cv2.putText(zero_array, ' Z :' + str(z), (int((width/40) * 29), int((height/20) * 10.5)), 1, 1.8, (255, 255, 0), 3)
	cv2.putText(zero_array, '[Eye pose]:', (int((width/40) * 29), int((height/20) * 13)), 1, 2, (0, 0, 255), 3)
	cv2.putText(zero_array, ' Left_x:' + str(round(left_gaze[1], 2)), (int((width/40) * 29), int((height/20) * 16)), 1, 1.8, (255, 255, 0), 3)
	cv2.putText(zero_array, ' Left_y :' + str(round(left_gaze[0], 2)), (int((width/40) * 29), int((height/20) * 14.5)), 1, 1.8, (255, 255, 0), 3)
	cv2.putText(zero_array, ' Right_x:' + str(round(right_gaze[1], 2)), (int((width/40) * 29), int((height/20) * 19)), 1, 1.8, (255, 255, 0), 3)
	cv2.putText(zero_array, ' Right_y:' + str(round(right_gaze[0], 2)), (int((width/40) * 29), int((height/20) * 17.5)), 1, 1.8, (255, 255, 0), 3)
	return zero_array

def draw_body_keypoints(frame, keypoints):
	colors = [(255, 0, 0),
	(0, 255, 0),
	(0, 0, 255),
	(255, 255, 0),
	(0, 255, 255)]
	for index, keypoint in enumerate(keypoints):
		cv2.circle(frame, (int(keypoint[0]), int(keypoint[1])), 3, colors[index], 3)
	return frame

def visualization(draw_frame, depth, human_info, text_visualization, flip_mode):
    height, width = draw_frame.shape[:2]
    flip_val = 1
    if flip_mode:
        flip_val = -1
    # apply colormap to depthmap
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=0.03), cv2.COLORMAP_JET)

    # Visualize head pose
    if human_info.head_pose_estimation_flag:
        draw_frame = draw_axis(draw_frame, flip_val * human_info.head_poses[-1][0], human_info.head_poses[-1][1], human_info.head_poses[-1][2], 
                          [640 - int((human_info.face_box[0][0] + human_info.face_box[0][2])/2), int(human_info.face_box[0][1] - 30)])

    # Visualize body pose
    if human_info.body_pose_estimation_flag:
        draw_frame = draw_axis(draw_frame, flip_val * human_info.body_poses[-1][0], human_info.body_poses[-1][1], human_info.body_poses[-1][2], 
                          [640 - int((human_info.left_shoulders[-1][0] + human_info.right_shoulders[-1][0])/2), int(human_info.left_shoulders[-1][1])], 
                          color1=(255,255,0), color2=(255,0,255), color3=(0,255,255))

    # Visualize eye pose
    if human_info.gaze_estimation_flag:
        for i, ep in enumerate([human_info.left_eye_landmark, human_info.right_eye_landmark]):
            for (x, y) in ep.landmarks[16:33]:
                color = (0, 255, 0)
                if ep.eye_sample.is_left:
                    color = (255, 0, 0)
                cv2.circle(draw_frame,(int(round(x)), int(round(y))), 1, color, -1, lineType=cv2.LINE_AA)
            gaze = [human_info.left_eye_gaze, human_info.right_eye_gaze][i]
            length = 60.0
            draw_gaze(draw_frame, ep.landmarks[-2], gaze, length=length, thickness=2)
			
	# Visualize the values of each poses
    if text_visualization:
        width = int(width * 1.5)
        zero_array = np.zeros((height, width, 3), dtype=np.uint8)
        if human_info.body_pose_estimation_flag:
            center_shoulder = (human_info.left_shoulders[-1] + human_info.right_shoulders[-1]) / 2
            zero_array= draw_body_information(zero_array, width, height, round(center_shoulder[0], 2), round(center_shoulder[1], 2), round(center_shoulder[2], 2), 
																	flip_val * round(human_info.body_poses[-1][0], 2), round(human_info.body_poses[-1][1], 2), round(human_info.body_poses[-1][2], 2))
        if human_info.head_pose_estimation_flag:
            zero_array = draw_face_information(zero_array, width, height, round(human_info.center_eyes[-1][0], 2), round(human_info.center_eyes[-1][1]),
																	round(human_info.center_eyes[-1][2]), flip_val * round(human_info.head_poses[-1][0], 2), round(human_info.head_poses[-1][1], 2),
																	round(human_info.head_poses[-1][2], 2))

        if human_info.gaze_estimation_flag:
            zero_array = draw_gaze_information(zero_array,width, height, round(human_info.center_eyes[-1][0], 2), round(human_info.center_eyes[-1][1], 2),
																	round(human_info.center_eyes[-1][2], 2), [human_info.left_eye_gaze, human_info.right_eye_gaze])
        draw_frame = np.concatenate([draw_frame, zero_array], axis=1)
    return draw_frame