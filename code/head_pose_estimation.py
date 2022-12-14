import os
import numpy as np
import cv2
import time
from multiprocessing import shared_memory
from estimators.face_detection_module.yolov5.detect import main

import estimators.head_pose_estimation_module.service as service
from estimators.main_user_classifier import main_user_classification, main_user_classification_filter
from user_information.human import HumanInfo
from estimators.calibrator import calibration
from total_visualization import draw_axis

def head_pose_estimation_func(frame, face_boxes, fa, handler):
	feed = frame.copy()

	# Estimate head pose
	head_poses = []
	for i in range(len(face_boxes)):
		# Face box coordinates
		x1, y1, x2, y2 = map(int, face_boxes[i][:])
		if x1 == 0 and y1 == 0 and x2 == 0 and y2 == 0:
			break
		# Get face landmarks from the face image for estmating user head poses
		for results in fa.get_landmarks(feed, np.expand_dims(np.array([int(x1), int(y1), int(x2), int(y2)]), axis=0)):
			pitch, yaw, roll = handler(frame, results, color=(125, 125, 125))
			head_poses.append([pitch, yaw, roll])
	return head_poses

def head_pose_estimation():
	ROOT = os.path.dirname(os.path.abspath(__file__))

	######################## Input ############################

	# frame shared memory
	frame_shape = (640, 640, 3)
	frame_shm = shared_memory.SharedMemory(name='frame')
	frame = np.ndarray(frame_shape, dtype=np.uint8, buffer=frame_shm.buf)
	draw_frame = np.ndarray(frame_shape, dtype=np.uint8)

	# depth shared memory
	depth_shape = (640, 640)
	depth_shm = shared_memory.SharedMemory(name='depth')
	depth = np.ndarray(depth_shape, dtype=np.uint64, buffer=depth_shm.buf)

	# face coordinate shared memory
	face_coordinate_shape = (10, 4)
	box_size_array = np.zeros(face_coordinate_shape, dtype=np.int64)
	face_coordinate_shm = shared_memory.SharedMemory(name = 'face_box_coordinate')
	face_coordinate_sh_array = np.ndarray(face_coordinate_shape, dtype=np.int64, buffer=face_coordinate_shm.buf)
	face_coordinate_array = np.ndarray(face_coordinate_shape, dtype=np.int64)

	# face box coordinate shared memory
	main_user_face_box_coordinate_shape = (1, 4) # for 20 peoples
	main_user_face_box_coordinate_shm = shared_memory.SharedMemory(name = 'main_user_face_box_coordinate')
	main_user_face_box_coordinate_sh_array = np.ndarray(main_user_face_box_coordinate_shape, dtype=np.int64, buffer=main_user_face_box_coordinate_shm.buf)

	# main user face center coordinate shared memory
	main_user_face_center_coordinate_shape = (1, 3) # for 20 peoples
	main_user_face_center_coordinate_shm = shared_memory.SharedMemory(name = 'main_user_face_center_coordinate')
	main_user_face_center_coordinate_sh_array = np.ndarray(main_user_face_center_coordinate_shape, dtype=np.int64, buffer=main_user_face_center_coordinate_shm.buf)

	# main user face center coordinate shared memory
	main_user_calib_face_center_coordinate_shape = (1, 3) # for 20 peoples
	size_array = np.zeros(main_user_calib_face_center_coordinate_shape, dtype=np.int64)
	main_user_calib_face_center_coordinate_shm = shared_memory.SharedMemory(name = 'main_user_calib_face_center_coordinate')
	main_user_calib_face_center_coordinate_sh_array = np.ndarray(main_user_face_center_coordinate_shape, dtype=np.int64, buffer=main_user_calib_face_center_coordinate_shm.buf)

	# network shm
	network_shape = (1)
	size_array = np.zeros(network_shape, dtype=np.uint8)
	network_shm = shared_memory.SharedMemory(name = 'networking')
	network_sh_array = np.ndarray(network_shape, dtype=np.uint8, buffer=network_shm.buf)

	######################## Output ###########################
	
    # head pose shm
	head_pose_shape = (3)
	size_array = np.zeros(head_pose_shape, dtype=np.int64)
	head_pose_shm = shared_memory.SharedMemory(name = 'head_pose')
	head_pose_sh_array = np.ndarray(head_pose_shape, dtype=np.int64, buffer=head_pose_shm.buf)

	# Initialize face detection module
	fa = service.DepthFacialLandmarks(os.path.join(ROOT, "estimators/head_pose_estimation_module/weights/sparse_face.tflite"))
	handler = getattr(service, 'pose')
	previous_length = 0
	previous_main_user_position = [0, 0, 0]
	tolerance = 10
	main_user_info = HumanInfo()

	while 1:
		start_time = time.time()
		face_center_coordinates = []
		# copy the face coordinates from the shared memory (for safety)
		face_coordinate_array[:] = face_coordinate_sh_array[:]
		# deal 10 people (maximum num)
		for i in range(10):
			x1, y1, x2, y2 = face_coordinate_sh_array[i][:]
			if int(x1) == 0 and int(x2) == 0:
				break
			# calculate the face 3D center coordinates from the face coordinates
			face_center_x, face_center_y = int((x1 + x2) / 2), int((y1 + y2) / 2)
			face_center_x = min(max(0, face_center_x), 639)
			face_center_y = min(max(0, face_center_y), 639)
			face_center_z = depth[face_center_y, face_center_x]
			face_center_coordinates.append([face_center_x, face_center_y, face_center_z])
		# network_sh_array means the mode of the renderer 
		# (1 - face tracking / 2 - main user classification / 3 - action recognition / 4 - hand gesture recognition)
		if network_sh_array < 2:
			main_user_face_box_coordinate_sh_array[0][:] = face_coordinate_sh_array[0][:]
			if len(face_center_coordinates) > 0:
				main_user_face_center_coordinate_sh_array[0][:] = face_center_coordinates[0][:]
			continue
		draw_frame[:] = frame[:]
		# head pose estimation
		head_poses = head_pose_estimation_func(frame, face_coordinate_sh_array, fa, handler)

		# if the head pose is estimated correctly, the main user classification process is proceeded.
		if len(head_poses) > 0 and len(head_poses) == len(face_center_coordinates):
			fps = int(1 / (time.time() - start_time))
			# main user classification
			main_user_index = main_user_classification(face_center_coordinates, head_poses, use_head_pose=False)
			# filter the main user information for stability
			main_user_index, tolerance = main_user_classification_filter(tolerance, previous_main_user_position, face_center_coordinates[main_user_index], main_user_index, face_center_coordinates, fps)
			# save the information of the main user
			previous_main_user_position = face_center_coordinates[main_user_index]
			# only store the head pose of main user information
			head_pose_sh_array[:] = np.array(head_poses[main_user_index])[:]
			# save the face box coordinate of the main user
			main_user_face_box_coordinate_sh_array[0][:] = face_coordinate_array[main_user_index][:]
			# draw the main user face
			cv2.rectangle(draw_frame, (face_coordinate_array[main_user_index][0], face_coordinate_array[main_user_index][1]),
			(face_coordinate_array[main_user_index][2], face_coordinate_array[main_user_index][3]), (255, 0, 0), 3)
			flip_val = 1
			# draw the head pose of the main user
			draw_frame = draw_axis(draw_frame, flip_val * head_pose_sh_array[1], head_pose_sh_array[0], flip_val * head_pose_sh_array[2], 
			[int(face_center_coordinates[main_user_index][0]), int(face_center_coordinates[main_user_index][1] - 30)])
			main_user_face_center_coordinate_sh_array[0][:] = face_center_coordinates[main_user_index][:]
		# if you don't use the action recognition function, you must calibrate the face coordinate
		# because the calibration process is included in the action recognition process
		if network_sh_array < 3 or network_sh_array == 4:
			main_user_info.face_box[:] = main_user_face_box_coordinate_sh_array[0][:]
			main_user_info._put_data([main_user_face_center_coordinate_sh_array[0][0], main_user_face_center_coordinate_sh_array[0][1], main_user_face_center_coordinate_sh_array[0][2]], 'center_eyes')
			# calibrate the user info
			calibration(main_user_info, True)
			main_user_calib_face_center_coordinate_sh_array[:] = main_user_info.calib_center_eyes[:]
			# draw the image of the camera with user info
			cv2.imshow('draw_frame', draw_frame)
			cv2.waitKey(1)
