import os
import numpy as np
import cv2
import time
from multiprocessing import shared_memory

import estimators.head_pose_estimation_module.service as service
from estimators.main_user_classifier import main_user_classification

def head_pose_estimation_func(frame, face_boxes, fa, handler):
	feed = frame.copy()
	# Estimate head pose
	head_poses = []
	for i in range(len(face_boxes)):
		x1, y1, x2, y2 = map(int, face_boxes[i][:])
		if x1 == 0 and y1 == 0 and x2 == 0 and y2 == 0:
			break
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

	# depth shared memory
	depth_shape = (640, 640)
	depth_shm = shared_memory.SharedMemory(name='depth')
	depth = np.ndarray(depth_shape, dtype=np.uint64, buffer=depth_shm.buf)

	# face coordinate shared memory
	face_coordinate_shape = (10, 4)
	box_size_array = np.zeros(face_coordinate_shape, dtype=np.int64)
	face_coordinate_shm = shared_memory.SharedMemory(name = 'face_box_coordinate')
	face_coordinate_sh_array = np.ndarray(face_coordinate_shape, dtype=np.int64, buffer=face_coordinate_shm.buf)

	# face box coordinate shared memory
	main_user_face_box_coordinate_shape = (1, 4) # for 20 peoples
	main_user_face_box_coordinate_shm = shared_memory.SharedMemory(name = 'main_user_face_box_coordinate')
	main_user_face_box_coordinate_sh_array = np.ndarray(main_user_face_box_coordinate_shape, dtype=np.int64, buffer=main_user_face_box_coordinate_shm.buf)

	# main user face center coordinate shared memory
	main_user_face_center_coordinate_shape = (1, 3) # for 20 peoples
	main_user_face_center_coordinate_shm = shared_memory.SharedMemory(name = 'main_user_face_center_coordinate')
	main_user_face_center_coordinate_sh_array = np.ndarray(main_user_face_center_coordinate_shape, dtype=np.int64, buffer=main_user_face_center_coordinate_shm.buf)

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

	while 1:
		start_time = time.time()
		face_center_coordinates = []
		for i in range(10):
			x1, y1, x2, y2 = face_coordinate_sh_array[i][:]
			if int(x1) == 0 and int(x2) == 0:
				break
			face_center_x, face_center_y = int((x1 + x2) / 2), int((y1 + y2) / 2)
			face_center_x = min(max(0, face_center_x), 639)
			face_center_y = min(max(0, face_center_y), 639)
			face_center_z = depth[face_center_y, face_center_x]
			face_center_coordinates.append([face_center_x, face_center_y, face_center_z])
		if network_sh_array < 2:
			main_user_face_box_coordinate_sh_array[0][:] = face_coordinate_sh_array[0][:]
			if len(face_center_coordinates) > 0:
				main_user_face_center_coordinate_sh_array[0][:] = face_center_coordinates[0][:]
			continue
		head_poses = head_pose_estimation_func(frame, face_coordinate_sh_array, fa, handler)
		if len(head_poses) > 0 and len(head_poses) == len(face_center_coordinates):
			main_user_index = main_user_classification(face_center_coordinates, head_poses)

			# only store main user information
			head_pose_sh_array[:] = np.array(head_poses[main_user_index])[:]
			main_user_face_box_coordinate_sh_array[0][:] = face_coordinate_sh_array[main_user_index][:]
			main_user_face_center_coordinate_sh_array[0][:] = face_center_coordinates[main_user_index][:]
			#print('Head pose estimation fps is', str(1/(time.time() - start_time)))
