import mediapipe as mp
import numpy as np
import cv2
import time

from multiprocessing import shared_memory
from copy import deepcopy

from estimators.action_recognizer import action_recognition_func
from user_information.human import HumanInfo
from total_visualization import visualization
from estimators.face_detector import calibration


def action_recognition():
	main_user_info = HumanInfo()
	mp_drawing = mp.solutions.drawing_utils
	mp_pose = mp.solutions.pose

	# frame shared memory
	frame_shape = (640, 640, 3)
	frame_shm = shared_memory.SharedMemory(name='frame')
	frame = np.ndarray(frame_shape, dtype=np.uint8, buffer=frame_shm.buf)
	draw_frame = np.zeros(frame.shape, dtype=np.uint8)

	# face coordinate shared memory
	main_user_face_box_coordinate_shape = (1, 4) # for 20 peoples
	main_user_face_box_coordinate_shm = shared_memory.SharedMemory(name = 'main_user_face_box_coordinate')
	main_user_face_coordinate_sh_array = np.ndarray(main_user_face_box_coordinate_shape, dtype=np.int64, buffer=main_user_face_box_coordinate_shm.buf)

	# main user face center coordinate shared memory
	main_user_face_center_coordinate_shape = (1, 3) # for 20 peoples
	main_user_face_center_coordinate_shm = shared_memory.SharedMemory(name = 'main_user_face_center_coordinate')
	main_user_face_center_coordinate_sh_array = np.ndarray(main_user_face_center_coordinate_shape, dtype=np.int64, buffer=main_user_face_center_coordinate_shm.buf)

	# main user face center coordinate shared memory
	main_user_calib_face_center_coordinate_shape = (1, 3) # for 20 peoples
	size_array = np.zeros(main_user_calib_face_center_coordinate_shape, dtype=np.int64)
	main_user_calib_face_center_coordinate_shm = shared_memory.SharedMemory(name = 'main_user_calib_face_center_coordinate')
	main_user_calib_face_center_coordinate_sh_array = np.ndarray(main_user_face_center_coordinate_shape, dtype=np.int64, buffer=main_user_calib_face_center_coordinate_shm.buf)

    # body pose shm
	body_pose_shape = (3)
	body_pose_shm = shared_memory.SharedMemory(name = 'body_pose')
	body_pose_size_array = np.zeros(body_pose_shape, dtype=np.int64)
	body_pose_sh_array = np.ndarray(body_pose_shape, dtype=np.int64, buffer=body_pose_shm.buf)

    # body coordinates shm
	body_coordinates_shape = (5, 3)
	body_coordinates_shm = shared_memory.SharedMemory(name = 'body_coordinates')
	body_coordinates_size_array = np.zeros(body_coordinates_shape, dtype=np.int64)
	body_coordinates_sh_array = np.ndarray(body_coordinates_shape, dtype=np.int64, buffer=body_coordinates_shm.buf)

    # head pose shm
	head_pose_shape = (3)
	head_pose_shm = shared_memory.SharedMemory(name = 'head_pose')
	head_pose_sh_array = np.ndarray(head_pose_shape, dtype=np.int64, buffer=head_pose_shm.buf)

    # action shm
	action_shape = (1)
	action_shm = shared_memory.SharedMemory(name = 'action')
	action_sh_array = np.chararray(action_shape, itemsize=10, buffer=action_shm.buf)

	# network shm
	network_shape = (1)
	size_array = np.zeros(network_shape, dtype=np.uint8)
	network_shm = shared_memory.SharedMemory(name = 'networking')
	network_sh_array = np.ndarray(network_shape, dtype=np.uint8, buffer=network_shm.buf)

	while 1:
		start_time = time.time()
		draw_frame[:] = frame[:]
		main_user_info.face_box[:] = main_user_face_coordinate_sh_array[0][:]
		main_user_info._put_data([main_user_face_center_coordinate_sh_array[0][0], main_user_face_center_coordinate_sh_array[0][1], main_user_face_center_coordinate_sh_array[0][2]], 'center_eyes')
		calibration(main_user_info, True)
		main_user_calib_face_center_coordinate_sh_array[:] = main_user_info.calib_center_eyes[:]
		if network_sh_array[:] > 1:
			main_user_info._put_data(body_coordinates_sh_array[1], 'center_mouths')
			main_user_info._put_data(body_coordinates_sh_array[2], 'left_shoulders')
			main_user_info._put_data(body_coordinates_sh_array[3], 'right_shoulders')
			main_user_info._put_data(body_coordinates_sh_array[4], 'center_stomachs')
			main_user_info._put_data([head_pose_sh_array[1], head_pose_sh_array[0], head_pose_sh_array[2]], 'head_poses')
			main_user_info._put_data(body_pose_sh_array[:], 'body_poses')
		
		if network_sh_array[:] > 2:
			action_recognition_func(main_user_info)
			action_sh_array[:] = main_user_info.human_state
			cv2.putText(draw_frame, main_user_info.human_state, (0, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
		draw_frame = visualization(draw_frame, main_user_info, network_sh_array[:], False)
		cv2.imshow('draw_frame', draw_frame)
		cv2.waitKey(1)
		print('Action recognition estimation fps is', str(1/(time.time() - start_time)))
		
