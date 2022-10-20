import os
import numpy as np
from multiprocessing import Process, shared_memory
from get_input_from_cam import get_input_from_cam
from face_detection import face_detection
from head_pose_estimation import head_pose_estimation
from body_pose_estimation import body_pose_estimation
from action_recognition import action_recognition
from networking import router_function

if __name__ == "__main__":
	##################### shared memory initialization ########################
	# frame shared memory
	test_array = np.zeros((640, 640, 3), dtype=np.uint8)
	frame_shm = shared_memory.SharedMemory(create=True, size=test_array.nbytes, name='frame')
	frame = np.ndarray(test_array.shape, dtype=np.uint8, buffer=frame_shm.buf)

	# depth shared memory
	depth_array = np.zeros((640, 640), dtype=np.uint64)
	depth_shm = shared_memory.SharedMemory(create=True, size=depth_array.nbytes, name='depth')
	depth = np.ndarray(depth_array.shape, dtype=np.uint64, buffer=depth_shm.buf)

	# face box coordinate shared memory
	face_box_coordinate_shape = (10, 4) # for 20 peoples
	size_array = np.zeros(face_box_coordinate_shape, dtype=np.int64)
	face_box_coordinate_shm = shared_memory.SharedMemory(create=True, size=size_array.nbytes, name = 'face_box_coordinate')

	# main user face box coordinate shared memory
	main_user_face_box_coordinate_shape = (1, 4) # for 20 peoples
	size_array = np.zeros(main_user_face_box_coordinate_shape, dtype=np.int64)
	main_user_face_box_coordinate_shm = shared_memory.SharedMemory(create=True, size=size_array.nbytes, name = 'main_user_face_box_coordinate')

	# main user face center coordinate shared memory
	main_user_face_center_coordinate_shape = (1, 3) # for 20 peoples
	size_array = np.zeros(main_user_face_center_coordinate_shape, dtype=np.int64)
	main_user_face_center_coordinate_shm = shared_memory.SharedMemory(create=True, size=size_array.nbytes, name = 'main_user_face_center_coordinate')

	# main user face center coordinate shared memory
	main_user_calib_face_center_coordinate_shape = (1, 3) # for 20 peoples
	size_array = np.zeros(main_user_calib_face_center_coordinate_shape, dtype=np.int64)
	main_user_calib_face_center_coordinate_shm = shared_memory.SharedMemory(create=True, size=size_array.nbytes, name = 'main_user_calib_face_center_coordinate')

    # head pose shm
	head_pose_shape = (3) # for 1 people
	size_array = np.zeros(head_pose_shape, dtype=np.int64)
	head_pose_shm = shared_memory.SharedMemory(create = True, size = size_array.nbytes, name = 'head_pose')
	head_pose_sh_array = np.ndarray(head_pose_shape, dtype=np.int64, buffer=head_pose_shm.buf)

    # body pose shm
	body_pose_shape = (3) # for 1 people
	size_array = np.zeros(body_pose_shape, dtype=np.int64)
	body_pose_shm = shared_memory.SharedMemory(create = True, size = size_array.nbytes, name = 'body_pose')
	body_pose_sh_array = np.ndarray(body_pose_shape, dtype=np.int64, buffer=body_pose_shm.buf)

    # body coordinates shm
	body_coordinates_shape = (5, 3) # for 1 people
	size_array = np.zeros(body_coordinates_shape, dtype=np.int64)
	body_coordinates_shm = shared_memory.SharedMemory(create = True, size = size_array.nbytes, name = 'body_coordinates')
	body_coordinates_sh_array = np.ndarray(body_pose_shape, dtype=np.int64, buffer=body_coordinates_shm.buf)

    # action shm
	action_shape = (1)
	size_array = np.chararray(action_shape, itemsize=10)
	action_shm = shared_memory.SharedMemory(create = True, size = size_array.nbytes, name = 'action')
	action_sh_array = np.chararray(action_shape, itemsize=1, buffer=action_shm.buf)

    # network shm
	network_shape = (1)
	size_array = np.zeros(network_shape, dtype=np.int64)
	network_shm = shared_memory.SharedMemory(create = True, size = size_array.nbytes, name = 'networking')
	network_sh_array = np.ndarray(network_shape, dtype=np.int64, buffer=network_shm.buf)
	network_sh_array[:] = 3

	#################### Multi processing #########################

	p1 = Process(target=get_input_from_cam)
	p2 = Process(target=face_detection)
	p3 = Process(target=head_pose_estimation)
	p4 = Process(target=body_pose_estimation)
	p5 = Process(target=action_recognition)
	p6 = Process(target=router_function)
	p1.start()
	print('p1 start')
	p2.start()
	print('p2 start')
	p3.start()
	print('p3 start')
	p4.start()
	print('p4 start')
	p5.start()
	print('p5 start')
	p6.start()
	print('p6 start')

	p1.join()
	print('p1 join')
	p2.join()
	print('p2 join')
	p3.join()
	print('p3 join')
	p4.join()
	print('p4 join')
	p5.join()
	print('p5 join')
	p6.join()
	print('p6 join')