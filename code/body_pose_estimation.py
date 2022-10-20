import mediapipe as mp
import numpy as np
from multiprocessing import shared_memory
from estimators.body_pose_estimator import body_pose_estimation_func


def body_pose_estimation():
	mp_drawing = mp.solutions.drawing_utils
	mp_pose = mp.solutions.pose

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

	# frame shared memory
	frame_shape = (640, 640, 3)
	frame_shm = shared_memory.SharedMemory(name='frame')
	frame = np.ndarray(frame_shape, dtype=np.uint8, buffer=frame_shm.buf)

	# depth shared memory
	frame_shape = (640, 640)
	depth_shm = shared_memory.SharedMemory(name='depth')
	depth = np.ndarray(frame_shape, dtype=np.uint64, buffer=depth_shm.buf)

	# network shm
	network_shape = (1)
	size_array = np.zeros(network_shape, dtype=np.uint8)
	network_shm = shared_memory.SharedMemory(name = 'networking')
	network_sh_array = np.ndarray(network_shape, dtype=np.uint8, buffer=network_shm.buf)

	with mp_pose.Pose(
		min_detection_confidence=0.5,
		min_tracking_confidence=0.5,
		model_complexity=1) as pose:
		while 1:
			if network_sh_array[:] < 3:
				continue
			body_poses, body_coordinates = body_pose_estimation_func(pose, frame, depth)
			body_pose_sh_array[:] = body_pose_size_array[:]
			body_pose_sh_array[:] = body_poses[:]
			body_coordinates_sh_array[:] = body_coordinates_size_array[:]
			body_coordinates_sh_array[:] = body_coordinates[:]
