import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
import time
from multiprocessing import shared_memory
import os
import time

os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Data collection
from user_information.data_collector import realsense_initialization, video_loader_initialization, get_input

def get_input_from_cam(video_folder_path = None):
	flip_mode = True
	# shared memory
	test_array = np.zeros((640, 640, 3), dtype=np.uint8)
	frame_shm = shared_memory.SharedMemory(name='frame')
	frame = np.ndarray(test_array.shape, dtype=np.uint8, buffer=frame_shm.buf)

	# shared memory
	depth_array = np.zeros((640, 640), dtype=np.uint64)
	depth_shm = shared_memory.SharedMemory(name='depth')
	depth = np.ndarray(depth_array.shape, dtype=np.uint64, buffer=depth_shm.buf)

	if not video_folder_path:
		pipeline, align = realsense_initialization()
	else: # video
		rgb_caps, depth_caps, total_video_num = video_loader_initialization(video_folder_path)
		current_video_index = 0
		if len(depth_caps) > 0:
			rgb_cap, depth_cap = rgb_caps[current_video_index], depth_caps[current_video_index]
		else:
			rgb_cap = rgb_caps[current_video_index]
	
	while 1:
		start_time = time.time()
		if not video_folder_path:
			temp_frame, temp_depth = get_input(pipeline=pipeline, align=align, video_path=video_folder_path) # 6~7ms when the detection fps is not less that 60 fps
		else: # Load next video automatically.
			(rgb_ret, temp_frame), (depth_ret, temp_depth) = rgb_cap.read(), depth_cap.read()
			if depth_ret:
				temp_depth = temp_depth[: ,:, 0]
			else:
				if current_video_index + 1 < total_video_num:
					current_video_index += 1
					rgb_cap, depth_cap = rgb_caps[current_video_index], depth_caps[current_video_index]
					continue
				else:
					break
		
		temp_frame = cv2.resize(temp_frame, (640, 640))
		temp_depth = cv2.resize(temp_depth, (640, 640))
		if flip_mode:
			temp_frame = cv2.flip(temp_frame, 1)
			temp_depth = cv2.flip(temp_depth, 1)

		frame[:,:,:] = temp_frame[:,:,:]
		depth[:,:] = (temp_depth[:,:])
		print('Input fps is', str(1/(time.time() - start_time)))
		#cv2.imshow('frame', frame)
		#cv2.waitKey(1)