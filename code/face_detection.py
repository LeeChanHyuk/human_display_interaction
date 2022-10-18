import os
from cv2 import drawFrameAxes
from matplotlib.pyplot import draw
import numpy as np
import cv2
from multiprocessing import shared_memory
import time

from estimators.face_detection_module.yolov5.detect import yolo_face_detection, yolo_initialization


def face_detection():
	ROOT = os.path.dirname(os.path.abspath(__file__))
	# face coordinate shared memory
	face_coordinate_shape = (10, 4)
	size_array = np.zeros(face_coordinate_shape, dtype=np.int64)
	face_coordinate_shm = shared_memory.SharedMemory(name = 'face_box_coordinate')
	face_coordinate_sh_array = np.ndarray(face_coordinate_shape, dtype=np.int64, buffer=face_coordinate_shm.buf)

	# frame shared memory
	frame_shape = (640, 640, 3)
	frame_shm = shared_memory.SharedMemory(name='frame')
	frame = np.ndarray(frame_shape, dtype=np.uint8, buffer=frame_shm.buf)
	draw_frame = np.zeros((frame_shape), dtype=np.uint8)


	model, dt, device = yolo_initialization(
		frame_shape = (640, 640, 3),
		weights= os.path.join(ROOT, 'estimators', 'face_detection_module', 'yolov5', 'weights', 'hue_rgb_strong_best.torchscript'),
		data = os.path.join(ROOT, 'estimators', 'face_detection_module', 'yolov5', 'data', 'coco128.yaml'),
	)

	while 1:
		start_time = time.time()
		face_coordinates = yolo_face_detection( # 17ms ~ 21ms
			im=frame, 
			dt=dt, 
			device = device, 
			model = model, 
			frame_shape = (640, 640, 3))

		# push the zero array into the shm array
		face_coordinate_sh_array[:] = size_array[:]

		# push the estimated face coordinates
		for i in range(len(face_coordinates)):
			face_coordinate_sh_array[i] = face_coordinates[i]
		"""draw_frame[:,:,:] = frame[:,:,:]
		for i in range(len(face_coordinates)):
			x1, y1, x2, y2 = face_coordinates[0]
			cv2.putText(draw_frame, "FPS:" + str(int(1/(time.time() - start_time))), (0, 100), cv2.FONT_HERSHEY_COMPLEX, 3, (0,0,255), 3)
			cv2.rectangle(draw_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)"""
		#cv2.imshow("face detection", draw_frame)
		#cv2.waitKey(1)
		#print('Face detection fps is', str(1/(time.time() - start_time)))