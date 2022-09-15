import os
import cv2
import numpy as np
from pathlib import WindowsPath
from copy import deepcopy

from estimators.face_detection_module.yolov5.detect import yolo_face_detection, yolo_initialization
from estimators.face_detector import face_box_visualization, face_detection, resnet_face_detection

ROOT = os.path.dirname(os.path.abspath(__file__))
video_folder_path = os.path.join(ROOT, 'testset')
flip_mode = False
aug_list = ['0.5', '0.7']
for aug in aug_list:
	print(aug)
	model_dict = dict()
	model_list = os.listdir(
		os.path.join('C:/Users/user/Desktop/git/human_display_interaction/code/estimators/face_detection_module/yolov5/weights/brightness',
		aug,
		'weights'))
	for model_name in model_list:
		model, dt, device = yolo_initialization(
			frame_shape = (480, 640, 3),
			weights= WindowsPath(os.path.join(ROOT, 'estimators', 'face_detection_module', 'yolov5', 'weights', 'brightness', aug, 'weights', '60.pt')),
				data = WindowsPath(os.path.join(ROOT, 'estimators', 'face_detection_module', 'yolov5', 'data', 'coco128.yaml')),
				depth_face_tracker=None
			)
		model_dict[model_name] = 0
		for video_name in os.listdir(video_folder_path):
			cap = cv2.VideoCapture(os.path.join(video_folder_path, video_name))
			depth = np.zeros((int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))))
			human_infos = None
			while True:
				ret, frame = cap.read()
				if frame is None:
					break
				draw_frame = deepcopy(frame)
				
				human_infos, face_num = yolo_face_detection( # 17ms ~ 21ms
					im=frame, 
					depth = depth, 
					dt=dt, 
					device = device, 
					model = model, 
					draw_frame=draw_frame, 
					view_img = True, 
					frame_shape = (3, 480, 640), 
					human_infos= human_infos,
					depth_face_tracker=None,
					model_dict = model_dict,
					model_name = model_name)
				draw_frame = face_box_visualization(draw_frame, human_infos, flip_mode) # 0.5ms
				cv2.imshow('frame', draw_frame)
				cv2.waitKey(1)
	print(model_dict)
