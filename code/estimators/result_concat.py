import cv2
import numpy as np

cap1 = cv2.VideoCapture('C:/Users/user/Desktop/repo_face_detection_result.avi')
cap2 = cv2.VideoCapture('C:/Users/user/Desktop/mediapipe_result.avi')
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('concat.avi', fourcc, 25.0, (1280,480))
while cap1.isOpened():
	ret1, frame1 = cap1.read()
	ret2, frame2 = cap2.read()
	if ret1 is False or ret2 is False:
		break
	concat_frame = np.concatenate([frame1, frame2], axis=1)
	out.write(concat_frame)
cap1.release()
cap2.release()
out.release()
