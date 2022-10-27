import cv2
import time
import numpy as np
import estimators.hand_gesture_recognizer as htm
from collections import deque
from multiprocessing import shared_memory
 
################################
wCam, hCam = 640, 640
################################

""" Target
- 크기 조절 기능 (두 손가락) -> 손을 펴고 있는지 (3개 이상) & 엄지와 검지 끝의 거리를 측정
- Translation 제어 기능 (주먹을 쥔 정도)
- Rotation 기능 (손바닥을 펴고 좌우로)
- 떠오르고, 사라지는 기능 (한 손)
- 분리되고, 합쳐지는 기능 (펼친 손)
"""
 

def show_image(img, pTime):
    # Frame rate
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (40, 50), cv2.FONT_HERSHEY_COMPLEX,
                1, (255, 0, 0), 3)
 
    cv2.imshow("Img", img)
    cv2.waitKey(1)
    return fps

def user_hand_state_initialization(user_hand_state):
    for i in range(200):
        user_hand_state.append('standard')
    return user_hand_state

def user_state_analysis(detector, user_hand_state, half_fps):
    user_state = detector.state
    user_hand_state.popleft()
    user_hand_state.append(user_state)
    same_with_user_state = True
    val = None
    for i in range(half_fps):
        if user_hand_state[-1-i] != user_state:
            same_with_user_state = False
            break
    if same_with_user_state:
        if user_state == 'scaling':
            val = detector.scaling_factor[-1-half_fps]
        elif user_state == 'translating':
            val = detector.translating_factor[-1-half_fps]
        elif user_state == 'rotating':
            val = detector.rotating_factor[-1-half_fps]
    return val, user_hand_state

def draw_hand(img, finger_list):
    for i in range(21):
        if i == 12:
            cv2.circle(img, (finger_list[i][1], finger_list[i][2]), 3, (0, 0, 255), 3)
        elif i == 11:
            cv2.circle(img, (finger_list[i][1], finger_list[i][2]), 3, (0, 255, 0), 3)
        else:
            cv2.circle(img, (finger_list[i][1], finger_list[i][2]), 3, (0, 255, 255), 3)
    return img

def hand_gesture_recognition():
    fps = 20
    detector = htm.handDetector(detectionCon=0.7, maxHands=10)
    user_hand_state = deque()

    # frame shared memory
    frame_shape = (640, 640, 3)
    frame_shm = shared_memory.SharedMemory(name='frame')
    frame = np.ndarray(frame_shape, dtype=np.uint8, buffer=frame_shm.buf)
    img = np.zeros(frame_shape, dtype=np.uint8)

    # depth shared memory
    depth_shape = (640, 640)
    depth_shm = shared_memory.SharedMemory(name='depth')
    depth = np.ndarray(depth_shape, dtype=np.uint64, buffer=depth_shm.buf)

    # main user face center coordinate shared memory
    main_user_face_center_coordinate_shape = (1, 3) # for 20 peoples
    main_user_face_center_coordinate_shm = shared_memory.SharedMemory(name = 'main_user_face_center_coordinate')
    main_user_face_center_coordinate_sh_array = np.ndarray(main_user_face_center_coordinate_shape, dtype=np.int64, buffer=main_user_face_center_coordinate_shm.buf)

    # hand_gesture shm
    hand_gesture_shape = (1)
    size_array = np.chararray(hand_gesture_shape, itemsize=10)
    hand_gesture_shm = shared_memory.SharedMemory(name = 'hand_gesture')
    hand_gesture_sh_array = np.chararray(hand_gesture_shape, itemsize=1, buffer=hand_gesture_shm.buf)

    fist_count = 0
    user_hand_state = user_hand_state_initialization(user_hand_state)
    while True:
        pTime = time.time()
        img[:] = frame[:]
        if main_user_face_center_coordinate_sh_array[0][0] == 0:
            continue
    
        # Find Hand
        img = detector.findHands(img, draw=False)

        finger_position_list = detector.find_main_user_hand(img, main_user_face_center_coordinate_sh_array[0], depth)
        if finger_position_list is not None:
            hand_center_position = [min(639,finger_position_list[9][1]), min(639,finger_position_list[9][2]), depth[min(639,finger_position_list[9][2]), min(639,finger_position_list[9][1])]]
            img = draw_hand(img, finger_position_list)
            fps = 20
            half_fps = int(fps // 3)
            #hand_fist_bool = detector.hand_fist(finger_position_list, main_user_face_center_coordinate_sh_array[0])
            hand_fist_bool, var_mean = detector.new_hand_fist(finger_position_list, hand_center_position)
            #print('grabbing', str(var_mean))
            if hand_fist_bool:
                if detector.state != 'standard':
                    if fist_count < 3:
                        fist_count += 1
                    else:
                        detector.state = 'standard'
                        fist_count = 0
            else:
                spread_hand_bool, index_finger_grab = detector.scale_manipulation(fps, finger_position_list)
                if detector.state == 'scaling':
                    scaling_factor = detector.scaling_factor[-half_fps-1]
                    #print(detector.state, str(scaling_factor))
                    fps = show_image(img, pTime)
                    val, user_hand_state = user_state_analysis(detector, user_hand_state, half_fps)
                    continue
                
                detector.translation_manipulation(hand_center_position, finger_position_list, fps)
                if detector.state == 'translating':
                    translating_factor = detector.translating_factor[-half_fps-1]
                    #print(detector.state, str(translating_factor))
                    fps = show_image(img, pTime)
                    val, user_hand_state = user_state_analysis(detector, user_hand_state, half_fps)
                    continue
                
                detector.rotation_manipulation(hand_center_position, fps)
                if detector.state == 'rotating':
                    rotating_factor = detector.rotating_factor[-half_fps-1]
                    #print(detector.state, str(rotating_factor))
                    fps = show_image(img, pTime)
                    val, user_hand_state = user_state_analysis(detector, user_hand_state, half_fps)
                    continue
        else:
            detector.state = 'standard'
            print('no hand is detected')
        
        hand_gesture_sh_array[:] = detector.state
        show_image(img, pTime)
        #print(spread_hand_bools)

