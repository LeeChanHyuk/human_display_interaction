from unittest import result
import cv2
import time
import numpy as np
import estimators.hand_gesture_recognizer as htm
from multiprocessing import shared_memory
from collections import deque
 
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
    return int(fps)

def user_hand_state_initialization(user_hand_state):
    for i in range(200):
        user_hand_state.append('standard')
    return user_hand_state

def user_state_analysis(detector, user_hand_state, fps, current_state, last_state):
    fps = int(fps)
    user_hand_state.popleft()
    user_hand_state.append(current_state)
    same_with_user_state = True
    val = None
    state = None
    for i in range(max(10,fps)):
        if user_hand_state[-1-i] != current_state:
            same_with_user_state = False
            break
    if same_with_user_state:
        #if last_state != current_state and current_state == 'scaling':
        #    print('=============================')
        #    print(detector.last_scaling_factor)
        if current_state == 'scaling':
            val = detector.scaling_factor[-1]
            state = 'scaling'
        elif current_state == 'translating':
            val = detector.translating_factor[-1]
            state = 'translating'
        elif current_state == 'rotating':
            val = detector.rotating_factor[-1]
            state = 'rotating'
        elif current_state == 'standard':
            state = 'standard'
        elif current_state == 'horizontal_flip_standby':
            state = 'horizontal_flip_standby'
        elif current_state == 'vertical_flip_standby':
            state = 'vertical_flip_standby'
        elif current_state == 'hand_shake':
            state = 'hand_shake'
    if state is None:
        state = last_state
    if state == 'scaling':
        val = detector.scaling_factor[-1]
    elif state == 'translating':
        val = detector.translating_factor[-1]
    elif state == 'rotating':
        val = detector.rotating_factor[-1]
    elif state == 'hand_shake':
        val = [0, 0, 0]
    #print('current_State', current_state, ' state', state)
    return val, user_hand_state, state

def draw_hand(img, finger_list):
    for i in range(21):
        if i == 12:
            cv2.circle(img, (finger_list[i][1], finger_list[i][2]), 3, (0, 0, 255), 3)
        elif i == 11:
            cv2.circle(img, (finger_list[i][1], finger_list[i][2]), 3, (0, 255, 0), 3)
        else:
            cv2.circle(img, (finger_list[i][1], finger_list[i][2]), 3, (0, 255, 255), 3)
    return img

def result_networking(hand_gesture_sh_array, hand_val_sh_array, state, detector, fps, val, user_hand_state):
    try:
        hand_gesture_sh_array[:] = state#
        if val:
            if state == 'standard':
                hand_val_sh_array[:] = [0, 0, 0]
            elif state == 'horizontal_hand_flip' or state == 'vertical_hand_flip':
                val, user_hand_state, state = user_state_analysis(detector, user_hand_state, fps, 'standard', 'standard')
                hand_val_sh_array[:] = [0, 0, 0]
            elif state == 'translating':
                hand_val_sh_array[:] = val[:]
            elif state == 'hand_shake':
                hand_val_sh_array[:] = val[:]
            else:
                hand_val_sh_array[:] = [val, 0, 0]
    except:
        print(1)
    return hand_gesture_sh_array, hand_val_sh_array, user_hand_state

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
    size_array = np.chararray(hand_gesture_shape, itemsize=30)
    hand_gesture_shm = shared_memory.SharedMemory(name = 'hand_gesture')
    hand_gesture_sh_array = np.chararray(hand_gesture_shape, itemsize=30, buffer=hand_gesture_shm.buf)

    # hand_gesture 
    hand_val_shape = (3)
    size_array = np.zeros(hand_val_shape, dtype=np.int64)
    hand_val_shm = shared_memory.SharedMemory(name = 'hand_val')
    hand_val_sh_array = np.ndarray(hand_val_shape, dtype=np.int64, buffer=hand_val_shm.buf)

    fist_count = 0
    state = 'standard'
    user_hand_state = user_hand_state_initialization(user_hand_state)
    last_hand_detected_time = float(time.time())
    val = 0
    while True:
        pTime = time.time()
        img[:] = frame[:]
    
        # Find Hand
        img = detector.findHands(img, draw=False)


        left_finger_position_list, right_finger_position_list, left_hand_position, right_hand_position = detector.find_main_user_two_hand(img, main_user_face_center_coordinate_sh_array[0], depth)
        if left_finger_position_list is None or right_finger_position_list is None:
            finger_position_list = detector.find_main_user_hand(img, main_user_face_center_coordinate_sh_array[0], depth)
        if left_hand_position is not None and right_hand_position is not None:
            detector.scale_manipulation_new(fps, left_finger_position_list, right_finger_position_list, left_hand_position, right_hand_position, depth, state)
            img = draw_hand(img, left_finger_position_list)
            img = draw_hand(img, right_finger_position_list)
            img = cv2.putText(img, 'Two hand', (300, 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            fps = show_image(img, pTime)
            scaling_factor = detector.scaling_factor[-fps-1]
            val, user_hand_state, state = user_state_analysis(detector, user_hand_state, fps, detector.state, state)
            hand_gesture_sh_array, hand_val_sh_array, user_hand_state = result_networking(hand_gesture_sh_array, hand_val_sh_array, state, detector, fps, val, user_hand_state)
            """if state == 'scaling':
                print(state, detector.scale_start_value)
            else:
                print(state)"""
            continue
        elif finger_position_list is not None:
            last_hand_detected_time = float(time.time())
            hand_center_position = [min(639,finger_position_list[9][1]), min(639,finger_position_list[9][2]), int(depth[min(639,finger_position_list[9][2]), min(639,finger_position_list[9][1])])]
            #print(hand_center_position)
            img = draw_hand(img, finger_position_list)
            hand_fist_bool, var_mean = detector.new_hand_fist(finger_position_list, hand_center_position)
            if hand_fist_bool:
                #print(var_mean)
                val, user_hand_state, state = user_state_analysis(detector, user_hand_state, fps, 'standard', state)
                hand_gesture_sh_array, hand_val_sh_array, user_hand_state = result_networking(hand_gesture_sh_array, hand_val_sh_array, state, detector, fps, val, user_hand_state)
                detector.state = state
                #print(state)
            else:
                horizontal_hand_flip = detector.horizontal_hand_flip_manipulation(fps, finger_position_list)
                vertical_hand_flip = detector.vertical_hand_flip_manipulation(fps, finger_position_list)
                if horizontal_hand_flip:
                    state = 'horizontal_hand_flip'
                if vertical_hand_flip:
                    state = 'vertical_hand_flip'
                if detector.state == 'horizontal_flip_standby' or detector.state == 'vertical_flip_standby':
                    val, user_hand_state, state = user_state_analysis(detector, user_hand_state, fps, detector.state, state)
                    fps = show_image(img, pTime)
                    hand_gesture_sh_array, hand_val_sh_array, user_hand_state = result_networking(hand_gesture_sh_array, hand_val_sh_array, state, detector, fps, val, user_hand_state)
                    #print(state)
                    continue
                
                detector.hand_state_estimation(fps, finger_position_list)
                #spread_hand_bool, index_finger_grab = detector.scale_manipulation(fps, finger_position_list)
                #if detector.state == 'scaling':
                #    scaling_factor = detector.scaling_factor[-fps-1]
                #    val, user_hand_state, state = user_state_analysis(detector, user_hand_state, fps, detector.state, state)
                #    fps = show_image(img, pTime)
                #    hand_gesture_sh_array, hand_val_sh_array, user_hand_state = result_networking(hand_gesture_sh_array, hand_val_sh_array, state, detector, fps, val, user_hand_state)
                #    """if state == 'scaling':
                #        print(state, detector.scale_start_value)
                #    else:
                #        print(state)"""
                #    continue
                detector.hand_shake_estimation(fps, finger_position_list, depth)
                if detector.state == 'hand_shake':
                    state = 'hand_shake'
                    val, user_hand_state, state = user_state_analysis(detector, user_hand_state, fps, detector.state, state)
                    fps = show_image(img, pTime)
                    hand_gesture_sh_array, hand_val_sh_array, user_hand_state = result_networking(hand_gesture_sh_array, hand_val_sh_array, state, detector, fps, val, user_hand_state)
                    print(state)
                    continue
                detector.translation_manipulation(hand_center_position, finger_position_list, fps)
                if detector.state == 'translating':
                    translating_factor = detector.translating_factor[-fps-1]
                    val, user_hand_state, state = user_state_analysis(detector, user_hand_state, fps, detector.state, state)
                    fps = show_image(img, pTime)
                    hand_gesture_sh_array, hand_val_sh_array, user_hand_state = result_networking(hand_gesture_sh_array, hand_val_sh_array, state, detector, fps, val, user_hand_state)
                    print(state)
                    """if state == 'translating':
                        print(state, detector.grab_start_value)
                    else:
                        print(state)"""
                    continue
                
                detector.rotation_manipulation(hand_center_position, fps)
                if detector.state == 'rotating':
                    rotating_factor = detector.rotating_factor[-fps-1]
                    val, user_hand_state, state = user_state_analysis(detector, user_hand_state, fps, detector.state, state)
                    fps = show_image(img, pTime)
                    hand_gesture_sh_array, hand_val_sh_array, user_hand_state = result_networking(hand_gesture_sh_array, hand_val_sh_array, state, detector, fps, val, user_hand_state)
                    print(state)
                    """if state == 'rotating':
                        print(state, detector.spread_start_value)
                    else:
                        print(state)"""
                    continue
                fps = show_image(img, pTime)
                val, user_hand_state, state = user_state_analysis(detector, user_hand_state, fps, 'standard', state)
                detector.state = state # because the detector cannot update the state in unfinding situation.
                print(state)
        else:
            #print('no hand is detected')
            if float(time.time()) - last_hand_detected_time > 2:
                val, user_hand_state, state = user_state_analysis(detector, user_hand_state, fps, 'standard', state)
                detector.state = state # because the detector cannot update the state in unfinding situation.
                #print(state)
        hand_gesture_sh_array, hand_val_sh_array, user_hand_state = result_networking(hand_gesture_sh_array, hand_val_sh_array, state, detector, fps, val, user_hand_state)

        
        fps = show_image(img, pTime)

