import os
import time
import zmq
import numpy as np
from multiprocessing import shared_memory


def router_function():
    context2 = zmq.Context()
    to_renderer = context2.socket(zmq.REP)
    to_renderer.bind("tcp://*:5558")

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

    # head pose shm
    head_pose_shape = (3)
    head_pose_shm = shared_memory.SharedMemory(name = 'head_pose')
    head_pose_sh_array = np.ndarray(head_pose_shape, dtype=np.int64, buffer=head_pose_shm.buf)

    # action shm
    action_shape = (1)
    size_array = np.zeros(action_shape, dtype=np.string_)
    action_shm = shared_memory.SharedMemory(name = 'action')
    action_sh_array = np.ndarray(action_shape, dtype=np.string_, buffer=action_shm.buf)

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

    # network shm
    network_shape = (1)
    size_array = np.zeros(network_shape, dtype=np.uint8)
    network_shm = shared_memory.SharedMemory(name = 'networking')
    network_sh_array = np.ndarray(network_shape, dtype=np.uint8, buffer=network_shm.buf)

    while True:
        start_time = time.time()
        message = to_renderer.recv()
        #message = '3'
        network_sh_array[:] = int(message)
        message = str(message)[2]
        face_center_info = str(main_user_calib_face_center_coordinate_sh_array[0][0])+ ' ' + str(main_user_calib_face_center_coordinate_sh_array[0][1]) + ' ' + \
                str(main_user_calib_face_center_coordinate_sh_array[0][2]) + ' ' + str(main_user_face_center_coordinate_sh_array[0][2])
        head_pose_info = str(head_pose_sh_array[0]) + ' ' +str(head_pose_sh_array[1]) + ' ' + str(head_pose_sh_array[2])
        action_info = str(action_sh_array[0])
        hand_info = str(hand_gesture_sh_array[0])
        hand_info = hand_info[2:len(hand_info)-1]
        hand_val = str(hand_val_sh_array[0]) + ' ' + str(hand_val_sh_array[1]) + ' ' + str(hand_val_sh_array[2])
        if message == '0':
            send_message = 'N'
        elif message == '1':
            send_message = 'D' + ' ' + face_center_info
        elif message == '2':
            send_message = 'E' + ' ' + face_center_info + ' ' + head_pose_info
        elif message == '3':
            send_message = 'A' + ' ' + face_center_info + ' ' + head_pose_info + ' ' + action_info
        elif message == '4':
            send_message = 'H' + ' ' + face_center_info + ' ' + head_pose_info + ' ' + hand_info + ' ' + hand_val
        to_renderer.send_string(send_message)
        """tmes = time.time() - start_time
        if tmes == 0:
            tmes = 1
        print('networking fps is', str(1/tmes))"""


def networking(human_info, mode, base_path):
    communication_write = open(os.path.join(base_path, 'communication.txt'), 'r+')
    communication_write.write(str(mode) + '\n')
    communication_write.write(str(round(human_info.calib_center_eyes[0])).zfill(3) + ' ' + str(round(human_info.calib_center_eyes[1]+20)).zfill(3)
                              + ' ' + str(round(human_info.calib_center_eyes[2])).zfill(3) + ' ' + str(round(human_info.center_eyes[-1][2])).zfill(3) + '\n')
    communication_write.write(str(round(human_info.head_poses[-1][1])).zfill(3) + ' ' + str(round(human_info.head_poses[-1][0])).zfill(3)
                              + ' ' + str(round(human_info.head_poses[-1][2])).zfill(3) + '\n')
    communication_write.write(human_info.human_state+'\n')
    communication_write.close()