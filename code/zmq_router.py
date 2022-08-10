import os
import time
import zmq

from cv2 import split
from collections import defaultdict

def router_function():
    context2 = zmq.Context()
    to_renderer = context2.socket(zmq.REP)
    to_renderer.bind("tcp://*:5558")
    line = ['0', '0 0 0', '0 0 0', 'standard'] # tracking_mode / eye_position / head_rotation / human action
    base_path = os.path.dirname(os.path.abspath(__file__))
    while True:
        # message read from txt file to communicate with tracker
        communication_read = open(os.path.join(base_path, 'communication.txt'), 'r')
        for i in range(4):
            line[i] = communication_read.readline()
        communication_read.close()
        message = to_renderer.recv()
        message = str(message)[2]

        # message write to txt file to communicate with tracker
        communication_write = open(os.path.join(base_path, 'communication.txt'), 'r+')
        communication_write.write(message)
        communication_write.close()
        if line[0].strip() == '0':
            send_message = 'N'
        elif line[0].strip() == '1':
            send_message = 'D' + ' ' + line[1].strip()
        elif line[0].strip() == '2':
            send_message = 'E' + ' ' + line[1].strip() + ' ' + line[2].strip()
        elif line[0].strip() == '3':
            send_message = 'A' + ' ' + line[1].strip() + ' ' + line[2].strip() + ' ' + line[3].strip()
        to_renderer.send_string(send_message)


def networking(human_info, mode, base_path):
    communication_write = open(os.path.join(base_path, 'communication.txt'), 'r+')
    communication_write.write(str(mode) + '\n')
    communication_write.write(str(round(human_info.calib_center_eyes[0])).zfill(3) + ' ' + str(round(human_info.calib_center_eyes[1]+20)).zfill(3)
                              + ' ' + str(round(human_info.calib_center_eyes[2])).zfill(3) + str(round(human_info.center_eyes[-1][2])).zfill(3) + '\n')
    communication_write.write(str(round(human_info.head_poses[-1][1])).zfill(3) + ' ' + str(round(human_info.head_poses[-1][0])).zfill(3)
                              + ' ' + str(round(human_info.head_poses[-1][2])).zfill(3) + '\n')
    communication_write.write(human_info.human_state+'\n')
    communication_write.close()