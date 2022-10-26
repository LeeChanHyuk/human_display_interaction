"""
Hand Tracking Module
By: Murtaza Hassan
Youtube: http://www.youtube.com/c/MurtazasWorkshopRoboticsandAI
Website: https://www.computervision.zone
"""
 
import cv2
import mediapipe as mp
import time
import math
import numpy as np
import itertools
from collections import deque
 
class handDetector():
    def __init__(self, mode=False, maxHands=10, detectionCon=0.5, trackCon=0.5):
        # Default
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        
        # Mediapipe
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]

        # Data
        self.index_thumb_distance = deque()
        self.hand_grab_bool = deque()
        self.hand_spread_bool = deque()
        self.scaling_factor = deque()
        self.translating_factor = deque()
        self.rotating_factor = deque()
        
        self.deque_initialization()

        # state
        self.object_made_time = float(time.time())
        self.state = 'standard'
        self.scale_start_value = 0
        self.grab_start_value = 0
        self.spread_start_value = 0
        self.hand_up_state = False
        self.scaling_tolerance = 10
        self.translating_tolerance = 10
        self.rotating_tolerance = 10

        self.last_state_changed_time = float(time.time())
        self.find_hand = False
    
    ###################################### hand data manipulation #########################################

    def deque_initialization(self):
        for i in range(200):
            self.index_thumb_distance.append(1000)
            self.hand_grab_bool.append(False)
            self.hand_spread_bool.append(False)
            self.scaling_factor.append(1.0)
            self.translating_factor.append(0.0)
            self.rotating_factor.append(0)
    
    def put_info(self, data, sort):
        if sort == 'index_thumb_distance':
            val = self.index_thumb_distance.popleft()
            self.index_thumb_distance.append(data)
        elif sort == 'hand_grab_bool':
            val = self.hand_grab_bool.popleft()
            self.hand_grab_bool.append(data)
        elif sort == 'scaling_factor':
            val = self.scaling_factor.popleft()
            self.scaling_factor.append(data)
        elif sort == 'translating_factor':
            val = self.translating_factor.popleft()
            self.translating_factor.append(data)
        elif sort == 'hand_spread_bool':
            val = self.hand_spread_bool.popleft()
            self.hand_spread_bool.append(data)
        elif sort == 'rotating_factor':
            val = self.rotating_factor.popleft()
            self.rotating_factor.append(data)

    ################################# hand detection ###############################

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            self.find_hand = True
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,self.mpHands.HAND_CONNECTIONS)
        else:
            self.find_hand = False
        return img
    
    def findPosition(self, img, handNo=0, draw=True):
        fingerPositionList = []
        if self.find_hand:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                fingerPositionList.append([id, cx, cy])
        return fingerPositionList

    def find_main_user_hand(self, img, face_center, depth):
        main_user_finger_list = []
        finger_distance = []
        if self.results.multi_hand_landmarks:
            for i in range(len(self.results.multi_hand_landmarks)):
                fingerPositionList = self.findPosition(img, i, draw = False)
                main_user_finger_list.append(fingerPositionList)
                finger_distance.append(self.get_3d_distance(
                    [fingerPositionList[9][1], fingerPositionList[9][2], depth[fingerPositionList[9][2], fingerPositionList[9][1]]],
                    face_center
                ))
            max_distance = 100000
            max_index = 0

            # check the two closest finger lists from the center of the main user's face
            # the list is consisted of left and right hand list of main user
            for i in range(len(finger_distance)):
                if finger_distance[i] < max_distance:
                    max_distance = finger_distance[i]
                    max_index = i

            # return main user left, right hands positions
            return main_user_finger_list[max_index]

    ############################################### hand state analysis functions ###############################################

    def index_finger_grab(self, finger_position_list):
        index_finger_vec1 = [finger_position_list[6][1] - finger_position_list[5][1], finger_position_list[6][2] - finger_position_list[5][2]]
        index_finger_vec2 = [finger_position_list[8][1] - finger_position_list[7][1], finger_position_list[8][2] - finger_position_list[7][2]]
        index_theta = np.inner(index_finger_vec1, index_finger_vec2) / ((math.sqrt(index_finger_vec1[0] ** 2 + index_finger_vec1[1] ** 2)) * \
            (math.sqrt(index_finger_vec2[0] ** 2 + index_finger_vec2[1] ** 2)))
        index_theta = np.arccos(index_theta)

        middle_finger_vec1 = [finger_position_list[10][1] - finger_position_list[9][1], finger_position_list[10][2] - finger_position_list[9][2]]
        middle_finger_vec2 = [finger_position_list[12][1] - finger_position_list[11][1], finger_position_list[12][2] - finger_position_list[11][2]]
        middle_theta = np.inner(middle_finger_vec1, middle_finger_vec2) / ((math.sqrt(middle_finger_vec1[0] ** 2 + middle_finger_vec1[1] ** 2)) * \
            (math.sqrt(middle_finger_vec2[0] ** 2 + middle_finger_vec2[1] ** 2)))
        middle_theta = np.arccos(middle_theta)

        if index_theta > 1 and middle_theta < 1:
            return True
        else:
            return False

    def fingersUp(self,finger_list):
        fingers = []
        # Thumb
        if finger_list[self.tipIds[0]][1] > finger_list[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        # 4 Fingers
        for id in range(1, 5):
            if finger_list[self.tipIds[id]][2] < finger_list[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers
    
    def handSpread(self, finger_list):
        spread_hands_bool = False
        fingers = self.fingersUp(finger_list)
        if sum(fingers) >= 3 and not self.handGrab(finger_list):
            spread_hands_bool = True
        return spread_hands_bool

    def handGrab(self, finger_lists):
        distance_1 = self.get_distance(finger_lists[9], finger_lists[10])
        distance_2 = self.get_distance(finger_lists[9], finger_lists[11])
        distance_3 = self.get_distance(finger_lists[9], finger_lists[12])
        distance_4 = self.get_distance(finger_lists[10], finger_lists[11])
        distance_5 = self.get_distance(finger_lists[10], finger_lists[12])
        distance_6 = self.get_distance(finger_lists[11], finger_lists[12])
        max_distance = max(distance_1, distance_2, distance_3, distance_4, distance_5, distance_6)
        min_distance = min(distance_1, distance_2, distance_3, distance_4, distance_5, distance_6)
        ratio = min_distance / max_distance
        if ratio < 0.2 or finger_lists[12][2] > finger_lists[11][2] or finger_lists[11][2] > finger_lists[10][2] or finger_lists[10][2] > finger_lists[9][2]:
            return True
        else:
            return False

    def hand_fist(self, finger_position_list):
        finger_x_list = [finger_position_list[2][1], finger_position_list[3][1], finger_position_list[4][1], \
            finger_position_list[6][1], finger_position_list[7][1], finger_position_list[8][1], \
            finger_position_list[10][1], finger_position_list[11][1], finger_position_list[12][1], \
            finger_position_list[14][1], finger_position_list[15][1], finger_position_list[16][1], \
            finger_position_list[18][1], finger_position_list[19][1], finger_position_list[20][1]]
        finger_y_list = [finger_position_list[2][2], finger_position_list[3][2], finger_position_list[4][2], \
            finger_position_list[6][2], finger_position_list[7][2], finger_position_list[8][2], \
            finger_position_list[10][2], finger_position_list[11][2], finger_position_list[12][2], \
            finger_position_list[14][2], finger_position_list[15][2], finger_position_list[16][2], \
            finger_position_list[18][2], finger_position_list[19][2], finger_position_list[20][2]]
        x_var = np.var(finger_x_list)
        y_var = np.var(finger_y_list)
        var_mean = (x_var + y_var) / 2
        if var_mean < 300:
            return True
        else:
            return False

    def hand_up(self, finger_position_list):
        up = True
        for i in range(1, 21):
            if finger_position_list[i][2] > finger_position_list[0][2]:
                up = False
        return up

    ####################################### hand motion classification #########################################

    def scale_manipulation(self, fps, finger_position_list):
        half_fps = int(fps // 2)
        spread_hand_bool = self.handSpread(finger_position_list)
        index_finger_grab = self.index_finger_grab(finger_position_list)
        index_thumb_distance = self.get_distance(finger_position_list[4], finger_position_list[8])
        grab_bool = self.handGrab(finger_position_list)
        self.put_info(grab_bool, 'hand_grab_bool')
        self.hand_up_state = self.hand_up(finger_position_list)

        # Put the info into the detector deques
        self.put_info(index_thumb_distance, 'index_thumb_distance')
        self.put_info(spread_hand_bool, 'hand_spread_bool')

        if self.state != 'scaling':
            self.tolerance = half_fps
            # distance between index and thumb finger for 0.5 sec
            distances = list(itertools.islice(self.index_thumb_distance, len(self.index_thumb_distance)-half_fps, len(self.index_thumb_distance), 1))
            diff = distances - np.mean(distances)
            # Classify whether the scaling motion is occurred
            if len(diff) > 0 and np.all(diff < 10) and index_finger_grab and spread_hand_bool and self.hand_up_state: # must be fixed
                self.state = 'scaling'
                self.last_state_changed_time = float(time.time())
                self.scale_start_value = distances[-1]
        elif self.state == 'scaling':
            diff = self.scale_start_value - self.index_thumb_distance[-1]
            # Classify whether the scaling motion is occurred
            if index_finger_grab is False or spread_hand_bool is False or self.hand_up_state is False:
                self.scaling_tolerance -= 1
            else:
                self.scaling_tolerance = min(self.scaling_tolerance + 1, half_fps)
            # Check the tolerance of scaling motion miss
            if self.scaling_tolerance <= 0:
                self.state = 'standard'
                self.scaling_tolerance = half_fps
            # scaling up
            if diff < 0:
                self.put_info((1+(2 * (abs(diff)/100.0))), 'scaling_factor')
                #print('S', str(self.scale_start_value), 'N', str(self.index_thumb_distance[-1]), 'F', str(1+(2*(abs(diff)/100.0))))
            # scaling down
            else:
                self.put_info(1-(2*(diff / 100.0)), 'scaling_factor')
                #print('S', str(self.scale_start_value), 'N', str(self.index_thumb_distance[-1]), 'F', str(1-(2*(abs(diff)/100.0))))
        return spread_hand_bool, index_finger_grab

    def translation_manipulation(self, hand_center_position, finger_position_list, fps):
        half_fps = int(fps // 2)
        if self.state != 'translating':
            half_fps = int(fps // 2)
            self.tolerance = half_fps
            grab = True
            for i in range(half_fps):
                if self.hand_grab_bool[-1-i] == False:
                    grab = False
            if grab and self.hand_up_state:
                self.state = 'translating'
                self.grab_start_value = hand_center_position
                self.last_state_changed_time = float(time.time())
        elif self.state == 'translating':
            start_angle = self.get_3d_angle(self.grab_start_value, 640, 69)
            end_angle = self.get_3d_angle(hand_center_position, 640, 69)
            grab_diff = self.get_3d_x_diff(start_angle, end_angle, self.grab_start_value[2], hand_center_position[2])
            self.put_info(grab_diff, 'translating_factor')
            if self.hand_grab_bool[-1] == False or self.hand_up_state == False:
                self.translating_tolerance -= 1
            else:
                self.translating_tolerance = min(self.translating_tolerance + 1, half_fps)
            if self.translating_tolerance <= 0:
                self.state = 'standard'
                self.translating_tolerance = int(fps // 2)
    
    def get_3d_angle(self, position, img_w, camera_angle):
        center_w = int(img_w // 2)
        diff_x = position[0] - center_w
        angle = camera_angle * (diff_x / center_w)
        return angle
    
    def get_3d_x_diff(self, angle_1, angle_2, depth_1, depth_2):
        x_1 = math.sin(math.radians(angle_1)) * depth_1
        x_2 = math.sin(math.radians(angle_2)) * depth_2
        return x_2 - x_1

    def rotation_manipulation(self, hand_center_position, fps):
        half_fps = int(fps // 2)
        if self.state != 'rotating':
            self.tolerance = half_fps
            spread = True
            for i in range(half_fps):
                if self.hand_spread_bool[-1-i] == False:
                    spread = False
            if spread and self.hand_up_state:
                self.state = 'rotating'
                self.spread_start_value = hand_center_position
                self.last_state_changed_time = float(time.time())
        elif self.state == 'rotating':
            self.tolerance = half_fps
            start_angle = self.get_3d_angle(self.spread_start_value, 640, 69)
            end_angle = self.get_3d_angle(hand_center_position, 640, 69)
            position_diff = self.get_3d_x_diff(start_angle, end_angle, self.grab_start_value[2], hand_center_position[2])
            self.put_info(position_diff, 'rotating_factor')
            if self.hand_spread_bool[-1] == False or self.hand_grab_bool[-1] is True or self.hand_up_state is False:
                self.rotating_tolerance -= 1
            else:
                self.rotating_tolerance = min(self.rotating_tolerance + 1, half_fps)
            if self.rotating_tolerance <= 0:
                self.state = 'standard'
                self.rotating_tolerance = int(fps // 2)

	######################################################## Utils ######################################################

    def get_distance(self, p1, p2):
        if type(p1) == list:
            if len(p1) == 3:
                length = math.hypot(p1[1] - p2[1], p1[2] - p2[2])
            else:
                length = math.hypot(p1[0] - p2[0], p1[1] - p2[1])
            return length
        elif len(p1.shape) == 3:
            length = math.hypot(p1[1] - p2[1], p1[2] - p2[2])
        else:
            length = math.hypot(p1[0] - p2[0], p1[1] - p2[1])
        return length

    def get_3d_distance(self, p1, p2):
        length = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2)
        return length

            
            
        
