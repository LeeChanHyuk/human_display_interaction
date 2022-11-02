"""
Hand Tracking Module
By: Murtaza Hassan
Youtube: http://www.youtube.com/c/MurtazasWorkshopRoboticsandAI
Website: https://www.computervision.zone
"""
 
import cv2
from matplotlib.axis import YAxis
from matplotlib.pyplot import pink
import mediapipe as mp
import time
import math
import numpy as np
import itertools
from collections import deque
 
class handDetector():
    def __init__(self, mode=False, maxHands=10, detectionCon=0.5, trackCon=0.3):
        # Default
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        
        # Mediapipe
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, 1, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]

        # Data
        self.index_thumb_distance = deque()
        self.hand_grab_bool = deque()
        self.hand_spread_bool = deque()
        self.horizontal_hand_flip_standby_bool = deque()
        self.vertical_hand_flip_standby_bool = deque()
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
        self.last_hand_position = [0, 0, 0]
        self.hand_up_state = False
        self.stopping_tolerance = 20
        self.horizontal_flip_standby_tolerance = 20
        self.vertical_flip_standby_tolerance = 20

        self.last_state_changed_time = float(time.time())
        self.vertical_flip_time = float(time.time())
        self.horizontal_flip_time = float(time.time())
        self.stopping_time = float(time.time())
        self.find_hand = False
    
    ###################################### hand data manipulation #########################################

    def deque_initialization(self):
        for i in range(200):
            self.index_thumb_distance.append(1000)
            self.hand_grab_bool.append(False)
            self.hand_spread_bool.append(False)
            self.scaling_factor.append(1.0)
            self.translating_factor.append([0.0, 0.0, 0.0])
            self.rotating_factor.append(0)
            self.horizontal_hand_flip_standby_bool.append(False)
            self.vertical_hand_flip_standby_bool.append(False)
    
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
        elif sort == 'horizontal_hand_flip_standby_bool':
            val = self.horizontal_hand_flip_standby_bool.popleft()
            self.horizontal_hand_flip_standby_bool.append(data)
        elif sort == 'vertical_hand_flip_standby_bool':
            val = self.vertical_hand_flip_standby_bool.popleft()
            self.vertical_hand_flip_standby_bool.append(data)

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
                finger_center_position = [min(639,fingerPositionList[9][1]), min(639,fingerPositionList[9][2]), depth[min(639,fingerPositionList[9][2]), min(639,fingerPositionList[9][1])]]
                distance = self.get_3d_distance(finger_center_position, face_center)
                if distance > 50:
                    main_user_finger_list.append(fingerPositionList)
                    finger_distance.append(distance)
            max_distance = 100000
            max_index = 0

            # check the two closest finger lists from the center of the main user's face
            # the list is consisted of left and right hand list of main user
            for i in range(len(finger_distance)):
                if finger_distance[i] < max_distance:
                    max_distance = finger_distance[i]
                    max_index = i

            # return main user left, right hands positions
            if len(main_user_finger_list) > 0:
                return main_user_finger_list[max_index]
            else:
                return None

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

    def fingersUp(self, finger_list, direction):
        fingers = []
        # up, right, down, left
        if direction == 'up':
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
        elif direction == 'left':
            # Thumb
            if finger_list[self.tipIds[0]][2] < finger_list[self.tipIds[0] - 1][2]:
                fingers.append(1)
            else:
                fingers.append(0)
            # 4 Fingers
            for id in range(1, 5):
                if finger_list[self.tipIds[id]][1] < finger_list[self.tipIds[id] - 2][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)
        elif direction == 'down':
            # Thumb
            if finger_list[self.tipIds[0]][1] > finger_list[self.tipIds[0] - 1][1]:
                fingers.append(1)
            else:
                fingers.append(0)
            # 4 Fingers
            for id in range(1, 5):
                if finger_list[self.tipIds[id]][2] > finger_list[self.tipIds[id] - 2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)
        else: # right
            # Thumb
            if finger_list[self.tipIds[0]][2] < finger_list[self.tipIds[0] - 1][2]:
                fingers.append(1)
            else:
                fingers.append(0)
            # 4 Fingers
            for id in range(1, 5):
                if finger_list[self.tipIds[id]][1] > finger_list[self.tipIds[id] - 2][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)
        return fingers

    def handSpread(self, finger_list, direction):
        spread_hands_bool = False
        fingers = self.fingersUp(finger_list, direction)
        if sum(fingers) >= 3 and not self.fingerGrab(finger_list, 9, direction):
            spread_hands_bool = True
        return spread_hands_bool

    def fingerGrab(self, finger_lists, index, direction):
        # middle
        distance_1 = self.get_distance(finger_lists[index], finger_lists[index+1])
        distance_2 = self.get_distance(finger_lists[index], finger_lists[index+2])
        distance_3 = self.get_distance(finger_lists[index], finger_lists[index+3])
        distance_4 = self.get_distance(finger_lists[index+1], finger_lists[index+2])
        distance_5 = self.get_distance(finger_lists[index+1], finger_lists[index+3])
        distance_6 = self.get_distance(finger_lists[index+2], finger_lists[index+3])
        max_distance = max(distance_1, distance_2, distance_3, distance_4, distance_5, distance_6)
        min_distance = min(distance_1, distance_2, distance_3, distance_4, distance_5, distance_6)
        ratio = min_distance / max_distance
        if direction == 'up':
            grab = finger_lists[index+3][2] > finger_lists[index+2][2] or finger_lists[index+2][2] > finger_lists[index+1][2] or finger_lists[index+1][2] > finger_lists[index][2]
        elif direction == 'left':
            grab = finger_lists[index+3][1] > finger_lists[index+2][1] or finger_lists[index+2][1] > finger_lists[index+1][1] or finger_lists[index+1][1] > finger_lists[index][1]
        elif direction == 'down':
            grab = finger_lists[index+3][2] < finger_lists[index+2][2] or finger_lists[index+2][2] < finger_lists[index+1][2] or finger_lists[index+1][2] < finger_lists[index][2]
        else:
            grab = finger_lists[index+3][1] < finger_lists[index+2][1] or finger_lists[index+2][1] < finger_lists[index+1][1] or finger_lists[index+1][1] < finger_lists[index][1]
        if ratio < 0.2 or grab:
            return True
        else:
            return False

    def handGrab(self, finger_lists, direction):
        index_grab = self.fingerGrab(finger_lists, 5, direction)
        middle_grab = self.fingerGrab(finger_lists, 9, direction)
        fourth_grab = self.fingerGrab(finger_lists, 13, direction)
        pinky_grab = self.fingerGrab(finger_lists, 17, direction)
        if index_grab or middle_grab or fourth_grab or pinky_grab:
            return True
        else:
            return False

    def hand_fist(self, finger_position_list, hand_center_position):
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
        var_mean /= hand_center_position[2]

        directions = ['up', 'left', 'right', 'down']
        for i in range(4):
            index_grab = self.fingerGrab(finger_position_list, 5, directions[i])
            middle_grab = self.fingerGrab(finger_position_list, 9, directions[i])
            fourth_grab = self.fingerGrab(finger_position_list, 13, directions[i])
            pinky_grab = self.fingerGrab(finger_position_list, 17, directions[i])
            grab = index_grab and middle_grab and fourth_grab and pinky_grab
            if grab:
                break
        if var_mean < 45000 or grab:
            return True, var_mean
        else:
            return False, var_mean
    
    def new_hand_fist(self, finger_position_list, hand_center_position):
        if hand_center_position[2] != 0:
            self.last_hand_position = hand_center_position
        else:
            hand_center_position[2] = self.last_hand_position[2]
        x_min, y_min, x_max, y_max = 999, 999, -1, -1
        for i in range(21):
            if finger_position_list[i][1] < x_min:
                x_min = finger_position_list[i][1]
            if finger_position_list[i][1] > x_max:
                x_max = finger_position_list[i][1]
            if finger_position_list[i][2] < y_min:
                y_min = finger_position_list[i][2]
            if finger_position_list[i][2] > y_max:
                y_max = finger_position_list[i][2]
        
        start_x_angle = self.get_3d_x_angle(x_min, 640, 69)
        end_x_angle = self.get_3d_x_angle(x_max, 640, 69)
        position_x_diff = self.get_3d_x_diff(start_x_angle, end_x_angle, hand_center_position[2], hand_center_position[2])

        start_y_angle = self.get_3d_y_angle(y_min, 640, 69)
        end_y_angle = self.get_3d_y_angle(y_max, 640, 69)
        position_y_diff = self.get_3d_x_diff(start_y_angle, end_y_angle, hand_center_position[2], hand_center_position[2])

        area = position_x_diff * position_y_diff
        if area <= 50000:
            return True, area
        else:
            return False, area

    def hand_up(self, finger_position_list, direction):
        success = True
        if direction == 'up':
            for i in range(1, 21):
                if finger_position_list[i][2] > finger_position_list[0][2]:
                    success = False
        elif direction == 'left':
            for i in range(1, 21):
                if finger_position_list[i][1] > finger_position_list[0][1]:
                    success = False
        elif direction == 'down':
            for i in range(1, 21):
                if finger_position_list[i][2] < finger_position_list[0][2]:
                    success = False
        else: # right
            for i in range(1, 21):
                if finger_position_list[i][1] < finger_position_list[0][1]:
                    success = False
        return success

    def translation_factor_calculation(self, hand_center_position, scale_ratio = 0.2):
        if hand_center_position[2] == 0:
            hand_center_position[2] = self.last_hand_position[2]
        start_x_angle = self.get_3d_x_angle(self.grab_start_value[0], 640, 69)
        end_x_angle = self.get_3d_x_angle(hand_center_position[0], 640, 69)
        position_x_diff = self.get_3d_x_diff(start_x_angle, end_x_angle, self.grab_start_value[2], hand_center_position[2])

        start_y_angle = self.get_3d_y_angle(self.grab_start_value[1], 640, 69)
        end_y_angle = self.get_3d_y_angle(hand_center_position[1], 640, 69)
        position_y_diff = self.get_3d_y_diff(start_y_angle, end_y_angle, hand_center_position[2], hand_center_position[2])

        position_z_diff = hand_center_position[2] - self.grab_start_value[2]

        #diff = [position_x_diff * scale_ratio, position_y_diff * scale_ratio, position_z_diff * scale_ratio]
        diff = [position_x_diff * scale_ratio, position_y_diff * scale_ratio, 0 * scale_ratio]
        return diff

    def rotation_factor_calculation(self, hand_center_position, scale_ratio = 0.15):
        if hand_center_position[2] == 0:
            hand_center_position[2] = self.last_hand_position[2]
        # the range of rotation control is -180 ~ 180
        start_angle = self.get_3d_x_angle(self.spread_start_value[0], 640, 69)
        end_angle = self.get_3d_x_angle(hand_center_position[0], 640, 69)
        position_diff = self.get_3d_x_diff(start_angle, end_angle, self.spread_start_value[2], hand_center_position[2])
        return position_diff * scale_ratio

    def scale_factor_calculation(self, scale_ratio = 1.0):
        diff = self.scale_start_value - self.index_thumb_distance[-1]
        # scaling down
        # The range of scaling is -5 ~ 5
        if diff < 0:
            scaling_factor = - (scale_ratio * (diff / 20.0))
        else:
            scaling_factor = - (scale_ratio * (diff / 20.0))
        return scaling_factor


    ####################################### hand motion classification #########################################

    def scale_manipulation(self, fps, finger_position_list):
        fps = int(fps)
        spread_hand_bool = self.handSpread(finger_position_list, direction='up')
        index_finger_grab = self.index_finger_grab(finger_position_list)
        index_thumb_distance = self.get_distance(finger_position_list[4], finger_position_list[8])
        grab_bool = self.handGrab(finger_position_list, 'up')
        self.put_info(grab_bool, 'hand_grab_bool')
        self.hand_up_state = self.hand_up(finger_position_list, direction='up')

        # Put the info into the detector deques
        self.put_info(index_thumb_distance, 'index_thumb_distance')
        self.put_info(spread_hand_bool, 'hand_spread_bool')

        if self.state != 'scaling':
            self.tolerance = fps
            # distance between index and thumb finger for 0.5 sec
            distances = list(itertools.islice(self.index_thumb_distance, len(self.index_thumb_distance)-fps, len(self.index_thumb_distance), 1))
            diff_list = list(distances - np.mean(distances))
            diff = distances - np.mean(distances)
            diff_var = np.var(diff_list)
            # Classify whether the scaling motion is occurred
            if len(diff) > 0 and diff_var < 10 and index_finger_grab and spread_hand_bool and self.hand_up_state and float(time.time()) - self.stopping_time > 1.5: # must be fixed
                print('scailing is started at', float(time.time()) - self.stopping_time)
                print('previous state is', self.state)
                self.state = 'scaling'
                self.last_state_changed_time = float(time.time())
                self.scale_start_value = distances[-1]
        elif self.state == 'scaling':
            distances = list(itertools.islice(self.index_thumb_distance, len(self.index_thumb_distance)-fps, len(self.index_thumb_distance), 1))
            diff_list = list(distances - np.mean(distances))
            diff = distances - np.mean(distances)
            diff_var = np.var(diff_list)
            stopping = False
            if diff_var < 100:
                print('var is low now')
                stopping = True
            # Classify whether the scaling motion is occurred
            if (index_finger_grab is False or spread_hand_bool is False or self.hand_up_state is False or stopping) and float(time.time()) - self.last_state_changed_time > 1.5:
                self.state = 'standard'
            # Check the tolerance of scaling motion miss
                if stopping:
                    print('stopping is occurred')
                    self.stopping_time = float(time.time())
            else:
                scaling_factor = self.scale_factor_calculation(scale_ratio=30.0)
                self.put_info(scaling_factor, 'scaling_factor')
        return spread_hand_bool, index_finger_grab

    def translation_manipulation(self, hand_center_position, finger_position_list, fps):
        fps = int(fps)
        if self.state != 'translating':
            self.tolerance = fps
            spread = False
            for i in range(fps):
                if self.hand_spread_bool[-1-i] == True:
                    spread = True
            grab = True
            for i in range(fps):
                if self.hand_grab_bool[-1-i] == False:
                    grab = False
            if grab and self.hand_up_state and float(time.time()) - self.stopping_time > 1.5 and spread is False:
                print('previous state is', self.state)
                self.state = 'translating'
                self.grab_start_value = hand_center_position
                self.last_state_changed_time = float(time.time())
        elif self.state == 'translating':
            translating_factor = self.translation_factor_calculation(hand_center_position, scale_ratio=0.5)
            self.put_info(translating_factor, 'translating_factor')
            if self.hand_grab_bool[-1] == False or self.hand_up_state == False:
                self.state = 'standard'
    
    def rotation_manipulation(self, hand_center_position, fps):
        fps = int(fps)
        if self.state != 'rotating':
            self.tolerance = fps
            spread = True
            for i in range(fps):
                if self.hand_spread_bool[-1-i] == False:
                    spread = False
            grab = False
            for i in range(fps):
                if self.hand_grab_bool[-1-i] == True:
                    grab = True
            if grab is False and spread and self.hand_up_state and float(time.time()) - self.stopping_time > 1.5:
                print('previous state is', self.state)
                self.state = 'rotating'
                self.spread_start_value = hand_center_position
                self.last_state_changed_time = float(time.time())
        elif self.state == 'rotating':
            rotation_factor = self.rotation_factor_calculation(hand_center_position, scale_ratio=0.15)
            self.put_info(rotation_factor, 'rotating_factor')
            """
            if self.hand_spread_bool[-1] == False or self.hand_grab_bool[-1] is True or self.hand_up_state is False:
                self.rotating_tolerance -= 1
            else:
                self.rotating_tolerance = min(self.rotating_tolerance + 1, fps)
            if self.rotating_tolerance <= 0:
                self.state = 'standard'
                self.rotating_tolerance = int(fps)
            else:
                self.last_state_changed_time = float(time.time())
            """
            if self.hand_spread_bool[-1] == False or self.hand_grab_bool[-1] is True or self.hand_up_state is False:
                self.state = 'standard'

    def horizontal_hand_flip_manipulation(self, fps, finger_position_list):
        flip_standby = self.handSpread(finger_position_list, direction='left') and self.hand_up(finger_position_list, direction = 'left')
        if flip_standby:
            self.horizontal_flip_time = float(time.time())
        self.put_info(flip_standby, 'horizontal_hand_flip_standby_bool')
        if self.state != 'horizontal_flip_standby':
            horizontal_standby = True
            for i in range(fps):
                if self.horizontal_hand_flip_standby_bool[-1-i] == False:
                    horizontal_standby = False
            if horizontal_standby and fps != 0:
                print('previous state is', self.state)
                self.state = 'horizontal_flip_standby'
        flip_time = float(time.time()) - self.horizontal_flip_time
        if flip_time < 1:
            right_flip = self.handSpread(finger_position_list, direction='right') and self.hand_up(finger_position_list, direction = 'right')
            if right_flip:
                return True
        if self.state == 'horizontal_flip_standby' and flip_time > 1:
            self.state = 'standard'
        return False

    def vertical_hand_flip_manipulation(self, fps, finger_position_list):
        flip_standby = self.handSpread(finger_position_list, direction='down') and self.hand_up(finger_position_list, direction = 'down')
        if flip_standby:
            self.vertical_flip_time = float(time.time())
        self.put_info(flip_standby, 'vertical_hand_flip_standby_bool')
        if self.state != 'vertical_flip_standby':
            vertical_standby = True
            for i in range(fps):
                if self.vertical_hand_flip_standby_bool[-1-i] == False:
                    vertical_standby = False
            if vertical_standby and fps != 0:
                print('previous state is', self.state)
                self.state = 'vertical_flip_standby'
                self.vertical_flip_time = float(time.time())
        flip_time = float(time.time()) - self.vertical_flip_time
        if flip_time < 1:
            up_flip = self.handSpread(finger_position_list, direction='up') and self.hand_up(finger_position_list, direction = 'up')
            if up_flip:
                return True
        if self.state == 'vertical_flip_standby' and flip_time > 1:
            self.state = 'standard'
        return False
                
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

    def get_3d_x_angle(self, position, img_w, camera_angle):
        center_w = int(img_w // 2)
        diff_x = position - center_w
        angle = camera_angle * (diff_x / center_w)
        return angle
    
    def get_3d_x_diff(self, angle_1, angle_2, depth_1, depth_2):
        x_1 = math.sin(math.radians(angle_1)) * depth_1
        x_2 = math.sin(math.radians(angle_2)) * depth_2
        return x_2 - x_1

    def get_3d_y_angle(self, position, img_h, camera_angle):
        center_h = int(img_h // 2)
        diff_h = position - center_h
        angle = camera_angle * (diff_h / center_h)
        return angle
    
    def get_3d_y_diff(self, angle_1, angle_2, depth_1, depth_2):
        y_1 = math.sin(math.radians(angle_1)) * depth_1
        y_2 = math.sin(math.radians(angle_2)) * depth_2
        return y_1 - y_2
    
    def mean(self, lists):
        sum_num = 0
        if len(lists):
            for i in range(len(lists)):
                sum_num += lists[i]
            return sum_num / len(lists)
        else:
            return 100

            
            
        
