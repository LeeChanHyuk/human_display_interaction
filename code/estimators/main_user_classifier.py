import cv2
import math

def main_user_drawing(frame, human_infos, main_user_index, flip_mode):
    height, width = frame.shape[:2]
    #main_user_index = random.randint(0, len(face_box_per_man)-1) # For random main user visualization
    for index, human_info in enumerate(human_infos):
        if index == main_user_index:
            #cv2.line(frame, (int(width/2), height-1), (int((human_info.face_box[0][0] + human_info.face_box[0][2])/2), int(human_info.face_box[0][3])), (0, 255, 0), 3)
            if flip_mode:
                cv2.rectangle(frame, (int(-1 * human_info.face_box[0][0]), int(human_info.face_box[0][1])), (-1 * int(human_info.face_box[0][2]), int(human_info.face_box[0][3])), (0,255,0), 2, cv2.LINE_AA)
                cv2.putText(frame, 'M', (int((-1 * human_info.face_box[0][0] + human_info.face_box[0][2])/2), int(-1 * human_info.face_box[0][1])), 1, 2, (0, 255, 0), 2)
            else:
                cv2.rectangle(frame, (int(human_info.face_box[0][0]), int(human_info.face_box[0][1])), (int(human_info.face_box[0][2]), int(human_info.face_box[0][3])), (0,255,0), 2, cv2.LINE_AA)
                cv2.putText(frame, 'M', (int((human_info.face_box[0][0] + human_info.face_box[0][2])/2), int(human_info.face_box[0][1])), 1, 2, (0, 255, 0), 2)
        else:
            #cv2.line(frame, (int(width/2), height-1), (int((human_info.face_box[0][0] + human_info.face_box[0][2])/2), int(human_info.face_box[0][3])), (255, 0, 0), 1)
            if flip_mode:
                cv2.rectangle(frame, (int(-1 * human_info.face_box[0][0]), int(human_info.face_box[0][1])), (-1 * int(human_info.face_box[0][2]), int(human_info.face_box[0][3])), (255,0,0), 2, cv2.LINE_AA)
            else:
                cv2.rectangle(frame, (int(human_info.face_box[0][0]), int(human_info.face_box[0][1])), (int(human_info.face_box[0][2]), int(human_info.face_box[0][3])), (255,0,0), 2, cv2.LINE_AA)
    return frame

def main_user_classification(face_center_coordinate, head_poses = None, use_head_pose = False, use_center_people = True):
    if head_poses is None:
        man_score = [8000] * len(face_center_coordinate) # the people must be in 4m range
        for index, human_info in enumerate(face_center_coordinate):
            man_score[index] -= (face_center_coordinate[index][2] * 1.5)
            if use_center_people:
                man_score[index] -= abs(face_center_coordinate[index][0] - 320)
        max_val = max(man_score)
        main_user_index = man_score.index(max_val)
        return main_user_index
    else:
        man_score = [8000] * len(head_poses) # the people must be in 4m range
        for index, human_info in enumerate(head_poses):
            man_score[index] -= (face_center_coordinate[index][2] * 1.5)
            if use_center_people:
                man_score[index] -= abs(face_center_coordinate[index][0] - 320)
            if use_head_pose and abs(head_poses[index][1]) > 30: # if yaw value of the man is over than 30 or -30, then we think that the man is not looking the display.
                man_score[index] = 0
        max_val = max(man_score)
        main_user_index = man_score.index(max_val)
        return main_user_index

def get_3d_distance(p1, p2):
    for i in range(3):
        p1[i] += 1
    length = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2)
    return length

def main_display_classification(human_position, display_positions, main_user_head_pose):
    angle_diff = []
    camera_horizontal_angle = 87 # RGB = 60
    camera_vertical_angle = 58 # RGB = 42
    for display_position in display_positions:
        i_width = 640
        eye_x = 320 - (i_width/2)
        detected_x_angle = (camera_horizontal_angle / 2) * (eye_x / (i_width/2))
        new_x = int(human_position[2]) * math.sin(math.radians(detected_x_angle))

        x_diff = new_x - display_position[0]
        z_diff = int(human_position[2]) - display_position[2]
        if z_diff < 10:
            z_diff = 600

        angle = math.atan(x_diff/z_diff) * 180 / math.pi
        abs_diff = abs(angle - main_user_head_pose[1])
        angle_diff.append(abs_diff)
    min_val = min(angle_diff)
    main_display_index = angle_diff.index(min_val)
    return main_display_index

def main_user_classification_filter(tolerance, previous_main_user_position, current_main_user_position, main_user_index, face_center_coordinates, fps):
    distance_threshold = 50
    distance = get_3d_distance(previous_main_user_position, current_main_user_position)
    if distance > distance_threshold and tolerance < 0:
        tolerance = int(fps)
    elif distance > distance_threshold:
        find_previous_main_user = False
        previous_main_user_index = 0
        for i in range(len(face_center_coordinates)):
            temp_distance = get_3d_distance(previous_main_user_position, face_center_coordinates[i])
            if temp_distance < distance_threshold:
                find_previous_main_user = True
                previous_main_user_index = i
                break
        if find_previous_main_user:
            main_user_index = previous_main_user_index
            tolerance -= 1
        else:
            tolerance = int(fps)
    else:
        tolerance = int(fps)

    return main_user_index, tolerance


def mmain_user_classification_filter(tolerance, previous_main_user_index, current_main_user_index, fps):
    main_user_index = previous_main_user_index
    if current_main_user_index != previous_main_user_index and tolerance < 0:
        main_user_index = current_main_user_index
    elif current_main_user_index != previous_main_user_index:
        tolerance -= 1
    elif current_main_user_index == previous_main_user_index:
        tolerance = int(fps / 2)
    return main_user_index, tolerance


