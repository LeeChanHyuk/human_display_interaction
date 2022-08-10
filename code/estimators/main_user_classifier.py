import cv2

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

def main_user_classification(draw_frame, human_infos, flip_mode = False):
    man_score = [4000] * len(human_infos)
    for index, human_info in enumerate(human_infos):
        man_score[index] -= human_info.center_eyes[-1][2]
        if abs(human_info.head_poses[-1][0]) > 30: # if yaw value of the man is over than 30 or -30, then we think that the man is not looking the display.
            man_score[index] = 0
    max_val = max(man_score)
    main_user_index = man_score.index(max_val)
    draw_frame = main_user_drawing(draw_frame, human_infos, main_user_index, flip_mode)
    return main_user_index, draw_frame