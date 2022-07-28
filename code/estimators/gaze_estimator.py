from estimators.gaze_estimation_module.gaze_estimation import estimate_gaze_from_face_image

def gaze_estimation(frame_copy, frame, human_info, visualization):
    frame, eyes = estimate_gaze_from_face_image(frame_copy, frame, human_info, visualization)
    left_eye, right_eye = eyes
    left_gaze = left_eye.gaze.copy()
    left_gaze[1] = -left_gaze[1]
    right_gaze = right_eye.gaze.copy()
    gazes = [left_gaze, right_gaze]

    human_info._put_data([gazes[0][0], gazes[0][1], gazes[1][0], gazes[1][1]], 'eye_poses')
    human_info.left_eye_landmark = left_eye
    human_info.right_eye_landmark = right_eye
    human_info.left_eye_gaze = left_gaze
    human_info.right_eye_gaze = right_gaze
    return frame