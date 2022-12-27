import math

display_positions = [[0, 0, 0], [-640, 0, 0], [640, 0, 0]]
for display_position in display_positions:
    human_position = [760, 0, 600]
    camera_horizontal_angle = 87 # RGB = 60

    i_width = 640
    eye_x = human_position[0] - (i_width/2)
    detected_x_angle = (camera_horizontal_angle / 2) * (eye_x / (i_width/2))
    new_x = int(human_position[2]) * math.sin(math.radians(detected_x_angle))
    print(new_x, end=' ')

    x_diff = new_x - display_position[0]
    z_diff = int(human_position[2]) - display_position[2]
    if z_diff < 10:
        z_diff = 600

    angle = math.atan(x_diff/z_diff) * 180 / math.pi
    print(angle)
