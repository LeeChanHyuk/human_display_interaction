import cv2
import pyrealsense2 as rs
import numpy as np

def realsense_initialization():
    align_to = rs.stream.color
    align = rs.align(align_to)
    pipeline = rs.pipeline()
    # Configure streams
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 60)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)

    # Start streaming
    pipeline.start(config)
    return pipeline, align
def get_input(pipeline=None, align=None, rgb_cap=None, depth_cap=None, video_path=None):
    # Get input
    if not video_path:
        frames = pipeline.wait_for_frames()
        align_frames = align.process(frames)
        frame = align_frames.get_color_frame()
        depth = align_frames.get_depth_frame()
        depth = np.array(depth.get_data())
        frame = np.array(frame.get_data())
    else:
        ret, frame = rgb_cap.read()
        ret, depth = depth_cap.read()
    return frame, depth
#pipeline, align = realsense_initialization()
#frame, depth = get_input(pipeline, align)
import usb
busses = usb.busses()
for bus in busses:
    devices = bus.devices
    for dev in devices:
        print("Device:", dev.filename)
        print("  idVendor: %d (0x%04x)" % (dev.idVendor, dev.idVendor))
        print("  idProduct: %d (0x%04x)" % (dev.idProduct, dev.idProduct))
cap = cv2.VideoCapture(0)
while 1:
    ret, frame = cap.read()
    print(ret)
    cv2.imshow('frame', frame)
    cv2.waitKey(1)