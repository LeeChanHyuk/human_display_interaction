# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     path/                           # directory
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s.xml                # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
"""

import argparse
import os
import platform
import sys
from pathlib import Path
from traceback import FrameSummary
from tracemalloc import start

import torch
import torch.backends.cudnn as cudnn
import numpy as np
import time

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from estimators.face_detection_module.yolov5.models.common import DetectMultiBackend
from estimators.face_detection_module.yolov5.utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from estimators.face_detection_module.yolov5.utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from estimators.face_detection_module.yolov5.utils.plots import Annotator, colors, save_one_box
from estimators.face_detection_module.yolov5.utils.torch_utils import select_device, smart_inference_mode
from user_information.human import HumanInfo
from scipy.stats import mode


@smart_inference_mode()
def run(
        weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
):
    source = str(source)
    webcam = source.isnumeric() or source.endswith('.txt')

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)  # batch_size

    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    dt = (Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh

                    if view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))

            # Stream results
            im0 = annotator.result()
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond


        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)

def human_info_deep_copy(human_infos, human_info):
	reference_human_info = human_infos[-1]
	human_info.center_eyes = reference_human_info.center_eyes
	human_info.center_mouths =reference_human_info.center_mouths
	human_info.left_shoulders =reference_human_info.left_shoulders
	human_info.right_shoulders =reference_human_info.right_shoulders
	human_info.center_stomachs =reference_human_info.center_stomachs
	human_info.face_box = reference_human_info.face_box
	human_info.left_eye_box = reference_human_info.left_eye_box
	human_info.right_eye_box = reference_human_info.right_eye_box

	human_info.head_poses = reference_human_info.head_poses
	human_info.body_poses =reference_human_info.body_poses
	human_info.eye_poses =reference_human_info.eye_poses
	human_info.left_eye_landmark =reference_human_info.left_eye_landmark
	human_info.right_eye_landmark =reference_human_info.right_eye_landmark
	human_info.left_eye_gaze =reference_human_info.left_eye_gaze
	human_info.right_eye_gaze =reference_human_info.right_eye_gaze
	human_info.calib_center_eyes =reference_human_info.calib_center_eyes
	human_info.human_state = reference_human_info.human_state # Action recognition result
	return human_info

def yolo_initialization(frame_shape, weights, data, depth_face_tracker):
    device = select_device('0')
    model = DetectMultiBackend(weights, device=device, dnn=False, data=data, fp16=False)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size((frame_shape[1], frame_shape[1]), s=stride)  # check image size

    # Run inference
    if depth_face_tracker:
        model.warmup(imgsz=(1 if pt else 1, 4, *imgsz))  # warmup
    else:
        model.warmup(imgsz=(1 if pt else 1, 3, *imgsz))  # warmup
    dt = (Profile(), Profile(), Profile())
    return model, dt, device


def yolo_face_detection(im, depth, dt, device, model, draw_frame, view_img, frame_shape, human_infos = None, depth_face_tracker = False, model_dict = None, model_name = None):
    if depth_face_tracker:
        depth = cv2.inRange(depth, 300, 2000)
        depth = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        #cv2.imshow('depth_image', depth)
        #cv2.waitKey(1)
        depth = np.expand_dims(depth, axis=2)
        im = np.concatenate([im, depth], axis=2)
    start_time = time.time()
    im = cv2.resize(im, (640, 640))
    depth = cv2.resize(depth, (640, 640))
    height, width, channel = im.shape
    gn = torch.tensor(im.shape)[[1, 0, 1, 0]]  # normalization gain whwh
    im = im[None, :]
    im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
    im = np.ascontiguousarray(im)  # contiguous
    # human_info objects
    if not human_infos:
        human_infos = []
    avg_time = 0
    with dt[0]:
        im = torch.from_numpy(im).to(device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
    # Inference
    with dt[1]:
        pred = model(im, augment=False, visualize=False)

    # NMS
    with dt[2]:
        pred = non_max_suppression(pred, 0.25, 0.45, None, False, 1000)

    # Second-stage classifier (optional)
    # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)
    # Process predictions
    det_result_num = 0
    for i, det in enumerate(pred):  # per image
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], (height, width, channel)).round()

            # Write results
            for *xyxy, conf, cls in reversed(det):
                if conf.item() < 0.5:
                    continue
                if model_dict:
                    model_dict[model_name] += 1
                det_result_num += 1
                # Check new user
                if det_result_num > len(human_infos):
                    human_info = HumanInfo()
                    if len(human_infos)>0:
                        human_info = human_info_deep_copy(human_infos, human_info)
                else:
                    human_info = human_infos[det_result_num-1]
                # Convert the x, y, w, h from normalized coordinate 
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                x, y, w, h = xywh[0] * width, xywh[1] * height, xywh[2] * width, xywh[3] * height
                x1, y1= int(x-(w/2)), int(y-(h/2))
                x2, y2 = int(x1 + w -1), int(y1 + h - 1)

				# Put the info into the human_info object
                human_info.face_box = np.array([x1, y1, x2, y2]) # face box is not used for action recognition. Thus, face_box is not list.
                human_info.face_box = np.expand_dims(human_info.face_box, axis=0)
                human_info.face_detection_confidence = round(conf.item(), 3)

                center_eyes_x = int((x1 + x2) / 2)
                center_eyes_y = int((y1 + y2) / 2)
                if depth_face_tracker:
                    center_eyes_z = depth[min(int(center_eyes_y), height-1), min(int(center_eyes_x), width-1)][0]
                else:
                    center_eyes_z = depth[min(int(center_eyes_y), height-1), min(int(center_eyes_x), width-1)]
                human_info._put_data([center_eyes_x, center_eyes_y, center_eyes_z], 'center_eyes')
                if det_result_num >= len(human_infos):
                    human_infos.append(human_info)
        # Stream results
    if False:
        cv2.imshow("Detection result", draw_frame)
        cv2.waitKey(1)  # 1 millisecond
    #print(1/(time.time() - start_time))
    #if time_point5 > 0:
    #    print(1/time_point5)
    if det_result_num > 0:
        return human_infos, det_result_num
    else:
        return human_infos, 0

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'weights/brightness_augmentation_best.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=0, help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
