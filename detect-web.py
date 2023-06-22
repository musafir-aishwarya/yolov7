# This file is altered from detect.py. 'source' argument is disabled.
# Instead, 'source' argument is provided in HTTP request by user
# There is two api
# /detect  : request should contain a image, response is bounding boxes
# /enlarge : request should contain a image, response is a super-resolutioned image

import argparse
from flask import Flask, request, send_file
from flask_cors import CORS
import os
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, set_logging, increment_path
from utils.torch_utils import select_device, load_classifier, TracedModel

# Import cropping feature and Latent Diffusion Models (LDM) for super-resolution
from utils.custom_features import Latent

app = Flask(__name__)
CORS(app)

parser = argparse.ArgumentParser()
parser.add_argument('--weights', nargs='+', type=str,
                    default='yolov7.pt', help='model.pt path(s)')
# file/folder, 0 for webcam
parser.add_argument('--source', type=str,
                    default='inference/images', help='source')
parser.add_argument('--img-size', type=int, default=640,
                    help='inference size (pixels)')
parser.add_argument('--conf-thres', type=float,
                    default=0.25, help='object confidence threshold')
parser.add_argument('--iou-thres', type=float,
                    default=0.45, help='IOU threshold for NMS')
parser.add_argument('--device', default='',
                    help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
parser.add_argument('--view-img', action='store_true',
                    help='display results')
parser.add_argument('--save-txt', action='store_true',
                    help='save results to *.txt')
parser.add_argument('--save-conf', action='store_true',
                    help='save confidences in --save-txt labels')
parser.add_argument('--nosave', action='store_true',
                    help='do not save images/videos')
parser.add_argument('--classes', nargs='+', type=int,
                    help='filter by class: --class 0, or --class 0 2 3')
parser.add_argument('--agnostic-nms', action='store_true',
                    help='class-agnostic NMS')
parser.add_argument('--augment', action='store_true',
                    help='augmented inference')
parser.add_argument('--update', action='store_true',
                    help='update all models')
parser.add_argument('--project', default='runs/detect',
                    help='save results to project/name')
parser.add_argument('--name', default='exp',
                    help='save results to project/name')
parser.add_argument('--exist-ok', action='store_true',
                    help='existing project/name ok, do not increment')
parser.add_argument('--no-trace', action='store_true',
                    help='don`t trace model')
parser.add_argument('--sr', action='store_true',
                    help='execute super resolution')
parser.add_argument('--sr-step', default=100, type=int)
opt = parser.parse_args()

source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
save_img = not opt.nosave and not source.endswith(
    '.txt')  # save inference images
webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
    ('rtsp://', 'rtmp://', 'http://', 'https://'))

# Directories
save_dir = Path(increment_path(Path(opt.project) / opt.name,
                exist_ok=opt.exist_ok))  # increment run
(save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True,
                                                      exist_ok=True)  # make dir

# Initialize latent SR
latent = Latent()

# Initialize
set_logging()
device = select_device(opt.device)
# device = torch.device("cuda:1")
print(device)
half = device.type != 'cpu'  # half precision only supported on CUDA

# Load model
model = attempt_load(weights, map_location=device)  # load FP32 model
stride = int(model.stride.max())  # model stride
imgsz = check_img_size(imgsz, s=stride)  # check img_size

if trace:
    model = TracedModel(model, device, opt.img_size)

if half:
    model.half()  # to FP16

# Second-stage classifier
classify = False
if classify:
    modelc = load_classifier(name='resnet101', n=2)  # initialize
    modelc.load_state_dict(torch.load(
        'weights/resnet101.pt', map_location=device)['model']).to(device).eval()

# Get names and colors
names = model.module.names if hasattr(model, 'module') else model.names
# ? Cityscapes has 8 classes, but I accidentally train 8 classes plus an extra redundant class.
# ? To avoid wrong labeling, I just use 8 classes here.
# ? Please remove this line if you are using the correct dataset classes.
names = ['person', 'car', 'truck', 'rider',
         'motorcycle', 'bicycle', 'bus', 'train']
colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

# Run inference
if device.type != 'cpu':
    model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(
        next(model.parameters())))  # run once
old_img_w = old_img_h = imgsz
old_img_b = 1

t0 = time.time()


@app.route("/detect", methods=['POST'])
def detect():
    file = request.files['file']
    source = str(os.path.join(save_dir, str(time.time()) + ".png"))
    file.save(source)

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    for _, img, im0s, _ in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        # if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
        #     old_img_b = img.shape[0]
        #     old_img_h = img.shape[2]
        #     old_img_w = img.shape[3]
        #     for i in range(3):
        #         model(img, augment=opt.augment)[0]

        # Inference
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(
            pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # pred = pred.detach()

        det = pred[0]
        # Rescale boxes from img_size to im0 size
        det[:, :4] = scale_coords(
            img.shape[2:], det[:, :4], im0s.shape).round()

        det = det.tolist()

        return [({"xyxy": dt[:4], "conf": dt[4], "class": dt[5]}) for dt in reversed(det)]


@app.route("/enlarge", methods=['POST'])
def superResolution():
    file = request.files['file']
    filepath = str(os.path.join(save_dir, str(time.time()) + ".png"))
    file.save(filepath)
    cropped_img = cv2.imread(filepath)
    with torch.no_grad():
        cropped_img_SR = latent.inference(cropped_img, opt.sr_step)
    # Path to save the cropped SR image
    sr_path = str(os.path.join(save_dir, str(time.time()) + ".png"))
    cv2.imwrite(sr_path, cropped_img_SR)
    # Save the cropped SR image
    return send_file(sr_path)


print('start serving')
app.run(port=30701, host='0.0.0.0')
