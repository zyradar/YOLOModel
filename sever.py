import cv2

import sys
from utils.general import *
from shapely.geometry import LineString
import numpy as np
import os
import torch
from pasta import augment
import sys
from pathlib import Path

from models.experimental import attempt_load
from utils.general import (check_img_size, non_max_suppression)
from utils.torch_utils import select_device, time_sync

__presentDir = os.path.join(os.getcwd(), os.path.dirname(__file__))
sys.path.insert(0, __presentDir)
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # 定义YOLOv5的根目录
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # 将YOLOv5的根目录添加到环境变量中（程序结束后删除）
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

weights = ROOT / 'C:/Users/HZY/Desktop/tem/polygon_last.pt'  # 权重文件地址   .pt文件
# data = ROOT / 'data/coco.yaml'  # 标签文件地址   .yaml文件

imgsz = 1184  # 输入图片的大小 默认640(pixels)
conf_thres = 0.25  # object置信度阈值 默认0.25  用在nms中
iou_thres = 0.45  # 做nms的iou阈值 默认0.45   用在nms中
max_det = 1000  # 每张图片最多的目标数量  用在nms中
device = '0'  # 设置代码执行的设备 cuda device, i.e. 0 or 0,1,2,3 or cpu
classes = None  # 在nms中是否是只保留某些特定的类 默认是None 就是所有类只要满足条件都可以保留 --class 0, or --class 0 2 3
agnostic_nms = False  # 进行nms是否也除去不同类别之间的框 默认False
augment = False  # 预测是否也要采用数据增强 TTA 默认False
visualize = False  # 特征图可视化 默认FALSE
half = False  # 是否使用半精度 Float16 推理 可以缩短推理时间 但是默认是False
dnn = False  # 使用OpenCV DNN进行ONNX推理

# 获取设备
device = select_device(device)
# 载入模型
model = attempt_load(weights, map_location=device)
# stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
stride = model.stride
imgsz = check_img_size(imgsz)  # 检查图片尺寸

names = ['0', '1', '2', '3', '4']


def detect(detimg):
    im0 = detimg
    im = detimg.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(im)
    t1 = time_sync()
    im = torch.from_numpy(im).to(device)
    im = im.half() if half else im.float()  # uint8 to fp16/32
    im /= 255  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim
    pred = model(im, augment=augment)[0]
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
    detections = []
    robot_point = []
    for i, det in enumerate(pred):  # per image 每张图片
        if len(det):
            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
            for *xyxy, conf, cls in reversed(det):
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4))).view(-1).tolist()
                xywh = [round(x) for x in xywh]
                xywh = [xywh[0] - xywh[2] // 2, xywh[1] - xywh[3] // 2, xywh[2],
                        xywh[3]]  # 检测到目标位置，格式：（left，top，w，h）
                xyxy = (torch.tensor(xyxy).view(1, 4)).view(-1).tolist()
                # xywh = [xyxy[0], xyxy[1], xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]]
                cls = names[int(cls)]
                conf = float(conf)
                detections.append({'class': cls, 'conf': conf, 'position': xyxy})
                # if cls:
                #     detections.append({'class': cls, 'conf': conf, 'position': xywh})
                if cls == names[0]:
                    cv2.rectangle(detimg, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (255, 0, 0),
                                  3)
                elif cls == names[1]:
                    cv2.rectangle(detimg, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 0, 255),
                                  3)
                # elif cls == names[12]:
                #     red.append(xyxy)
                # cv2.rectangle(img, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 0, 255), 3)
                # if cls == names[5]:
                #     if xyxy[0] >= 0 and xyxy[1] >= 0:
                #         if xyxy[2] >= 0 and xyxy[3] >= 0:
                #             carpai = [x for x in xyxy]
                #     # carpai.append(xyxy)
                # robot_point.append(xyxy.insert(0, cls))
                # robot[0] = cls
                robot = [cls, xywh[0] + xywh[2] * 0.5, xywh[1] + xywh[3] * 2]
                # robot[2] = (xyxy[1]+xyxy[3])/2*0.8
                # xyxy.insert(0, cls)
                robot_point.append(robot)
                # print(robot)
    return detimg, robot_point
