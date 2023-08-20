import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box, \
    polygon_non_max_suppression, polygon_scale_coords
from utils.plots import colors, plot_one_box, polygon_plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

import subprocess as sp
import os


class mathdetect():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        # self.weights = 'C:/Users/HZY/Desktop/tem/polygon_last.pt',
        self.augment = False
        self.max_det = 1000
        self.img = cv2.imread("E:/RobotCenter/dataSet/3.10image/moveroi/1.jpg")
        self.classes = None
        self.agnostic_nms = False
        self.half = False
        self.path = 'C:/Users/HZY/Pictures/Saved.avi'
        self.device = select_device('0')
        self.model = attempt_load('C:/Users/HZY/Desktop/exp445/polygon_last.pt', map_location=self.device)  # load FP32 model
        # self.parser.add_argument('--source', type=str,
        #                     default='C:/Users/HZY/Pictures/Saved.avi',
        #                     help='file/dir/URL/glob, 0 for webcam')
        self.imgsz = 640
        self.conf_thres = 0.25
        self.iou_thres=0.45
        self.save_crop=False

        self.opt = self.parser.parse_args()

    def run(self):
        cap = cv2.VideoCapture(self.path)
        while True:
            ret, self.img = cap.read()
            self.detect(self.img)

    def detect(self, img):
        stride = int(self.model.stride.max())  # model stride
        self.imgsz = check_img_size(self.imgsz, s=stride)  # check image size
        print("check over!!")
        names = ['1', '2', '3', '4', '5']  # get class names
        img = cv2.resize(img, (640, 640))
        im0 = img
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        pred = self.model(img, augment=self.augment)[0]
        pred = polygon_non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det)
        for i, det in enumerate(pred):  # detections per image
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0, 1, 0, 1, 0]]  # normalization gain xyxyxyxy
            if len(det):
                det[:, :8] = polygon_scale_coords(img.shape[1:3], det[:, :8], im0.shape).round()
                for *xyxyxyxy, conf, cls in reversed(det):
                    cls = names[int(cls)]  # Write to file
                    xyxyxyxyn = (torch.tensor(xyxyxyxy).view(1, 8)).view(-1).tolist()  # normalized xyxyxyxy
                    xyxyxyxyn = [xyxyxyxyn[0], xyxyxyxyn[1], xyxyxyxyn[6], xyxyxyxyn[7], xyxyxyxyn[4], xyxyxyxyn[5],
                                 xyxyxyxyn[2], xyxyxyxyn[3]]
                    c1 = (int(xyxyxyxyn[0]), int(xyxyxyxyn[1]))
                    c2 = (int(xyxyxyxyn[6]), int(xyxyxyxyn[7]))
                    c3 = (int(xyxyxyxyn[4]), int(xyxyxyxyn[5]))
                    c4 = (int(xyxyxyxyn[2]), int(xyxyxyxyn[3]))
                    cv2.rectangle(im0, c1, c3, (255, 0, 0), thickness=1, lineType=cv2.LINE_AA)
                    # cv2.line(im0, c1, c3, (255, 255, 255), 3)
                    # cv2.line(im0, c2, (0, 0), (255, 255, 255), 3)
                    cv2.putText(im0, cls, c1, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), thickness=2,
                                lineType=cv2.LINE_AA)
                    cv2.imwrite("C:/Users/HZY/Desktop/3.12show/2.jpg", im0)
                    print("xyxyxyxyn", xyxyxyxyn)
                    cv2.namedWindow("im0", cv2.WINDOW_NORMAL)
                    cv2.imshow("im0", im0)
                    cv2.waitKey(0)  # 1 millisecond


if __name__ == "__main__":
    x = mathdetect()
    x.run()

