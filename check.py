import argparse
import time
from pathlib import Path

import cv2
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
        self.parser.add_argument('--source', type=str,
                                 default='C:/Users/HZY/Pictures/Saved.avi',
                                 help='file/dir/URL/glob, 0 for webcam')
        self.parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640,
                                 help='inference size (pixels)')
        self.parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
        self.parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
        self.parser.add_argument('--max-det', type=int, default=200, help='maximum detections per image')
        self.parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        self.parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
        self.parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
        self.parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
        self.parser.add_argument('--augment', action='store_true', help='augmented inference')
        self.parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')

        self.opt = self.parser.parse_args()

    def run(self):
        self.detect(**vars(self.opt))

    def detect(self,
               weights='C:/Users/HZY/Desktop/tem/polygon_last.pt',  # model.pt path(s)names = ['1', '2', '3', '4', '5']
               source='data/images',  # file/dir/URL/glob, 0 for webcam
               imgsz=640,  # inference size (pixels)
               conf_thres=0.25,  # confidence threshold
               iou_thres=0.45,  # NMS IOU threshold
               max_det=1000,  # maximum detections per image
               device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
               save_crop=False,  # save cropped prediction boxes
               classes=None,  # filter by class: --class 0, or --class 0 2 3
               agnostic_nms=False,  # class-agnostic NMS
               augment=False,  # augmented inference
               half=False,  # use FP16 half-precision inference
               ):
        # webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        #     ('rtsp://', 'rtmp://', 'http://', 'https://'))

        device = select_device(device)
        # half &= device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        model = attempt_load(weights, map_location=device)  # load FP32 model
        stride = int(model.stride.max())  # model stride
        imgsz = check_img_size(imgsz, s=stride)  # check image size
        names = ['1', '2', '3', '4', '5']  # get class names
        # if half:
        #     model.half()  # to FP16
        # get_gpu_memory()

        # Polygon does not support second-stage classifier
        # classify = False
        # assert not classify, "polygon does not support second-stage classifier"

        # Set Dataloader
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        for path, img, im0s, vid_cap in dataset:
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0

            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            pred = model(img, augment=augment)[0]
            # Apply polygon NMS
            pred = polygon_non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                # if webcam:  # batch_size >= 1
                #     p, s, im0, frame = path[i], f'{i}: ', im0s[i].copy(), dataset.count
                # else:
                p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

                p = Path(p)  # to Path
                s += '%gx%g ' % img.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0, 1, 0, 1, 0]]  # normalization gain xyxyxyxy

                assert not save_crop, "polygon does not support save_crop"
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    """mark"""
                    det[:, :8] = polygon_scale_coords(img.shape[1:3], det[:, :8], im0.shape).round()

                    # Write results

                    for *xyxyxyxy, conf, cls in reversed(det):
                        cls = names[int(cls)]  # Write to file
                        print('666', int(cls))
                        xyxyxyxyn = (torch.tensor(xyxyxyxy).view(1, 8) / gn).view(-1).tolist()  # normalized xyxyxyxy
                        # xyxyxyxyn = (torch.tensor(xyxyxyxy).view(1, 8)).view(-1).tolist()  # normalized xyxyxyxy
                        xyxyxyxyn = [xyxyxyxyn[0], xyxyxyxyn[1], xyxyxyxyn[6], xyxyxyxyn[7], xyxyxyxyn[4], xyxyxyxyn[5],
                                     xyxyxyxyn[2], xyxyxyxyn[3]]
                        c1 = (int(xyxyxyxyn[0]), int(xyxyxyxyn[1]))
                        c2 = (int(xyxyxyxyn[6]), int(xyxyxyxyn[7]))
                        c3 = (int(xyxyxyxyn[4]), int(xyxyxyxyn[5]))
                        c4 = (int(xyxyxyxyn[2]), int(xyxyxyxyn[3]))
                        cv2.rectangle(im0, c1, c3, (255, 0, 0), thickness=4, lineType=cv2.LINE_AA)
                        cv2.putText(im0, cls, c1, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), thickness=2,
                                    lineType=cv2.LINE_AA)
                        print("xyxyxyxyn", xyxyxyxyn)
                        cv2.namedWindow("im0", cv2.WINDOW_NORMAL)
                        cv2.imshow("im0", im0)
                        cv2.waitKey(1)  # 1 millisecond


if __name__ == "__main__":
    x = mathdetect()
    x.run()

