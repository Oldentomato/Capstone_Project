import torch
import cv2
import numpy as np
import sys

sys.path.append(r"/app")
from models.experimental import attempt_load
from utils.general import non_max_suppression,scale_coords
from utils.torch_utils import time_synchronized
from modules.get_img import letterbox


class Detect_Image:
    def __init__(self, model_name:str, img_path:str, weight_path:str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.img_path = img_path
        self.weight_path = weight_path



    def detect(self):
        model = attempt_load(self.weight_path, map_location = self.device)
        print("model loaded")
        image = cv2.imread(self.img_path)
        stride = int(model.stride.max())
        # Padded resize
        img = letterbox(image, 512, stride=stride)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        result_image = torch.from_numpy(img).to(self.device)
        result_image = result_image.float()
        result_image /= 255.0
        if result_image.ndimension() == 3:
            result_image = result_image.unsqueeze(0)

        print("image loaded")

        t1 = time_synchronized()
        with torch.no_grad():
            pred = model(result_image)[0]
        t2 = time_synchronized()

        pred = non_max_suppression(pred, 0.01, 0.45) # conf_thres, iou_thres
        t3 = time_synchronized()

        for i, det in enumerate(pred): #detections per image
            if len(det):
                result_cls = []
                result_conf = []

                for *xyxy,conf,cls in reversed(det):
                    result_cls.append(int(cls.item()))
                    result_conf.append(conf.item())

                max_conf_index = result_conf.index(max(result_conf))
                return result_cls[max_conf_index]
            else:
                return None




