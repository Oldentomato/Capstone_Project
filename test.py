import cv2
import torch 
from predict import Detect_Image
from data_type import category

detect_module = Detect_Image(model_name = "best_model",
                             img_path = "test_image/",
                             img_name = "img_8.jpg", 
                             weight_path = "model/best.pt")

print(category[detect_module.detect()])