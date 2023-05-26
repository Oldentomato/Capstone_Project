import cv2
import torch 
from api.predict import Detect_Image
from api.data_type import category

detect_module = Detect_Image(model_name = "best_model",
                             img_path = "test_image/",
                             img_name = "img_9.jpeg", 
                             weight_path = "model/best_2.pt")

print(category[detect_module.detect()])