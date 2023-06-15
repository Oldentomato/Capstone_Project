from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import tensorflow as tf
import os 

class Tensor_Detect():
    def __init__(self, model_name:str, img_path:str, model_path:str):
        self.model_name = model_name
        self.img_path = img_path
        self.model_path = model_path

    def Predict(self):
        model = load_model(os.path.join(self.model_path,self.model_name))
        img = image.load_img(self.img_path,target_size=(512,512))
        img_tensor = image.img_to_array(img)
        img_tensor = np.expand_dims(img_tensor,axis=0)
        img_tensor /= 255.

        predict = model(img_tensor)

        return tf.argmax(predict[0],0).numpy()