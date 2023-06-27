# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image
import json 
import requests
import cv2
import numpy as np
import sys
sys.path.append(r"D:/yaming_dataset/Yaming_AI/api")
from api.data_type import category


def Predict(img_path):
    image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    image = cv2.resize(image, dsize=(256,256),interpolation=cv2.INTER_LINEAR)

    image = image / 255.0
    image_arr = np.array(image)
    image_arr = np.expand_dims(image_arr,axis=0)

    data = json.dumps({"signature_name":"serving_default","instances":image_arr.tolist()})

    headers = {"content-type": "application/json"}
    json_response = requests.post('http://localhost:8501/v1/models/model:predict', data=data, headers=headers)
    prediction = json.loads(json_response.text)

    predict_arr = np.array(prediction['predictions'])
    result = np.argmax(predict_arr)

    print(category[result])

    return category[result]

#NOT USE
# class Tensor_Detect():
#     def __init__(self, model_name:str, img_path:str, model_path:str):
#         self.model_name = model_name
#         self.img_path = img_path
#         self.model_path = model_path

#     def Predict(self):
#         model = load_model(os.path.join(self.model_path,self.model_name))
#         img = image.load_img(self.img_path,target_size=(512,512))
#         img_tensor = image.img_to_array(img)
#         img_tensor = np.expand_dims(img_tensor,axis=0)
#         print(np.shape(img_tensor))
#         img_tensor /= 255.

#         predict = model(img_tensor)

#         return tf.argmax(predict[0],0).numpy()
    

#debug 
# if __name__ == "__main__":
#     detect_model = Tensor_Detect("best.h5","../test_image/1000/test.jpg","../tensorflow/tensor_model")
#     result = detect_model.Predict()
#     print(result)