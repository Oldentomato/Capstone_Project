import json 
import requests
import cv2
import numpy as np
import sys
sys.path.append(r"D:/yaming_dataset/Yaming_AI/api")
from api.data_type import category

#검증할때 배경작업 해야함
def TestModel():
    image = cv2.imread("test_image/1000/test4.jpg", cv2.IMREAD_UNCHANGED)
    image = cv2.resize(image, dsize=(256,256),interpolation=cv2.INTER_LINEAR)

    image = image / 255.0
    image_arr = np.array(image)
    image_arr = np.expand_dims(image_arr,axis=0)

    data = json.dumps({"signature_name":"serving_default","instances":image_arr.tolist()})

    headers = {"content-type": "application/json"}
    json_response = requests.post('http://localhost:8501/v1/models/model:predict', data=data, headers=headers)
    prediction = json.loads(json_response.text)

    predict_arr = np.array(prediction['predictions'])
    result = np.argmax(predict_arr[0])

    print(category[result])



if __name__ == "__main__":
    TestModel()

