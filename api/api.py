import sys
sys.path.append(r"/app")
from fastapi import FastAPI, UploadFile, status
from api.database import send_query
import os
import uuid
from pydantic import BaseModel
from api.predict import Detect_Image
from api.data_type import category


app = FastAPI()


class User(BaseModel):
    id: int
    name: str
    nickname: str
    user_id: str
    user_password: str
    user_email: str
    user_phone: str

class Food(BaseModel):
    foodname: str


@app.get("/test")
async def test():
    result = send_query("select ifnull(max(name),0) name from user where name='woosung'")
    return result[0][0]




@app.post("/imagetest/")
async def imagetest(file: UploadFile, userid: int):
    UPLOAD_DIR = "test_image"
    result = send_query("select ifnull(max(name),0) name from user where usernum='"+str(userid)+"'")
    username = result[0][0] #이중 튜플로 되어있어서 풀어줌

    if username != '0': #유저 조회 중 있을 경우
        if str(userid) not in os.listdir(UPLOAD_DIR): #이미지 디렉토리에 유저가 없을 경우
            os.mkdir(os.path.join(UPLOAD_DIR,str(userid)))

        content = await file.read()
        filename = f"{str(uuid.uuid4())}.jpg" #uuid로 유니크한 파일명으로 변경
        with open(os.path.join(UPLOAD_DIR,str(userid),filename),"wb") as fp:
            fp.write(content) #서버 로컬 스토리지에 이미지 저장


        detect_module = Detect_Image(model_name = "best_2",
                             img_path = os.path.join(UPLOAD_DIR,str(userid),filename),
                             weight_path = "/data/model/best_2.pt")


        model_result = detect_module.detect()
        return {"filename": filename,"success": True, "category": category[model_result], "food_code":model_result}

    else: #유저가 없을 경우
        return {"error": "not checked user","success": False}


@app.post("/adduser", status_code=status.HTTP_200_OK)
async def adduser(user: User):
    send_query(f"INSERT INTO user VALUES({user.id},'{user.user_id}','{user.user_password}','{user.name}','{user.nickname}','{user.user_email}','{user.user_phone}');")
    return {"success":True}




@app.post("/searchfood")
async def searchfood(food: Food):
    result = send_query(f"SELECT * FROM FoodData WHERE 음식이름='{food.foodname}'")
    return {"success":True, "result":result}