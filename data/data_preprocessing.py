from data_type import category,code
import os 
import xmltodict
from bisect import bisect_left
from sys import stdout
import shutil


def Parsing_XML():
    xml_path = 'D:\\yaming_dataset\\[라벨]음식분류_TRAIN\\xml'
    txt_path = 'D:\\yaming_dataset\\Yaming_AI\\yolov7\\dataset\\labels'
    dir_list = os.listdir(xml_path+"/")
    data_max_count = 499
    passed_arr = []
    
    for dir in dir_list:
        file_list = os.listdir(xml_path+"/"+dir+"/")
        for index,file in enumerate(file_list):
            if index > data_max_count:
                break
            target = xmltodict.parse(open(xml_path+"/"+dir+"/"+file ,encoding="UTF-8").read())

            #몇몇 파일에 이름이 깨지는 경우가 있어서 파일명으로 카테고리 지정으로 변경
            class_index = bisect_left(code,file.split("_")[2])
            #간혼 size 가 없는 xml 이 있을경우 예외처리
            try:
                Image_Height = float(target['annotation']['size']['height'])
                Image_Width  = float(target['annotation']['size']['width'])
            except:
                passed_arr.append(file)
                continue
            with(open(txt_path + "/" + file[:-4]+".txt",mode="w")) as label_file:
                for obj in target['annotation']['object']:
                    
                    # class의 index 휙득(이진탐색으로 검색)
                    if obj['name'] == "dish":
                        # class_index = 401
                        continue
                        
                    # min, max좌표 얻기
                    x_min = float(obj['bndbox']['xmin']) 
                    y_min = float(obj['bndbox']['ymin'])
                    x_max = float(obj['bndbox']['xmax']) 
                    y_max = float(obj['bndbox']['ymax'])

                    xc = (x_min + x_max) / 2.
                    yc = (y_max + y_min) / 2.

                    x = xc / Image_Width
                    y = yc / Image_Height

                    w = (x_max - x_min) / Image_Width
                    h = (y_max - y_min) / Image_Height
            
                    label_file.write(
                        f"{class_index} {x} {y} {w} {h}\n"
                    )

                stdout.write("\r===== In  "+dir+"  DIR : "+str(index+1)+" progressed=====")
                stdout.flush()

    with(open("./Passed_arr.txt",mode="w")) as file:
        for i in passed_arr:
            file.write(
                f"{i}\n"
            )

#카테고리별로 500개씩 파일을 옮김
def ImageFileMove():
    img_path = "D:\\yaming_dataset\\음식 이미지 및 영양정보 텍스트\\Training/"
    dir_path = "D:\\yaming_dataset\\dataset\\train\\images/"
    file_list = os.listdir(img_path)
    pass_count = 0


    for file in file_list:
        imgs = os.listdir(img_path+file+"/")
        for count,img in enumerate(imgs):
            if count > 499:
                print("total: "+str(pass_count)+"passed")
                break
            
            shutil.move(img_path+file+"/"+img, dir_path)
            stdout.write("\r===== "+ str(count+1) +" : "+str(500)+" progressed=====")
            stdout.flush()



def RemovePassImage():
    pass_arr = []
    img_path = "D:\\yaming_dataset\\dataset\\train\\images/"
    file_list = os.listdir(img_path)
    with(open("./Passed_arr.txt",mode="r")) as file:#통과된 파일리스트 txt 불러오고 배열에 담기
        pass_arr = list(map(lambda s: s.strip(), file))

    for file in pass_arr:
        result = file in pass_arr #.jpg과 숫자 두개 더 지우고 .xml로 바꾼 뒤 검색
        if result == True:
            #운좋게 한번에 잘 실행됬지만 remove는 휴지통이동이 아니라 아예 삭제이기때문에
            #다음에는 파일이동으로 바꿀것
            try:
                os.remove(img_path+file[:-4]+".jpg")
            except:
                os.remove(img_path+file[:-4]+".jpeg")

#txt 파일내 카테고리번호가 이상치가 있는 것 같아서 파일들을 순회하면 카테고리코드 확인
def Debuging():
    txt_path = 'D:\\yaming_dataset\\Yaming_AI\\yolov7\\dataset\\labels/'
    file_list = os.listdir(txt_path)
    for count,file in enumerate(file_list):
        with(open(txt_path+file,mode="r")) as text:
            line = text.readline()
            line = line.split()
            if (int(line[0]) > 399) or (int(line[0]) < 0):
                print("trouble detected in "+file+" value: "+line[0])
        stdout.write("\r===== "+ str(count+1) +" : "+str(len(file_list))+" progressed=====")
        stdout.flush()
    
#이상치가 확인이되어 xml파일내에 정보가 잘못되었는지 확인해봄   
#결과로 이름이 깨져있어서 xml파일명에 있는 음식코드를 통해 인덱스조정으로 수정함
def test():
    xml_path = 'D:\\yaming_dataset\\[라벨]음식분류_TRAIN\\xml/01014007/01_014_01014007_160311628224367_0.xml'

    target = xmltodict.parse(open(xml_path ,encoding="UTF-8").read())
    print(target['annotation']['object'][1]['name'])

#500개의 파일이 너무많아 카테고리별로 100개씩 줄이는 코드 (20만개에서 4만개로 줄임)
def Data_Reduction():
    txt_path = 'D:\\yaming_dataset\\Yaming_AI\\yolov7\\dataset\\labels/'
    img_path = 'D:\\yaming_dataset\\Yaming_AI\\yolov7\\dataset\\images/'

    temp_txt_path = "D:\\yaming_dataset\\dataset\\temp\\labels/"
    temp_img_path = "D:\\yaming_dataset\\dataset\\temp\\images/"

    count = 0

    txt_list = os.listdir(temp_txt_path)
    img_list = os.listdir(temp_img_path)

    for code_str in code:
        for file in txt_list:
            if count > 99:
                break
            if file.split("_")[2] == code_str:
                shutil.move(temp_txt_path+file, txt_path)
                count += 1
                stdout.write("\r===== "+code_str+" : "+str(count)+" progressed=====")
                stdout.flush()

        count = 0

    print("text_file : done")
    for code_str in code:
        for file in img_list:
            if count > 99:
                break
            if file.split("_")[2] == code_str:
                shutil.move(temp_img_path+file, img_path)
                count += 1
                stdout.write("\r===== "+code_str+" : "+str(count)+" progressed=====")
                stdout.flush()

        count = 0


if __name__ == "__main__":
    # Parsing_XML()
    # ImageFileMove()
    # RemovePassImage()
    # Debuging()
    Data_Reduction()
