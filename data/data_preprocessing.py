import os 
import xmltodict
from bisect import bisect_left
from sys import stdout
import shutil
from use_list import food_list


def Search_Img(find_file) -> int:
    img_path = "D:\\yaming_dataset\\Yaming_AI\\yolov7\\dataset\\images"
    img_path2 = "D:\\yaming_dataset\\dataset\\temp\\images"
    file_list = os.listdir(img_path+"/")
    file_list2 = os.listdir(img_path2+"/")
    if (find_file+".jpg" in file_list) or (find_file+".jpeg" in file_list) or (find_file+".JPG" in file_list) or (find_file+".JPEG" in file_list):
        return 0
    elif (find_file+".jpg" in file_list2) or (find_file+".jpeg" in file_list2) or (find_file+".JPG" in file_list2) or (find_file+".JPEG" in file_list2):
        return 1
    else:
        return 2


def Test():
    # print(Search_Img("01_012_01012003_160553522146917_0"))
    shutil.copy("D:\\yaming_dataset\\dataset\\temp\\images/01_012_01012003_160553522146917_0.jpg", "D:\\yaming_dataset\\dataset_2\\image/01_012_01012003_160553522146917_0.jpg")



def Parsing_XML():
    xml_path = 'D:\\yaming_dataset\\[라벨]음식분류_TRAIN\\xml/'
    img_path = 'D:\\yaming_dataset\\음식 이미지 및 영양정보 텍스트\\Training/'
    
    txt_dir_path = 'D:\\yaming_dataset\\dataset_2\\label'
    img_dir_path = 'D:\\yaming_dataset\\dataset_2\\image'

    dir_list = os.listdir(xml_path+"/")
    data_max_count = 499
    
    for dir in dir_list:
        file_list = os.listdir(xml_path+"/"+dir+"/")
        for index,file in enumerate(file_list):
            #특정 음식코드이거나 500개가 넘어간다면
            if (index > data_max_count) or (dir in food_list):
                break
            target = xmltodict.parse(open(xml_path+"/"+dir+"/"+file ,encoding="UTF-8").read())

            #몇몇 파일에 이름이 깨지는 경우가 있어서 파일명으로 카테고리 지정으로 변경
            # class의 index 휙득(이진탐색으로 검색)
            class_index = bisect_left(food_list,file.split("_")[2])
            #간혼 size 가 없는 xml 이 있을경우 예외처리
            try:
                Image_Height = float(target['annotation']['size']['height'])
                Image_Width  = float(target['annotation']['size']['width'])
            except:
                # passed_arr.append(file)
                continue
            with(open(txt_dir_path + "/" + file[:-4]+".txt",mode="w")) as label_file:
                for obj in target['annotation']['object']:
                    #접시 정보는 패스
                    if obj['name'] == "dish":
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
                #이미지 이동(jpg or jpeg)
                # 이미지를 이미 옮긴 이력이 있어서 파일 검색 후 처리
                search_result = Search_Img(file[:-4])

                if search_result == 0:
                    try:
                        shutil.copy("D:\\yaming_dataset\\Yaming_AI\\yolov7\\dataset\\images/"+file[:-4]+".jpg", img_dir_path+"/"+file[:-4]+".jpg")
                    except:
                        shutil.copy("D:\\yaming_dataset\\Yaming_AI\\yolov7\\dataset\\images/"+file[:-4]+".jpeg", img_dir_path+"/"+file[:-4]+".jpeg")
                elif search_result == 1:
                    try:
                        shutil.copy("D:\\yaming_dataset\\dataset\\temp\\images/"+file[:-4]+".jpg", img_dir_path+"/"+file[:-4]+".jpg")
                    except:
                        shutil.copy("D:\\yaming_dataset\\dataset\\temp\\images/"+file[:-4]+".jpeg", img_dir_path+"/"+file[:-4]+".jpeg")
                else:
                    try:
                        shutil.copy(img_path+dir+"/"+file[:-4]+".jpg", img_dir_path+"/"+file[:-4]+".jpg")
                    except:
                        shutil.copy(img_path+dir+"/"+file[:-4]+".jpeg", img_dir_path+"/"+file[:-4]+".jpeg")


                stdout.write("\r===== In  "+dir+"  DIR : "+str(index+1)+" progressed=====")
                stdout.flush()


if __name__ == "__main__":
    Parsing_XML()
    # Test()

