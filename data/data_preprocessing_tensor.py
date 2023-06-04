import os 
import shutil 
from sys import stdout
from rembg import remove
from PIL import Image


def Move2Dir(): #하나로 뭉쳐진 파일들을 폴더별로 분리(필요없어짐)
    IMG_DATA_DIR = "D:\\yaming_dataset\\Yaming_AI\\yolov7\\dataset\\images/"

    list_img = os.listdir(IMG_DATA_DIR)


    namelist_img = list(map(lambda x: x.split('_')[2], list_img))

    namelist_img = list(set(namelist_img))

    for i in namelist_img: #i = 숫자코드
        os.mkdir(IMG_DATA_DIR+i)
        for count,j in enumerate(list_img): #j = 파일명
            if j.split("_")[2] == i:
                shutil.move(IMG_DATA_DIR+j,IMG_DATA_DIR+i+"/"+j)
        
            stdout.write("\r===== In  "+i+"  COUNT : "+str(count+1)+" progressed=====")
            stdout.flush()

def RemoveImgBG():
    IMG_DATA_DIR = "D:\\yaming_dataset\\Yaming_AI\\yolov7\\dataset\\images/"
    SAVE_IMG_DIR = "D:\\yaming_dataset\\Yaming_AI\\yolov7\\dataset\\nonbg_images/"

    except_count_list = []

    list_img = os.listdir(IMG_DATA_DIR)
    for i in list_img:
        os.mkdir(SAVE_IMG_DIR+i)
        except_count = 0
        for count,j in enumerate(os.listdir(IMG_DATA_DIR+i)):  
            img = Image.open(os.path.join(IMG_DATA_DIR,i,j))
            try:
                out = remove(img)
            except:
                except_count += 1
                continue
            else:
                out = out.convert("RGB")
                out.save(os.path.join(SAVE_IMG_DIR,i,j))
            finally:
                stdout.write("\r===== In  "+i+"  COUNT : "+str(count+1)+" progressed=====")
                stdout.flush()

        except_count_list.append({i : 1000 - except_count})
    
    print(except_count_list)





if __name__ == "__main__":
    # Move2Dir()
    RemoveImgBG()
