import pandas as pd
import os


DATA_DIR = "D:\\yaming_dataset\\dataset\\temp"


LABEL_DIR = DATA_DIR+"/labels/"
IMAGE_DIR = DATA_DIR+"/images/"

y_col = []
x_col = []

dirlist = os.listdir(LABEL_DIR)

for file in dirlist:
    with(open(LABEL_DIR+file,mode="r")) as label_file:
        label = label_file.read().split(" ")[0]
        x_col.append(label)

print(x_col)