import os
from tensorflow.keras.applications.vgg16 import VGG16
# from tensorflow.keras.applications.resnet50 import ResNet18
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import splitfolders



class BaseModel:

    def Set_Dataset_Generator(self,batch_size:int,img_dir:str,train_dir:str,img_size:int):
        if 'train' not in os.listdir(train_dir):
            splitfolders.ratio(img_dir, self.train_dir,seed=1337, ratio=(0.8,0.2))

        # 학습에 사용될 이미지 데이터 생성
        train_datagen= ImageDataGenerator( #여기다가 여러 파라미터를 넣어서 새로운 이미지들을 만들어야한다
            rescale = 1. /255, # 각픽셀을 최대값 255을 중심으로 0과 1사이의 값으로 재구성
            rotation_range = 40,
            width_shift_range= 0.2,
            vertical_flip = True, 
            height_shift_range = 0.2, 
            shear_range = 0.2,
            zoom_range = 0.2,
            horizontal_flip = True,
            fill_mode='nearest'
        )

        # 검증에 사용될 이미지 데이터 생성
        valid_datagen = ImageDataGenerator(
            rescale = 1. /255,
        )

        # 학습에 사용될 데이터 생성
        #rgb 혹은 rgba 확인하기
        train_generator = train_datagen.flow_from_directory( #디렉토리에서 가져온 데이터를 flow시키는 것
            train_dir+'train',
            target_size = (img_size,img_size), # (image_size, image_size)
            color_mode = 'rgb',
            class_mode = 'sparse', #class를 어떻게 읽는지 설정
            # binary: 이진 레이블의 1D numpy 배열
            # categorical: one-hot 인코딩된 레이블의 2D numpy 배열입니다. 멀티 라벨 출력을 지원합니다.
            # sparse: 정수 레이블의 1D numpy 배열,
            # input: 입력 이미지와 동일한 이미지(주로 오토인코더 작업에 사용됨),
            # other: y_col 데이터의 numpy 배열,
            # None: 대상이 반환되지 않습니다(생성기는 에서 사용하는 데 유용한 이미지 데이터의 배치만 생성합니다 model.predict_generator()).
            shuffle = False, # 섞는다는 뜻. 순서를 무작위로 적용한다.
            batch_size = batch_size, # 배치 size는 한번에 gpu를 몇 개 보는가. 한번에 8장씩 학습시킨다
        )


        # 검증에 사용될 데이터 생성
        valid_generator = valid_datagen.flow_from_directory(
            train_dir+'val',
            target_size = (img_size,img_size),
            color_mode = 'rgb',
            class_mode = 'sparse',
            shuffle = False,
            batch_size = batch_size,
        )

        return train_generator, valid_generator

