from ..basemodel import BaseModel
from tensorflow.keras.applications.densenet import DenseNet121
import os
import numpy as np


class DenseNet(BaseModel):
    def __init__(self,batch_size, bottle_dir, img_size,img_dir,train_dir):
        self.img_size = img_size
        self.bottle_dir = bottle_dir
        self.batch_size = batch_size
        self.img_dir = img_dir
        self.train_dir = train_dir

    def __call__(self):
        train,valid = super().Set_Dataset_Generator(batch_size=self.batch_size,img_dir=self.img_dir,train_dir=self.train_dir,img_size=self.img_size)
        model = DenseNet121(weights='imagenet', include_top=False, input_shape=(self.img_size,self.img_size,3))

        model.summary()

        bottleneck_dir = os.listdir("./bottleneck_features/")

        if self.bottle_dir not in bottleneck_dir:
            os.mkdir(f"./bottleneck_features/{self.bottle_dir}")

        print("start extract train data\n")
        bottleneck_features_train = model.predict_generator(train)
        print("start extract valid data\n")
        bottleneck_features_valid = model.predict_generator(valid)

        # 매칭된 라벨 저장
        train_labels = train.classes
        valid_labels = valid.classes

        # Bottleneck features와 라벨을 한 쌍으로 저장
        print("saving\n")
        np.save(open(f'./bottleneck_features/{self.bottle_dir}/bottleneck_features_train.npy', 'wb'), bottleneck_features_train)
        np.save(open(f'./bottleneck_features/{self.bottle_dir}/bottleneck_features_valid.npy', 'wb'), bottleneck_features_valid)
        np.save(open(f'./bottleneck_features/{self.bottle_dir}/train_labels.npy', 'wb'), train_labels)
        np.save(open(f'./bottleneck_features/{self.bottle_dir}/valid_labels.npy', 'wb'), valid_labels)

        print("feature extracted\n")

