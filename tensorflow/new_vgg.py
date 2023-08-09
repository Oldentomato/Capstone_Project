import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Input, Dropout, Flatten,GlobalAveragePooling2D
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
import splitfolders
import os
from tqdm.keras import TqdmCallback
import numpy as np

class VGG_MODEL():
    def __init__(self, img_size:int, img_dir:str, train_dir:str, classes:int):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 경고 메시지를 숨기는 설정
        tf.get_logger().setLevel('ERROR')
        self.IMG_SIZE = img_size
        self.IMG_DIRECTORY = img_dir
        self.TRAIN_DIRECTORY = train_dir
        self.CLASSES = classes
        self.SAVE_PARAM = None

        # check experiment
        try:
            exp_dir = os.listdir("./experiment/")
        except:
            print("no experiment dir")
        else:
            exp_count = len(exp_dir) + 1
            os.mkdir(f"./experiment/{exp_count}")
            self.exp_path = f"./experiment/{exp_count}"

    def Use_GPU(self) -> None:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                # Currently, memory growth needs to be the same across GPUs
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(e)

    def __save_log(self):
        pass

    def __Set_Dataset_Generator(self,batch_size:int):
        if 'train' not in os.listdir(self.TRAIN_DIRECTORY):
            splitfolders.ratio(self.IMG_DIRECTORY, self.TRAIN_DIRECTORY,seed=1337, ratio=(0.8,0.2))

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
            self.TRAIN_DIRECTORY+'train',
            target_size = (self.IMG_SIZE,self.IMG_SIZE), # (image_size, image_size)
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
            self.TRAIN_DIRECTORY+'val',
            target_size = (self.IMG_SIZE,self.IMG_SIZE),
            color_mode = 'rgb',
            class_mode = 'sparse',
            shuffle = False,
            batch_size = batch_size,
        )

        return train_generator, valid_generator

    def Set_Callbacks(self, batch_size):
        checkpoint = ModelCheckpoint(
            str({self.exp_path})+"/{epoch:03d}best.h5", #모델 저장 경로
            monitor='val_accuracy', #모델을 저장할 때 기준이 되는 값
            verbose = 0, # 1이면 저장되었다고 화면에 뜨고 0이면 안뜸
            save_best_only=True,
            mode = 'max',
            #val_acc인 경우, 정확도이기 때문에 클수록 좋으므로 max를 쓰고, val_loss일 경우, loss값이기 떄문에 작을수록 좋으므로 min을써야한다
            #auto일 경우 모델이 알아서 min,max를 판단하여 모델을 저장한다
            save_weights_only=False, #가중치만 저장할것인가 아닌가
            save_freq = 1*batch_size #3번째 에포크마다 가중치를 저장 period를 안쓰고 save_freq룰 씀
        )

        lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_accuracy', factor=0.1, patience= 2*batch_size, verbose=0, mode='auto',
            min_delta=0.0001, cooldown=0, min_lr=0
        )

        earlystop = EarlyStopping(
            monitor='val_accuracy',
            min_delta = 0.05,
            patience = 3*batch_size,
            verbose = 1,
            mode='auto'
        )

        callbacks = [checkpoint,earlystop,lr_scheduler,TqdmCallback(verbose=1)]
        return callbacks
    

    def Generate_bottleneck(self, batch_size):

        train,valid = self.__Set_Dataset_Generator(batch_size)

        vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(self.IMG_SIZE,self.IMG_SIZE,3))

        print("start extract train data\n")
        bottleneck_features_train = vgg16.predict_generator(train)
        print("start extract valid data\n")
        bottleneck_features_valid = vgg16.predict_generator(valid)

        # 매칭된 라벨 저장
        train_labels = train.classes
        valid_labels = valid.classes

        # Bottleneck features와 라벨을 한 쌍으로 저장
        print("saving\n")
        np.save(open('bottleneck_features/bottleneck_features_train.npy', 'wb'), bottleneck_features_train)
        np.save(open('bottleneck_features/bottleneck_features_valid.npy', 'wb'), bottleneck_features_valid)
        np.save(open('bottleneck_features/train_labels.npy', 'wb'), train_labels)
        np.save(open('bottleneck_features/valid_labels.npy', 'wb'), valid_labels)

        print("feature extracted\n")


    
    def Run_Training(self, **kargs):
        """
        Run Training
        batch_size (int): batch size

        layers (tuplelist): [(layer_num, activation)]
        learning_rate (float): learning_rate
        train_epochs (int): train epochs 
        (opt)model_path (str): load model path
        """

        #혹시 이미 그려둔 그래프가 있다면 clear
        tf.keras.backend.clear_session()

        bottleneck_features_train = np.load('bottleneck_features/bottleneck_features_train.npy')
        bottleneck_features_valid = np.load('bottleneck_features/bottleneck_features_valid.npy')
        train_labels = np.load('bottleneck_features/train_labels.npy')
        valid_labels = np.load('bottleneck_features/valid_labels.npy')


        input_shape = bottleneck_features_train.shape[1:]  # bottleneck features의 형태를 가져옴

        inputs = Input(shape=input_shape)
        x = GlobalAveragePooling2D()(inputs)

        for layer in kargs["layers"]:
            Add_layer = Dense(units=layer[0], activation = layer[1])(x)

        #이부분을 계층적 출력층으로 변경해야함
        # class Node:
        #     def __init__(self, id):
        #         self.id = id
        #         self.left = None
        #         self.right = None
            
        #     def is_leaf(self):
        #         return self.left is None and self.right is None

        # # 가상의 트리 생성
        # root = Node("Root")
        # A = Node("A")
        # B = Node("B")
        # C = Node("C")
        # D = Node("D")
        # E = Node("E")
        # F = Node("F")

        # root.left = A
        # root.right = B
        # A.left = C
        # A.right = D
        # B.left = E
        # B.right = F

        # # 트리 탐색 코드 수정
        # tree = root

        # def traverse_tree(node, code=()):
        #     if node is None:
        #         return []
        #     if node.is_leaf():
        #         return [(node.id, code)]
            
        #     left_code = traverse_tree(node.left, code + (0,))
        #     right_code = traverse_tree(node.right, code + (1,))
        #     return left_code + right_code

        # # 코드 탐색
        # codes = traverse_tree(tree)
        # print(codes)
        outputs = Dense(units=int(self.CLASSES), activation = 'softmax')(Add_layer)
        

        model = Model(inputs,outputs)

        if kargs['model_path']:
            model.load_weights(kargs['model_path'])

        model.summary() #모델 구성을 보여줌


        model.compile(optimizer=tf.keras.optimizers.Adam(lr=kargs["learning_rate"]) ,loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
        with tf.device("/gpu:0"):
            history = model.fit(bottleneck_features_train,
                                train_labels,
                                epochs=kargs["train_epochs"],
                                steps_per_epoch=len(bottleneck_features_train),
                                validation_data=(bottleneck_features_valid,valid_labels),
                                validation_steps=len(bottleneck_features_valid),
                                callbacks = self.Set_Callbacks(kargs['batch_size']),
                                verbose=0,
                                shuffle=True)
            
            
        return history
    

    def Draw_Graph(self, history):
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train_loss','val_loss'])
        plt.savefig(f"{self.exp_path}/loss.png")
        
        plt.clf()

        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train_accuracy','val_accuracy'])
        plt.savefig(f"{self.exp_path}/acc.png")