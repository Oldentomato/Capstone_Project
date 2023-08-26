import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Dropout, Flatten,GlobalAveragePooling2D
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
import splitfolders
import os
import kerastuner as kt
from tqdm.keras import TqdmCallback
import numpy as np



class VGG_MODEL():
    def __init__(self, img_size:int, img_dir:str, train_dir:str, save_model_dir:str, classes:int,save_graph_dir:str=None):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 경고 메시지를 숨기는 설정
        tf.get_logger().setLevel('ERROR')
        self.IMG_SIZE = img_size
        self.IMG_DIRECTORY = img_dir
        self.TRAIN_DIRECTORY = train_dir
        self.SAVE_MODEL_DIRECTORY = save_model_dir,
        self.CLASSES = classes,
        self.SAVE_GRAPH_DIRECTORY = save_graph_dir
        self.SAVE_PARAM = None

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
            '/data/api/tensorflow/tensor_ckpt/model-{epoch:04d}.h5', #모델 저장 경로
            monitor='val_accuracy', #모델을 저장할 때 기준이 되는 값
            verbose = 0, # 1이면 저장되었다고 화면에 뜨고 0이면 안뜸
            save_best_only=False,
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
    
    def Generate_feature(self, batch_size):
        train_generator, valid_generator = self.__Set_Dataset()
        vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(self.IMG_SIZE,self.IMG_SIZE,3)) #3채널만 됨
            #imagenet에서 이미 학습된 가중치를 가져옴. 모델 커스터마이징 하려면 false로. 

        print("train_predict processed....")
        bottleneck_features_train = vgg16.predict_generator(
            train_generator, batch_size
        )
        np.save(open('bottleneck_features/bottleneck_features_train.npy','wb'),
                bottleneck_features_train)
        
        print("valid_predict processed....")
        bottleneck_features_valid = vgg16.predict_generator(
            valid_generator, batch_size
        )
        np.save(open('bottleneck_features/bottleneck_features_valid.npy','wb'),
                bottleneck_features_valid)


    def Run_Training(self, **kargs):
        train,valid = self.__Set_Dataset_Generator(kargs['batch_size'])
        
        vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(self.IMG_SIZE,self.IMG_SIZE,3)) #3채널만 됨

        vgg16.trainable = False 

        flat = GlobalAveragePooling2D()(vgg16.output)

        Add_layer = Dense(units=kargs["layer_1"], activation = kargs["activation"])(flat)
        Add_layer = Dense(53, activation = 'softmax')(Add_layer)
        model = Model(inputs=vgg16.input, outputs=Add_layer)

        model.load_weights(kargs['model_path'])

        model.summary() #모델 구성을 보여줌


        model.compile(optimizer=tf.keras.optimizers.Adam(lr=kargs["learning_rate"]) ,loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
        with tf.device("/gpu:0"):
            history = model.fit(train,
                                epochs=kargs["train_epochs"],
                                steps_per_epoch=len(train),
                                validation_data=valid,
                                validation_steps=len(valid),
                                callbacks = self.Set_Callbacks(kargs['batch_size']),
                                verbose=0,
                                shuffle=True)
            
            
        return history


    def Run_Training_Tuner(self, objective, search_max_epochs, dir, project_name, search_epochs, batch_size, train_epochs,LAYER_INFO, isoverwrite):
        train,valid = self.__Set_Dataset_Generator(batch_size)
        # train_list = []
        # valid_list = []


        # for i,(_, labels )in enumerate(train):
        #     train_list.append(list(labels))

        # for _,label in train:
        #     temp = []
        #     temp.append(label)
        #     count += 1
        #     if count == 8:
        #         train_list.append(temp)
        #         count  = 0
        #         break
            
        
        # for _ in range(len(valid) // 8):
        #     _,vlabel = valid
        #     valid_list.append(vlabel)
        

        # train_labels = np.array(train_list)
        # train_data = np.load(open('bottleneck_features/bottleneck_features_train.npy','rb'))
        # valid_labels = np.array(valid_list)
        # valid_data = np.load(open('bottleneck_features/bottleneck_features_valid.npy','rb'))


        # print(np.shape(train))
        # print(np.shape(valid))
        # return 

        train_data, train_labels = train.next()
        valid_data, valid_labels = valid.next()


        def model_builder(hp):

            hp_units_1 = hp.Int('units_1', min_value = LAYER_INFO["min_value"], max_value = LAYER_INFO["max_value"], step = LAYER_INFO["step"])
            # hp_units_2 = hp.Int('units_2', min_value = LAYER_INFO["min_value"], max_value = LAYER_INFO["max_value"], step = LAYER_INFO["step"])
            # hp_units_3 = hp.Int('units_3', min_value = LAYER_INFO["min_value"], max_value = LAYER_INFO["max_value"], step = LAYER_INFO["step"])
            hp_activate = hp.Choice('activate', values = LAYER_INFO["activates"])
            hp_learning_rate = hp.Choice('learning_rate',values = LAYER_INFO["learning_rate"])

            vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(self.IMG_SIZE,self.IMG_SIZE,3)) #3채널만 됨

            vgg16.trainable = False 

            flat = GlobalAveragePooling2D()(vgg16.output)

            Add_layer = Dense(units=hp_units_1, activation = hp_activate)(flat)
            # Add_layer = Dense(units=hp_units_2, activation = hp_activate)(Add_layer)
            # Add_layer = Dense(units=hp_units_3, activation = hp_activate)(Add_layer)
            Add_layer = Dense(25, activation = 'softmax')(Add_layer)
            model = Model(inputs=vgg16.input, outputs=Add_layer)

            model.summary() #모델 구성을 보여줌

            model.compile(optimizer=tf.keras.optimizers.SGD(lr=hp_learning_rate) ,loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])

            return model
        
        
        tuner = kt.Hyperband(hypermodel = model_builder,
                             objective = objective,
                             max_epochs = search_max_epochs,
                             directory = dir,
                             overwrite = isoverwrite,
                             project_name = project_name)
        
        tuner.search(train_data, train_labels, epochs=search_epochs, validation_data=(valid_data,valid_labels))

        best_hps = tuner.get_best_hyperparameters(num_trials = 1)[0]

        f = open('/data/api/tensorflow/tensor_ckpt/hyper.txt','w')

        print(f"""
        The hyperparameter search is complete. The optimal number of units in the first densely-connected
        # layer is {best_hps.get('units_1')}, and the optimal learning rate for the optimizer
        is {best_hps.get('learning_rate')}.
        activate is {best_hps.get('activate')}. batch_size is {batch_size}.
        """, file=f)

        f.close()

        model = tuner.hypermodel.build(best_hps)
        self.SAVE_PARAM = best_hps


        with tf.device("/gpu:0"):
            history = model.fit(train,
                                epochs=train_epochs,
                                steps_per_epoch=len(train),
                                validation_data=valid,
                                validation_steps=len(valid),
                                callbacks = self.Set_Callbacks(batch_size),
                                verbose=0,
                                shuffle=True)
            
            
        return history
    

                            
    def Draw_Graph(self, history):
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train_loss','val_loss'])
        if self.SAVE_GRAPH_DIRECTORY != None:
            plt.savefig(str(self.SAVE_GRAPH_DIRECTORY)+"loss.png")
        
        plt.clf()

        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train_accuracy','val_accuracy'])
        if self.SAVE_GRAPH_DIRECTORY != None:
            plt.savefig(str(self.SAVE_GRAPH_DIRECTORY)+"acc.png")



        

