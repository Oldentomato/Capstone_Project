import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Input, Dropout, GlobalAveragePooling2D, BatchNormalization, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
import splitfolders
import os
from tqdm.keras import TqdmCallback
import numpy as np
import datetime
from models.vgg import VGG
from models.resnet import ResNet
from models.densenet import DenseNet


class Train():
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

        # config = tf.ConfigProto()
        # config.gpu_options.allow_growth = True #탄력적 gpu memory 사용
        # config.gpu_options.per_process_gpu_memory_fraction = 0.4
        # session = tf.Session(config=config)
        # tf.keras.backend.tensorflow_backend.set_session(session)



    def Set_Callbacks(self):
        checkpoint = ModelCheckpoint(
            self.exp_path+"/best.h5", #모델 저장 경로
            monitor='val_accuracy', #모델을 저장할 때 기준이 되는 값
            verbose = 0, # 1이면 저장되었다고 화면에 뜨고 0이면 안뜸
            save_best_only=False,
            mode = 'max',
            #val_acc인 경우, 정확도이기 때문에 클수록 좋으므로 max를 쓰고, val_loss일 경우, loss값이기 떄문에 작을수록 좋으므로 min을써야한다
            #auto일 경우 모델이 알아서 min,max를 판단하여 모델을 저장한다
            save_weights_only=False, #가중치만 저장할것인가 아닌가
            save_freq = 'epoch' #3번째 에포크마다 가중치를 저장 period를 안쓰고 save_freq룰 씀
        )

        lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_accuracy', factor=0.1, patience= 1, verbose=1, mode='max',
            min_delta=0.1, cooldown=0, min_lr=0.00001
        )

        tensorboard = tf.keras.callbacks.TensorBoard(log_dir="logs/fit/"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

        earlystop = EarlyStopping(
            monitor='val_accuracy',
            min_delta = 0.1,
            patience = 1,
            verbose = 1,
            mode='max'
        )

        callbacks = [checkpoint,earlystop,lr_scheduler,tensorboard,TqdmCallback(verbose=1)]
        return callbacks
    



    
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

        if kargs["generate_feature"] == True:
            if kargs["model_name"] == "vgg":
                vgg_model = VGG(batch_size=8,bottle_dir="test",img_size=256,train_dir='/dataset/',img_dir='/dataset/nonbg_images/')
                vgg_model()
            elif kargs["model_name"] == "resnet":
                resnet_model = ResNet(batch_size=kargs["batch_size"],bottle_dir=kargs["bottle_dir"],img_size=kargs["img_size"],train_dir=self.TRAIN_DIRECTORY,img_dir=self.IMG_DIRECTORY)
                resnet_model()
            elif kargs["model_name"] == "densenet":
                densenet_model = DenseNet(batch_size=kargs["batch_size"],bottle_dir=kargs["bottle_dir"],img_size=kargs["img_size"],train_dir=self.TRAIN_DIRECTORY,img_dir=self.IMG_DIRECTORY)
                densenet_model()

            

        bottleneck_features_train = np.load(f'bottleneck_features/{kargs["bottle_dir"]}/bottleneck_features_train.npy')
        bottleneck_features_valid = np.load(f'bottleneck_features/{kargs["bottle_dir"]}/bottleneck_features_valid.npy')
        train_labels = np.load(f'bottleneck_features/{kargs["bottle_dir"]}/train_labels.npy')
        valid_labels = np.load(f'bottleneck_features/{kargs["bottle_dir"]}/valid_labels.npy')



        input_shape = bottleneck_features_train.shape[1:]  # bottleneck features의 형태를 가져옴

        inputs = Input(shape=input_shape)
        x = GlobalAveragePooling2D()(inputs)

        if len(kargs["layers"]) >= 1:
            is_first_layer = True
            for layer in kargs["layers"]:
                if is_first_layer == True:
                    Add_layer = Dense(units=layer[0])(x)
                    if kargs["batch_normal"]:
                        Add_layer = BatchNormalization()(Add_layer)
                    Add_layer = Activation(layer[1])(Add_layer)
                    is_first_layer = False
                else:
                    if layer[1] == "dropout":
                        Add_layer = Dropout(layer[0])(Add_layer)
                    else:
                        Add_layer = Dense(units=layer[0])(Add_layer)
                        if kargs["batch_normal"]:
                            Add_layer = BatchNormalization()(Add_layer)
                        Add_layer = Activation(layer[1])(Add_layer)

            outputs = Dense(units=int(self.CLASSES), activation = 'softmax')(Add_layer)
        else:
            outputs = Dense(units=int(self.CLASSES), activation = 'softmax')(x)


        
        
        model = Model(inputs,outputs)

        # if kargs['model_path']:
        #     model.load_weights(kargs['model_path'])

        model.summary() #모델 구성을 보여줌


        model.compile(optimizer=tf.keras.optimizers.Adam(lr=kargs["learning_rate"]) ,loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
        with tf.device("/gpu:0"):
            history = model.fit(bottleneck_features_train,
                                train_labels,
                                epochs=kargs["train_epochs"],
                                steps_per_epoch=len(bottleneck_features_train),
                                validation_data=(bottleneck_features_valid,valid_labels),
                                validation_steps=len(bottleneck_features_valid),
                                callbacks = self.Set_Callbacks(),
                                verbose=0,
                                shuffle=True)
            
            
        return history
    

    def save_log(self, history):

        acc = history.history["sparse_categorical_accuracy"]
        val_acc = history.history["val_sparse_categorical_accuracy"]
        loss = history.history["loss"]
        val_loss = history.history["val_loss"]
        
        min_acc = min(acc)
        max_acc = max(acc)


        min_val_acc = min(val_acc)
        max_val_acc = max(val_acc)

        min_loss = min(loss)
        max_loss = max(loss)

        min_val_loss = min(val_loss)
        max_val_loss = max(val_loss)

        str_acc = list(map(str, acc))
        str_val_acc = list(map(str, val_acc))
        str_loss = list(map(str, loss))
        str_val_loss = list(map(str, val_loss))


        with open(f'{self.exp_path}/acctxt.txt','w') as acctxt:
            acctxt.write("\n".join(str_acc))
        with open(f'{self.exp_path}/val_acctxt.txt','w') as val_acctxt:
            val_acctxt.write("\n".join(str_val_acc))
        with open(f'{self.exp_path}/losstxt.txt','w') as losstxt:
            losstxt.write("\n".join(str_loss))
        with open(f'{self.exp_path}/val_losstxt.txt','w') as val_losstxt:
            val_losstxt.write("\n".join(str_val_loss))

        with open(f'{self.exp_path}/integrated analysis.txt','w') as int_ana:
            int_ana.write(f"""
                          
                        min accuracy : {min_acc} in {acc.index(min_acc)} \n
                        max accuracy : {max_acc} in {acc.index(max_acc)} \n
                        min val_accuracy : {min_val_acc} in {val_acc.index(min_val_acc)} \n
                        max val_accuracy : {max_val_acc} in {val_acc.index(max_val_acc)} \n

                        min loss : {min_loss} in {loss.index(min_loss)} \n
                        max loss : {max_loss} in {loss.index(max_loss)} \n
                        min val_loss : {min_val_loss} in {val_loss.index(min_val_loss)} \n
                        max val_loss : {max_val_loss} in {val_loss.index(max_val_loss)} \n
                          """)
    

    def Draw_Graph(self, history):
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train_loss','val_loss'])
        plt.savefig(f"{self.exp_path}/loss.png")
        
        plt.clf()

        plt.plot(history.history['sparse_categorical_accuracy'])
        plt.plot(history.history['val_sparse_categorical_accuracy'])
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train_accuracy','val_accuracy'])
        plt.savefig(f"{self.exp_path}/acc.png")