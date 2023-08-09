# from vgg_model import VGG_MODEL
#새로 구성한 모델 bottle featureneck 적용, keras tuner 미사용, 
from new_vgg import VGG_MODEL

def Training():

    create_model = VGG_MODEL(
        img_size = 512,
        img_dir = '/dataset/nonbg_images/',
        train_dir = '/dataset/',
        classes=25
    )

    create_model.Use_GPU()

    create_model.Generate_bottleneck(batch_size=8)#최초 한번만 실행할것


    # history = create_model.Run_Training_Tuner(
    #                         objective='val_accuracy',
    #                         search_max_epochs=10,
    #                         dir='/data/api/tensorflow/tensor_hyper/',
    #                         project_name='experience_6',
    #                         search_epochs=10,
    #                         train_epochs=100,
    #                         batch_size=16,
    #                         isoverwrite = False,
    #                         LAYER_INFO={
    #                                 "min_value": 256,
    #                                 "max_value": 4096,
    #                                 "step": 256,
    #                                 "activates": ['relu'],
    #                                 "learning_rate":[0.01,0.001,0.0001]})

    history = create_model.Run_Training(batch_size=16, 
                                        layers=[(
                                            512, 'relu'
                                        )], 
                                        learning_rate=0.001,
                                        train_epochs=100)

    create_model.Draw_Graph(history)


  


if __name__ == "__main__":
    Training()