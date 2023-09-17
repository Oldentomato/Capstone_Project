# from vgg_model import VGG_MODEL
#새로 구성한 모델 bottle featureneck 적용, keras tuner 미사용, 
from train import Train

def Training():
    img_size = 128
    batch_size = 4
    generate_feature = False
    model_name = "densenet"
    bottle_dir = "dense_size256_batch2"

    run_model = Train(
        img_size = img_size,
        img_dir = '/dataset/nonbg_images/',
        train_dir = '/dataset/',
        classes=22
    )

    run_model.Use_GPU()


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

    if generate_feature:
        history = run_model.Run_Training(generate_feature = generate_feature,
                                        batch_size = batch_size,
                                        img_size = img_size,
                                        model_name = model_name,
                                            batch_normal = True,
                                            layers=[(
                                                512, "relu"
                                            )], 
                                            bottle_dir = bottle_dir,
                                            learning_rate=0.001,
                                            train_epochs=100)
    else:
        history = run_model.Run_Training(generate_feature = generate_feature,
                                        batch_normal = False,
                                        model_name = model_name,
                                        img_size = img_size,
                                        layers=[(
                                            1024, "relu"
                                        )], 
                                        bottle_dir = bottle_dir,
                                        learning_rate=0.001,
                                        train_epochs=100)
    

    run_model.save_log(history)

    run_model.Draw_Graph(history)


  


if __name__ == "__main__":
    Training()