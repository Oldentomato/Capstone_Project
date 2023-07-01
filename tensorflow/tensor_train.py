from vgg_model import VGG_MODEL

def Training():

    create_model = VGG_MODEL(
        img_size = 256,
        img_dir = '/dataset/nonbg_images/',
        train_dir = '/dataset/',
        save_model_dir = '/data/api/tensorflow/tensor_model/' ,
        classes=53,
        save_graph_dir="/data/api/tensorflow/tensor_result_graph/"
    )

    create_model.Use_GPU()

    # create_model.Generate_feature()#최초 한번만 실행할것


    # history = create_model.Run_Training_Tuner(
    #                         objective='val_accuracy',
    #                         search_max_epochs=10,
    #                         dir='/data/api/tensorflow/tensor_hyper/',
    #                         project_name='experience_4',
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
                                        layer_1=768, 
                                        activation='relu', 
                                        model_path="model-0077.h5", 
                                        learning_rate=0.001,
                                        train_epochs=100)

    create_model.Draw_Graph(history)


  


if __name__ == "__main__":
    Training()