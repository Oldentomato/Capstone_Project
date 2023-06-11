from vgg_model import VGG_MODEL

def Training():

    create_model = VGG_MODEL(
        img_size = 256,
        img_dir = '/dataset/nonbg_images/',
        train_dir = '/dataset/',
        save_model_dir = '/data/api/tensor_model/' ,
        classes=53,
        save_graph_dir="/data/api/tensor_result_graph/"
    )

    create_model.Use_GPU()

    # create_model.Generate_feature()#최초 한번만 실행할것


    history = create_model.Run_Training(
                            objective='val_accuracy',
                            search_max_epochs=10,
                            dir='/data/api/tensor_hyper/',
                            project_name='experience_1',
                            search_epochs=10,
                            train_epochs=100,
                            isoverwrite = False,
                            LAYER_INFO={
                                    "min_value": 32,
                                    "max_value": 512,
                                    "step": 32,
                                    "activates": ['relu','tanh'],
                                    "learning_rate":[0.01,0.001,0.0001]})

    create_model.Draw_Graph(history)


if __name__ == "__main__":
    Training()