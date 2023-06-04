from vgg_model import VGG_MODEL

def Training():

    create_model = VGG_MODEL(
        img_size = 512,
        img_dir = 'D:\\yaming_dataset\\Yaming_AI\\yolov7\\dataset\\nonbg_images/',
        train_dir = 'D:\\yaming_dataset\\Yaming_AI\\yolov7\\dataset/',
        save_model_dir = 'D:\\yaming_dataset\\Yaming_AI\\api\\tensor_model/' ,
        classes=98,
        save_graph_dir="D:\\yaming_dataset\\Yaming_AI\\api\\tensor_result_graph/"
    )

    create_model.Use_GPU()

    train_generator, valid_generator = create_model.Set_Dataset()

    model = create_model.model_builder(LAYER_INFO={
                                    "min_value": 32,
                                    "max_value": 512,
                                    "step": 32,
                                    "activates": ['relu','tanh'],
                                    "learning_rate":[0.01,0.001,0.0001]}
                                    )

    history = create_model.Run_Training(model = model,
                            train_generator = train_generator,
                            valid_generator = valid_generator,
                            callbacks = create_model.Set_Callbacks(),
                            objective='val_accuracy',
                            search_max_epochs=10,
                            dir='D:\\yaming_dataset\\Yaming_AI\\api\\tensor_hyper/',
                            project_name='experience_1',
                            search_epochs=10,
                            train_epochs=100)

    create_model.Draw_Graph(history)


if __name__ == "__main__":
    Training()