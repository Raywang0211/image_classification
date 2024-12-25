from so_out.Model import MyModel



if __name__=="__main__":
    output_class = 5
    batch_size = 100
    lr = 0.0001
    save_model_name = "ft_model"
    MM = MyModel(output_class, batch_size, lr)
    
    # root_dir_train = "data/SLT03缺點圖片收集"
    # root_dir_test = "/home/trx50/project/image_classification/data/vechicles/test"
    # epoch = 10
    # MM.start_train(root_dir_train, epoch, save_model_name)
    # model_path = "/home/trx50/project/image_classification/ft_model_01.pth"
    # batch inference
    # test = "/home/trx50/project/image_classification/data/vechicles/test"
    # test = "data/ttt"
    # MM.start_inference(model_path, test)
    
    
    # inference single image
    model = "/home/trx50/project/image_classification/ft_model.pth"
    filename = "/home/trx50/project/mytrainingGUI/projects/Myproject/Dataset/mark/Mark (1).jpg"
    result = MM.start_inference_single(filename, model)
    
    
