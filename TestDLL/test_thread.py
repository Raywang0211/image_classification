from dll_out.AImodelInference import MyModel
import os
import time


if __name__=="__main__":
    output_class = 5
    batch_size = 100
    lr = 0.0001
    model = "C:\\Users\\USER\\Desktop\\project\\image_classification\\TestDLL\\ft_model.pth"
    def callback(result):
        print("callback = ",result)
    MM = MyModel(output_class, batch_size, lr, model)
    
    # root_dir_train = "data/SLT03缺點圖片收集"
    # root_dir_test = "/home/trx50/project/image_classification/data/vechicles/test"
    # epoch = 10
    # MM.start_train(root_dir_train, epoch, save_model_name)
    # model_path = "/home/trx50/project/image_classification/ft_model_01.pth"
    # batch inference
    # test = "/home/trx50/project/image_classification/data/vechicles/test"
    # test = "data/ttt"
    # MM.start_inference(model_path, test)
    path = "C:\\Users\\USER\\Desktop\\project\\RAW\\SLT03缺點圖片收集\\膜邊\\"
    images = os.listdir(path)
    target_class = "class_0"
    s1 = time.time()
    class_dic = {
            "class_0" : 0,
            "class_1" : 0,
            "class_2" : 0,
            "class_3" : 0,
            "class_4" : 0,
    }

    for image in images:
        filename = path + image        
        result = MM.start_inference_single_thread(filename,callback)
        if result == "0":
            class_dic["class_0"]+=1
        elif result == "1":
            class_dic["class_1"]+=1
        elif result == "2":
            class_dic["class_2"]+=1
        elif result == "3":
            class_dic["class_3"]+=1
        elif result == "4":
            class_dic["class_4"]+=1
    




    Acuracy = class_dic[target_class]/len(images)
    print("Acuracy === ",Acuracy)
    print("Total time = ",time.time()-s1)
    print("Average inference time = ",(time.time()-s1)/len(images))
    for key, value in class_dic.items():
        print(f"{key:<10}: {value}")
    


        
        
