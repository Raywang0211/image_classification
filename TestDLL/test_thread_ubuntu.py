from so_out.inference_server import MyModel
import os
import time
import cv2


if __name__=="__main__":
    output_class = 5
    batch_size = 100
    lr = 0.0001
    model = "/home/trx50/project/image_classification/ft_model.pth"

    def callback(result):
        print("callback = ",result)
    MM = MyModel(output_class, batch_size, lr, model)
    result = MM.start_inference_single_thread(callback)
    
    path = "/home/trx50/project/image_classification/data/2024-12-12_缺點圖片收集/Mark/"
    images = os.listdir(path)
    target_class = "class_4"
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
        input_image = cv2.imread(filename) 
        MM.input_image.put(input_image)
        result = MM.output_result.get()
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
    


        
        
