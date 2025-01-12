# from torchvision.models import mobilenet_v3_small
# from torchvision.models import MobileNet_V3_Small_Weights
from torchvision.models import efficientnet_v2_l
from torchvision.models import EfficientNet_V2_L_Weights
from torch.utils.data import random_split
from torch import nn

from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import torch

# from MyDataset import MyImageDataset
# from .MyDataset import MyImageDataset # for build .DLL
import matplotlib.pyplot as plt
import threading
from PIL import Image
import cv2
import numpy as np
import time

stop_event = threading.Event()
input_image = None
test_value = "a"


class MyModel():
    def __init__(self, output_class, batch_size=100, lr=0.01, inference_model=None):
        self.output_class = output_class
        self.batch_size = batch_size
        self.learningrate = lr
        self.writer = SummaryWriter("./log") # for tensor board
        self.training_device = "cuda:0"
        self.iterate_time = 0 # for estimate training time
        self.data_split_rate = 0.8
        self.train_size = 0
        self.val_size = 0
        if inference_model!=None:
            self.load_inference_model(inference_model)
        print(f"Using {self.training_device} device")


    def load_model(self):
        # Load model
        self.model = efficientnet_v2_l(weights=EfficientNet_V2_L_Weights.DEFAULT)
        self.model.classifier = nn.Sequential(
            nn.Linear(1280, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, self.output_class)
        )
        for name, param in self.model.named_parameters():
            if param.numel() == 0:
                print(f"Layer {name} has zero parameters")
        # print(self.model)

    def model_freeze(self):
        """
        unfreeze last few layer for transfer learning 
        """

        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.model.classifier[0:].parameters():
            param.requires_grad = True

    
    def load_single_img(self, image_path):
        """
        load single image 
        """
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        print("laod_single_imag: 1 ")

        # 讀入影像並進行預處理
        image = Image.open(image_path).convert('RGB')
        print("laod_single_imag: 2 ")
        input_tensor = transform(image).unsqueeze(0)  # 新增 batch 維度
        print("laod_single_imag: 3")
        input_tensor = input_tensor.to(self.training_device)
        print("laod_single_imag: 4")
        return input_tensor

    
    def load_single_img_cv2(self, image_path):
        """
        load single image 
        """
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        print("laod_single_imag: 1 ")

        # 讀入影像並進行預處理
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print("laod_single_imag: 2 ")
        image = Image.fromarray(image)
        input_tensor = transform(image).unsqueeze(0)  # 新增 batch 維度
        print("laod_single_imag: 3")
        input_tensor = input_tensor.to(self.training_device)
        print("laod_single_imag: 4")
        return input_tensor


    def load_dataset(self, root_dir_train):
        """
        setup train and valadation data loader
        train data with augmentation
        
        """

        train_augmentation = transforms.Compose(
            [
                transforms.Resize((224,224)),
                transforms.RandomRotation(15), 
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
        )

        full_train = MyImageDataset(root_dir_train, transform=train_augmentation)
        self.train_size = int(self.data_split_rate * len(full_train))
        self.val_size = len(full_train) - self.train_size
        train_dataset, val_dataset = random_split(full_train, [self.train_size, self.val_size])
        
        
        full_train.save_label_pare() # save train label and class name pare
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Validation dataset size: {len(val_dataset)}")



    def load_testdata(self, root_dir_test):
        
        test_augmentation = transforms.Compose(
            [
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
        )
        test = MyImageDataset(root_dir_test, transform=test_augmentation)
        self.val_loader = DataLoader(test, batch_size=self.batch_size)
        

    def init_loss_optimizer(self):
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learningrate)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.1)


    def train(self, dataloader, model, loss_fn, optimizer):
        size = len(dataloader.dataset)
        model = model.to(self.training_device)
        model.train()
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(self.training_device), y.to(self.training_device)
            
            # Compute prediction error
            pred = model(X)
            loss = loss_fn(pred, y)

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            print("123-123",batch)
            if batch % 100 == 0:
                loss, current = loss.item(), (batch + 1) * len(X)  
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        return loss


    def validation(self, dataloader, model, loss_fn):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(self.training_device), y.to(self.training_device)
                pred = model(X)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= size 
        print(f"Validation: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
        return (100*correct)
    
    
    def inference_test(self, dataloader, model, loss_fn):
        """
        batch inference calculate acc
        """
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        model = model.to(self.training_device)
        model.eval()
        test_loss, correct = 0, 0
            
        with torch.no_grad():
            
            # for batch in dataloader:
            #     print(len(batch))
            #     # print(batch)
            #     print(len(batch[1]))
            #     print(batch[1])
            #     print(len(batch[2]))
            #     print(batch[2])
                
            
            
            
            for X, y, image_name in dataloader:
                X, y = X.to(self.training_device), y.to(self.training_device)
                pred = model(X)
                probabilities = torch.softmax(pred,dim = 1)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
                
        test_loss /= num_batches
        correct /= size 
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
        return (100*correct)
    
    
    def save_model(self, model_name):
        model_name = model_name + ".pth"
        torch.save(self.model.state_dict(), model_name)
        print("save model")
    
    def start_train(self,train_dataset, epoch, model_name):
        self.load_model()
        self.model_freeze()

        self.load_dataset(train_dataset)
        self.init_loss_optimizer()
        self.acc_pre = 0
        for t in range(epoch):
            t1 = time.time()
            print(f"Epoch {t+1}\n-------------------------------")
            loss = self.train(self.train_loader, self.model, self.loss_fn, self.optimizer)
            
            if t%20==0:
                acc = self.validation(self.val_loader, self.model, self.loss_fn)
                if self.acc_pre<= acc:
                    self.save_model(model_name)
                    self.acc_pre = acc
                
                self.writer.add_scalar("accuracy", acc, t)
                
            self.writer.add_scalar("loss", loss, t)
            
            self.scheduler.step()
            self.iterate_time = time.time() - t1 # for estimating training time
        print("Done!")
        self.writer.close()
        


    def start_inference(self, model_path, test_Dataset):
        """
        batch inference
        """

        self.load_model()
        self.model.load_state_dict(torch.load(model_path, weights_only=True))
        self.init_loss_optimizer()
        self.load_testdata(test_Dataset)
        acc = self.inference_test(self.val_loader, self.model, self.loss_fn)
        print(acc)
    
    
    def load_inference_model(self, model_path):
        """
        preload model for inference

        """
        self.load_model()
        self.model.load_state_dict(torch.load(model_path, weights_only=True))
        self.model.to(self.training_device)
        self.model.eval()
        
    
    def start_inference_single(self, test_image):
        """
        single inference
        
        """
        print("Inside start_inference_single")
        img_tensor = self.load_single_img(test_image)
        print("Load image finished")
        with torch.no_grad():  # 關閉梯度計算以加速推論
            
            s1 = time.time()
            output = self.model(img_tensor)
            print("model inference finished")
            _, predicted_class = torch.max(output, 1)
            print("torch max finished")
            result = str(predicted_class[0].item())
            print("class = ", result)
        print("InferenceTime:",time.time()-s1)
        return result

    def start_inference_single_withcallback(self, test_image, callback):
        """
        single inference callback
        
        """
        print("Inside start_inference_single_withcallback: 1")
        img_tensor = self.load_single_img(test_image)
        print("Inside start_inference_single_withcallback: 2")
        with torch.no_grad():  # 關閉梯度計算以加速推論
            
            s1 = time.time()
            output = self.model(img_tensor)
            print("Inside start_inference_single_withcallback: 3")
            _, predicted_class = torch.max(output, 1)
            print("Inside start_inference_single_withcallback: 4")
            result = str(predicted_class[0].item())
            print("Inside start_inference_single_withcallback: 5")
            print("class = ", result)
            callback(result)
            
        print("InferenceTime:",time.time()-s1)
        return result

    def start_inference_single_withcallback_image(self, callback):
        """
        single inference callback
        
        """
        global test_value
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        while not stop_event.is_set():
            if input_image is None:
                print("Test =",test_value )
                pass
            else:
                print("Inside start_inference_single_withcallback: 1")
                image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
                print("laod_single_imag: 2 ")
                image = Image.fromarray(image)
                input_tensor = transform(image).unsqueeze(0)  # 新增 batch 維度
                print("laod_single_imag: 3")
                input_tensor = input_tensor.to(self.training_device)
                print("laod_single_imag: 4")
                print("Inside start_inference_single_withcallback: 2")
                with torch.no_grad():  # 關閉梯度計算以加速推論
                    
                    s1 = time.time()
                    output = self.model(input_tensor)
                    print("Inside start_inference_single_withcallback: 3")
                    _, predicted_class = torch.max(output, 1)
                    print("Inside start_inference_single_withcallback: 4")
                    result = str(predicted_class[0].item())
                    print("Inside start_inference_single_withcallback: 5")
                    print("class = ", result)
                    callback(result)
                    
                print("InferenceTime:",time.time()-s1)
                test_image = None
        print("Close inference") 



    def start_inference_single_thread(self, callback):
        """
        single inference thread inference
        
        """
        
        inference_threading = threading.Thread(target=self.start_inference_single_withcallback_image,
                                               args=(callback,))
        inference_threading.start()
        
        return 0
    
    def close_inference(self):
        stop_event.set()
    def send_image(self, image, v):
        input_image = image
        test_value = v
        print("send = ",test_value)


if __name__=="__main__":
    output_class = 5
    batch_size = 100
    lr = 0.0001
    save_model_name = "ft_model"
    model = "/home/trx50/project/image_classification/ft_model.pth"
    # if you only want to inference just add model, 
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
    
    # inference single image
    def callback(result):
        print("callback = ",result)
    
    filename = "/home/trx50/project/image_classification/data/2024-12-12_缺點圖片收集/毛絲/3649.jpg"
    filename2 = "/home/trx50/project/image_classification/data/2024-12-12_缺點圖片收集/Mark/2597.jpg"
    # result = MM.start_inference_single(filename)
    # result = MM.start_inference_single(filename)
    # result = MM.start_inference_single(filename)
    # result = MM.start_inference_single(filename)
    image = cv2.imread(filename)
    image2 = cv2.imread(filename2)
    result = MM.start_inference_single_thread(callback)
    time.sleep(1)
    MM.send_image(image, "123")
    time.sleep(1)
    MM.send_image(image2, "456")
    time.sleep(1)
    MM.close_inference()
    



        
