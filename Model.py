import torchvision.models as models
from torchvision.models import mobilenet_v3_small
from torchvision.models import MobileNet_V3_Small_Weights
from torchvision.models import efficientnet_v2_l
from torchvision.models import EfficientNet_V2_L_Weights
from torch import nn
from MyDataset import MyImageDataset
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import torch
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import cv2
import time

class MyModel():
    def __init__(self, output_class, batch_size=100, lr=0.01):
        self.output_class = output_class
        self.batch_size = batch_size
        self.learningrate = lr
        self.writer = SummaryWriter("./log")
        self.training_device = "cuda:0"
        self.iterate_time = 0
        print(f"Using {self.training_device} device")

    # def load_model(self):
    #     # Load model
    #     self.model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
    #     self.model.classifier = nn.Sequential(
    #         nn.Linear(576, 1024),  # 第一層全連接層：1024 -> 512
    #         nn.ReLU(),             # ReLU 激活函數
    #         nn.Dropout(0.5),       # Dropout 防止過擬合
    #         nn.Linear(1024, 256),  # 第二層全連接層：512 -> 2（假設是 2 類分類）
    #         nn.ReLU(),             # ReLU 激活函數
    #         nn.Dropout(0.5),       # Dropout 防止過擬合
    #         nn.Linear(256, self.output_class)
    #     )
    #     print(self.model)

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
        # print(self.model)

    def model_freeze(self):
        layers = 0
        # for name, param in self.model.named_parameters():
        #     layers+=1
        #     print(f"Layer: {name} | Size: {param.size()} | Requires Grad: {param.requires_grad}")
        # print("layers = ",layers)
        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.model.classifier[0:].parameters():
            param.requires_grad = True

        # for name, param in self.model.named_parameters():
        #     layers+=1
        #     print(f"Layer: {name} | Size: {param.size()} | Requires Grad: {param.requires_grad}")
        # print("layers = ",layers)
    
    def load_single_img(self, image_path):
        """
        load single image 
        """
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # 讀入影像並進行預處理
        image = Image.open(image_path).convert('RGB')
        input_tensor = transform(image).unsqueeze(0)  # 新增 batch 維度
        input_tensor = input_tensor.to(self.training_device)
        return input_tensor
    
    
    def load_dataset(self, root_dir_train, root_dir_test):
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
        test_augmentation = transforms.Compose(
            [
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
        )

        train = MyImageDataset(root_dir_train,transform=train_augmentation)
        train.save_label_pare() # save train label and class name pare
        test = MyImageDataset(root_dir_test, transform=test_augmentation)
        self.train_loader = DataLoader(train, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(test, batch_size=self.batch_size)

        # image, label = train[5]
        # image_np = image.permute(1, 2, 0).numpy()
        # image_np = (image_np * 255).astype(np.uint8)
        # print(image_np.shape)
        # print(label)
        # plt.figure("123")
        # plt.imshow(image_np)
        # plt.show()


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
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        model = model.to(self.training_device)
        model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(self.training_device), y.to(self.training_device)
                pred = model(X)
                probabilities = torch.softmax(pred,dim = 1)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= size 
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
        return (100*correct)
    
    # def inference_image(self, image):
    #     self.load_model()
    #     self.model = self.model.to(self.training_device)
    #     self.model.eval()
        
    #     preprocess = transforms.Compose(            [
    #             transforms.Resize((224,224)),
    #             transforms.ToTensor(),
    #             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    #         ])
    #     input_tensor = preprocess(image)
    #     input_batch = input_tensor.unsqueeze(0) 
    #     with torch.no_grad():
    #         input_batch = input_batch.to(self.training_device)
    #         output = self.model(input_batch)
    #     probabilities = torch.nn.functional.softmax(output[0], dim=0)
    #     print(len(probabilities))
    #     print(max(probabilities))

    #     # Read the categories
    #     with open("imagenet_classes.txt", "r") as f:
    #         categories = [s.strip() for s in f.readlines()]
    #     # Show top categories per image
    #     top5_prob, top5_catid = torch.topk(probabilities, 5)
    #     for i in range(top5_prob.size(0)):
    #         print(categories[top5_catid[i]], top5_prob[i].item())
                
    
    
    def save_model(self, model_name):
        model_name = model_name + ".pth"
        torch.save(self.model.state_dict(), model_name)
    
    def start_train(self,train_dataset, val_dataset, epoch, model_name):
        self.load_model()
        self.model_freeze()

        self.load_dataset(train_dataset, val_dataset)
        self.init_loss_optimizer()
        self.acc_pre = 0
        for t in range(epoch):
            t1 = time.time()
            print(f"Epoch {t+1}\n-------------------------------")
            loss = self.train(self.train_loader, self.model, self.loss_fn, self.optimizer)
            
            if t%20==0:
                acc = self.validation(self.val_loader, self.model, self.loss_fn)
                if self.acc_pre>= acc:
                    self.save_model(model_name)
                
                self.writer.add_scalar("accuracy", acc, t)
                
            self.writer.add_scalar("loss", loss, t)
            
            self.scheduler.step()
            self.iterate_time = time.time() - t1 # for estimating training time
        print("Done!")
        self.writer.close()
        


    def start_inference(self, test_Dataset):
        """
        batch inference
        """

        self.load_model()
        self.model.load_state_dict(torch.load('model_weight.pth', weights_only=True))
        self.init_loss_optimizer()
        self.load_testdata(test_Dataset)
        acc = self.inference_test(self.val_loader, self.model, self.loss_fn)
        print(acc)
    
    def start_inference_single(self, test_image):
        """
        single inference
        
        """
        
        img_tensor = self.load_single_img(test_image)
        
        self.load_model()
        self.model.load_state_dict(torch.load('model_weight.pth', weights_only=True))
        self.model.to(self.training_device)
        self.model.eval()
        
        with torch.no_grad():  # 關閉梯度計算以加速推論
            
            for i in range(1000):
                s1 = time.time()
                output = self.model(img_tensor)
                _, predicted_class = torch.max(output, 1)
                print("class = ", predicted_class)

        
    
    

if __name__=="__main__":
    output_class = 10
    batch_size = 100
    lr = 0.0001
    save_model_name = "ft_model"
    MM = MyModel(output_class, batch_size, lr)
    
    # root_dir_train = "/home/trx50/project/image_classification/data/vechicles/train"
    # root_dir_test = "/home/trx50/project/image_classification/data/vechicles/test"
    # MM.load_dataset(root_dir_train,root_dir_test)
    # train_layer = 100
    # epoch = 500
    # MM.start_train(root_dir_train, root_dir_test, epoch, save_model_name)
    
    # batch inference
    # test = "/home/trx50/project/image_classification/data/vechicles/test"
    # MM.start_inference(test)
    
    
    # inference single image
    filename = "/home/trx50/project/image_classification/data/vechicles/train/truck/images (22).jpg"
    MM.start_inference_single(filename)
    
    



        
