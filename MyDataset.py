import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt

class MyImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_name = []
        self.image_paths = []
        self.labels = []
        self.label_name= []
        

        # 假設資料夾名稱是類別標籤
        for label, class_dir in enumerate(os.listdir(root_dir)):
            class_path = os.path.join(root_dir, class_dir)
            self.label_name.append(class_dir)
            for img_file in os.listdir(class_path):
                self.image_name.append(img_file)
                self.image_paths.append(os.path.join(class_path, img_file))
                self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]
        image_name = self.image_name[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
    
    def save_label_pare(self):
        my_dict = {i: self.label_name[i] for i in range(len(self.label_name))}
        with open('output.txt', 'w') as file:
            for key, value in my_dict.items():
                file.write(f"{key}: {value}\n")

        print("List has been saved to output.txt")
        
        

if __name__=="__main__":

    root_dir_train = "/home/trx50/project/image_classification/data/vechicles/train"
    data = MyImageDataset(root_dir_train)
    image, label, label_list= data[108]
    data.save_label_pare()
    # image = image.permute(1, 2, 0).numpy()
    # print(image.shape)
    print(label)
    print(label_list)
    plt.imshow(image)
    plt.show()