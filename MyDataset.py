import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt

class MyImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # 假設資料夾名稱是類別標籤
        for label, class_dir in enumerate(os.listdir(root_dir)):
            class_path = os.path.join(root_dir, class_dir)
            for img_file in os.listdir(class_path):
                self.image_paths.append(os.path.join(class_path, img_file))
                self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

if __name__=="__main__":

    root_dir_train = "/home/ray/project/image_classification/data/vechicles/train"
    data = MyImageDataset(root_dir_train)
    image, label = data[108]
    # image = image.permute(1, 2, 0).numpy()
    # print(image.shape)
    print(label)
    plt.imshow(image)
    plt.show()