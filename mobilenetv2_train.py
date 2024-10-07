import torchvision.models as models
from torchvision.models import mobilenet_v3_small
from torchvision.models import MobileNet_V3_Small_Weights
from torch import nn
from MyDataset import MyImageDataset
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import torch

def show_image():
    # check data
    image, label = train[2]
    image = image.permute(1, 2, 0).numpy()
    print(image.shape)
    plt.imshow(image)
    plt.show()



# Load model
model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
model.classifier[3] = nn.Linear(1024,2)
print(model)

# root_dir_train = "/home/ray/project/image_classification/data/image/train"
# root_dir_test = "/home/ray/project/image_classification/data/image/test"

# train_augmentation = transforms.Compose(
#     [
#         transforms.Resize((224,224)),
#         transforms.RandomHorizontalFlip(p=0.99),
#         transforms.RandomVerticalFlip(p=1),
#         transforms.ToTensor()
#     ]
# )
# test_augmentation = transforms.Compose(
#     [
#         transforms.Resize((224,224)),
#         transforms.ToTensor()
#     ]
# )


# train = MyImageDataset(root_dir_train,transform=train_augmentation)
# test = MyImageDataset(root_dir_test, transform=test_augmentation)

# batch_size = 100
# train_loader = DataLoader(train, batch_size=batch_size)
# test_loader = DataLoader(test, batch_size=batch_size)


# loss_fn = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

# device = (
#     "cuda"
#     if torch.cuda.is_available()
#     else "mps"
#     if torch.backends.mps.is_available()
#     else "cpu"
# )
# print(f"Using {device} device")


# def train(dataloader, model, loss_fn, optimizer):
#     size = len(dataloader.dataset)
#     model.train()
#     for batch, (X, y) in enumerate(dataloader):
#         X, y = X.to(device), y.to(device)
        
#         # Compute prediction error
#         pred = model(X)
#         loss = loss_fn(pred, y)

#         # Backpropagation
#         loss.backward()
#         optimizer.step()
#         optimizer.zero_grad()

#         if batch % 100 == 0:
#             loss, current = loss.item(), (batch + 1) * len(X)
#             print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


# def test(dataloader, model, loss_fn):
#     size = len(dataloader.dataset)
#     num_batches = len(dataloader)
#     model.eval()
#     test_loss, correct = 0, 0
#     with torch.no_grad():
#         for X, y in dataloader:
#             X, y = X.to(device), y.to(device)
#             pred = model(X)
#             test_loss += loss_fn(pred, y).item()
#             correct += (pred.argmax(1) == y).type(torch.float).sum().item()
#     test_loss /= num_batches
#     correct /= size
#     print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


# if __name__ == "__main__":
#     epochs = 100
#     for t in range(epochs):
#         print(f"Epoch {t+1}\n-------------------------------")
#         train(train_loader, model, loss_fn, optimizer)
#         test(test_loader, model, loss_fn)
#     print("Done!")

