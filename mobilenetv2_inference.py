import torchvision.models as models

# Load model
mobilenet_v2_model = models.mobilenet_v2(pretrained=True)
print(mobilenet_v2_model)
# Set model to eval mode
mobilenet_v2_model.eval()


from PIL import Image
from torchvision import transforms
filename = "/home/ray/project/image_classification/t1.jpg"
input_image = Image.open(filename)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0) 

import torch

with torch.no_grad():
    output = mobilenet_v2_model(input_batch)
# Tensor of shape 1000, with confidence scores over ImageNet's 1000 classes
# print(output[0])
# The output has unnormalized scores. To get probabilities, you can run a softmax on it.
probabilities = torch.nn.functional.softmax(output[0], dim=0)
print(len(probabilities))
print(max(probabilities))

# Read the categories
with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]
# Show top categories per image
top5_prob, top5_catid = torch.topk(probabilities, 5)
for i in range(top5_prob.size(0)):
    print(categories[top5_catid[i]], top5_prob[i].item())