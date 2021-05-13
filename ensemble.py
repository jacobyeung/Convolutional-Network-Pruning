import sys
import csv
import torch
import torchvision
from torch import nn
from torchvision import transforms
from PIL import Image

device = 'cuda:0'

data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.4805, 0.4483, 0.3978), (0.263, 0.257, 0.267)),
])
softmax = nn.Softmax(dim=1)


def predict(image_path, model_list):
    image = data_transforms(Image.open(image_path).convert('RGB')).unsqueeze(0).to(device)
    scores = torch.zeros((1, 200)).to(device)
    for model in model_list:
        pred = softmax(model(image))
        scores += pred
    return torch.argmax(pred).item()


resnet50 = torchvision.models.resnet50()
resnet50.fc = nn.Linear(2048, 200)
resnet50.load_state_dict(torch.load('outputs/best_models/resnet50.pt'))
resnet50.to(device)

resnet101 = torchvision.models.resnet101()
resnet101.fc = nn.Linear(2048, 200)
resnet101.load_state_dict(torch.load('outputs/best_models/resnet101.pt'))
resnet101.to(device)

input_file = open(sys.argv[1])
output_file = open('eval_classified.csv', 'w', newline='')
reader = csv.reader(input_file)
writer = csv.writer(output_file)
for row in reader:
    image_id = row[0]
    image_path = row[1]
    pred = predict(image_path, [resnet50, resnet101])
    writer.writerow([image_id, pred])

