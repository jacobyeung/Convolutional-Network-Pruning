import sys
import csv
import pathlib
import torch
import torchvision
from torch import nn
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

device = 'cuda:0'

data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.4805, 0.4483, 0.3978), (0.263, 0.257, 0.267)),
])
softmax = nn.Softmax(dim=1)
data_dir = pathlib.Path('./data/tiny-imagenet-200/train/')
CLASSES = sorted([item.name for item in data_dir.glob('*')])


def predict(image_path, model_list):
    image = data_transforms(Image.open(image_path).convert('RGB')).unsqueeze(0).to(device)
    scores = torch.zeros((1, 200)).to(device)
    for model in model_list:
        pred = softmax(model(image))
        scores += pred
    return torch.argmax(scores).item()


model_names = ['resnet50_orig', 'resnet50_adv',
               'resnet101_orig', 'resnet101_adv',
               'wide_resnet50_orig', 'wide_resnet50_adv']

models = [torchvision.models.resnet50(), torchvision.models.resnet50(),
          torchvision.models.resnet101(), torchvision.models.resnet101(),
          torchvision.models.wide_resnet50_2(), torchvision.models.wide_resnet50_2()]

for i in range(6):
    model = models[i]
    model_name = model_names[i]
    model.fc = nn.Linear(2048, 200)
    model.load_state_dict(torch.load(f'outputs/best_models/{model_name}.pt'))
    model.to(device)
    model.eval()

input_file = open(sys.argv[1])
output_file = open('eval_classified.csv', 'w', newline='')
reader = csv.reader(input_file)
writer = csv.writer(output_file)
for row in tqdm(reader):
    image_id = row[0]
    image_path = row[1]
    pred = predict(image_path, models)
    writer.writerow([image_id, CLASSES[pred]])
