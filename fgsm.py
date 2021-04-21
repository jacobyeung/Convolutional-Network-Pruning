import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
from torchvision.utils import save_image
import numpy as np
from pathlib import Path
from tqdm import tqdm
import os
import shutil

epsilon = 0.05  # TODO: tune this hyperparameter if needed
model_path = 'outputs/experiment_101/data/101_101/101_101_model_valid_accuracy=0.6475.pt'  # TODO: change to model path
data_path = 'data/tiny-imagenet-200/train'  # TODO: change to data path
class_names = os.listdir(data_path)
data_dir = Path(data_path)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = torchvision.models.resnet50()
model.fc = nn.Linear(2048, 200)  # TODO: change these two lines to match model type
model = model.to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

print('model loaded')

batch_size = 500  # TODO: change to number of images per class, code below assumes each batch comes from same class

norm_data_transforms = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0, 0, 0), tuple(np.sqrt((255, 255, 255))))])
norm_train_set = torchvision.datasets.ImageFolder(data_dir, norm_data_transforms)
norm_train_loader = torch.utils.data.DataLoader(norm_train_set, batch_size=batch_size, shuffle=False, pin_memory=True)
norm_train_loader = iter(norm_train_loader)

orig_train_set = torchvision.datasets.ImageFolder(data_dir, transforms.ToTensor())
orig_train_loader = torch.utils.data.DataLoader(orig_train_set, batch_size=batch_size, shuffle=False, pin_memory=True)
orig_train_loader = iter(orig_train_loader)

print('dataset loaded')

save_path = 'data/tiny-imagenet-200/adversarial'  # TODO: change this to desired save location
save_path = Path(save_path)
if os.path.exists(save_path):
    shutil.rmtree(save_path)
os.mkdir(save_path)

softmax = nn.Softmax(dim=1)
loss_fn = nn.CrossEntropyLoss()

def attack(images, epsilon, image_grads):
    sign_image_grads = image_grads.sign()
    adversarial_images = images + epsilon * sign_image_grads
    adversarial_images = torch.clamp(adversarial_images, 0, 1)
    return adversarial_images


for i in tqdm(range(200)):
    # since all images from the same batch come from the same class, i is just the label
    data = next(norm_train_loader)
    original_images = next(orig_train_loader)[0].to(device)
    images = data[0].to(device)  # data[1] is the label
    label = torch.ones(batch_size, dtype=torch.long).to(device) * i
    images.requires_grad = True
    pred = softmax(model(images))
    loss = loss_fn(pred, label)
    loss.backward()
    images_grads = images.grad.data
    adversarial_images = attack(original_images, epsilon, images_grads)
    class_name = class_names[i]
    save_folder = save_path/class_name
    os.mkdir(save_folder)
    for img_num in range(adversarial_images.shape[0]):
        adv_img = adversarial_images[img_num]
        img_name = class_name + '_adv_' + str(img_num) + '.JPEG'
        save_image(adv_img, save_folder/img_name)


