from PIL import Image
import torchvision
import random
import os
from pathlib import *
from os import walk

directory_path = "data/tiny-imagenet-200/train"
files = []
for (dirpath, dirnames, filenames) in walk(directory_path):
    for file in filenames:
        if file[-5:] == ".JPEG":
            files.append(file)


def transform(file):
    file = "data/tiny-imagenet-200/train/" + file[:9] + "/images/" + file
    im = Image.open(file)
    resize = torchvision.transforms.Resize((64, 64))
    colorjitter1 = torchvision.transforms.ColorJitter(0.4, 0.3, 0.2, 0.2)
    greyscale2 = torchvision.transforms.RandomGrayscale(1)
    rot3 = torchvision.transforms.RandomRotation(25)
    centercrop4 = torchvision.transforms.Compose([torchvision.transforms.CenterCrop(random.randint(50, 60)),
                                                  resize])
    colorjitter5 = torchvision.transforms.ColorJitter(0.3, 0.4, 0.3, 0.3)
    #gaus6 = torchvision.transforms.GaussianBlur(3)
    fliph7 = torchvision.transforms.RandomHorizontalFlip(0.9)
    flipv7 = torchvision.transforms.RandomVerticalFlip(0.1)
    flip7 = torchvision.transforms.Compose([fliph7, flipv7])
    crop8 = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop((random.randint(50, 60), random.randint(50, 60))),
        resize])

    transforms = [colorjitter1, greyscale2, rot3, centercrop4, colorjitter5, flip7, crop8]

    rand9 = torchvision.transforms.Compose(random.sample(transforms, 3))
    rand10 = torchvision.transforms.Compose(random.sample(transforms, random.randint(2, 5)))

    transforms.extend([rand9, rand10])

    for i in range(9):
        new_im = transforms[i](im)
        new_im.save(file[:-5] + "_" + str(i) + ".JPEG")


i = 0
for file in files:
    i += 1
    if i % 1000 == 0:
        print(str(i // 1000) + "/100")
    transform(file)
