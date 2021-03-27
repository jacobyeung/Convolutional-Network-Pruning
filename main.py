import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms


def main():
    data_dir = pathlib.Path('./data/tiny-imagenet-200')
    image_count = len(list(data_dir.glob('**/*.JPEG')))
    CLASS_NAMES = np.array(
        [item.name for item in (data_dir / 'train').glob('*')])
    print('Discovered {} images'.format(image_count))

    train_set = torchvision.datasets.ImageFolder(
        data_dir / 'train', data_transforms)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                               shuffle=True, num_workers=4, pin_memory=True)

    batch_size = 32
    im_height = 64
    im_width = 64
    num_epochs = 1

    data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0, 0, 0), tuple(np.sqrt((255, 255, 255)))),
    ])
