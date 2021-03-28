import argparse
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torchvision
import torchvision.transforms as transforms
from train_loop import train_loop
from models import ResNet50


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=int,
                        help="Experiment number")
    parser.add_argument('--data_path', type=str, default='./data/tiny-imagenet-200',
                        help="path to data")
    parser.add_argument('--model_path', type=str, default=None,
                        help="model path")
    parser.add_argument('--seed', type=int,
                        help="numpy and pytorch seed")
    parser.add_argument('--device', type=str, default='cpu',
                        help='cpu or cuda')
    parser.add_argument('-f', '--ff', help="Dummy arg")
    args = vars(parser.parse_args())

    experiment = args['experiment']
    data_path = args['data_path']
    model_path = args['model_path']
    seed = args['seed']
    device = args['device']
    batch_size = 128
    model_id = f"{experiment}_{seed}"

    rng = np.random.RandomState(seed)
    int_info = np.iinfo(int)
    torch.manual_seed(rng.randint(int_info.min, int_info.max))

    model = torchvision.models.resnet50(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(2048, 200)
    model.to(device)
    # model.load_state_dict(torch.load(model_path))

    data_dir = Path(data_path)
    image_count = len(list(data_dir.glob('**/*.JPEG')))
    CLASS_NAMES = np.array(
        [item.name for item in (data_dir / 'train').glob('*')])
    print('Discovered {} images'.format(image_count))

    data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0, 0, 0), tuple(np.sqrt((255, 255, 255)))),
    ])

    train_set = torchvision.datasets.ImageFolder(
        data_dir / 'train', data_transforms)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                               shuffle=True, num_workers=4, pin_memory=True)
    valid_set = torchvision.datasets.ImageFolder(
        data_dir / 'val', data_transforms)
    valid_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                               shuffle=True, num_workers=4, pin_memory=True)
    ds = [train_loader, valid_loader]
    batch_size = 32
    im_height = 64
    im_width = 64
    num_epochs = 1

    params = {}
    params['lr'] = 0.00001
    params['momentum'] = 0.98
    params['l2_wd'] = 3.4e-5

    Path('outputs').mkdir(exist_ok=True)
    base = f"outputs/experiment_{experiment}"
    Path(base).mkdir(exist_ok=True)
    base_data = f"{base}/data"
    Path(base_data).mkdir(exist_ok=True)

    train_loop(model, params, ds, base_data, model_id, device=device)


if __name__ == '__main__':
    main()
