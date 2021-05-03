import argparse
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torchvision
import torchvision.transforms as transforms
from train_loop import train_loop
from adv_train_loop import adv_train_loop
from models import ResNet50


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=int,
                        help="Experiment number")
    parser.add_argument('--data_path', type=str, default='tiny-imagenet-200',
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
    batch_size = 32
    model_id = f"{experiment}_{seed}"
    num_workers = 0

    rng = np.random.RandomState(seed)
    int_info = np.iinfo(int)
    torch.manual_seed(rng.randint(int_info.min, int_info.max))
    # model = EN.from_pretrained('efficientnet-b5', num_classes = 200)
    model = torchvision.models.wide_resnet50_2(pretrained=True)
    if model_path:
        model.load_state_dict(torch.load(model_path)['model'])
    else:
        model.fc = nn.Linear(2048, 200)
    model.to(device)

    data_dir = Path(data_path)
    image_count = len(list(data_dir.glob('**/*.JPEG')))
    CLASS_NAMES = np.array(
        [item.name for item in (data_dir / 'train').glob('*')])
    print('Discovered {} images'.format(image_count))

    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.4805, 0.4483, 0.3978), (0.263, 0.257, 0.267)),
    ])

    train_set = torchvision.datasets.ImageFolder(
        data_dir / 'train', data_transforms)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=1,
                                               shuffle=False, pin_memory=True)
    lowest_train_label = next(iter(train_loader))[1].item()  # sometimes the labels are not zero-indexed
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                               shuffle=True, num_workers=num_workers, pin_memory=True)

    valid_set = torchvision.datasets.ImageFolder(
        data_dir / 'val', data_transforms)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=1,
                                               shuffle=False, pin_memory=True)
    lowest_valid_label = next(iter(valid_loader))[1].item()
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size,
                                               shuffle=False, num_workers=num_workers, pin_memory=True)

    ds = [train_loader, valid_loader]
    min_y = [lowest_train_label, lowest_valid_label]
    batch_size = 32
    im_height = 64
    im_width = 64
    num_epochs = 1

    params = {}
    params['lr'] = 1e-4
    params['momentum'] = 0.98
    params['l2_wd'] = 3.4e-5
    params['batch_size'] = batch_size
    Path('outputs').mkdir(exist_ok=True)
    base = f"outputs/experiment_{experiment}"
    Path(base).mkdir(exist_ok=True)
    base_data = f"{base}/data"
    Path(base_data).mkdir(exist_ok=True)

    print('\nstarting training on augmented dataset\n')

    train_loop(model, params, ds, min_y, base_data, model_id, device, batch_size, 1)

    print('\nstarting training adversarial models\n')


#     for attack_type in ["fgsm", "bim", "carlini", "deepfool"]:
#         adv_train_loop(model, params, ds, base_data, model_id, attack_type, device, batch_size, 1)

#     adv_train_loop(model, params, ds, min_y, base_data, model_id, 'bim', device, batch_size, 1)


if __name__ == '__main__':
    main()
