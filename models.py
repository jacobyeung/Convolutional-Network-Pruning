import torchvision
import torchvision.models
from torch import nn
class ResNet50:
    def __init__(self, dim):
        model = torchvision.models.resnet50(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        model.fc = nn.Linear(2048, dim)
        self = model