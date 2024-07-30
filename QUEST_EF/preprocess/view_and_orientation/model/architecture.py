import torch
import torch.nn as nn
import torchvision.models as models

class ResNeXt50Module(nn.Module):
    def __init__(self, num_classes):
        super(ResNeXt50Module, self).__init__()
        self.out_dim = num_classes
        self.resnext50 = models.resnext50_32x4d(pretrained=True)
        self.resnext50.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.resnext50(x)
        return x
