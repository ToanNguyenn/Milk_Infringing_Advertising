from torch import nn
from efficientNet import EfficientNet
from resnet import ResNet

class Net(nn.Module):
    def __init__(self, net_version, num_classes):
        super(Net, self).__init__()
        self.backbone = EfficientNet(version= net_version, num_classes=num_classes)
        self.backbone._fc = nn.Sequential(
            nn.Linear(1280, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)