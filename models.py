import torch
import torch.nn as nn 
from torchvision.models.mobilenetv3 import mobilenet_v3_small


class AutoCorrectorBaseLine(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=7),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )


        self.pre_regression_layers = nn.Sequential(
            nn.Linear(64*59*59, 1024),
            nn.ReLU()
        )

        self.reg_layers = nn.Sequential(
            nn.Dropout(.5),
            nn.Linear(1024, 512),
            nn.Dropout(.3),
            nn.ReLU(),
            nn.Linear(512, 4)      
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.shape[0], -1)
        x = self.pre_regression_layers(x)
        x = self.reg_layers(x)

        return x


class MobileNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_layers = nn.Sequential(
            mobilenet_v3_small(pretrained=True, requires_grad=False, eval=True).features,
            nn.AdaptiveAvgPool2d(output_size=1),
        )

        self.pre_regression_layers = nn.Sequential(
            nn.Linear(576, 1024),
            nn.ReLU(),
            nn.Dropout(.2)
        )

        self.reg_layers = nn.Sequential(
            nn.Linear(1024 + 50, 512),
            nn.ReLU(),
            nn.Dropout(.3),
            nn.Linear(512, 4)      
        )


    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.shape[0], -1)
        x = self.pre_regression_layers(x)
        x = self.reg_layers(x)

        return x


def fetch_model(name):
    name = name.lower()
    models = {
        'baseline': AutoCorrectorBaseLine,
        'mobilenet': MobileNet,

    }

    for model_name in list(models.keys()):
        if model_name in name:
            return models[model_name]

    
    raise ValueError(f'`{name}` is not a model.\n\t Available -> {list(models.keys())}')


if __name__ == '__main__':
    x = torch.zeros((1, 3, 256, 256))

    n = AutoCorrectorBaseLine()

    print(n(x))

    print(fetch_model('autosomthing baseline'))