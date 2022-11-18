import torch
import torch.nn as nn 
from torchvision.models.mobilenetv3 import mobilenet_v3_small
from torchvision.models import vgg16
from transforms import vgg_transform


class ImageSelector:
    def __init__(self, rho=.2) -> None:
        self.rho = rho
        self.transform = vgg_transform
        self.model = vgg16(pretrained=True).eval()
        self.model.requires_grad_(False)

    def predict(self, x):

        x = self.transform(x).unsqueeze(0)

        x = self.model(x).softmax(1)

        return x

    def is_a_good_image(self, x):
        p = self.predict(x).topk(1).values

        is_good = (p>=self.rho).tolist()

        return is_good

    def __call__(self, x):
        return self.is_a_good_image(x)
        


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


class SmallNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=20, stride=3, kernel_size=7),
            nn.BatchNorm2d(20),
            nn.ReLU(),

            nn.Conv2d(in_channels=20, out_channels=30,  stride=3, kernel_size=7),
            nn.BatchNorm2d(30),
            nn.ReLU(),

            nn.Conv2d(in_channels=30, out_channels=38, stride=2, kernel_size=5),
            nn.BatchNorm2d(38),
            nn.ReLU(),

            nn.MaxPool2d(2,2),
        )

        self.fc = nn.Sequential(
            nn.Linear(38*5*5, 1024),
            nn.ReLU(), 
            nn.Dropout(.5),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(.3),
            nn.Linear(256, 4)
        )



    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)

        return x


class SmallNet2(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=40, stride=2, kernel_size=5, padding=2),
            nn.BatchNorm2d(40),
            nn.ReLU(),

            nn.Conv2d(in_channels=40, out_channels=64,  stride=2, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.MaxPool2d(2,2),


            nn.Conv2d(in_channels=64, out_channels=128, stride=1, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(in_channels=128, out_channels=128, stride=1, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.MaxPool2d(2,2),
        )

        self.fc = nn.Sequential(
            nn.Linear(128*14*14, 4096),
            nn.ReLU(), 
            nn.Dropout(.5),

            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.Dropout(.3),

            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(.2),

            nn.Linear(256, 4)
        )


    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)

        return x


class LessSmallNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU()

        )

        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU()
        )


        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3),
            nn.ReLU(),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3),
            nn.ReLU()
        )


        self.block4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3),
            nn.ReLU(),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3),
            nn.ReLU()
        )


        self.fc = nn.Sequential(
            nn.Linear(256*11*11, 5160),
            nn.ReLU(),
            nn.Linear(5160, 4)
        )

        self.maxpool = nn.MaxPool2d(2,2)


    def forward(self, x):

        x = self.block1(x)
        x = self.maxpool(x)

        x = self.block2(x)
        x = self.maxpool(x)

        x = self.block3(x)
        x = self.maxpool(x)

        x = self.block4(x)
        x = self.maxpool(x)
        
        x = x.view(x.shape[0], -1)

        x = self.fc(x)

        return x



class MobileNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_layers = nn.Sequential(
            mobilenet_v3_small(pretrained=True, requires_grad=False, eval=True).features,
            nn.AdaptiveAvgPool2d(output_size=1),
        )

        self.fc = nn.Linear(576, 4)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)

        return x


class MapModel(nn.Module):
    def __init__(self, output_channels=2):
        super().__init__()

        self.E1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.E2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

        )
        self.E3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

        )
        self.E4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),

        )
        self.E5 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),

        )


        self.D1 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.D2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.D3 = nn.Sequential
        (
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.D4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.D5 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=output_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )


        self.E_maxpool = nn.MaxPool2d(2,2, return_indices=True)
        self.D_maxpool = nn.MaxUnpool2d(2,2)


    def forward(self, x):

        # Encoder

        size_1 = x.shape()[-2:]
        x = self.E1(x)
        ind1 = self.E_maxpool(x)

        size_2 = x.shape()[-2:]
        x = self.E2(x)
        ind2 = self.E_maxpool(x)
        
        size_3 = x.shape()[-2:]
        x = self.E3(x)
        ind3 = self.E_maxpool(x)

        size_4 = x.shape()[-2:]
        x = self.E4(x)
        ind4 = self.E_maxpool(x)

        size_5 = x.shape()[-2:]
        x = self.E5(x)
        ind5 = self.E_maxpool(x)


        # Decoder

        x = self.D_maxpool(x, ind5, output_size=size_5)
        x = self.D1(x)

        x = self.D_maxpool(x, ind4, output_size=size_4)
        x = self.D2(x)

        x = self.D_maxpool(x, ind3, output_size=size_3)
        x = self.D3(x)

        x = self.D_maxpool(x, ind2, output_size=size_2)
        x = self.D4(x)

        x = self.D_maxpool(x, ind1, output_size=size_1)
        x = self.D5(x)


        return x



def fetch_model(name):
    name = name.lower()
    models = {
        'baseline': AutoCorrectorBaseLine,
        'mobilenet': MobileNet,
        'smallnet2': SmallNet2,
        'smallnet': SmallNet,
        'mapmodel': MapModel,
        'map': MapModel,


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
