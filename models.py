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
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

        )
        self.E5 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

        )


        self.D1 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.D2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.D3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.D4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.D5 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=output_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(),
        )


        self.E_maxpool = nn.MaxPool2d(2,2, return_indices=True)
        self.D_maxpool = nn.MaxUnpool2d(2,2)


    def forward(self, x):

        # Encoder

        size_1 = x.size()
        x = self.E1(x)
        x, ind1 = self.E_maxpool(x)

        size_2 = x.size()
        x = self.E2(x)
        x, ind2 = self.E_maxpool(x)
        
        size_3 = x.size()
        x = self.E3(x)
        x, ind3 = self.E_maxpool(x)

        size_4 = x.size()
        x = self.E4(x)
        x, ind4 = self.E_maxpool(x)

        size_5 = x.size()
        x = self.E5(x)
        x, ind5 = self.E_maxpool(x)


        # Decoder

        x = self.D_maxpool(x, indices=ind5, output_size=size_5)
        x = self.D1(x)

        x = self.D_maxpool(x, indices=ind4, output_size=size_4)
        x = self.D2(x)

        x = self.D_maxpool(x, indices=ind3, output_size=size_3)
        x = self.D3(x)

        x = self.D_maxpool(x, indices=ind2, output_size=size_2)
        x = self.D4(x)

        x = self.D_maxpool(x, indices=ind1, output_size=size_1)
        x = self.D5(x)


        return x

class MapModelPretrainedVGG(nn.Module):
    def __init__(self, output_channels=2):
        super().__init__()
        feature_extractor = vgg16(pretrained=True).eval().requires_grad_(False).features

        self.encoder_maxpool = nn.MaxPool2d(2,2, return_indices=True) # fuck vgg maxpool
        self.decoder_maxpool = nn.MaxUnpool2d(2,2)

        #Pretrained encoder

        self.E1 = nn.Sequential(
            feature_extractor[0],
            feature_extractor[1],
            feature_extractor[2],
            feature_extractor[3],
        ).eval().requires_grad_(False)

        self.E2 = nn.Sequential(
            feature_extractor[5],
            feature_extractor[6],
            feature_extractor[7],
            feature_extractor[8],
        ).eval().requires_grad_(False)

        self.E3 = nn.Sequential(
            feature_extractor[10],
            feature_extractor[11],
            feature_extractor[12],
            feature_extractor[13],
            feature_extractor[14],
            feature_extractor[15],
        ).eval().requires_grad_(False)

        self.E4 = nn.Sequential(
            feature_extractor[17],
            feature_extractor[18],
            feature_extractor[19],
            feature_extractor[20],
            feature_extractor[21],
            feature_extractor[22],
        ).eval().requires_grad_(False)

        self.E5 = nn.Sequential(
            feature_extractor[24],
            feature_extractor[25],
            feature_extractor[26],
            feature_extractor[27],
            feature_extractor[28],
            feature_extractor[29],
        ).eval().requires_grad_(False)



        # Decoder does not have to be symetric
        # because the encoder net uses same convolutions to keep the
        # spatial dimensions the same througout the block until a maxpool layer


        self.D1 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        self.D2 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.D3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.D4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.D5 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=output_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(),
        )



    def forward(self, x):
        output1 = x.size()
        x = self.E1(x)
        x, ind1 = self.encoder_maxpool(x)

        output2 = x.size()
        x = self.E2(x)
        x, ind2 = self.encoder_maxpool(x)

        output3 = x.size()
        x = self.E3(x)
        x, ind3 = self.encoder_maxpool(x)

        output4 = x.size()
        x = self.E4(x)
        x, ind4 = self.encoder_maxpool(x)

        output5 = x.size()
        x = self.E5(x)
        x, ind5 = self.encoder_maxpool(x)


        x = self.decoder_maxpool(x, indices=ind5, output_size=output5)
        x = self.D1(x)

        x = self.decoder_maxpool(x, indices=ind4, output_size=output4)
        x = self.D2(x)
        
        x = self.decoder_maxpool(x, indices=ind3, output_size=output3)
        x = self.D3(x)
        
        x = self.decoder_maxpool(x, indices=ind2, output_size=output2)
        x = self.D4(x)
        
        x = self.decoder_maxpool(x, indices=ind1, output_size=output1)
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
        'mapvgg':MapModelPretrainedVGG
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
