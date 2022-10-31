import torch
import torch.nn as nn 
from torchvision.models.mobilenetv3 import mobilenet_v3_small


class NetBase(nn.Module):
    def __init__(self):
        super().__init__()


    def forward(self, img, label):
        
        x_img = self.forward_image(img)
        x_label = self.forward_label(label)
        joined = self.join_image_label(x_img, x_label)
        out = self.regression(joined)

        return out


    def forward_image(self, x):
        pass

    def forward_label(self, x):
        pass

    def join_image_label(self, intermidiery, label):
        pass

    def regression(self, x):
        pass



class AutoCorrectorBaseLine(NetBase):
    def __init__(self, num_labels=1):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=7),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.label_embed = nn.Embedding(num_embeddings=num_labels, embedding_dim=50)

        self.pre_regression_layers = nn.Sequential(
            nn.Linear(64*59*59, 1024),
            nn.ReLU()
        )

        self.reg_layers = nn.Sequential(
            nn.Dropout(.5),
            nn.Linear(1024 + 50, 512),
            nn.Dropout(.5),
            nn.ReLU(),
            nn.Linear(512, 4)      
        )


    def forward_image(self, x):
        x = self.conv_layers(x)
        x = x.view(x.shape[0], -1)
        x = self.pre_regression_layers(x)
        return x


    def forward_label(self, label):
        return self.label_embed(label)


    def join_image_label(self, inter, label):
        return torch.cat([inter, label], axis=1)


    def regression(self, x):
        return self.reg_layers(x)


class MobileNet(NetBase):
    def __init__(self, num_labels=1):
        super().__init__()

        self.conv_layers = nn.Sequential(
            mobilenet_v3_small(pretrained=True, requires_grad=False, eval=True).features,
            nn.AdaptiveAvgPool2d(output_size=1),
        )

        self.label_embed = nn.Embedding(num_embeddings=num_labels, embedding_dim=50)

        self.pre_regression_layers = nn.Sequential(
            nn.Linear(576, 1024),
            nn.ReLU(),
            nn.Dropout(.2)
        )

        self.reg_layers = nn.Sequential(
            nn.Linear(1024 + 50, 512),
            nn.ReLU(),
            nn.Dropout(.2),
            nn.Linear(512, 4)      
        )


    def forward_image(self, x):
        x = self.conv_layers(x)
        x = x.view(x.shape[0], -1)
        x = self.pre_regression_layers(x)
        return x


    def forward_label(self, label):
        return self.label_embed(label)


    def join_image_label(self, inter, label):
        return torch.cat([inter, label], axis=1)


    def regression(self, x):
        return self.reg_layers(x)


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
    # x = torch.zeros((1, 3, 256, 256))

    # n = AutoCorrectorBaseLine()

    # print(n(x, torch.Tensor([0]).long()))

    print(fetch_model('autosomthing baseline'))