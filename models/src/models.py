import torch
from torch import nn
from typing import Tuple

class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.relu(x)
        return x


class DoubleConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.convs = nn.Sequential(
            ConvBlock(in_channels=in_channels, out_channels=out_channels),
            ConvBlock(in_channels=out_channels, out_channels=out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.convs(x)

        return x


class DeConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv(x))
        return x


class WbEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block1 = DoubleConvBlock(3, 24)
        self.block2 = DoubleConvBlock(24, 48)
        self.block3 = DoubleConvBlock(48, 96)
        self.block4 = DoubleConvBlock(96, 192)
        self.end = ConvBlock(192, 384)

        self.maxpool = nn.MaxPool2d(2, 2)

    def forward(self, x: torch.Tensor) -> dict:
        x1 = self.block1(x)
        x = self.maxpool(x1)

        x2 = self.block2(x)
        x = self.maxpool(x2)

        x3 = self.block3(x)
        x = self.maxpool(x3)

        x4 = self.block4(x)
        x = self.maxpool(x4)

        x = self.end(x)

        return {
            "b1": x1,
            "b2": x2,
            "b3": x3,
            "b4": x4,
            "end": x
        }


class WBDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.conv = DoubleConvBlock(
            in_channels=in_channels*2, out_channels=in_channels)
        self.up = DeConvBlock(in_channels=in_channels,
                              out_channels=out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.up(x)

        return x


class WBDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=384,
                      kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=384, out_channels=192, kernel_size=2, stride=2),
            nn.ReLU()
        )

        self.out = nn.Sequential(
            DoubleConvBlock(in_channels=24*2, out_channels=24),
            nn.Conv2d(in_channels=24, out_channels=3, kernel_size=1)
        )

        self.block1 = WBDecoderBlock(192, 96)
        self.block2 = WBDecoderBlock(96, 48)
        self.block3 = WBDecoderBlock(48, 24)

    def forward(self, output_dict: dict) -> torch.Tensor:
        x = self.head(torch.clone(output_dict["end"]))

        x = torch.cat([torch.clone(output_dict["b4"]), x], dim=1)
        x = self.block1(x)

        x = torch.cat([torch.clone(output_dict["b3"]), x], dim=1)
        x = self.block2(x)

        x = torch.cat([torch.clone(output_dict["b2"]), x], dim=1)
        x = self.block3(x)

        x = torch.cat([torch.clone(output_dict["b1"]), x], dim=1)
        x = self.out(x)

        return x


class WBNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.encoder = WbEncoder()
        self.awb_decoder = WBDecoder()
        self.tungsten_decoder = WBDecoder()
        self.shade_decoder = WBDecoder()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        encoded = self.encoder(x)

        awb = self.awb_decoder(encoded)
        tungsten = self.tungsten_decoder(encoded)
        shade = self.shade_decoder(encoded)

        return awb, tungsten, shade


if __name__ == "__main__":
    x = torch.zeros((1, 3, 128, 128))

    net = WBNet()

    print(net.named_parameters().__next__())

    awb, tungsten, shade = net(x)

    print(awb.shape, tungsten.shape, shade.shape)