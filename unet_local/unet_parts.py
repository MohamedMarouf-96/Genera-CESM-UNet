""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False)
        )

    def forward(self, x):
        return self.double_conv(x)


class DoubleConv3d(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Down3d(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv3d(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class Up3d(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            self.conv = DoubleConv3d(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv3d(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class OutConv3d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv3d, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class ConfidenceScorer2D(nn.Module):
    def __init__(self,input_channels,intermediate_channels):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(input_channels,intermediate_channels,3,2,1,bias=False),nn.BatchNorm2d(intermediate_channels),
            nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(intermediate_channels,intermediate_channels,3,2,1,bias=False),nn.BatchNorm2d(intermediate_channels),
            nn.ReLU(inplace=True))
        self.fc = nn.Linear(intermediate_channels,intermediate_channels)
        self.last = nn.Linear(intermediate_channels,1)

    def forward(self,x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = out.mean(dim=(2,3))
        out = out.squeeze(-1).squeeze(-1)
        out = self.fc(out)
        out = self.last(out)
        return torch.sigmoid(out.unsqueeze(-1).unsqueeze(-1))


class ConfidenceScorer3D(nn.Module):
    def __init__(self,input_channels,intermediate_channels):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv3d(input_channels,intermediate_channels,3,2,1,bias=False),nn.BatchNorm3d(intermediate_channels),
            nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv3d(intermediate_channels,intermediate_channels,3,2,1,bias=False),nn.BatchNorm3d(intermediate_channels),
            nn.ReLU(inplace=True))
        self.fc_patient = nn.Linear(intermediate_channels,intermediate_channels)
        self.last_patient = nn.Linear(intermediate_channels,1)

    def forward(self,x):
        out = self.conv1(x)
        out = self.conv2(out)

        out_patient = out.mean(dim=(2,3,4))
        out_patient = self.fc_patient(out_patient)
        out_patient = self.last_patient(out_patient)
        out_patient = torch.sigmoid(out_patient.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1))
        return out_patient

        
class ConfidenceScorer3DSlice(nn.Module):
    def __init__(self,input_channels,intermediate_channels):
        super().__init__()
        self.fc_slice = nn.Linear(input_channels,intermediate_channels)
        self.last_slice = nn.Linear(intermediate_channels,1)

    def forward(self,x):
        out = x.mean(dim = (3,4)).transpose(1,2)
        out = self.fc_slice(out)
        out = self.last_slice(out)
        out = torch.sigmoid(out).unsqueeze(-1).unsqueeze(-1).transpose(1,2)
        return out