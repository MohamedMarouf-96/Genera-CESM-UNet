import torch
from torch import nn
from .unet_parts import ConfidenceScorer3D

class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv3d(in_size, out_size, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.BatchNorm3d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # print('in', x.shape)
        # print('out', self.model(x).shape)
        return self.model(x)



class UNetMid(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetMid, self).__init__()
        layers = [
            nn.Conv3d(in_size, out_size, 4, 1, 1, bias=False),
            nn.BatchNorm3d(out_size),
            nn.LeakyReLU(0.2)
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)


    def forward(self, x, skip_input):
        # print(x.shape)
        x = torch.cat((x, skip_input), 1)
        x = self.model(x)
        x =  nn.functional.pad(x, (1,0,1,0,1,0))

        return x

class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose3d(in_size, out_size, 4, 2, 1, bias=False),
            nn.BatchNorm3d(out_size),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        # print('new')
        # print(x.shape)
        # print(skip_input.shape)
        x = self.model(x)
        # print(x.shape)
        x = torch.cat((x, skip_input), 1)

        return x


class Vox2Vox(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(Vox2Vox, self).__init__()

        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512)
        self.mid1 = UNetMid(1024, 512, dropout=0.2)
        self.mid2 = UNetMid(1024, 512, dropout=0.2)
        self.mid3 = UNetMid(1024, 512, dropout=0.2)
        self.mid4 = UNetMid(1024, 256, dropout=0.2)
        self.up1 = UNetUp(256, 256)
        self.up2 = UNetUp(512, 128)
        self.up3 = UNetUp(256, 64)
        self.confidence_generator = ConfidenceScorer3D(256,256)
        # self.us =   nn.Upsample(scale_factor=2)

        self.final = nn.Sequential(
            # nn.Conv3d(128, out_channels, 4, padding=1),
            # nn.Tanh(),
            nn.ConvTranspose3d(128, out_channels, 4, 2, 1),
        )

    def forward(self, x):
        # U-Net generator with skip connections from encoder to decoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        m1 = self.mid1(d4, d4)
        m2 = self.mid2(m1, m1)
        m3 = self.mid3(m2, m2)
        m4 = self.mid4(m3, m3)
        confidence = self.confidence_generator(m4.detach())
        u1 = self.up1(m4, d3)
        u2 = self.up2(u1, d2)
        u3 = self.up3(u2, d1)
        # u7 = self.up7(u6, d1)
        # u7 = self.us(u7)
        # u7 = nn.functional.pad(u7, pad=(1,0,1,0,1,0))
        # # print(self.final(u7).shape)
        return self.final(u3), confidence