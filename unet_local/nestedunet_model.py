import torch
from torch import nn
from .unet_parts import ConfidenceScorer2D
from torch.nn import functional as F

class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, norm_layer, noisy = False):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = norm_layer(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = norm_layer(out_channels)
        self.noisy = noisy

    def forward(self, x):
        out = self.conv1(x)
        if self.noisy :
            out = out + torch.randn_like(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if self.noisy :
            out = out + torch.randn_like(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out

class ProgressiveVGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, norm_layer,output_nc = 3,is_final = False,noisy = False,wavelet = False, is_first = False):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = norm_layer(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = norm_layer(out_channels)
        self.to_image = ToImage(out_channels,output_nc,is_final)


    def forward(self, x, image):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        image = self.to_image(out,image)

        return out, image

class ToImage(nn.Module):
    def __init__(self, in_channels, out_channels, is_final = False):
        super().__init__()
        self.conv_image = nn.Conv2d(in_channels, out_channels, kernel_size= 1)
        self.upsample = nn.Upsample(scale_factor=2,mode = 'bilinear',align_corners=True)

    def forward(self, x, image = None):
        x_image = self.conv_image(x)
        if image != None :
            image = (self.upsample(image) + x_image) * (2 ** 0.5)
        else :
            image = x_image

        return image


class NestedUNet(nn.Module):
    def __init__(self, n_channels, n_classes, with_confidence = True):
        super().__init__()
        self.bilinear = True
        self.n_channels = n_channels
        self.n_classes = n_classes

        nb_filter = [32, 64, 128, 256, 512]
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        norm_layer = nn.BatchNorm2d

        self.conv0_0 = VGGBlock(self.n_channels, nb_filter[0], nb_filter[0],norm_layer)
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1],norm_layer)
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2],norm_layer)
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3],norm_layer)
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4],norm_layer)

        self.conv0_1 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0],norm_layer)
        self.conv1_1 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1],norm_layer)
        self.conv2_1 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2],norm_layer)
        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3],norm_layer)

        self.conv0_2 = VGGBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0],norm_layer)
        self.conv1_2 = VGGBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1],norm_layer)
        self.conv2_2 = VGGBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2],norm_layer)

        self.conv0_3 = VGGBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0],norm_layer)
        self.conv1_3 = VGGBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1],norm_layer)

        self.conv0_4 = VGGBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0],norm_layer)
        self.final = nn.Conv2d(nb_filter[0], self.n_classes, kernel_size=1)


        self.with_confidence = with_confidence
        if with_confidence :
            self.confidence_scorer = ConfidenceScorer2D(nb_filter[4] , nb_filter[4])


    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        logits = self.final(x0_4)
        classes = self.confidence_scorer(x4_0)
        return torch.sigmoid(logits), classes


class ProgressiveNestedUNet(nn.Module):
    def __init__(self, n_channels, n_classes, with_confidence = True):
        super().__init__()
        noisy = False
        self.bilinear = True
        nb_filter = [32, 64, 128, 256, 512]
        self.n_channels = n_channels
        self.n_classes = n_classes
        

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        norm_layer = nn.BatchNorm2d
        norm_layer_up = nn.BatchNorm2d

        self.conv0_0 = VGGBlock(n_channels, nb_filter[0], nb_filter[0],norm_layer)
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1],norm_layer)
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2],norm_layer)
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3],norm_layer)
        self.conv4_0 = ProgressiveVGGBlock(nb_filter[3], nb_filter[4], nb_filter[4],norm_layer_up,n_classes)

        self.conv0_1 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0],norm_layer,noisy= noisy)
        self.conv1_1 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1],norm_layer,noisy= noisy)
        self.conv2_1 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2],norm_layer,noisy= noisy)
        self.conv3_1 = ProgressiveVGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3],norm_layer_up,n_classes,noisy= noisy)

        self.conv0_2 = VGGBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0],norm_layer,noisy= noisy)
        self.conv1_2 = VGGBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1],norm_layer,noisy= noisy)
        self.conv2_2 = ProgressiveVGGBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2],norm_layer_up,n_classes,noisy= noisy)

        self.conv0_3 = VGGBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0],norm_layer,noisy= noisy)
        self.conv1_3 = ProgressiveVGGBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1],norm_layer_up,n_classes,noisy= noisy)

        self.conv0_4 = ProgressiveVGGBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0],norm_layer_up,n_classes,noisy= noisy)

        self.with_confidence = with_confidence
        if with_confidence :
            self.confidence_scorer = ConfidenceScorer2D(nb_filter[4] , nb_filter[4])


    def forward(self, input):
        dropout = False
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(F.dropout(x0_0,0.5,self.training and dropout)))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(F.dropout(x1_0,0.5,self.training and dropout)))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(F.dropout(x2_0,0.5,self.training and dropout)))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0, image = self.conv4_0(self.pool(F.dropout(x3_0,0.5,self.training and dropout)), None)
        x3_1, image = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1), image)
        x2_2, image = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1), image)
        x1_3, image = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1), image)
        x0_4, image = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1), image)

        logits = image
        classes = self.confidence_scorer(x4_0)
        return torch.sigmoid(logits), classes

