from telnetlib import DO
import torch
import torch.nn as nn
import pywt
import torch
from torch.autograd import Variable
from .unet_parts import ConfidenceScorer2D, DoubleConv

w=pywt.Wavelet('db1')

dec_hi = torch.Tensor(w.dec_hi[::-1]) 
dec_lo = torch.Tensor(w.dec_lo[::-1])
rec_hi = torch.Tensor(w.rec_hi)
rec_lo = torch.Tensor(w.rec_lo)

filters = torch.stack([dec_lo.unsqueeze(0)*dec_lo.unsqueeze(1)/2.0,
                       dec_lo.unsqueeze(0)*dec_hi.unsqueeze(1),
                       dec_hi.unsqueeze(0)*dec_lo.unsqueeze(1),
                       dec_hi.unsqueeze(0)*dec_hi.unsqueeze(1)], dim=0)

inv_filters = torch.stack([rec_lo.unsqueeze(0)*rec_lo.unsqueeze(1)*2.0,
                           rec_lo.unsqueeze(0)*rec_hi.unsqueeze(1),
                           rec_hi.unsqueeze(0)*rec_lo.unsqueeze(1),
                           rec_hi.unsqueeze(0)*rec_hi.unsqueeze(1)], dim=0)

def wt(vimg):
    padded = vimg
    res = torch.zeros(vimg.shape[0],4*vimg.shape[1],int(vimg.shape[2]/2),int(vimg.shape[3]/2))
    res = res.cuda()
    for i in range(padded.shape[1]):
        res[:,4*i:4*i+4] = torch.nn.functional.conv2d(padded[:,i:i+1], Variable(filters[:,None].cuda(),requires_grad=True),stride=2)
        res[:,4*i+1:4*i+4] = (res[:,4*i+1:4*i+4]+1)/2.0

    return res

def iwt(vres):
    res = torch.zeros(vres.shape[0],int(vres.shape[1]/4),int(vres.shape[2]*2),int(vres.shape[3]*2))
    res = res.cuda()
    for i in range(res.shape[1]):
        vres[:,4*i+1:4*i+4]=2*vres[:,4*i+1:4*i+4]-1
        temp = torch.nn.functional.conv_transpose2d(vres[:,4*i:4*i+4], Variable(inv_filters[:,None].cuda(),requires_grad=True),stride=2)
        res[:,i:i+1,:,:] = temp
    return res


class Waveletnet(nn.Module):
    def __init__(self,n_channels= 1,n_classes = 1, with_confidence = True):
        super(Waveletnet, self).__init__()
        self.bilinear = True
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.with_confidence = with_confidence
        self.num=1
        c=16
        self.conv1 = nn.Conv2d(4*self.n_channels,c,3, 1,padding=1)
        self.conv2 = nn.Conv2d(4*c,4*c,3, 1,padding=1)        
        self.conv3 = nn.Conv2d(16*c,16*c,3, 1,padding=1)  
        self.conv4 = nn.Conv2d(64*c,64*c,3, 1,padding=1)
        self.bn = nn.BatchNorm2d(320) 
        self.convd1 = nn.Conv2d(c,4*self.n_classes,3, 1,padding=1)
        self.convd2 = nn.Conv2d(2*c,c,3, 1,padding=1) 
        self.convd3 = nn.Conv2d(8*c,4*c,3, 1,padding=1)        
        self.convd4 = nn.Conv2d(32*c,16*c,3, 1,padding=1)  
        self.relu = nn.LeakyReLU(0.2)

        self.with_confidence = with_confidence
        if with_confidence :
            self.confidence_scorer = ConfidenceScorer2D(64*c , 64*c)

    def forward(self, x):

        w1=wt(x)
        c1=self.relu(self.conv1(w1))
        w2=wt(c1)
        c2=self.relu(self.conv2(w2))
        w3=wt(c2)
        c3=self.relu(self.conv3(w3))
        w4=wt(c3)
        c4=self.relu(self.conv4(w4))
        c5=self.relu(self.conv4(c4))
        c6=(self.conv4(c5))
        ic4=self.relu(c6+w4)
        iw4=iwt(ic4)
        iw4=torch.cat([c3,iw4],1)
        ic3=self.relu(self.convd4(iw4))
        iw3=iwt(ic3)
        iw3=torch.cat([c2,iw3],1)
        ic2=self.relu(self.convd3(iw3))
        iw2=iwt(ic2)
        iw2=torch.cat([c1,iw2],1)
        ic1=self.relu(self.convd2(iw2))
        iw1=self.relu(self.convd1(ic1))

        logits = iwt(iw1) 
        if self.with_confidence :
            classes = self.confidence_scorer(c6)
        return torch.sigmoid(logits), classes


class WaveletnetFull(nn.Module):
    def __init__(self,n_channels= 1,n_classes = 1, with_confidence = True):
        super(WaveletnetFull, self).__init__()
        self.bilinear = True
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.with_confidence = with_confidence
        self.num=1
        c=16
        self.conv1 = DoubleConv(4*self.n_channels,c)
        self.conv2 = DoubleConv(4*c,4*c)        
        self.conv3 = DoubleConv(16*c,16*c)  
        self.conv4 = DoubleConv(64*c,64*c)
        self.convd2 = DoubleConv(2*c,c) 
        self.convd3 = DoubleConv(8*c,4*c)        
        self.convd4 = DoubleConv(32*c,16*c)  
        self.outc = nn.Conv2d(c,4*self.n_classes,1)

        self.with_confidence = with_confidence
        if with_confidence :
            self.confidence_scorer = ConfidenceScorer2D(64*c , 64*c)

    def forward(self, x):
        w1=wt(x)
        c1=self.conv1(w1)
        w2=wt(c1)
        c2=self.conv2(w2)
        w3=wt(c2)
        c3=self.conv3(w3)
        w4=wt(c3)
        c4=self.conv4(w4)
        iw4=iwt(c4)
        iw4=torch.cat([c3,iw4],1)
        ic3=self.convd3(iw4)
        iw3=iwt(ic3)
        iw3=torch.cat([c2,iw3],1)
        ic2=self.convd2(iw3)
        iw2=iwt(ic2)
        iw2=torch.cat([c1,iw2],1)
        ic1=self.convd1(iw2)
        iw1=self.outc(ic1)
        logits = iwt(iw1) 
        if self.with_confidence :
            classes = self.confidence_scorer(c4)
        return torch.sigmoid(logits), classes



class ACT(nn.Module):
    def __init__(self):
        super(ACT, self).__init__()
        self.net = Waveletnet()
        self.c = torch.nn.Conv2d(3,3,1,padding=0, bias=False)
      
    def forward(self, x):
        x = self.net(x)
        x1 = self.c(x)
        x2 =x + x1
    
        return x