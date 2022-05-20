from multiprocessing import reduction
from turtle import forward
import torch
from torch import nn
from torch.nn import functional as F


def balanced_binary_cross_entropy(input, label_orig):
    label = torch.nn.functional.one_hot(label_orig.long(), num_classes = label_orig.max().long().item()+1)
    class_occurence = torch.sum(label, dim=[i for i in range(len(label_orig.size()))]).float()
    num_of_classes = (class_occurence > 0).sum()
    coefficients = torch.reciprocal(class_occurence) * torch.numel(label) / (num_of_classes * label.shape[1])
    integers = torch.argmax(label, dim=-1, keepdim=True)
    weight_map = coefficients[integers].squeeze(-1)
    loss = torch.nn.functional.binary_cross_entropy(input,label_orig,reduction='none')
    loss = ( loss * weight_map ).mean()
    return loss

class BBCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_computer = nn.BCELoss(reduction='none')
    def forward(self,input,label_orig):
        label = torch.nn.functional.one_hot(label_orig.long(), num_classes = label_orig.max().long().item()+1)
        class_occurence = torch.sum(label, dim=[i for i in range(len(label_orig.size()))]).float()
        num_of_classes = (class_occurence > 0).sum()
        coefficients = torch.reciprocal(class_occurence) * torch.numel(label) / (num_of_classes * label.shape[1])
        integers = torch.argmax(label, dim=-1, keepdim=True)
        weight_map = coefficients[integers].squeeze(-1)
        loss = self.loss_computer(input,label_orig)
        loss = ( loss * weight_map ).mean()
        return loss