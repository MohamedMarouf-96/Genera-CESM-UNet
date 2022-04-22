import numpy as np 
import torch
from torch.utils.data import WeightedRandomSampler

def get_balanced_weighted_sampler(elements_classes):
    y_train_indices = range(len(elements_classes))
    y_train = elements_classes
    class_sample_count = np.array(
    [len(np.where(y_train == t)[0]) for t in np.unique(y_train)])
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[t] for t in y_train])
    samples_weight = torch.from_numpy(samples_weight)
    sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))
    return sampler
