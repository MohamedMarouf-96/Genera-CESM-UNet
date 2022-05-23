from enum import unique
import numpy as np 
import torch
from torch.utils.data import WeightedRandomSampler, SubsetRandomSampler, Sampler, RandomSampler
import typing
def get_balanced_weighted_sampler(elements_classes, epoch_size):
    y_train_indices = range(len(elements_classes))
    y_train = elements_classes
    class_sample_count = np.array(
    [len(np.where(y_train == t)[0]) for t in np.unique(y_train)])
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[t] for t in y_train])
    samples_weight = torch.from_numpy(samples_weight)
    sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'),replacement= False,num_samples= epoch_size)
    return sampler
# def get_balanced_weighted_sampler(elements_classes):
#     y_train_indices = range(len(elements_classes))
#     y_train = elements_classes
#     class_sample_count = np.array(
#     [len(np.where(y_train == t)[0]) for t in np.unique(y_train)])
#     weight = 1. / class_sample_count
#     samples_weight = np.array([weight[t] for t in y_train])
#     samples_weight = torch.from_numpy(samples_weight)
#     sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))
#     return sampler



class ClassBalancedRandomSampler(Sampler[int]):
    r"""Samples elements from ``[0,..,len(weights)-1]`` with given probabilities (weights).

    Args:
        weights (sequence)   : a sequence of weights, not necessary summing up to one
        num_samples (int): number of samples to draw
        replacement (bool): if ``True``, samples are drawn with replacement.
            If not, they are drawn without replacement, which means that when a
            sample index is drawn for a row, it cannot be drawn again for that row.
        generator (Generator): Generator used in sampling.

    Example:
        >>> list(WeightedRandomSampler([0.1, 0.9, 0.4, 0.7, 3.0, 0.6], 5, replacement=True))
        [4, 4, 1, 4, 5]
        >>> list(WeightedRandomSampler([0.9, 0.4, 0.05, 0.2, 0.3, 0.1], 5, replacement=False))
        [0, 1, 4, 3, 2]
    """
    weights: torch.Tensor
    num_samples: int
    replacement: bool

    def __init__(self, elemets_classes: typing.Sequence[int]) -> None:
        unique_classes = np.unique(elemets_classes)
        self.category_sampler = iter(RandomSampler(unique_classes,replacement= True, num_samples= len(elemets_classes)))
        class_samples_indecies = [np.where(elemets_classes == t) for t in unique_classes]
        self.class_samplers = [iter(SubsetRandomSampler(t)) for t in class_samples_indecies]
        print(self.class_samplers)


    def __iter__(self):
        return self
    
    def __next__(self):
        selected_category = next(self.category_sampler)
        print([x for x in self.class_samplers[selected_category]])
        print(next(self.class_samplers[selected_category]))
        exit()
        for x in self.class_samplers[selected_category] :
            print(x)
        exit()

        return next(self.class_samplers[selected_category])

    def __len__(self):
        return self.num_samples

class SubsetRandomSampler(Sampler[int]):
    r"""Samples elements randomly. If without replacement, then sample from a shuffled dataset.
    If with replacement, then user can specify :attr:`num_samples` to draw.

    Arguments:
        data_source (Dataset): dataset to sample from
        replacement (bool): samples are drawn on-demand with replacement if ``True``, default=``False``
        num_samples (int): number of samples to draw, default=`len(dataset)`. This argument
            is supposed to be specified only when `replacement` is ``True``.
        generator (Generator): Generator used in sampling.
    """
    data_source: typing.Sized
    replacement: bool

    def __init__(self, data_source: typing.Sized, replacement: bool = False,
                 num_samples: typing.Optional[int] = None, generator=None) -> None:
        self.data_source = data_source
        self.replacement = replacement
        self._num_samples = num_samples
        self.generator = generator

        if not isinstance(self.replacement, bool):
            raise TypeError("replacement should be a boolean value, but got "
                            "replacement={}".format(self.replacement))

        if self._num_samples is not None and not replacement:
            raise ValueError("With replacement=False, num_samples should not be specified, "
                             "since a random permute will be performed.")

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError("num_samples should be a positive integer "
                             "value, but got num_samples={}".format(self.num_samples))

    @property
    def num_samples(self) -> int:
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self):
        n = len(self.data_source)
        if self.generator is None:
            generator = torch.Generator()
            generator.manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))
        else:
            generator = self.generator
        if self.replacement:
            for _ in range(self.num_samples // 32):
                yield from torch.randint(high=n, size=(32,), dtype=torch.int64, generator=generator).tolist()
            yield from torch.randint(high=n, size=(self.num_samples % 32,), dtype=torch.int64, generator=generator).tolist()
        else:
            yield from torch.randperm(n, generator=self.generator).tolist()

    def __len__(self):
        return self.num_samples



# class InfSubsetRandomSampler(Sampler[int]):
#     r"""Samples elements randomly from a given list of indices, without replacement.

#     Arguments:
#         indices (sequence): a sequence of indices
#         generator (Generator): Generator used in sampling.
#     """
#     indices: typing.Sequence[int]


#     def __init__(self, indices: typing.Sequence[int], generator=None) -> None:
#         self.indices = indices
#         self.generator = generator

    
#     def __iter__(self):
#         return self
#     def __next__(self):
#         n = len(self.indices)
#         if self.generator is None:
#             generator = torch.Generator()
#             generator.manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))
#         else:
#             generator = self.generator
#         schuffled = [self.indices[i] for i in torch.randperm(n, generator=self.generator)]
#         print(f'schuffled : {schuffled}')
#         exit()

#         for i,index in enumerate(schuffled) :
#             if i >= 50 :
#                 yield index


#     def __len__(self):
#         return len(self.indices)





