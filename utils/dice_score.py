import torch
from torch import Tensor


def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6, threeD = False, reduce_batch = True):
    if threeD:
        n_dim = 3
    else:
        n_dim = 2
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    if input.dim() == n_dim and reduce_batch_first:
        raise ValueError(f'Dice: asked to reduce batch but got tensor without batch dimension (shape {input.shape})')

    if reduce_batch :
        if input.dim() == n_dim or reduce_batch_first:
            inter = torch.dot(input.reshape(-1), target.reshape(-1))
            sets_sum = torch.sum(input) + torch.sum(target)
            if sets_sum.item() == 0:
                sets_sum = 2 * inter

            return (2 * inter + epsilon) / (sets_sum + epsilon)
        else:
            # compute and average metric for each batch element
            dice = 0
            for i in range(input.shape[0]):
                dice += dice_coeff(input[i, ...], target[i, ...],threeD)
            return dice / input.shape[0]
    else :
        if input.dim() == n_dim :
            raise ValueError(f'Dice: asked to not reduce batch but got tensor without batch dimension (shape {input.shape})')
        else:
            # compute and average metric for each batch element
            dice = []
            for i in range(input.shape[0]):
                dice.append(dice_coeff(input[i, ...], target[i, ...],threeD))
            return torch.stack(dice,dim=0)

def dice_loss(input: Tensor, target: Tensor, threeD= False):
    # Dice loss (objective to minimize) between 0 and 1
    assert input.size() == target.size()
    return 1 - dice_coeff(input, target, reduce_batch_first=True, threeD = threeD)
