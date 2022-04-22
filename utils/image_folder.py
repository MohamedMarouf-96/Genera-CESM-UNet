"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

###############################################################################
# Code from
# https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py
# Modified the original code so that it also loads images from the current
# directory as well as the subdirectories
###############################################################################
import os
import glob


def make_duke_dataset(dir_mri,read_cache=False, write_cache=False):
    patient_inputs = sorted(list(glob.glob(f'{dir_mri}/*')))
    pre_paths = list()
    post_paths = list()
    segmentation_paths = list()
    breast_mask_paths = list()
    bbox_paths =  list()
    for file in patient_inputs:
        filename = os.path.basename(file)
        if 'pre' in filename :
            pre_paths.append(file)
        elif 'post' in filename :
            post_paths.append(file)
        elif 'tumor' in filename :
            segmentation_paths.append(file)
        elif 'breast_mask' in filename :
            breast_mask_paths.append(file)
        elif 'bbox' in filename :
            bbox_paths.append(file) 
    return pre_paths,post_paths,segmentation_paths,breast_mask_paths,bbox_paths