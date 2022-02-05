import logging
from os import listdir
from os.path import splitext
from pathlib import Path
from matplotlib.pyplot import axis

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class BasicDataset(Dataset):
    def __init__(self, images_dir: str, masks_dir: str, scale: float = 1.0, mask_suffix: str = ''):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix

        self.ids = [splitext(file)[0] for file in listdir(images_dir) if not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img_ndarray = np.asarray(pil_img)

        if img_ndarray.ndim == 2 and not is_mask:
            img_ndarray = img_ndarray[np.newaxis, ...]
        elif not is_mask:
            img_ndarray = img_ndarray.transpose((2, 0, 1))

        if not is_mask:
            img_ndarray = img_ndarray / 127.5 - 1

        return img_ndarray

    @classmethod
    def load(cls, filename):
        ext = splitext(filename)[1]
        if ext in ['.npz', '.npy']:
            return Image.fromarray(np.load(filename))
        elif ext in ['.pt', '.pth']:
            return Image.fromarray(torch.load(filename).numpy())
        else:
            return Image.open(filename)

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.masks_dir.glob(name + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        mask = self.load(mask_file[0])
        img = self.load(img_file[0])

        assert img.size == mask.size, \
            'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img, self.scale, is_mask=False)
        mask = self.preprocess(mask, self.scale, is_mask=True)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }


class CarvanaDataset(BasicDataset):
    def __init__(self, images_dir, masks_dir, scale=1):
        super().__init__(images_dir, masks_dir, scale, mask_suffix='_mask')



import os.path
from utils.image_folder import make_dataset

class TumorSliceSubsetDataset():
    """ "
    a class used to load data from final training data provided by kasper available in 'bucket_dca_1/data/axial/png/tumor_slice_subset/unetpp_specnorm_tts_w_segm_and_class'
    and convert it to the same format as BasicDataset. This dataset is split into train set and test set each in different folder in the main directory. Each
    image consists of five channels attach width-wise. In this dataset we only care about the first and second channels which represent pre-contrast and post-contrast iamges
    respectivley.

    Attributes
    ----------
    opt : BaseOptions
        an object containing run related options
    root : str
        dataset root in the file system.
    dir_image : str
        path of directory containining the images of the subset.
    image_path : list(str)
        list of paths for all instances inside the selected subset.

    """
    def __init__(self, data_root: str, phase: str, scale: float = 1.0, unregistered = False, dataset_mode = 'B'):

        self.root = data_root
        dir_image = "axial/png/tumor_slice_subset/third_iteration/registered_images"
        self.dir_image = os.path.join(self.root, dir_image, phase)
        self.image_paths = sorted(make_dataset(self.dir_image))
        self.patient_ids = []

        for img_path in self.image_paths :
            patient_id = img_path[img_path.find('patient-')+len('patient-'):][:3]
            self.patient_ids.append(patient_id)
        
        self.registered = not unregistered

        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.dataset_mode = dataset_mode

        logging.info(f'Creating dataset with {len(self.image_paths)} examples')



        self.dataset_size = len(self.image_paths)

    def __getitem__(self, index):
        """
        Parameters
        ----------
        index : int
            the index of the item to fetch
        """
        # input image (both pre and post contrast) where A and B are side by side
        image_path = self.image_paths[index]
        image = Image.open(image_path)
        _ , image_h = image.size

        # input A (pre contrast image) 
        offset = 3 if self.registered else 0 
        A = image.crop((offset * image_h, 0, (offset+1) * image_h, image_h))
        # A = A.convert("RGB")
 

        B_tensor = difference_tensor = breast_mask_tensor = tumor_mask_tensor = 0
        # input B (post contrast image) which is the second crop in image
        # if self.opt.isTrain or self.opt.use_vae:
        B = image.crop((image_h, 0, 2 * image_h, image_h))
        # B = B.convert("RGB")

        # input C (difference map image) 
        offset = 4 if self.registered else 2 
        C = image.crop((offset * image_h, 0, (offset+1) * image_h, image_h))
        # C = C.convert("RGB")

        # input D (breast mask image) 
        offset = 5 
        D = image.crop((offset * image_h, 0, (offset+1) * image_h, image_h))
        D = D.convert("RGB")

        # input E (tumor mask image) 
        offset = 6 
        E = image.crop((offset * image_h, 0, (offset+1) * image_h, image_h))

        img = A
        mask = E
        assert img.size == mask.size, \
            'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'
            

        if self.dataset_mode == 'A' :
            img = A.convert('RGB')
            img = self.preprocess(img, self.scale, is_mask=False)
        elif self.dataset_mode == 'B' :
            img = B.convert('RGB')
            img = self.preprocess(img, self.scale, is_mask=False)
        elif self.dataset_mode == 'ABD' :
            imgA = self.preprocess(A, self.scale, is_mask=False)
            imgB = self.preprocess(B, self.scale, is_mask=False)
            imgC = self.preprocess(C, self.scale, is_mask=False)
            img = np.concatenate([imgA,imgB,imgC],axis=0)
        
        mask = self.preprocess(mask, self.scale, is_mask=True) / 255

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous(),
            'patient_id': self.patient_ids[index]
        }

    @classmethod
    def preprocess(cls, pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img_ndarray = np.asarray(pil_img)

        if img_ndarray.ndim == 2 and not is_mask:
            img_ndarray = img_ndarray[np.newaxis, ...]
        elif not is_mask:
            img_ndarray = img_ndarray.transpose((2, 0, 1))

        if not is_mask:
            img_ndarray = img_ndarray / 127.5 - 1

        return img_ndarray

    def __len__(self):
        return len(self.image_paths)
