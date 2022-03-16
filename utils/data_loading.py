import logging
from os import listdir
from os.path import splitext
from pathlib import Path
from turtle import shape
from matplotlib.pyplot import axis

import numpy as np
from pandas import Index
import torch
from PIL import Image
from torch.utils.data import Dataset
import SimpleITK as sitk
from scipy import ndimage as nd 
import torchio as tio


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
from utils.image_folder import make_dataset, make_duke_dataset

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
        dir_image = "data/axial/png/tumor_slice_subset/third_iteration/registered_images"
        if phase == 'val':
            self.dir_image = os.path.join(self.root, dir_image, 'train')
        else :
            self.dir_image = os.path.join(self.root, dir_image, phase)
        self.image_paths = sorted(make_dataset(self.dir_image))

        self.image_paths = self.filter_according_to_phase(self.image_paths,phase)
        self.patient_ids = []

        for img_path in self.image_paths :
            patient_id = img_path[img_path.find('patient-')+len('patient-'):][:5]
            self.patient_ids.append(patient_id)
        
        self.registered = not unregistered

        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.dataset_mode = dataset_mode

        logging.info(f'Creating dataset with {len(self.image_paths)} examples')


        self.dataset_size = len(self.image_paths)

    def filter_according_to_phase(self,patients,phase):
        np.random.shuffle(patients)
        if phase == 'train':
            return sorted(patients[len(patients)//9:])
        elif phase == 'val' :
            return sorted(patients[0:len(patients)//9])
        elif phase == 'test' :
            return sorted(patients)
        else :
            raise Exception('split "{}" is not defined'.format(phase))

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





class DukeBreastCancerMRIDataset():
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
        dir_mri = "Duke-Breast-Cancer-MRI/manifest-1607053360376/Segmentation/"
        self.dir_mri = os.path.join(self.root, dir_mri)
        self.patients= make_duke_dataset(self.dir_mri)
        self.patients= self.filter_according_to_phase(self.patients,phase)
        self.patient_ids = []
        self.common_spacing = np.asfarray([2.1,1.4,1.4])


        for img_path in self.patients :
            patient_id = img_path[img_path.find('MRI_')+len('MRI_'):][:3]
            self.patient_ids.append(patient_id+'-L')
            self.patient_ids.append(patient_id+'-R')

        
        self.registered = not unregistered

        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.dataset_mode = dataset_mode

        logging.info(f'Creating dataset with {len(self.patients)} examples')



        self.dataset_size = len(self.patients)
        self.fix_shape = tio.EnsureShapeMultiple(16)

    def filter_according_to_phase(self,patients,phase):
        np.random.shuffle(patients)
        if phase == 'train':
            return sorted(patients[150:])
        elif phase == 'test' :
            return sorted(patients[0:150])
        else :
            raise Exception('split "{}" is not defined'.format(phase))

    def __getitem__(self, index):
        """
        Parameters
        ----------
        index : int
            the index of the item to fetch
        """
        # each patient provides 0 examples left and right
        patient_index = index // 2
        # 0 for left and 1 for right, right breast is gonna be mirrored
        left_or_right = index % 2
        patient_dir = self.patients[patient_index]
        I_pre = sitk.ReadImage(os.path.join(patient_dir,'pre.nii'))
        I_post = sitk.ReadImage(os.path.join(patient_dir,'post.nii'))
        I_bbox = sitk.ReadImage(os.path.join(patient_dir,'bbox.nii'))
        I_breast_mask = sitk.ReadImage(os.path.join(patient_dir,'breast_mask.nii'))
        I_tumor_mask = sitk.ReadImage(os.path.join(patient_dir,'prob_2nd.nii'))

        spacing_subject = I_breast_mask.GetSpacing()
        img_pre = np.array(sitk.GetArrayFromImage(I_pre))
        img_post = np.array(sitk.GetArrayFromImage(I_post))
        img_bbox = np.array(sitk.GetArrayFromImage(I_bbox))
        img_breast_mask = np.array(sitk.GetArrayFromImage(I_breast_mask))
        img_tumor_mask = np.array(sitk.GetArrayFromImage(I_tumor_mask))

        #select breast
        img_pre = np.split(img_pre,2,axis = 2)[left_or_right]
        img_post = np.split(img_post,2,axis = 2)[left_or_right]
        img_bbox= np.split(img_bbox,2,axis = 2)[left_or_right]
        img_breast_mask= np.split(img_breast_mask,2,axis = 2)[left_or_right]
        img_tumor_mask= np.split(img_tumor_mask,2,axis = 2)[left_or_right]






        spacing_subject = spacing_subject[::-1]/self.common_spacing

        img_pre = nd.interpolation.zoom(img_pre,spacing_subject,order=1)
        img_post = nd.interpolation.zoom(img_post,spacing_subject,order=1)
        img_bbox = nd.interpolation.zoom(img_bbox,spacing_subject,order=0)
        img_breast_mask = nd.interpolation.zoom(img_breast_mask,spacing_subject,order=0)
        img_tumor_mask = nd.interpolation.zoom(img_tumor_mask,spacing_subject,order=0)


        assert img_pre.shape == img_post.shape
        assert img_pre.shape == img_bbox.shape
        assert img_pre.shape == img_breast_mask.shape
        assert img_pre.shape == img_tumor_mask.shape


        coronal_surface_mask = img_breast_mask.sum(axis= (0,2))  # NOTE: ax, cr, sg --> cr, sg <=> 0 --> 1
        coronal_surface_box = img_bbox.sum(axis = (0,2))
        coronal_cutoff_mask = np.argmax(coronal_surface_mask)
        coronal_cutoff_box = np.argwhere(coronal_surface_box == np.amax(coronal_surface_box))[0][0] if np.sum(coronal_surface_box) != 0 else 0
        coronal_cutoff =  np.max([coronal_cutoff_mask, coronal_cutoff_box])

        # Optional: Sagittal cutoff on edges?
        # Optional: Remove axial slices with no values in breast mask, or can this backfire?

        img_pre = img_pre[:, :coronal_cutoff, :]
        img_post = img_post[:, :coronal_cutoff, :]
        img_breast_mask = img_breast_mask[:, :coronal_cutoff, :]
        img_tumor_mask = img_tumor_mask[:, :coronal_cutoff, :]

        try :
            img_pre = Norm_Zscore(imgnorm(img_pre))
        except :
            print(index)
            print(patient_index)
            print(patient_dir)
            print(left_or_right)
            print(coronal_cutoff)
            print(np.array(sitk.GetArrayFromImage(I_pre)).shape)
            print(np.array(sitk.GetArrayFromImage(I_breast_mask)).shape)
            print(img_pre.shape)
            raise IndexError

        img_post = Norm_Zscore(imgnorm(img_post)) 


            

        if self.dataset_mode == 'A' :
            img = img_pre
        elif self.dataset_mode == 'B' :
            img = img_post
        
        img = img[np.newaxis, ...]
        mask = img_tumor_mask

        assert img.size == mask.size, \
        'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        return {
            'image': self.fix_shape(torch.as_tensor(img.copy()).float().contiguous()),
            'mask': self.fix_shape(torch.as_tensor(mask.copy()).long().contiguous().unsqueeze(0))[0],
            'patient_id': self.patient_ids[index]
        }

    @classmethod
    def preprocess(cls, np_array,  is_mask):

        if img_ndarray.ndim == 2 and not is_mask:
            img_ndarray = img_ndarray[np.newaxis, ...]
        elif not is_mask:
            img_ndarray = img_ndarray.transpose((2, 0, 1))

        return img_ndarray

    def __len__(self):
        return len(self.patient_ids)

def Norm_Zscore(img):
    img= (img-np.mean(img))/np.std(img) 
    return img


def imgnorm(N_I,index1=0.001,index2=0.001):
    N_I = N_I.astype(np.float32)
    I_sort = np.sort(N_I.flatten())
    I_min = I_sort[int(index1*len(I_sort))]
    I_max = I_sort[-int(index2*len(I_sort))]
    
    N_I =1.0*(N_I-I_min)/(I_max-I_min)
    N_I[N_I>1.0]=1.0
    N_I[N_I<0.0]=0.0

    
    return N_I