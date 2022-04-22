import logging

import numpy as np
import torch
import SimpleITK as sitk
from scipy import ndimage as nd 
import torchio as tio
from utils.dbcmri_slice_dataset import DBCMRIDataset

import os.path
from utils.image_folder import make_duke_dataset


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
        dir_mri = f"Duke-Breast-Cancer-MRI/manifest-1607053360376/3D-Dataset/{phase}"
        self.dir_mri = os.path.join(self.root, dir_mri)
        self.pre_paths,self.post_paths,self.segmentation_paths,self.breast_mask_paths, self.bbox_paths = make_duke_dataset(self.dir_mri)
        assert len(self.pre_paths) == len(self.post_paths)
        assert len(self.pre_paths) == len(self.segmentation_paths)
        assert len(self.pre_paths) == len(self.breast_mask_paths) 
        assert len(self.pre_paths) == len(self.bbox_paths)


        self.patient_ids = []
        self.common_spacing = np.asfarray([2.1,1.4,1.4])


        for img_path in self.pre_paths :
            patient_id = os.path.basename(img_path)[:5]
            self.patient_ids.append(patient_id)
        
        self.registered = not unregistered

        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.dataset_mode = dataset_mode

        logging.info(f'Creating dataset with {len(self.pre_paths)} examples and mode {self.dataset_mode}')



        self.dataset_size = len(self.pre_paths)
        self.fix_shape = tio.EnsureShapeMultiple(16)

    # def filter_according_to_phase(self,patients,phase):
    #     np.random.shuffle(patients)
    #     if phase == 'train':
    #         return sorted(patients[150:])
    #     elif phase == 'test' :
    #         return sorted(patients[0:150])
    #     else :
    #         raise Exception('split "{}" is not defined'.format(phase))

    def __getitem__(self, index):
        """
        Parameters
        ----------
        index : int
            the index of the item to fetch
        """
        I_pre = sitk.ReadImage(self.pre_paths[index])
        I_post = sitk.ReadImage(self.post_paths[index])
        I_bbox = sitk.ReadImage(self.bbox_paths[index])
        I_breast_mask = sitk.ReadImage(self.breast_mask_paths[index])
        I_tumor_mask = sitk.ReadImage(self.segmentation_paths[index])

        
        img_pre = np.array(sitk.GetArrayFromImage(I_pre))
        img_post = np.array(sitk.GetArrayFromImage(I_post))
        img_bbox = np.array(sitk.GetArrayFromImage(I_bbox))
        img_breast_mask = np.array(sitk.GetArrayFromImage(I_breast_mask))
        img_tumor_mask = np.array(sitk.GetArrayFromImage(I_tumor_mask))




        # get spacing and rescale to common spacing
        spacing_subject = I_breast_mask.GetSpacing()
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

        try :
            # img_pre = Norm_Zscore(imgnorm(img_pre))
            img_pre = imgnorm(img_pre)
        except :
            print(index)
            print(self.pre_paths[index])
            print(np.array(sitk.GetArrayFromImage(I_pre)).shape)
            print(np.array(sitk.GetArrayFromImage(I_breast_mask)).shape)
            print(img_pre.shape)
            raise IndexError

        img_post = imgnorm(img_post)


        if self.dataset_mode == 'A' :
            img = img_pre
            img = img[np.newaxis, ...]
        elif self.dataset_mode == 'B' :
            img = img_post
            img = img[np.newaxis, ...]
        elif self.dataset_mode == 'ABD' :
            img = np.stack([Norm_Zscore(img_pre),Norm_Zscore(img_post),Norm_Zscore(img_pre-img_post)])
        
        mask = img_tumor_mask

        assert img.shape[-3:] == mask.shape[-3:], \
        'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        return {
            'image': self.fix_shape(torch.as_tensor(img.copy()).float().contiguous()),
            'mask': self.fix_shape(torch.as_tensor(mask.copy()).long().contiguous().unsqueeze(0))[0],
            'patient_id': self.patient_ids[index]
        }

    # def __getitem__(self, index):
    #     """
    #     Parameters
    #     ----------
    #     index : int
    #         the index of the item to fetch
    #     """
    #     # each patient provides 0 examples left and right
    #     patient_index = index // 2
    #     # 0 for left and 1 for right, right breast is gonna be mirrored
    #     left_or_right = index % 2
    #     patient_dir = self.patients[patient_index]
    #     I_pre = sitk.ReadImage(os.path.join(patient_dir,'pre.nii'))
    #     I_post = sitk.ReadImage(os.path.join(patient_dir,'post.nii'))
    #     I_bbox = sitk.ReadImage(os.path.join(patient_dir,'bbox.nii'))
    #     I_breast_mask = sitk.ReadImage(os.path.join(patient_dir,'breast_mask.nii'))
    #     I_tumor_mask = sitk.ReadImage(os.path.join(patient_dir,'prob_2nd.nii'))

    #     spacing_subject = I_breast_mask.GetSpacing()
    #     img_pre = np.array(sitk.GetArrayFromImage(I_pre))
    #     img_post = np.array(sitk.GetArrayFromImage(I_post))
    #     img_bbox = np.array(sitk.GetArrayFromImage(I_bbox))
    #     img_breast_mask = np.array(sitk.GetArrayFromImage(I_breast_mask))
    #     img_tumor_mask = np.array(sitk.GetArrayFromImage(I_tumor_mask))

    #     #select breast
    #     img_pre = np.split(img_pre,2,axis = 2)[left_or_right]
    #     img_post = np.split(img_post,2,axis = 2)[left_or_right]
    #     img_bbox= np.split(img_bbox,2,axis = 2)[left_or_right]
    #     img_breast_mask= np.split(img_breast_mask,2,axis = 2)[left_or_right]
    #     img_tumor_mask= np.split(img_tumor_mask,2,axis = 2)[left_or_right]






    #     spacing_subject = spacing_subject[::-1]/self.common_spacing

    #     img_pre = nd.interpolation.zoom(img_pre,spacing_subject,order=1)
    #     img_post = nd.interpolation.zoom(img_post,spacing_subject,order=1)
    #     img_bbox = nd.interpolation.zoom(img_bbox,spacing_subject,order=0)
    #     img_breast_mask = nd.interpolation.zoom(img_breast_mask,spacing_subject,order=0)
    #     img_tumor_mask = nd.interpolation.zoom(img_tumor_mask,spacing_subject,order=0)


    #     assert img_pre.shape == img_post.shape
    #     assert img_pre.shape == img_bbox.shape
    #     assert img_pre.shape == img_breast_mask.shape
    #     assert img_pre.shape == img_tumor_mask.shape


    #     coronal_surface_mask = img_breast_mask.sum(axis= (0,2))  # NOTE: ax, cr, sg --> cr, sg <=> 0 --> 1
    #     coronal_surface_box = img_bbox.sum(axis = (0,2))
    #     coronal_cutoff_mask = np.argmax(coronal_surface_mask)
    #     coronal_cutoff_box = np.argwhere(coronal_surface_box == np.amax(coronal_surface_box))[0][0] if np.sum(coronal_surface_box) != 0 else 0
    #     coronal_cutoff =  np.max([coronal_cutoff_mask, coronal_cutoff_box])

    #     # Optional: Sagittal cutoff on edges?
    #     # Optional: Remove axial slices with no values in breast mask, or can this backfire?

    #     img_pre = img_pre[:, :coronal_cutoff, :]
    #     img_post = img_post[:, :coronal_cutoff, :]
    #     img_breast_mask = img_breast_mask[:, :coronal_cutoff, :]
    #     img_tumor_mask = img_tumor_mask[:, :coronal_cutoff, :]

    #     try :
    #         img_pre = Norm_Zscore(imgnorm(img_pre))
    #     except :
    #         print(index)
    #         print(patient_index)
    #         print(patient_dir)
    #         print(left_or_right)
    #         print(coronal_cutoff)
    #         print(np.array(sitk.GetArrayFromImage(I_pre)).shape)
    #         print(np.array(sitk.GetArrayFromImage(I_breast_mask)).shape)
    #         print(img_pre.shape)
    #         raise IndexError

    #     img_post = Norm_Zscore(imgnorm(img_post)) 


            

    #     if self.dataset_mode == 'A' :
    #         img = img_pre
    #     elif self.dataset_mode == 'B' :
    #         img = img_post
        
    #     img = img[np.newaxis, ...]
    #     mask = img_tumor_mask

    #     assert img.size == mask.size, \
    #     'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

    #     return {
    #         'image': self.fix_shape(torch.as_tensor(img.copy()).float().contiguous()),
    #         'mask': self.fix_shape(torch.as_tensor(mask.copy()).long().contiguous().unsqueeze(0))[0],
    #         'patient_id': self.patient_ids[index]
    #     }

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