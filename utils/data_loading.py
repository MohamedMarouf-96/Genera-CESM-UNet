import itertools
import logging
import numpy as np
import torch
import SimpleITK as sitk
from scipy import ndimage as nd 
import torchio as tio
from tqdm import tqdm
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
    def __init__(self, data_root: str, phase: str, args):

        self.phase = phase
        self.root = data_root
        if phase in ['train','val']:
            dir_mri = f"Duke-Breast-Cancer-MRI/manifest-1607053360376/3D-Dataset/train"
        elif phase in ['test']:
            dir_mri = f"Duke-Breast-Cancer-MRI/manifest-1607053360376/3D-Dataset/test"
        else :
            raise Exception('phase is not defined')      
        self.dir_mri = os.path.join(self.root, dir_mri)
        self.pre_paths,self.post_paths,self.segmentation_paths,self.breast_mask_paths, self.bbox_paths = make_duke_dataset(self.dir_mri)
        assert len(self.pre_paths) == len(self.post_paths)
        assert len(self.pre_paths) == len(self.segmentation_paths)
        assert len(self.pre_paths) == len(self.breast_mask_paths) 
        assert len(self.pre_paths) == len(self.bbox_paths)


        index = self.get_subset_indecies(len(self.pre_paths),phase)

        self.pre_paths = list(itertools.compress(self.pre_paths,index))
        self.post_paths = list(itertools.compress(self.post_paths,index))
        self.segmentation_paths = list(itertools.compress(self.segmentation_paths,index))
        self.breast_mask_paths = list(itertools.compress(self.breast_mask_paths,index))
        self.bbox_paths = list(itertools.compress(self.bbox_paths,index))

        self.args = args
        self.patient_ids = []

        for img_path in self.pre_paths :
            patient_id = os.path.basename(img_path)[:5]
            self.patient_ids.append(patient_id)
        
        # resample or resize
        if self.args.resample :
            self.common_spacing = np.asfarray([2.1,1.4,1.4])
        else :
            self.resize_nearest = lambda x,size: torch.nn.functional.interpolate(torch.as_tensor(x).unsqueeze(0).unsqueeze(0),size,mode = 'nearest')[0,0].numpy()
            self.resize_bilinear = lambda x,size: torch.nn.functional.interpolate(torch.as_tensor(x).unsqueeze(0),size,mode = 'bilinear', align_corners = True)[0].numpy()
        
        self.registered = not self.args.unregistered
        self.dataset_mode = self.args.dataset_mode

        self.axial_size = self.args.axial_size
        self.coronal_size = self.args.coronal_size
        self.sagital_size = self.args.sagital_size

        self.patient_id_category, self.index_of_patient_id, self.patient_id_per_example, self.start_slice_per_example, self.example_category, self.slice_category_per_example = self.get_positive_stats()

        logging.info(f'Creating dataset with {len(self.start_slice_per_example)} examples and mode {self.dataset_mode}')

        assert len(self.patient_id_per_example) == len(self.start_slice_per_example)
        assert len(self.example_category) == len(self.start_slice_per_example)
        assert len(self.slice_category_per_example) == len(self.start_slice_per_example)


        self.dataset_size = len(self.start_slice_per_example)
        self.fix_shape = tio.EnsureShapeMultiple(16)



    def get_positive_stats(self):
        patient_id_category = list()
        index_of_patient_id = dict()
        patient_id_per_example = list()
        start_slice_numbers_per_example = list()
        example_category = list()
        slices_category_per_example = list()
        for i,mask_path in tqdm(enumerate(self.segmentation_paths),total=len(self.segmentation_paths)) :
            I_tumor_mask = sitk.ReadImage(mask_path)
            img_tumor_mask = np.array(sitk.GetArrayFromImage(I_tumor_mask))
            if len(img_tumor_mask) < self.axial_size // 2 :
                raise Exception(f'patient {mask_path} does not have enough slices')
            slices_category_of_patient = [False for _ in range(-self.axial_size //4,0)] + list(img_tumor_mask.sum(axis= (1,2)) > 0) + [False for _ in range(0,self.axial_size //4)]
            start_slice_numbers_of_patient = list(range(0 ,len(img_tumor_mask) - (self.axial_size // 2),self.axial_size // 2))
            start_slice_numbers_per_example += (start_slice_numbers_of_patient)
            patient_id_per_example += [self.patient_ids[i]]*len(start_slice_numbers_of_patient)
            index_of_patient_id[self.patient_ids[i]] = i
            for start_index in start_slice_numbers_of_patient :
                example_category += [True in slices_category_of_patient[start_index + self.axial_size // 4 :start_index + 3 * self.axial_size // 4]]
                slices_category_per_example.append(slices_category_of_patient[start_index:start_index + self.axial_size])
        return  patient_id_category, index_of_patient_id,patient_id_per_example, start_slice_numbers_per_example, example_category, slices_category_per_example

    def get_subset_indecies(self,length,phase):
        index = np.random.RandomState(seed=42).permutation(length)
        if phase == 'train':
            index = sorted(index[length//10:])
        elif phase == 'val' :
            index = sorted(index[0:length//10:])
        elif phase == 'test' :
            index = sorted(index)#[0:20]
        else :
            raise Exception('split "{}" is not defined'.format(phase))
        index = [True if i in index else False for i in range(length)]
        return index

    def __getitem__(self, example_index):
        """
        Parameters
        ----------
        index : int
            the index of the item to fetch
        """
        patient_id = self.patient_id_per_example[example_index]
        patient_index = self.index_of_patient_id[patient_id]
        start_slice = self.start_slice_per_example[example_index]

        I_pre = sitk.ReadImage(self.pre_paths[patient_index])
        I_post = sitk.ReadImage(self.post_paths[patient_index])
        I_bbox = sitk.ReadImage(self.bbox_paths[patient_index])
        I_breast_mask = sitk.ReadImage(self.breast_mask_paths[patient_index])
        I_tumor_mask = sitk.ReadImage(self.segmentation_paths[patient_index])

        
        img_pre = np.array(sitk.GetArrayFromImage(I_pre))
        img_post = np.array(sitk.GetArrayFromImage(I_post))
        img_bbox = np.array(sitk.GetArrayFromImage(I_bbox))
        img_breast_mask = np.array(sitk.GetArrayFromImage(I_breast_mask))
        img_tumor_mask = np.array(sitk.GetArrayFromImage(I_tumor_mask))

        pad_before_select = lambda x,size: np.concatenate([np.zeros_like(x)[0:size],x,np.zeros_like(x)[0:size]],axis = 0)

        img_pre = imgnorm(pad_before_select(img_pre,self.axial_size//4))[start_slice : start_slice + self.axial_size ]
        img_post = imgnorm(pad_before_select(img_post,self.axial_size//4))[start_slice : start_slice + self.axial_size ]
        img_bbox = pad_before_select(img_bbox,self.axial_size//4)[start_slice : start_slice + self.axial_size ]
        img_breast_mask = pad_before_select(img_breast_mask,self.axial_size//4)[start_slice : start_slice + self.axial_size ]
        img_tumor_mask = pad_before_select(img_tumor_mask,self.axial_size//4)[start_slice : start_slice + self.axial_size ]




        # get spacing and rescale to common spacing
        if self.args.resample :
            spacing_subject = I_breast_mask.GetSpacing()
            spacing_subject = spacing_subject[::-1]/self.common_spacing

            img_pre = nd.interpolation.zoom(img_pre,spacing_subject,order=1)
            img_post = nd.interpolation.zoom(img_post,spacing_subject,order=1)
            img_bbox = nd.interpolation.zoom(img_bbox,spacing_subject,order=0)
            img_breast_mask = nd.interpolation.zoom(img_breast_mask,spacing_subject,order=0)
            img_tumor_mask = nd.interpolation.zoom(img_tumor_mask,spacing_subject,order=0)
        else :
            img_pre = self.resize_bilinear(img_pre,(self.coronal_size,self.sagital_size))
            img_post = self.resize_bilinear(img_post,(self.coronal_size,self.sagital_size))
            img_bbox = self.resize_nearest(img_bbox,(img_bbox.shape[0],self.coronal_size,self.sagital_size))
            if self.phase in ['test']:
                img_breast_mask = self.resize_nearest(img_breast_mask,(img_breast_mask.shape[0],256,256))
                img_tumor_mask = self.resize_nearest(img_tumor_mask,(img_tumor_mask.shape[0],256,256))
            else:
                img_breast_mask = self.resize_nearest(img_breast_mask,(img_breast_mask.shape[0],self.coronal_size,self.sagital_size))
                img_tumor_mask = self.resize_nearest(img_tumor_mask,(img_tumor_mask.shape[0],self.coronal_size,self.sagital_size))




        assert img_pre.shape == img_post.shape
        assert img_pre.shape == img_bbox.shape
        assert img_pre.shape == img_breast_mask.shape
        assert img_pre.shape == img_tumor_mask.shape

        # add new axis
        img_pre = img_pre[np.newaxis]
        img_post = img_post[np.newaxis]
        img_bbox = img_bbox[np.newaxis]
        img_breast_mask = img_breast_mask[np.newaxis]
        img_tumor_mask = img_tumor_mask[np.newaxis]

        # convert to tensor
        img_pre = torch.as_tensor(img_pre.copy()).float().contiguous()
        img_post = torch.as_tensor(img_post.copy()).float().contiguous()
        img_bbox = torch.as_tensor(img_bbox.copy()).long().contiguous()
        img_breast_mask = torch.as_tensor(img_breast_mask.copy()).long().contiguous()
        img_tumor_mask = torch.as_tensor(img_tumor_mask.copy()).long().contiguous()

        # select which modality according to dataset mode
        if self.dataset_mode == 'A' :
            img = img_pre
        elif self.dataset_mode == 'B' :
            img = img_post
        elif self.dataset_mode == 'ABD' :
            img = torch.cat([img_pre, img_post, img_post-img_pre],axis = 0)
        
        mask = img_tumor_mask

        assert img.shape[-3:] == mask.shape[-3:], \
        'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        return {
            'image': img,
            'mask': mask,
            'patient_id': self.patient_id_per_example[example_index]
        }

    def get_labels(self):
        return [int(x) for x in self.example_category]


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

    def __len__(self):
        return len(self.start_slice_per_example)

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