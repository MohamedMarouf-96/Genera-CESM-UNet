import itertools
import os.path

import numpy as np
from tqdm import tqdm
import torch
import glob
import SimpleITK as sitk
from scipy import ndimage as nd
import skimage

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


def bbox2_3D(img):

    z = np.any(img, axis=(1, 2))
    r = np.any(img, axis=(0, 2))
    c = np.any(img, axis=(0, 1))

    zmin, zmax = np.where(z)[0][[0, -1]]
    rmin, rmax = np.where(r)[0][[0, -1]]
    cmin, cmax = np.where(c)[0][[0, -1]]

    return zmin, zmax, rmin, rmax, cmin, cmax 

class DBCMRIDataset():
    """ "
    a class used to load 3D data from the patients included in kasper's final training set. it has the same test set and training set for backward compatability. the available slices can be a bit different due to the use of different segmentation masks during cropping however this shouldn't cause 
    huge deviations in the results.

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

    Methods
    -------
    initialize(self, opt)
        intializes the dataset instance and loads the paths of the images in the subset

    __getitem__(self, index)
        returns the item in the dataset at index 'index'

    __len__(self)
        returns the nubmer of batches needed for one full epoch of the dataset

    name(self):
        returns name of the dataset class
    """

    def __init__(self, dataroot, phase , dataset_mode = 'B'):
        """
        Parameters
        ----------
        opt : BaseOptions
            the options for this training run
        """
        self.root = dataroot


        dir_mri = f"Duke-Breast-Cancer-MRI/manifest-1607053360376/3D-Dataset/"
        if phase == 'val':
            self.dir_mri = os.path.join(self.root, dir_mri, 'train')
        else :
            self.dir_mri = os.path.join(self.root, dir_mri, phase)

        self.dataset_mode = dataset_mode




        # to make loading determistic for consistent metrics
        self.for_validation = phase == 'val' or phase == 'test'
       
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

        self.patient_ids = []
        self.common_spacing = np.asfarray([2.1,1.4,1.4])
        self.coronal_size = 256
        self.sagital_size = 256

        for img_path in self.pre_paths :
            patient_id = os.path.basename(img_path)[:5]
            self.patient_ids.append(patient_id)

        self.patient_id_category, self.index_of_patient_id, self.patient_id_per_example, self.slice_numbers_per_example, self.example_category = self.get_positive_stats()

        print('only positive examples are used : {}'.format(not (False in self.patient_id_category)))

        assert len(self.patient_id_per_example) == len(self.slice_numbers_per_example)
        assert len(self.example_category) == len(self.slice_numbers_per_example)


        self.dataset_size = len(self.slice_numbers_per_example)
    
    
    def get_subset_indecies(self,length,phase):
        index = np.random.RandomState(seed=42).permutation(length)
        if phase == 'train':
            index = sorted(index[length//10:])#[0:4]
        elif phase == 'val' :
            index = sorted(index[0:length//10:])#[0:4]
        elif phase == 'test' :
            index = sorted(index)#[0:20]
        else :
            raise Exception('split "{}" is not defined'.format(phase))
        index = [True if i in index else False for i in range(length)]
        return index


    def get_positive_stats(self):
        patient_id_category = list()
        index_of_patient_id = dict()
        patient_id_per_example = list()
        slice_numbers_per_example = list()
        example_category = list()
        for i,mask_path in tqdm(enumerate(self.segmentation_paths),total=len(self.segmentation_paths)) :
            I_tumor_mask = sitk.ReadImage(mask_path)
            img_tumor_mask = np.array(sitk.GetArrayFromImage(I_tumor_mask))
            slice_numbers_per_example += (list(range(len(img_tumor_mask))))
            patient_id_per_example += [self.patient_ids[i]]*len(img_tumor_mask)
            index_of_patient_id[self.patient_ids[i]] = i
            example_category += list(img_tumor_mask.sum(axis= (1,2)) > 0)
            patient_id_category.append(img_tumor_mask.sum() > 0)
        return  patient_id_category, index_of_patient_id,patient_id_per_example, slice_numbers_per_example, example_category


    def __getitem__(self, example_index):
        """
        Parameters
        ----------
        index : int
            the index of the item to fetch
        """
        patient_id = self.patient_id_per_example[example_index]
        slice_number = self.slice_numbers_per_example[example_index]
        index = self.index_of_patient_id[patient_id]



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


        # normalize to range [0,1]
        img_pre = imgnorm(img_pre)
        img_post = imgnorm(img_post)

        # change range to [-1,1]
        img_pre = img_pre * 2 - 1
        img_post = img_post * 2 - 1

        assert img_pre.shape == img_post.shape
        assert img_pre.shape == img_bbox.shape
        assert img_pre.shape == img_breast_mask.shape
        assert img_pre.shape == img_tumor_mask.shape

        # get spacing and rescale to common spacing
        # spacing_subject = I_breast_mask.GetSpacing()
        # spacing_subject = spacing_subject[::-1]/self.common_spacing


        # img_pre = nd.interpolation.zoom(img_pre,spacing_subject,order=1)
        # img_post = nd.interpolation.zoom(img_post,spacing_subject,order=1)
        # img_bbox = nd.interpolation.zoom(img_bbox,spacing_subject,order=0)
        # img_breast_mask = nd.interpolation.zoom(img_breast_mask,spacing_subject,order=0)
        # img_tumor_mask = nd.interpolation.zoom(img_tumor_mask,spacing_subject,order=0)

        # if volume has more slices than maximum number of slices then select some. if slices are smaller than max number of slices then do zero padding to size
        slice_pre = img_pre[slice_number]
        slice_post = img_post[slice_number]
        slice_bbox = img_bbox[slice_number]
        slice_breast_mask = img_breast_mask[slice_number]
        slice_tumor_mask = img_tumor_mask[slice_number]
        resize_nearest = lambda x,size: torch.nn.functional.interpolate(torch.as_tensor(x).unsqueeze(0).unsqueeze(0),size,mode = 'nearest')[0,0].numpy()
    
        slice_pre = skimage.transform.resize(slice_pre,[self.coronal_size,self.sagital_size],order = 2).clip(-1,1)
        slice_post = skimage.transform.resize(slice_post,[self.coronal_size,self.sagital_size],order = 2).clip(-1,1)
        slice_bbox = resize_nearest(slice_bbox,[self.coronal_size,self.sagital_size])
        slice_breast_mask = resize_nearest(slice_breast_mask,[self.coronal_size,self.sagital_size])
        slice_tumor_mask = resize_nearest(slice_tumor_mask,[self.coronal_size,self.sagital_size])



        assert slice_pre.shape == slice_post.shape
        assert slice_pre.shape == slice_bbox.shape
        assert slice_pre.shape == slice_breast_mask.shape
        assert slice_pre.shape == slice_tumor_mask.shape

        # add channel dimension
        slice_pre = slice_pre[np.newaxis]
        slice_post = slice_post[np.newaxis]
        slice_bbox = slice_bbox[np.newaxis]
        slice_breast_mask = slice_breast_mask[np.newaxis]
        slice_tumor_mask = slice_tumor_mask[np.newaxis]

        slice_pre = torch.as_tensor(slice_pre.copy()).float().contiguous()
        slice_post = torch.as_tensor(slice_post.copy()).float().contiguous()
        slice_bbox = torch.as_tensor(slice_bbox.copy()).long().contiguous()
        slice_breast_mask = torch.as_tensor(slice_breast_mask.copy()).long().contiguous()
        slice_tumor_mask = torch.as_tensor(slice_tumor_mask.copy()).long().contiguous()
        
        if self.dataset_mode == 'A' :
            img = slice_pre
        elif self.dataset_mode == 'B' :
            img = slice_post
        elif self.dataset_mode == 'ABD' :
            img = torch.cat([slice_pre, slice_post, slice_post-slice_pre],axis = 0)
        
        mask = slice_tumor_mask

        assert img.shape[-2:] == mask.shape[-2:], \
        f'Image and mask {example_index} should be the same size, but are {img.shape} and {mask.shape}'


        return {
            'image': img,
            'mask': mask,
            'patient_id': self.patient_ids[index]
        }

    def __len__(self):
        """
        Parameters
        ----------

        """
        return self.dataset_size


    def name(self):
        """
        Parameters
        ----------

        """
        return "DukeBreastCancerMRISubsetDataset"

    def get_labels(self):
        return [int(x) for x in self.example_category]


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