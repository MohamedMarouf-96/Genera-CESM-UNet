import itertools
import os.path
from random import random
import typing
from xml.etree.ElementInclude import include

import numpy as np
from tqdm import tqdm
import torch
import glob
import SimpleITK as sitk
from scipy import ndimage as nd
import skimage
import json
import random
import torchdatasets as td

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


class KasperNormADataset():
    """ "
    a class used to load 3D data from the patients and slices included in kasper's final training set. it has the same patient test set and training set for backward compatability. the dataset has both the options to load all slices included in the breast mask or only the slices
    included in kasper's split or x random slices where x is the number of slices in kasper's dataset. the slices in this dataset have only normA applied without n4 bias field correction or normB

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

    def __init__(self, dataroot, phase , args):
        """
        Parameters
        ----------
        opt : BaseOptions
            the options for this training run
        """
        dataset_mode = args.dataset_mode
        experimet_type = args.experiment_type
        self.root = dataroot
        self.args = args


        dir_mri = f"Duke-Breast-Cancer-MRI/manifest-1607053360376/3D-Dataset-Kasper/"
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

        with open('kasper_slices.json','r') as fh :
            kasper_slices_tmp  = json.load(fh)
            kasper_slices = dict()
            kasper_slices['patients']  = kasper_slices_tmp[f'{"test" if phase == "test" else "train"}_patients']
            kasper_slices['slices']  = kasper_slices_tmp[f'{"test" if phase == "test" else "train"}_slices']
            kasper_slices['categorys']  = kasper_slices_tmp[f'{"test" if phase == "test" else "train"}_category']



        self.patient_id_category, self.index_of_patient_id, self.patient_id_per_example, self.slice_numbers_per_example, self.example_category = self.get_positive_stats()

        if not self.args.full_set :

            kasper_examples_ordered_pairs = [(x,int(y),z) for x,y,z in zip(kasper_slices['patients'],kasper_slices['slices'], kasper_slices['categorys'])]
            all_examples_ordered_pairs = [(x,int(y),z) for x,y,z in zip(self.patient_id_per_example,self.slice_numbers_per_example,self.example_category)]
            subset_examples_ordered_pairs = list(set(kasper_examples_ordered_pairs) & set(all_examples_ordered_pairs))
            not_included_examples_ordered_paris = list(set(kasper_examples_ordered_pairs) - set(subset_examples_ordered_pairs))
            all_examples_ordered_pairs_filtered = list(set(all_examples_ordered_pairs) - set(not_included_examples_ordered_paris))
            positive_examples_ordered_pairs = [x for x in all_examples_ordered_pairs_filtered if x[2]]
            negative_examples_ordered_pairs = [x for x in all_examples_ordered_pairs_filtered if not x[2]]
            
            if experimet_type == 'expA' :
                selected = subset_examples_ordered_pairs
            elif experimet_type == 'expB' :
                positive_number = sum([int(x[2]) for x in subset_examples_ordered_pairs])
                negative_nubmer = len(subset_examples_ordered_pairs) - positive_number
                random.seed(42)
                selected_positives = random.sample(positive_examples_ordered_pairs,positive_number)
                selected_negatives = random.sample(negative_examples_ordered_pairs,negative_nubmer)
                selected = selected_negatives + selected_positives
            elif experimet_type == 'expC' :
                positive_number = sum([int(x[2]) for x in all_examples_ordered_pairs_filtered])
                negative_nubmer = positive_number
                random.seed(42)
                selected_positives = random.sample(positive_examples_ordered_pairs,positive_number)
                selected_negatives = random.sample(negative_examples_ordered_pairs,negative_nubmer)
                selected = selected_negatives + selected_positives
            elif experimet_type in ['expE','expD','expF']  :
                if phase == 'train':
                    self.positive_number = sum([int(x[2]) for x in all_examples_ordered_pairs_filtered])
                    selected = all_examples_ordered_pairs_filtered
                else :
                    positive_number = sum([int(x[2]) for x in all_examples_ordered_pairs_filtered])
                    negative_nubmer = positive_number
                    random.seed(42)
                    selected_positives = random.sample(positive_examples_ordered_pairs,positive_number)
                    selected_negatives = random.sample(negative_examples_ordered_pairs,negative_nubmer)
                    selected = selected_negatives + selected_positives


            else : 
                raise Exception('not implemeted error')        


            self.patient_id_per_example = [x[0] for x in selected]
            self.slice_numbers_per_example = [x[1] for x in selected]
            self.example_category = [x[2] for x in selected]


        print('only positive examples are used : {}'.format(not (False in self.patient_id_category)))

        assert len(self.patient_id_per_example) == len(self.slice_numbers_per_example)
        assert len(self.example_category) == len(self.slice_numbers_per_example)


        self.dataset_size = len(self.slice_numbers_per_example)
    
    
    def get_subset_indecies(self,length,phase):
        index = np.random.RandomState(seed=42).permutation(length)
        if phase == 'train':
            # index = sorted(index)
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
        include_example = list()
        for i,mask_path in tqdm(enumerate(self.segmentation_paths),total=len(self.segmentation_paths)) :
            I_breast_mask = sitk.ReadImage(self.breast_mask_paths[i])
            img_breast_mask = np.array(sitk.GetArrayFromImage(I_breast_mask))
            I_tumor_mask = sitk.ReadImage(mask_path)
            img_tumor_mask = np.array(sitk.GetArrayFromImage(I_tumor_mask))
            slice_numbers_per_example += (list(range(len(img_tumor_mask))))
            patient_id_per_example += [self.patient_ids[i]]*len(img_tumor_mask)
            index_of_patient_id[self.patient_ids[i]] = i
            example_category += list(img_tumor_mask.sum(axis= (1,2)) > 0)
            include_example += list(np.amax(img_breast_mask,axis= (1,2)) > 0)
            patient_id_category.append(img_tumor_mask.sum() > 0)


        if self.args.full_set :
            patient_id_per_example = [patient_id_per_example[i] for i in range(len(patient_id_per_example)) if include_example[i]]
            slice_numbers_per_example = [slice_numbers_per_example[i] for i in range(len(slice_numbers_per_example)) if include_example[i]]
            example_category = [example_category[i] for i in range(len(example_category)) if include_example[i]]
            
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

        # TODO : this is normalization for each and every slice alone
        slice_pre = imgnorm(slice_pre) * 2 - 1
        slice_post = imgnorm(slice_post) * 2 - 1

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
            'patient_id': self.patient_ids[index],
            'valid_from': 0,
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
        
class KasperN4Dataset():
    """ "
    a class used to load 3D data from the patients and slices included in kasper's final training set. it has the same patient test set and training set for backward compatability. the dataset has both the options to load all slices included in the breast mask or only the slices
    included in kasper's split or x random slices where x is the number of slices in kasper's dataset. the slices in this dataset have  ormA and n4 bias field applied without normB

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

    def __init__(self, dataroot, phase , args):
        """
        Parameters
        ----------
        opt : BaseOptions
            the options for this training run
        """
        dataset_mode = args.dataset_mode
        experimet_type = args.experiment_type
        self.root = dataroot
        self.args = args


        dir_mri = f"Duke-Breast-Cancer-MRI/manifest-1607053360376/3D-Dataset-KasperN4/"
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

        with open('kasper_slices.json','r') as fh :
            kasper_slices_tmp  = json.load(fh)
            kasper_slices = dict()
            kasper_slices['patients']  = kasper_slices_tmp[f'{"test" if phase == "test" else "train"}_patients']
            kasper_slices['slices']  = kasper_slices_tmp[f'{"test" if phase == "test" else "train"}_slices']
            kasper_slices['categorys']  = kasper_slices_tmp[f'{"test" if phase == "test" else "train"}_category']



        self.patient_id_category, self.index_of_patient_id, self.patient_id_per_example, self.slice_numbers_per_example, self.example_category = self.get_positive_stats()
        if not self.args.full_set :

            kasper_examples_ordered_pairs = [(x,int(y),z) for x,y,z in zip(kasper_slices['patients'],kasper_slices['slices'], kasper_slices['categorys'])]
            all_examples_ordered_pairs = [(x,int(y),z) for x,y,z in zip(self.patient_id_per_example,self.slice_numbers_per_example,self.example_category)]
            subset_examples_ordered_pairs = list(set(kasper_examples_ordered_pairs) & set(all_examples_ordered_pairs))
            not_included_examples_ordered_paris = list(set(kasper_examples_ordered_pairs) - set(subset_examples_ordered_pairs))
            all_examples_ordered_pairs_filtered = list(set(all_examples_ordered_pairs) - set(not_included_examples_ordered_paris))
            positive_examples_ordered_pairs = [x for x in all_examples_ordered_pairs_filtered if x[2]]
            negative_examples_ordered_pairs = [x for x in all_examples_ordered_pairs_filtered if not x[2]]
            
            if experimet_type == 'expA' :
                selected = subset_examples_ordered_pairs
            elif experimet_type == 'expB' :
                positive_number = sum([int(x[2]) for x in subset_examples_ordered_pairs])
                negative_nubmer = len(subset_examples_ordered_pairs) - positive_number
                random.seed(42)
                selected_positives = random.sample(positive_examples_ordered_pairs,positive_number)
                selected_negatives = random.sample(negative_examples_ordered_pairs,negative_nubmer)
                selected = selected_negatives + selected_positives
            elif experimet_type == 'expC' :
                positive_number = sum([int(x[2]) for x in all_examples_ordered_pairs_filtered])
                negative_nubmer = positive_number
                random.seed(42)
                selected_positives = random.sample(positive_examples_ordered_pairs,positive_number)
                selected_negatives = random.sample(negative_examples_ordered_pairs,negative_nubmer)
                selected = selected_negatives + selected_positives
            elif experimet_type in ['expE','expD','expF']  :
                if phase == 'train':
                    self.positive_number = sum([int(x[2]) for x in all_examples_ordered_pairs_filtered])
                    selected = all_examples_ordered_pairs_filtered
                else :
                    positive_number = sum([int(x[2]) for x in all_examples_ordered_pairs_filtered])
                    negative_nubmer = positive_number
                    random.seed(42)
                    selected_positives = random.sample(positive_examples_ordered_pairs,positive_number)
                    selected_negatives = random.sample(negative_examples_ordered_pairs,negative_nubmer)
                    selected = selected_negatives + selected_positives
            else : 
                raise Exception('not implemeted error')        


            self.patient_id_per_example = [x[0] for x in selected]
            self.slice_numbers_per_example = [x[1] for x in selected]
            self.example_category = [x[2] for x in selected]


        print('only positive examples are used : {}'.format(not (False in self.patient_id_category)))

        assert len(self.patient_id_per_example) == len(self.slice_numbers_per_example)
        assert len(self.example_category) == len(self.slice_numbers_per_example)


        if hasattr(self,'positive_number') :
            self.dataset_size = 2* self.positive_number
        else :
            self.dataset_size = len(self.slice_numbers_per_example)
    
    
    def get_subset_indecies(self,length,phase):
        index = np.random.RandomState(seed=42).permutation(length)
        if phase == 'train':
            # index = sorted(index)
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
        include_example = list()
        for i,mask_path in tqdm(enumerate(self.segmentation_paths),total=len(self.segmentation_paths)) :
            I_breast_mask = sitk.ReadImage(self.breast_mask_paths[i])
            img_breast_mask = np.array(sitk.GetArrayFromImage(I_breast_mask))
            I_tumor_mask = sitk.ReadImage(mask_path)
            img_tumor_mask = np.array(sitk.GetArrayFromImage(I_tumor_mask))
            slice_numbers_per_example += (list(range(len(img_tumor_mask))))
            patient_id_per_example += [self.patient_ids[i]]*len(img_tumor_mask)
            index_of_patient_id[self.patient_ids[i]] = i
            example_category += list(img_tumor_mask.sum(axis= (1,2)) > 0)
            include_example += list(np.amax(img_breast_mask,axis= (1,2)) > 0)
            patient_id_category.append(img_tumor_mask.sum() > 0)

        if self.args.full_set:

            patient_id_per_example = [patient_id_per_example[i] for i in range(len(patient_id_per_example)) if include_example[i]]
            slice_numbers_per_example = [slice_numbers_per_example[i] for i in range(len(slice_numbers_per_example)) if include_example[i]]
            example_category = [example_category[i] for i in range(len(example_category)) if include_example[i]]
        
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
            'patient_id': self.patient_ids[index],
            'valid_from': 0
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

'''class KasperN4Dataset():
    """ "
    a class used to load 3D data from the patients and slices included in kasper's final training set. it has the same patient test set and training set for backward compatability. the dataset has both the options to load all slices included in the breast mask or only the slices
    included in kasper's split or x random slices where x is the number of slices in kasper's dataset. the slices in this dataset have normA and N4 bias bias field correction applied to them without normB

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

    def __init__(self, dataroot, phase , args):
        """
        Parameters
        ----------
        opt : BaseOptions
            the options for this training run
        """
        dataset_mode = args.dataset_mode
        experimet_type = args.experiment_type
        self.root = dataroot
        self.args = args


        dir_mri = f"Duke-Breast-Cancer-MRI/manifest-1607053360376/3D-Dataset-KasperN4/"
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

        with open('kasper_slices.json','r') as fh :
            kasper_slices_tmp  = json.load(fh)
            kasper_slices = dict()
            kasper_slices['patients']  = kasper_slices_tmp[f'{"test" if phase == "test" else "train"}_patients']
            kasper_slices['slices']  = kasper_slices_tmp[f'{"test" if phase == "test" else "train"}_slices']
            kasper_slices['categorys']  = kasper_slices_tmp[f'{"test" if phase == "test" else "train"}_category']



        self.patient_id_category, self.index_of_patient_id, self.patient_id_per_example, self.slice_numbers_per_example, self.example_category = self.get_positive_stats()

        kasper_examples_ordered_pairs = [(x,int(y),z) for x,y,z in zip(kasper_slices['patients'],kasper_slices['slices'], kasper_slices['categorys'])]
        all_examples_ordered_pairs = [(x,int(y),z) for x,y,z in zip(self.patient_id_per_example,self.slice_numbers_per_example,self.example_category)]

        subset_examples_ordered_pairs = list(set(kasper_examples_ordered_pairs) & set(all_examples_ordered_pairs))
        not_included_examples_ordered_paris = list(set(kasper_examples_ordered_pairs) - set(subset_examples_ordered_pairs))


        all_examples_ordered_pairs_filtered = list(set(all_examples_ordered_pairs) - set(not_included_examples_ordered_paris))
        positive_examples_ordered_pairs = [x for x in all_examples_ordered_pairs_filtered if x[2]]
        negative_examples_ordered_pairs = [x for x in all_examples_ordered_pairs_filtered if not x[2]]
        
        if experimet_type == 'expA' :
            selected = subset_examples_ordered_pairs
        elif experimet_type == 'expB' :
            positive_number = sum([int(x[2]) for x in subset_examples_ordered_pairs])
            negative_nubmer = len(subset_examples_ordered_pairs) - positive_number
            random.seed(42)
            selected_positives = random.sample(positive_examples_ordered_pairs,positive_number)
            selected_negatives = random.sample(negative_examples_ordered_pairs,negative_nubmer)
            selected = selected_negatives + selected_positives
        elif experimet_type == 'expC' :
            positive_number = sum([int(x[2]) for x in all_examples_ordered_pairs_filtered])
            negative_nubmer = positive_number
            random.seed(42)
            selected_positives = random.sample(positive_examples_ordered_pairs,positive_number)
            selected_negatives = random.sample(negative_examples_ordered_pairs,negative_nubmer)
            selected = selected_negatives + selected_positives
        else : 
            raise Exception('not implemeted error')        


        self.patient_id_per_example = [x[0] for x in selected]
        self.slice_numbers_per_example = [x[1] for x in selected]
        self.example_category = [x[1] for x in selected]

        self.selected_slices_per_patient = dict()

        for example in selected :
            paitent_list = self.selected_slices_per_patient.get(example[0],list())
            paitent_list.append(example)
            self.selected_slices_per_patient[example[0]] = paitent_list

        print('only positive examples are used : {}'.format(not (False in self.patient_id_category)))

        assert len(self.patient_id_per_example) == len(self.slice_numbers_per_example)
        assert len(self.example_category) == len(self.slice_numbers_per_example)


        self.dataset_size = len(self.pre_paths)
    
    
    def get_subset_indecies(self,length,phase):
        index = np.random.RandomState(seed=42).permutation(length)
        if phase == 'train':
            # index = sorted(index)
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
        include_example = list()
        for i,mask_path in tqdm(enumerate(self.segmentation_paths),total=len(self.segmentation_paths)) :
            I_breast_mask = sitk.ReadImage(self.breast_mask_paths[i])
            img_breast_mask = np.array(sitk.GetArrayFromImage(I_breast_mask))
            I_tumor_mask = sitk.ReadImage(mask_path)
            img_tumor_mask = np.array(sitk.GetArrayFromImage(I_tumor_mask))
            slice_numbers_per_example += (list(range(len(img_tumor_mask))))
            patient_id_per_example += [self.patient_ids[i]]*len(img_tumor_mask)
            index_of_patient_id[self.patient_ids[i]] = i
            example_category += list(img_tumor_mask.sum(axis= (1,2)) > 0)
            include_example += list(np.amax(img_breast_mask,axis= (1,2)) > 0)
            patient_id_category.append(img_tumor_mask.sum() > 0)


        # patient_id_per_example = [patient_id_per_example[i] for i in range(len(patient_id_per_example)) if include_example[i]]
        # slice_numbers_per_example = [slice_numbers_per_example[i] for i in range(len(slice_numbers_per_example)) if include_example[i]]
        # example_category = [example_category[i] for i in range(len(example_category)) if include_example[i]]
        
        return  patient_id_category, index_of_patient_id,patient_id_per_example, slice_numbers_per_example, example_category


    def __getitem__(self, index):

        patient_id = self.patient_ids[index]

        assert patient_id in self.pre_paths[index]

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


        # resize sagital and coronal dimension to common size
        zoom_factor = (1,self.coronal_size / img_pre.shape[1] ,self.sagital_size / img_pre.shape[2])

        img_pre = nd.interpolation.zoom(img_pre,zoom_factor,order=1)
        img_post = nd.interpolation.zoom(img_post,zoom_factor,order=1)
        img_bbox = nd.interpolation.zoom(img_bbox,zoom_factor,order=0)
        img_breast_mask = nd.interpolation.zoom(img_breast_mask,zoom_factor,order=0)
        img_tumor_mask = nd.interpolation.zoom(img_tumor_mask,zoom_factor,order=0)

        selected_slices = self.selected_slices_per_patient[patient_id]
        selected_slices_indecies = [int(x[1]) for x in selected_slices]

        if len(selected_slices_indecies) == 0 :
            return ListDataset([])

            
        slices_pre = torch.as_tensor(img_pre[selected_slices_indecies,np.newaxis].copy()).float().contiguous()
        slices_post = torch.as_tensor(img_post[selected_slices_indecies,np.newaxis].copy()).float().contiguous()
        slices_bbox = torch.as_tensor(img_bbox[selected_slices_indecies,np.newaxis].copy()).long().contiguous()
        slices_breast_mask = torch.as_tensor(img_breast_mask[selected_slices_indecies,np.newaxis].copy()).long().contiguous()
        slices_tumor_mask = torch.as_tensor(img_tumor_mask[selected_slices_indecies,np.newaxis].copy()).long().contiguous()


        if self.dataset_mode == 'A' :
            slices_img = slices_pre
        elif self.dataset_mode == 'B' :
            slices_img = slices_post
        elif self.dataset_mode == 'ABD' :
            slices_img = torch.cat([slices_pre, slices_post, slices_post-slices_pre],axis = 1)
        
        slices_mask = slices_tumor_mask

        assert slices_img.shape[-2:] == slices_mask.shape[-2:], \
        f'Image and mask {index} should be the same size, but are {slices_img.shape} and {slices_mask.shape}'


        list_of_examples = []
        for i in range(len(slices_img)):
            list_of_examples.append(
                {'image': slices_img[i],
                'mask': slices_mask[i],
                'patient_id': patient_id}
                )


        return ListDataset(list_of_examples)
        

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
        return "KasperN4Dataset"

    def get_labels(self):
        return [int(x) for x in self.example_category]'''

class KasperNormBDataset():
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

class TorchListDataset(torch.utils.data.Dataset):
    r"""Dataset wrapping tensors.

    Each sample will be retrieved by indexing tensors along the first dimension.

    Args:
        *lists (List): lists that have the same size of the first dimension.
    """
    def __init__(self, *lists: typing.List) -> None:
        assert all(len(lists[0]) == len(cur_list) for cur_list in lists), "Size mismatch between lists"
        self.lists = lists

    def __getitem__(self, index):
        if len(self.lists) == 1 :
            return self.lists[0][index]
        else :
            return tuple(cur_list[index] for cur_list in self.lists)

    def __len__(self):
        return len(self.lists[0])

class ListDataset(TorchListDataset, td.Dataset):
    r"""**Dataset wrapping** `torch.tensors` **.**
    `cache`, `map` etc. enabled version of `torch.utils.data.TensorDataset <https://pytorch.org/docs/stable/data.html#torch.utils.data.TensorDataset>`__.
    Parameters:
    -----------
    *tensors : torch.Tensor
            List of `tensors` to be wrapped.
    """

    def __init__(self, *tensors):
        td.Dataset.__init__(self)
        TorchListDataset.__init__(self, *tensors)

def Norm_Zscore(img):
    img= (img-np.mean(img))/np.std(img) 
    return img


def imgnorm(N_I,index1=0.001,index2=0.001):
    N_I = N_I.astype(np.float32)
    I_sort = np.sort(N_I.flatten())
    I_min = I_sort[int(index1*len(I_sort))]
    I_max = I_sort[-int(index2*len(I_sort))]
    
    N_I =1.0*(N_I-I_min)/(I_max-I_min + 1e-10)
    N_I[N_I>1.0]=1.0
    N_I[N_I<0.0]=0.0
    
    return N_I