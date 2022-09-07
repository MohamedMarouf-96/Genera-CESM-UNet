import copy
import numpy as np
import os
import pandas as pd
from tensorflow.keras.utils import Sequence
from PIL import Image, ImageDraw
from skimage.transform import resize
import itertools
import json
import torch
from tqdm import tqdm
class CESM():
    def __init__(self,dataroot,phase,args):
        self.args = args
        self.dataset_mode = args.dataset_mode
        self.root = dataroot
        self.target_size = (256,256)
        self.label_columns =  ['Pathology Classification/ Follow up']
        self.class_names = ['Normal', 'Benign', 'Malignant']
        self.multi_label_classification = True
        # self.dataset_csv_file = os.path.join(self.root,'Radiology_hand_drawn_segmentations_v2.csv')
        # self.dataset_csv_file = os.path.join(self.root,'Radiology manual annotations.xlsx')

        self.dataset_csv_file = os.path.join(self.root,'annotations.csv')
        self.annotations_csv_file = os.path.join(self.root,'Radiology_hand_drawn_segmentations_v2.csv')
        self.dataset_df = pd.read_csv(self.dataset_csv_file)
        self.annotations_df = pd.read_csv(self.annotations_csv_file)

        self.source_image_dir_low = os.path.join(self.root,'images/Low energy images of CDD-CESM')
        self.source_image_dir_sub = os.path.join(self.root,'images/Subtracted images of CDD-CESM')
        self.dataset_df_low = self.dataset_df.loc[['DM' in x for x in self.dataset_df['Image_name']]].sort_values(by=['Image_name'])
        self.dataset_df_sub = self.dataset_df.loc[['CM' in x for x in self.dataset_df['Image_name']]].sort_values(by=['Image_name']) 
        # self.verify_dataset()
        self.low_path, self.sub_path = self.get_images_path(self.dataset_df_low["Image_name"].values),self.get_images_path(self.dataset_df_sub["Image_name"].values)
        
        y_low = self.convert_labels_to_numbers(self.dataset_df_low[self.label_columns].values)
        y_high = self.convert_labels_to_numbers(self.dataset_df_sub[self.label_columns].values)
        self.y = copy.deepcopy(y_low)
        self.y[y_high > y_low] = y_high[y_high > y_low] 
        
        self.side, self.low_image_names, self.sub_image_names =  self.dataset_df['Side'].values, self.dataset_df_low['Image_name'].values,self.dataset_df_sub['Image_name'].values
        
        index = self.get_subset_indecies(len(self.low_path),phase)
        self.low_path = list(itertools.compress(self.low_path,index))
        self.sub_path = list(itertools.compress(self.sub_path,index))
        self.y = list(itertools.compress(self.y,index))
        self.side = list(itertools.compress(self.side,index))
        self.low_image_names = list(itertools.compress(self.low_image_names,index))
        self.sub_image_names = list(itertools.compress(self.sub_image_names,index))
        
        
        self.expected_number = {}
        index = self.filter_cases_with_no_mask()
        self.low_path = list(itertools.compress(self.low_path,index))
        self.sub_path = list(itertools.compress(self.sub_path,index))
        self.y = list(itertools.compress(self.y,index))
        self.side = list(itertools.compress(self.side,index))
        self.low_image_names = list(itertools.compress(self.low_image_names,index))
        self.sub_image_names = list(itertools.compress(self.sub_image_names,index))

        self.augmentor = None

                                                               
    def get_subset_indecies(self,length,phase):
        index = np.random.RandomState(seed=42).permutation(length)
        if phase == 'train':
            # index = sorted(index)
            if self.args.debug :
                index = sorted(index[length//10 * 3:])[0:4]
            else :
                index = sorted(index[length//10 * 3:])
        elif phase == 'val' :
            if self.args.debug :
                index = sorted(index[0:length//10])[0:4]
            else :
                index = sorted(index[0:length//10])
        elif phase == 'test' :
            if self.args.debug :
                index = sorted(index[length//10: length//10 * 3])[0:20]
            else :
                index = sorted(index[length//10: length//10 * 3])
        else :
            raise Exception('split "{}" is not defined'.format(phase))
        index = [True if i in index else False for i in range(length)]
        return index

    def filter_cases_with_no_mask(self):
        selected_indecies = []
        for idx in tqdm(range(len(self.low_path))):
            x_low, x_sub ,original_size = self.load_pair(idx)
            y = self.y[idx]
            if y == 0 :
                selected_indecies.append(idx)
            else :
                original_mask,mask = self.get_segmented_image(original_size, idx)
                if mask.sum() != 0 :
                    self.expected_number[self.low_image_names[idx]] = mask.sum()
                    selected_indecies.append(idx)
        
        selected_indecies = [True if i in selected_indecies else False for i in range(len(self.low_path))]
        return selected_indecies


    def __getitem__(self, idx):
        x_low, x_sub ,original_size = self.load_pair(idx)

        y = self.y[idx]
        if y == 0 :
            mask = np.zeros_like(x_low)
        else :
            original_mask,mask = self.get_segmented_image(original_size, idx)
            # original_mask,mask = self.get_segmented_image(original_size, masks_sub)
            # original_mask = np.array(original_mask)
            # mask = np.array(mask)
            if mask.sum() == 0 :
                # print(mask)
                print(f'image {self.low_image_names[idx]} has no tumor pixels after resizing and {original_mask.sum()} pixels before resizing and exptected {self.expected_number[self.low_image_names[idx]]}')
            # else :
            #     print(f'image {self.image_names[idx]} has {mask.sum()} tumor pixels after resizing and {original_mask.sum()} pixels before resizing')
            
        # x = x_low[np.newaxis]
        x = x_sub[np.newaxis]
        mask = mask[np.newaxis]
        
        x = x * 2 - 1
        x = torch.as_tensor(x.copy()).float().contiguous()
        mask = torch.as_tensor(mask.copy()).long().contiguous()
        
        return {
            'image': x,
            'mask': mask,
            'patient_id': self.dataset_df_low[self.dataset_df_low['Image_name'] == self.low_image_names[idx]]['Patient_ID'].values[0],
            'valid_from' : 0,
            'class' : y} 


    def __len__(self):
        return len(self.low_path)

    # def __getitem__(self, idx):
    #     batch_x_path = self.x_path[idx * self.batch_size:(idx + 1) * self.batch_size]
    #     batch_sides = self.side[idx * self.batch_size:(idx + 1) * self.batch_size]
    #     batch = zip(batch_x_path, batch_sides)
    #     batch_x = np.asarray([self.load_image(x_path, side) for x_path, side in batch])
    #     batch_x = self.transform_batch_images(batch_x)
    #     batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
    #     return batch_x, batch_y

    def load_image(self, image_file, side):
        image_path = os.path.join(self.source_image_dir, image_file)
        image = Image.open(image_path)
        image_array = np.asarray(image.convert("L"))
        image_array = image_array / 255.
        original_size = image_array.shape[:2]
        image_array = resize(image_array, self.target_size)
        return image_array, original_size

    def load_pair(self, idx):
        low_image_path = os.path.join(self.source_image_dir_low, self.low_path[idx])
        sub_image_path = os.path.join(self.source_image_dir_sub, self.sub_path[idx])
        low_image = Image.open(low_image_path)
        sub_image = Image.open(sub_image_path)

        low_image_array = np.asarray(low_image.convert("L"))
        low_image_array = low_image_array / 255.
        low_original_size = low_image_array.shape[:2]
        low_image_array = resize(low_image_array, self.target_size)

        sub_image_array = np.asarray(sub_image.convert("L"))
        sub_image_array = sub_image_array / 255.
        sub_original_size = sub_image_array.shape[:2]
        sub_image_array = resize(sub_image_array, self.target_size)

        assert low_image_array.shape == sub_image_array.shape
        return low_image_array, sub_image_array, low_original_size

    def transform_batch_images(self, batch_x):
        if self.augmenter is not None:
            batch_x = self.augmenter.augment_images(batch_x)
        imagenet_mean = np.array([0.485, 0.456, 0.406])
        imagenet_std = np.array([0.229, 0.224, 0.225])
        batch_x = (batch_x - imagenet_mean) / imagenet_std
        return batch_x

    def get_y_true(self):
        """
        Use this function to get y_true for predict_generator
        In order to get correct y, you have to set shuffle_on_epoch_end=False.

        """
        if self.shuffle:
            raise ValueError("""
            You're trying run get_y_true() when generator option 'shuffle_on_epoch_end' is True.
            """)
        if self.multi_label_classification:
            return self.y[:self.steps * self.batch_size, :]
        else:
            return self.y[:self.steps * self.batch_size]

    def get_class_counts(self):
        return self.class_counts

    def get_sparse_labels(self, y):
        labels = np.zeros(y.shape[0], dtype=int)
        index = 0

        for label in y:
            labels[index] = self.class_names.index(label)
            self.class_counts[labels[index]] += 1
            index += 1

        return labels

    def get_onehot_labels(self, y):
        onehot = np.zeros((y.shape[0], len(self.class_names)))
        index = 0
        for label in y:
            labels = str(label[0]).split("$")
            for l in labels:
                ind = self.class_names.index(l)
                onehot[index, ind] = 1
            index += 1
        return onehot
    
    def get_numerical_labels(self, y):
        onehot = np.zeros((y.shape[0]))
        index = 0
        for label in y:
            labels = str(label[0]).split("$")
            for l in labels:
                ind = self.class_names.index(l)
                onehot[index] = ind
            index += 1
        return onehot

    def convert_labels_to_numbers(self, y):
        if self.multi_label_classification:
            return self.get_numerical_labels(y)
        else:
            return self.get_sparse_labels(y)

    def get_images_names(self):
        return self.image_names

    def get_images_path(self, image_names):
        for i in range(image_names.shape[0]):
            image_names[i] = image_names[i].strip() + '.jpg'
        return image_names

    def verify_dataset(self):
        low_images_names = list(self.dataset_df_low["Image_name"].values)
        sub_images_names = list(self.dataset_df_sub["Image_name"].values)
        assert len(low_images_names) == len(sub_images_names)
        for i in range(len(low_images_names)):
            temp1 = low_images_names[i].replace('DM','').strip()
            temp2 = sub_images_names[i].replace('CM','').strip()
            assert temp1 == temp2, f'{temp1}  ,{temp2}'
        low_images_class = self.convert_labels_to_numbers(self.dataset_df_low[self.label_columns].values)
        sub_images_class = self.convert_labels_to_numbers(self.dataset_df_sub[self.label_columns].values)
        print(low_images_class)
        j = 0
        is_patients_consistent = dict()
        patient_class = dict()
        for i in range(len(low_images_class)):
            temp1 = low_images_class[i]
            temp2 = sub_images_class[i]
            if temp1 != temp2 :
                j+=1
            #     print(temp1,temp2, f'{low_images_names[i]}  ,{sub_images_names[i]}' )
            patient_id = low_images_names[i].split('_')[0]+low_images_names[i].split('_')[1]
            if patient_id in list(patient_class.keys()):
                if temp2 != temp1 :
                    is_patients_consistent.pop(patient_id)
                    patient_class.pop(patient_id)
                else :
                    is_patients_consistent[patient_id] = temp1 == patient_class[patient_id]
                # is_patients_consistent[patient_id] = temp1 == patient_class[patient_id] if temp1 == temp2 else is_patients_consistent[patient_id]
            else :
                if temp1 == temp2 :
                    # is_patients_consistent[patient_id] = temp1 == temp2
                    is_patients_consistent[patient_id] = True
                    patient_class[patient_id] = temp1
            # assert temp1 == temp2, f'{low_images_names[i]}  ,{sub_images_names[i]}'
        print(j)
        print(sum([1 if not x else 0 for x in list(is_patients_consistent.values())]))
        print(len(is_patients_consistent.values()))
        print(len(patient_class))
        exit()


    def on_epoch_end(self):
        if self.shuffle:
            self.random_state += 1
            self.prepare_dataset()


    def get_polygon_formatted(self,x_points, y_points):
        points = []
        for i in range(len(x_points)):
            points.append((x_points[i], y_points[i]))
        return points


    def get_segmented_image(self,image_shape, idx):
        low_masks =  self.annotations_df[self.annotations_df['#filename'] == self.low_image_names[idx]]['region_shape_attributes']
        sub_masks =  self.annotations_df[self.annotations_df['#filename'] == self.sub_image_names[idx]]['region_shape_attributes']
        img_mask = Image.new('L', (image_shape[1], image_shape[0]), 0)
        for low_mask, sub_mask in zip(low_masks, sub_masks):
            if low_mask == '{}' and sub_mask == '{}' :
                continue
            elif low_mask == '{}' and sub_mask != '{}' :
                mask = sub_mask
            elif low_mask != '{}' and sub_mask == '{}' :
                mask = low_mask
            else :
                mask = low_mask
                if json.loads(low_mask) != json.loads(sub_mask): 
                    # print(f'inconsistent masks : \n{json.loads(low_mask)}\n{json.loads(sub_mask)}\n{self.low_image_names[idx]},{self.sub_image_names[idx]}')
                    mask_low = json.loads(low_mask)
                    # mask_sub = json.loads(sub_mask)
                    img_mask_low = self.get_mask_from_dict(mask_low,img_mask)
                    # img_mask_sub = self.get_mask_from_dict(mask_sub,img_mask)
                    img_mask_low = np.array(img_mask_low,dtype= bool)
                    # img_mask_sub = np.array(img_mask_sub,dtype = bool)
                    # # print(f'intersection : {np.sum(img_mask_sub*img_mask_low)}')
                    # # print(f'union : {np.sum(img_mask_sub + img_mask_low)}')
                    # # iou = np.sum(img_mask_sub*img_mask_low) / float(np.sum(img_mask_low + img_mask_sub))
                    # print(np.sum(img_mask_sub),np.sum(img_mask_low))
                    if np.sum(img_mask_low) == 0 :
                        mask = sub_mask
            mask = json.loads(mask)
            img_mask = self.get_mask_from_dict(mask,img_mask)
        img_mask_orig = np.array(img_mask)
        img_mask = resize(img_mask_orig, self.target_size,order=1)
        img_mask[img_mask > 0] = 1
        return img_mask_orig, img_mask

    def get_mask_from_dict(self,mask,img_mask):
        img_mask = copy.deepcopy(img_mask)
        if mask['name'] == 'polygon':
            poly = self.get_polygon_formatted(mask['all_points_x'], mask['all_points_y'])
            ImageDraw.Draw(img_mask).polygon(poly, outline=1, fill=1)
        elif mask['name'] == 'ellipse' or mask['name'] == 'circle' or mask['name'] == 'point':
            if mask['name'] == 'circle':
                mask['rx'] = mask['ry'] = mask['r']
            elif mask['name'] == 'point':
                mask['rx'] = mask['ry'] = 25
            ellipse = [(mask['cx'] - mask['rx'], mask['cy'] - mask['ry']),
                    (mask['cx'] + mask['rx'], mask['cy'] + mask['ry'])]
            ImageDraw.Draw(img_mask).ellipse(ellipse, outline=1, fill=1)
        return img_mask