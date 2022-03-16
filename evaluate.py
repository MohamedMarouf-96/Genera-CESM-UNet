import torch
import torch.nn.functional as F
from tqdm import tqdm
import argparse
import logging
import sklearn
from sklearn.metrics import roc_auc_score, average_precision_score
from metrics import fn_based_dcg
import numpy as np


from utils.dice_score import multiclass_dice_coeff, dice_coeff


def evaluate(net, net_single_device, dataloader, device, threeD = False):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    accuracy = 0.0
    predictions_all = []
    gt_all = []

    # iterate over the validation set
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        image, mask_true = batch['image'], batch['mask']
        # move images and labels to correct device and type
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.long)
        is_slice_positive = mask_true.amax(dim=(1,2)).float()

        if threeD :
            mask_true = F.one_hot(mask_true, net_single_device.n_classes).permute(0, 4, 1, 2, 3).float()
        else :
            mask_true = F.one_hot(mask_true, net_single_device.n_classes).permute(0, 3, 1, 2).float()

        with torch.no_grad():
            # predict the mask
            mask_pred, confidence = net(image)

            # convert to one-hot format
            if net_single_device.n_classes == 1:
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                # compute the Dice score
                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False, threeD=threeD)
            else:
                if threeD :
                    mask_pred = F.one_hot(mask_pred.argmax(dim=1), net_single_device.n_classes).permute(0, 4, 1, 2, 3).float()
                else :
                    mask_pred = F.one_hot(mask_pred.argmax(dim=1), net_single_device.n_classes).permute(0, 3, 1, 2).float()
    
                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:, ...], mask_true[:, 1:, ...], reduce_batch_first=False, threeD=threeD)
            accuracy += (((confidence > 0.5) == is_slice_positive)*1.0).mean()
            predictions_all.append(confidence)
            gt_all.append(is_slice_positive)


    

           

    net.train()
    predictions_all = torch.cat(predictions_all).detach().cpu().numpy()
    gt_all = torch.cat(gt_all).detach().cpu().numpy()
    roc_auc = roc_auc_score(gt_all,predictions_all)
    pr_auc = average_precision_score(gt_all,predictions_all)
    logging.info('roc_auc : {}'.format(roc_auc))
    logging.info('PR auc : {}'.format(pr_auc))
    logging.info('accuracy : {}'.format(accuracy/num_val_batches))

    # Fixes a potential division by zero error
    if num_val_batches == 0:
        return dice_score
    return dice_score / num_val_batches


def evaluate_metrics(net,net_single_device, dataset, device, dice_positive_threshold =0.15, dice_negative_threshold=200):
    net.eval()
    num_val_images = len(dataset)
    dice_score = 0
    accuracy = 0.0
    predictions_all = []
    gt_all = []
    dice_all = []


    # iterate over the validation set
    tp_count_image = 0
    fp_count_image = 0
    tn_count_image = 0
    fn_count_image = 0

    gt_patient = dict()
    prediction_patient = dict()

    for i,batch in tqdm(enumerate(dataset), total=num_val_images, desc='Validation round', unit='batch', leave=False):
        image, mask_true, patient_id = batch['image'], batch['mask'], batch['patient_id'][0]
        # move images and labels to correct device and type
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.long)
        is_slice_positive = mask_true.amax(dim=(1,2)).float()
        mask_true = F.one_hot(mask_true, net_single_device.n_classes).permute(0, 3, 1, 2).float()

        with torch.no_grad():
            # predict the mask
            mask_pred, confidence = net(image)

            # convert to one-hot format
            if net_single_device.n_classes == 1:
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                # compute the Dice score
                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
            else:
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net_single_device.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_score_one_image = multiclass_dice_coeff(mask_pred[:, 1:, ...], mask_true[:, 1:, ...], reduce_batch_first=False)

            accuracy += (((confidence > 0.5) == is_slice_positive)*1.0).mean()
            predictions_all.append(confidence)
            gt_all.append(is_slice_positive)
            dice_all.append(dice_score_one_image.unsqueeze(0))
            dice_score += dice_score_one_image
            
            # compute prediction for each image 
            if mask_true[:, 1, ...].sum() > 0 :
                gt_patient[patient_id] = True
                prediction_patient[patient_id] = prediction_patient.get(patient_id,False) or (dice_score_one_image >=  dice_positive_threshold)

                tp_count_image +=  1 if dice_score_one_image >=  dice_positive_threshold  else 0
                fn_count_image +=  0 if dice_score_one_image >=  dice_positive_threshold  else 1

            else :
                gt_patient[patient_id] = gt_patient.get(patient_id, False)
                prediction_patient[patient_id] = prediction_patient.get(patient_id,False) or (mask_pred[:, 1, ...].sum() >=  dice_negative_threshold)

                fp_count_image +=  1 if  mask_pred[:, 1, ...].sum() >=  dice_negative_threshold  else 0
                tn_count_image +=  0 if  mask_pred[:, 1, ...].sum() >=  dice_negative_threshold  else 1

            
    # print(tp_count_image,tn_count_image,fp_count_image,fn_count_image)
    # print(gt_patient)
    # print(prediction_patient)
    tp_count_patient = 0
    fp_count_patient = 0
    tn_count_patient = 0
    fn_count_patient = 0

    for key in gt_patient.keys():
        tp_count_patient += int(gt_patient[key] and prediction_patient[key])
        fp_count_patient += int(not gt_patient[key] and prediction_patient[key])
        tn_count_patient += int(not gt_patient[key] and not prediction_patient[key])
        fn_count_patient += int(gt_patient[key] and not prediction_patient[key])


    net.train()
    predictions_all = torch.cat(predictions_all).detach().cpu().numpy()
    gt_all = torch.cat(gt_all).detach().cpu().numpy()
    dice_all = torch.cat(dice_all).detach().cpu().numpy()
    roc_auc = roc_auc_score(gt_all,predictions_all)
    pr_auc = average_precision_score(gt_all,predictions_all)
    fnddg = fn_based_dcg(gt_all,predictions_all,dice_all)
    positive_only_dice = dice_all.dot(gt_all)/np.sum(gt_all)
    logging.info('roc_auc : {}'.format(roc_auc))
    logging.info('PR auc : {}'.format(pr_auc))
    logging.info('accuracy : {}'.format(accuracy/num_val_images))
    logging.info('fnddg : {}'.format(fnddg))
    logging.info('DPO : {}'.format(positive_only_dice))


    return [dice_score / num_val_images , tp_count_image / (tp_count_image + fn_count_image), fp_count_image / (tn_count_image + fp_count_image),
            tp_count_patient / (tp_count_patient + fn_count_patient), fp_count_patient / (tn_count_patient + fp_count_patient)]

