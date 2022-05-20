import einops
import torch
import torch.nn.functional as F
from tqdm import tqdm
import argparse
import logging
import sklearn
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, average_precision_score
from metrics import fn_based_dcg
import numpy as np


from utils.dice_score import dice_coeff


def evaluate(net, net_single_device, dataloader, device, args):

    net.eval()
    num_val_batches = len(dataloader)
    localization_only_dice_all = []

    dice_all = []
    predictions_all = []
    gt_all = []


    # iterate over the validation set
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        image, mask_true = batch['image'], batch['mask']
        # move images and labels to correct device and type
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.float)

        if args.d3 :
            mask_true = mask_true[:,:,args.axial_size // 4: - args.axial_size //4]
            is_slice_positive = mask_true.amax(dim=(1,3,4)).float()
            mask_true = einops.rearrange(mask_true,'b c z ... -> (b z) c ...')
            is_slice_positive = einops.rearrange(is_slice_positive,'b z -> (b z)')
            # is_patient_positive = is_slice_positive.amax(dim= 1).float()
        else :
            is_slice_positive = mask_true.amax(dim=(1,2,3)).float()


        with torch.no_grad():
            # predict the mask
            if args.d3 :
                masks_pred, slice_class, patient_class = net(image)
                masks_pred = masks_pred[:,:,args.axial_size // 4: - args.axial_size //4]
                slice_class = slice_class[:,:, args.axial_size // 4: - args.axial_size //4]
                slice_class = slice_class * patient_class
                masks_pred = einops.rearrange(masks_pred,'b c z ... -> (b z) c ...')
                slice_class = einops.rearrange(slice_class,'b c z ... -> (b z) c ...')
            else :
                masks_pred, slice_class = net(image)
            
            # convert to one-hot format
            masks_pred_final = (masks_pred*slice_class >= 0.5).float()
            confidence = slice_class

            # compute the Dice score
            dice_score_tmp = dice_coeff(masks_pred_final, mask_true, reduce_batch_first=False, reduce_batch=False)
            dice_all.append(dice_score_tmp)

            dice_score_tmp = dice_coeff((masks_pred >= 0.5).float(), mask_true, reduce_batch_first=False, reduce_batch=False)
            localization_only_dice_all.append(dice_score_tmp)

            predictions_all.append(confidence)
            gt_all.append(is_slice_positive)



    

           

    net.train()
    predictions_all = torch.cat(predictions_all).reshape(-1)
    pred_class_all = (predictions_all >= 0.5).float().detach().cpu().numpy()
    predictions_all = predictions_all.detach().cpu().numpy()
    gt_all = torch.cat(gt_all).detach().cpu().numpy()
    dice_all = torch.cat(dice_all,dim=0).detach().cpu().numpy()
    localization_only_dice_all = torch.cat(localization_only_dice_all,dim=0).detach().cpu().numpy()

    roc_auc = roc_auc_score(gt_all,predictions_all)
    pr_auc = average_precision_score(gt_all,predictions_all)
    accuracy = (pred_class_all == gt_all).mean()
    balanced_accuracy = balanced_accuracy_score(gt_all, pred_class_all)

    tpr = (pred_class_all*gt_all)[gt_all > 0].mean()
    fpr = ( pred_class_all !=  gt_all)[gt_all == 0].mean()
    logging.info('accuracy : {}'.format(accuracy))
    logging.info('balanced accuracy : {}'.format(balanced_accuracy))
    logging.info('roc_auc : {}'.format(roc_auc))
    logging.info('PR auc : {}'.format(pr_auc))
    logging.info(f'TPR : {tpr}')
    logging.info(f'FPR : {fpr}')
    logging.info(f'POD : {(dice_all*gt_all)[gt_all > 0].mean()}')
    logging.info(f'localization POD : {(localization_only_dice_all*gt_all)[gt_all > 0].mean()}')

    # Fixes a potential division by zero error
    return dice_all.mean()

def evaluate_metrics(net, net_single_device, dataloader, device, args,dice_positive_threshold =0.15, dice_negative_threshold=200):

    net.eval()
    num_val_batches = len(dataloader)
    localization_only_dice_all = []

    dice_all = []
    predictions_all = []
    gt_all = []

    predictions_patient_all = dict()
    gt_patient_all = dict()


    # iterate over the validation set
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        image, mask_true, patient_ids, valid_from = batch['image'], batch['mask'], batch['patient_id'],batch['valid_from'] 
        # move images and labels to correct device and type
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.float)

        if args.d3 :
            patient_ids_temp = []
            for patient_id,slices_valid_from in zip(patient_ids,valid_from):
                patient_ids_temp += [patient_id]* (args.axial_size//2 - slices_valid_from)
            patient_ids = patient_ids_temp
            mask_true = mask_true[:,:,args.axial_size // 4: - args.axial_size //4]
            is_slice_positive = mask_true.amax(dim=(1,3,4)).float()
            mask_true_temp = []
            is_slice_positive_temp = []
            for patient_mask_true,patient_is_slice_positive,patient_valid_from in zip(mask_true, is_slice_positive,valid_from) :
                mask_true_temp.append(einops.rearrange(patient_mask_true,'c z ... -> z c ...')[patient_valid_from:])
                is_slice_positive_temp.append(patient_is_slice_positive[patient_valid_from:])
            mask_true = torch.cat(mask_true_temp,dim=0)
            is_slice_positive = torch.cat(is_slice_positive_temp,dim=0)
        else :
            is_slice_positive = mask_true.amax(dim=(1,2,3)).float()

        # mask_true = torch.nn.functional.interpolate(mask_true,size = (256,256),mode = 'nearest')

        with torch.no_grad():
            # predict the mask
            if args.d3 :
                masks_pred, slice_class, patient_class = net(image)
                masks_pred = masks_pred[:,:,args.axial_size // 4: - args.axial_size //4]
                slice_class = slice_class[:,:, args.axial_size // 4: - args.axial_size //4]
                slice_class = slice_class * patient_class
                mask_pred_temp = []
                slice_class_temp = []
                for patient_mask_pred,patient_slice_class,patient_valid_from in zip(masks_pred, slice_class, valid_from) :
                    mask_pred_temp.append(einops.rearrange(patient_mask_pred,'c z ... -> z c ...')[patient_valid_from:])
                    slice_class_temp.append(einops.rearrange(patient_slice_class,'c z ... -> z c ...')[patient_valid_from:])
                masks_pred = torch.cat(mask_pred_temp,dim=0)
                slice_class = torch.cat(slice_class_temp,dim=0)
                # masks_pred = einops.rearrange(masks_pred,'b c z ... -> (b z) c ...')
                # slice_class = einops.rearrange(slice_class,'b c z ... -> (b z) c ...')
            else :
                masks_pred, slice_class = net(image)
            
            masks_pred = torch.nn.functional.interpolate(masks_pred,size = (256,256), mode = 'nearest')

            
            # convert to one-hot format
            masks_pred_final = (masks_pred*slice_class >= 0.5).float()
            confidence = slice_class

            # compute the Dice score
            dice_score_tmp = dice_coeff((masks_pred >= 0.5).float(), mask_true, reduce_batch_first=False, threeD=args.d3, reduce_batch=False)
            localization_only_dice_all.append(dice_score_tmp)

            dice_score_tmp = dice_coeff(masks_pred_final, mask_true, reduce_batch_first=False, threeD=args.d3, reduce_batch=False)
            dice_all.append(dice_score_tmp)



            predictions_all.append(confidence)
            gt_all.append(is_slice_positive)
            for i,patient_id in enumerate(patient_ids) :
                gt_patient_all[patient_id] = is_slice_positive[i] or gt_patient_all.get(patient_id,torch.as_tensor(0,device='cuda'))
                if bool(is_slice_positive[i]) :
                    predictions_patient_all[patient_id] = predictions_patient_all.get(patient_id,False) or (dice_score_tmp[i] >=  dice_positive_threshold)
                else :
                    predictions_patient_all[patient_id] = predictions_patient_all.get(patient_id,False) or (masks_pred_final[i, 0, ...].sum() >=  dice_negative_threshold)



    
    predictions_patient_all_list = list()
    gt_patient_all_list = list()
    for patient_id in predictions_patient_all.keys():
        predictions_patient_all_list.append(predictions_patient_all[patient_id])
        gt_patient_all_list.append(gt_patient_all[patient_id])

    
    net.train()
    predictions_all = torch.cat(predictions_all).reshape(-1)
    pred_class_all = (predictions_all >= 0.5).float().detach().cpu().numpy()
    predictions_all = predictions_all.detach().cpu().numpy()
    gt_all = torch.cat(gt_all).detach().cpu().numpy()
    dice_all = torch.cat(dice_all,dim=0).detach().cpu().numpy()
    localization_only_dice_all = torch.cat(localization_only_dice_all,dim=0).detach().cpu().numpy()


    patient_pred_class_all = torch.stack(predictions_patient_all_list).reshape(-1)
    patient_pred_class_all = (patient_pred_class_all >= 0.5).float().detach().cpu().numpy()
    patient_gt_all = torch.stack(gt_patient_all_list).detach().cpu().numpy()


    roc_auc = roc_auc_score(gt_all,predictions_all)
    pr_auc = average_precision_score(gt_all,predictions_all)

    accuracy = (pred_class_all == gt_all).mean()
    balanced_accuracy = balanced_accuracy_score(gt_all, pred_class_all)
    tpr = (pred_class_all*gt_all)[gt_all > 0].mean()
    fpr = (pred_class_all != gt_all)[gt_all == 0].mean()

    # logging.info(f'number of patients is {len(patient_gt_all)}')
    # logging.info(f'number of slices is {len(gt_all)}')

    patient_accuracy = (patient_pred_class_all == patient_gt_all).mean()
    patient_balanced_accuracy = balanced_accuracy_score(patient_gt_all, patient_pred_class_all)
    patient_tpr = (patient_pred_class_all*patient_gt_all)[patient_gt_all > 0].mean()
    patient_fpr = (patient_pred_class_all != patient_gt_all)[patient_gt_all == 0].mean()

    # logging.info('accuracy : {}'.format(accuracy))
    logging.info('balanced accuracy : {}'.format(balanced_accuracy))
    # logging.info('roc_auc : {}'.format(roc_auc))
    # logging.info('PR auc : {}'.format(pr_auc))
    logging.info(f'TPR : {tpr}')
    logging.info(f'FPR : {fpr}')
    # logging.info('patient accuracy : {}'.format(patient_accuracy))
    logging.info('patient balanced accuracy : {}'.format(patient_balanced_accuracy))
    logging.info(f'patient TPR : {patient_tpr}')
    logging.info(f'patient FPR : {patient_fpr}')
    logging.info(f'POD : {(dice_all*gt_all)[gt_all > 0].mean()}')
    logging.info(f'localization POD : {(localization_only_dice_all*gt_all)[gt_all > 0].mean()}')

    # Fixes a potential division by zero error
    return [dice_all.mean(), tpr, fpr,patient_tpr,patient_fpr]



def evaluate_metrics_new(net,net_single_device, dataset, device, dice_positive_threshold =0.15, dice_negative_threshold=200):
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
    logging.info(f'TP : {(predictions_all >= 0.5)[gt_all > 0].mean()}')
    logging.info(f'TN : {(predictions_all < 0.5)[gt_all == 0].mean()}')
    logging.info('fnddg : {}'.format(fnddg))
    logging.info('DPO : {}'.format(positive_only_dice))


    return [dice_score / num_val_images , tp_count_image / (tp_count_image + fn_count_image), fp_count_image / (tn_count_image + fp_count_image),
            tp_count_patient / (tp_count_patient + fn_count_patient), fp_count_patient / (tn_count_patient + fp_count_patient)]

