import argparse
import copy
import logging
import os
import sys
from pathlib import Path

import numpy as np
import utils

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.data_loading import DukeBreastCancerMRIDataset, DBCMRIDataset
from utils.kasper_datasets import KasperN4Dataset,KasperNormADataset,KasperNormBDataset
from utils.dice_score import dice_loss
from evaluate import evaluate, evaluate_metrics
from unet_local import UNet3d
from unet_local import UNet
# from unet_local import UNet, UNet3d
from utils.my_collate_fn import my_collate_fn
from utils.balanced_sampling import get_balanced_weighted_sampler, ClassBalancedRandomSampler
import torchdatasets as td

dir_root = Path('/data/')

def evaluate_net(net,
                 net_single_device,
                 device,
                 args):
    
    if args.dataset == 'duke3d' :
        val_set = DukeBreastCancerMRIDataset(dir_root, 'test', args)
    elif args.dataset == 'duke2d' :
        val_set = DBCMRIDataset(dir_root, 'test', dataset_mode = args.dataset_mode)
    elif args.dataset == 'kaspernormA' :
        val_set = KasperNormADataset(dir_root, 'test', args = args)
    elif args.dataset == 'kaspern4' :
        val_set = KasperN4Dataset(dir_root, 'test', args = args)
    elif args.dataset == 'kaspernormb' :
        val_set = KasperNormBDataset(dir_root, 'test', args = args)
    else :
        raise Exception(f"dataset {args.dataset} is not implemented")
    n_val = len(val_set)

    # 3. Create data loaders
    loader_args = dict(batch_size=args.batch_size, num_workers=32, pin_memory=True)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=False, **loader_args)

    logging.info('''Starting evaluation for checkpoint {}'''.format(args.load))
    metrics = evaluate_metrics(net,net_single_device, val_loader, device,args)
    logging.info('Validation Dice score: {}'.format(metrics[0]))
    logging.info('TPR: {}'.format(metrics[1]))
    logging.info('FPR: {}'.format(metrics[2]))
    logging.info('TPR Patient: {}'.format(metrics[3]))
    logging.info('FPR Patient: {}'.format(metrics[4]))


def train_net(net,
              net_single_device,
              device,
              args,
              epochs: int = 5,
              batch_size: int = 1,
              learning_rate: float = 0.001,
              save_checkpoint: bool = True,
              amp: bool = False):
    # 1. Create dataset
    if args.dataset == 'duke3d' :
        train_set = DukeBreastCancerMRIDataset(dir_root, 'train', args=args)
        val_set = DukeBreastCancerMRIDataset(dir_root, 'val', args=args)
        n_train = len(train_set)
        n_val = len(val_set)

    elif args.dataset == 'duke2d' :
        train_set = DBCMRIDataset(dir_root, 'train', dataset_mode = args.dataset_mode)
        val_set = DBCMRIDataset(dir_root, 'val', dataset_mode = args.dataset_mode)
        n_train = len(train_set)
        n_val = len(val_set)

    elif args.dataset == 'kaspernormA' :
        train_set = KasperNormADataset(dir_root, 'train', args = args)
        val_set = KasperNormADataset(dir_root, 'val', args = args)
        n_train = len(train_set)
        n_val = len(val_set)
        train_set = td.datasets.WrapDataset(train_set).cache(td.cachers.Pickle(Path("./cache_train")))
        val_set = td.datasets.WrapDataset(val_set).cache(td.cachers.Pickle(Path("./cache_val")))


    elif args.dataset == 'kaspern4' :
        train_set = KasperN4Dataset(dir_root, 'train', args = args)
        val_set = KasperN4Dataset(dir_root, 'val', args = args)
        n_train = len(train_set)
        n_val = len(val_set)
        train_set = td.datasets.WrapDataset(train_set).cache(td.cachers.Pickle(Path("./cache_train")))
        val_set = td.datasets.WrapDataset(val_set).cache(td.cachers.Pickle(Path("./cache_val")))
        # train_set = KasperN4Dataset(dir_root, 'train', args = args)
        # n_train = len(train_set.slice_numbers_per_example)
        # # train_set = torch.utils.data.ConcatDataset(train_set)
        # train_set = td.datasets.WrapDataset(train_set).cache(td.cachers.Pickle(Path("./cache_train")))
        # train_set = td.datasets.ChainDataset(train_set)#.cache(td.cachers.Pickle(Path("./cache_train")))

        # val_set = KasperN4Dataset(dir_root, 'val', args = args)
        # n_val = len(val_set.slice_numbers_per_example)
        # # val_set = torch.utils.data.ConcatDataset(val_set)
        # val_set = td.datasets.WrapDataset(val_set).cache(td.cachers.Pickle(Path("./cache_val")))
        # val_set = td.datasets.ChainDataset(val_set)#.cache(td.cachers.Pickle(Path("./cache_val")))
    elif args.dataset == 'kaspernormb' :
        train_set = KasperNormBDataset(dir_root, 'train', args = args)
        val_set = KasperNormBDataset(dir_root, 'val', args = args)
        n_train = len(train_set)
        n_val = len(val_set)
        train_set = td.datasets.WrapDataset(train_set).cache(td.cachers.Pickle(Path("./cache_train")))
        val_set = td.datasets.WrapDataset(val_set).cache(td.cachers.Pickle(Path("./cache_val")))
        
    else :
        raise Exception('dataset not defined')

    
    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=32, pin_memory=True)

    if args.balanced :
        # balanced_sampler = get_balanced_weighted_sampler(train_set.get_labels())
        balanced_sampler = get_balanced_weighted_sampler(train_set.get_labels(),len(train_set))
        # balanced_sampler = ClassBalancedRandomSampler(train_set.get_labels())
        # for x in balanced_sampler :
        #     print(x)
        # exit()
        # train_loader = DataLoader(train_set,collate_fn= my_collate_fn, sampler= balanced_sampler,**loader_args)
        train_loader_eval = train_loader = DataLoader(train_set, shuffle=True,collate_fn= my_collate_fn,**loader_args)
    else :
        train_loader = DataLoader(train_set, shuffle=True,collate_fn= my_collate_fn,**loader_args)
        train_loader_eval = DataLoader(train_set, shuffle=True,collate_fn= my_collate_fn,**loader_args)

    val_loader = DataLoader(val_set, shuffle=False, drop_last=True,collate_fn= my_collate_fn, **loader_args)

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(net_single_device.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.BCELoss()
    # criterion = utils.BBCELoss()
    global_step = 0

    # 5. Begin training
    for epoch in range(epochs):
        net.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images = batch['image']
                true_masks = batch['mask']

                assert images.shape[1] == net_single_device.n_channels, \
                    f'Network has been defined with {net_single_device.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.float)

                if args.d3 :
                    is_slice_positive = true_masks[:,:,args.axial_size // 4:-args.axial_size // 4].amax(dim=(3,4),keepdim = True).float()
                    is_patient_positive = is_slice_positive.amax(dim = 2, keepdim = True).float()
                else :
                    is_slice_positive = true_masks.amax(dim=(2,3), keepdim = True).float()

                with torch.cuda.amp.autocast(enabled=amp):
                    if args.d3 :
                        masks_pred, slice_class, patient_class = net(images)
                        confidence = slice_class*patient_class
                        confidence = confidence[:,:,args.axial_size // 4:-args.axial_size // 4]
                        masks_pred = masks_pred[:,:,args.axial_size // 4:-args.axial_size // 4]
                        true_masks = true_masks[:,:,args.axial_size // 4:-args.axial_size // 4]
                    else :
                        masks_pred, slice_class = net(images)
                        confidence = slice_class

                    # loss = dice_loss(masks_pred*confidence,
                    #             true_masks, threeD=args.d3) \
                    # + utils.balanced_binary_cross_entropy(confidence, is_slice_positive)
                    loss = criterion(masks_pred*confidence, true_masks) \
                    + dice_loss(masks_pred*confidence,
                                true_masks, threeD=args.d3) \
                    + F.binary_cross_entropy(confidence, is_slice_positive)
                    # + utils.balanced_binary_cross_entropy(confidence, is_slice_positive)
                    # if args.d3 :
                    #     loss += F.binary_cross_entropy(patient_class,is_patient_positive)



                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round
                division_step = (n_train // (10 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        histograms = {}
                        for tag, value in net_single_device.named_parameters():
                            tag = tag.replace('/', '.')
                            histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                            histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                        
                        logging.info('Evaluating for test set')
                        val_score = evaluate(net,net_single_device, val_loader, device,args=args)
                        scheduler.step(val_score)

                        logging.info('Validation Dice score: {}'.format(val_score))

                        # logging.info('Evaluating for train set')
                        # val_score = evaluate(net,net_single_device, train_loader_eval, device,threeD=args.d3)
                        # logging.info('Validation Dice score: {}'.format(val_score))
                        # experiment.log({
                        #     'learning rate': optimizer.param_groups[0]['lr'],
                        #     'validation Dice': val_score,
                        #     'images': wandb.Image(images[0].cpu()),
                        #     'masks': {
                        #         'true': wandb.Image(true_masks[0].float().cpu()),
                        #         'pred': wandb.Image(torch.softmax(masks_pred, dim=1).argmax(dim=1)[0].float().cpu()),
                        #     },
                        #     'step': global_step,
                        #     'epoch': epoch,
                        #     **histograms
                        # })

        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            torch.save(net_single_device.state_dict(), str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch + 1)))
            logging.info(f'Checkpoint {epoch + 1} saved!')


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=0.00001,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--d3',action='store_true',help='if specified, use 3d data and 3d unet model')
    parser.add_argument('--name', '-n', dest='name', type=str, default='B',
                        help='name of the checkpoint subfolder')
    parser.add_argument('--dataset',type=str,default='kasper',help='which dataset to use for the training should be in (kasper|duke3d|duke2d)')
    parser.add_argument('--dataset_mode',type=str, default='B',help='type of input to the network')
    parser.add_argument('--unregistered', action='store_true', default=False, help='Use unregistered dataset')
    parser.add_argument('--eval_only',action='store_true',help='only evaluate given checkpoint')
    parser.add_argument('--gpu_ids',type=str, default="0",help='type of input to the network')
    parser.add_argument('--input_channels', type=int, default= 3)
    parser.add_argument('--classes', type=int, default= 1)
    parser.add_argument('--balanced',action='store_true',help='use a balanced random sampler to create a balanced training')
    parser.add_argument('--resample',action='store_true',help='use resampling instead of resizing for 3d volumes')
    parser.add_argument('--axial_size',type= int, default=32)
    parser.add_argument('--sagital_size',type=int, default=256)
    parser.add_argument('--coronal_size',type=int,default=256)
    parser.add_argument('--experiment_type', type=str ,default='expA', help='which experiment to perform to debug the model')
    parser.add_argument('--full_set', action='store_true', help='testing the model using all the images in the test set')



    args = parser.parse_args()
    str_ids = args.gpu_ids.split(',')
    args.gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
           args.gpu_ids.append(id)

    return args
if __name__ == '__main__':
    args = get_args()
    dir_checkpoint = Path('./checkpoint/{}/'.format(args.name))
    os.makedirs('./logs/',exist_ok = True)
    logging_file_name = './logs/{}.txt/'.format(args.name)

    logging.basicConfig(filename=logging_file_name,
                            filemode='a',level=logging.INFO, format='%(levelname)s: %(message)s')
    logging.getLogger().addHandler(logging.StreamHandler())
    # device = torch.device('cpu')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_classes is the number of probabilities you want to get per pixel
    if not args.d3:
        net = UNet(args.input_channels, n_classes=args.classes, bilinear=True)
    else :
        net = UNet3d(args.input_channels,args.classes,bilinear=True)

    
    logging.info(f'expirement args: {args}')
    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')


    if args.load:
        try:
            net.load_state_dict(torch.load(args.load, map_location=device))
        except:   
            pretrained_dict = torch.load(args.load, map_location=device)                
            model_dict = net.state_dict()
            try:
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}                    
                net.load_state_dict(pretrained_dict)
                logging.info('Pretrained network %s has excessive layers; Only loading layers that are used' % type(net))
            except:
                logging.info('Pretrained network %s has fewer layers; The following are not initialized:' % type(net))
                for k, v in pretrained_dict.items():                      
                    if v.size() == model_dict[k].size():
                        model_dict[k] = v

                if sys.version_info >= (3,0):
                    not_initialized = set()
                else:
                    from sets import Set
                    not_initialized = Set()                    

                for k, v in model_dict.items():
                    if k not in pretrained_dict or v.size() != pretrained_dict[k].size():
                        not_initialized.add(k.split('.')[0])
                
                logging.info(sorted(not_initialized))
                net.load_state_dict(model_dict)

        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)
    net_single_device = net
    net = torch.nn.DataParallel(net,args.gpu_ids)

    try:
        if args.eval_only :
            assert args.load, 'to evaluate a model, you need to give pth path'
            evaluate_net(net,net_single_device,device,args)
        else :
            train_net(net=net,
                    net_single_device= net_single_device,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    learning_rate=args.lr,
                    device=device,
                    amp=args.amp,
                    args= args)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        sys.exit(0)
