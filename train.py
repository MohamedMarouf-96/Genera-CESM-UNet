import argparse
import logging
import sys
from pathlib import Path
from unittest.util import strclass
from matplotlib.animation import ImageMagickBase

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from utils.data_loading import BasicDataset, CarvanaDataset, TumorSliceSubsetDataset, DukeBreastCancerMRIDataset
from utils.dice_score import dice_loss
from evaluate import evaluate, evaluate_metrics
from unet_local import UNet, UNet3d
import unet
from utils.my_collate_fn import my_collate_fn

dir_root = Path('/data/')

def evaluate_net(net,
                 net_single_device,
                 device,
                 amp: bool = False,
                 d3 = False):
    
    val_set = TumorSliceSubsetDataset(dir_root, 'test', dataset_mode = args.dataset_mode, unregistered= args.unregistered)
    n_val = len(val_set)

    # 3. Create data loaders
    loader_args = dict(batch_size=1, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=False, **loader_args)

    logging.info('''Starting evaluation for checkpoint {}'''.format(args.load))
    metrics = evaluate_metrics(net,net_single_device, val_loader, device)
    logging.info('Validation Dice score: {}'.format(metrics[0]))
    logging.info('TPR: {}'.format(metrics[1]))
    logging.info('FPR: {}'.format(metrics[2]))
    logging.info('TPR Patient: {}'.format(metrics[3]))
    logging.info('FPR Patient: {}'.format(metrics[4]))


def train_net(net,
              net_single_device,
              device,
              epochs: int = 5,
              batch_size: int = 1,
              learning_rate: float = 0.001,
              val_percent: float = 0.1,
              save_checkpoint: bool = True,
              img_scale: float = 0.5,
              amp: bool = False,
              d3 = False):
    # 1. Create dataset
    if d3 :
        dataset = DukeBreastCancerMRIDataset(dir_root, 'train', dataset_mode = args.dataset_mode, unregistered= args.unregistered)
        # test_set = DukeBreastCancerMRIDataset(dir_root, 'test', unregistered= args.unregistered)
        n_val = int(len(dataset) * val_percent)
        n_train = len(dataset) - n_val
        train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))
        n_train = len(train_set)
        n_val = len(val_set)
    else :
        train_set = TumorSliceSubsetDataset(dir_root, 'train', dataset_mode = args.dataset_mode, unregistered= args.unregistered)
        val_set = TumorSliceSubsetDataset(dir_root, 'val', dataset_mode = args.dataset_mode , unregistered= args.unregistered)
        n_train = len(train_set)
        n_val = len(val_set)

    # try:
    #     dataset = CarvanaDataset(dir_img, dir_mask, img_scale)
    # except (AssertionError, RuntimeError):
    #     dataset = BasicDataset(dir_img, dir_mask, img_scale)

    # 2. Split into train / validation partitions
    # n_val = int(len(dataset) * val_percent)
    # n_train = len(dataset) - n_val
    # train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=True)

    
    # train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    # val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)
    train_loader = DataLoader(train_set, shuffle=True,collate_fn= my_collate_fn, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True,collate_fn= my_collate_fn, **loader_args)

    # # (Initialize logging)
    # if args.d3: 
    #     experiment = wandb.init(project='U-Net-3D', resume='allow', anonymous='must')
    # else :
    #     experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')

    # experiment.config.update(dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
    #                               val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale,
    #                               amp=amp))

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(net_single_device.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss()
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
                true_masks = true_masks.to(device=device, dtype=torch.long)
                is_slice_positive = true_masks.amax(dim=(1,2)).float()

                if args.d3 :
                    true_masks_one_hot = F.one_hot(true_masks, net_single_device.n_classes).permute(0, 4, 1, 2, 3).float()
                else :
                    true_masks_one_hot = F.one_hot(true_masks, net_single_device.n_classes).permute(0, 3, 1, 2).float()

                with torch.cuda.amp.autocast(enabled=amp):
                    masks_pred, confidence = net(images)
                    if args.stage == 'seg':
                        loss = criterion(masks_pred, true_masks) \
                       + dice_loss(F.softmax(masks_pred, dim=1).float(),
                                   true_masks_one_hot,
                                   multiclass=True, threeD=args.d3) \
                       + F.binary_cross_entropy(confidence, is_slice_positive)

                    elif args.stage == 'conf':
                        loss = F.binary_cross_entropy(confidence, is_slice_positive)



                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                # experiment.log({
                #     'train loss': loss.item(),
                #     'step': global_step,
                #     'epoch': epoch
                # })
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

                        val_score = evaluate(net,net_single_device, val_loader, device,threeD=args.d3)
                        scheduler.step(val_score)

                        logging.info('Validation Dice score: {}'.format(val_score))
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
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=0.00001,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=1.0, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--d3',action='store_true',help='if specified, use 3d data and 3d unet model')
    parser.add_argument('--name', '-n', dest='name', type=str, default='B',
                        help='name of the checkpoint subfolder')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--unregistered', action='store_true', default=False, help='Use unregistered dataset')
    parser.add_argument('--eval_only',action='store_true',help='only evaluate given checkpoint')
    parser.add_argument('--dataset_mode',type=str, default='B',help='type of input to the network')
    parser.add_argument('--gpu_ids',type=str, default="0",help='type of input to the network')
    parser.add_argument('--stage',type=str,default='seg',help='stage of training the model, should be seg | conf for training segmentation ,confidence score for metrics ')



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
    logging_file_name = './logs/{}.txt/'.format(args.name)

    logging.basicConfig(filename=logging_file_name,
                            filemode='a',level=logging.INFO, format='%(levelname)s: %(message)s')
    logging.getLogger().addHandler(logging.StreamHandler())
    # device = torch.device('cpu')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    if not args.d3:
        net = UNet(n_channels=3, n_classes=2, bilinear=True)
    else :
        # net = UNet3d(n_channels=1,n_classes=2,bilinear=True) 
        net = unet.UNet3D(padding=1)
        net.n_channels =1
        net.n_classes =2
        net.bilinear = False

    

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
            evaluate_net(net,net_single_device,device,args.amp,args.d3)
        else :
            train_net(net=net,
                    net_single_device= net_single_device,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    learning_rate=args.lr,
                    device=device,
                    img_scale=args.scale,
                    val_percent=args.val / 100,
                    amp=args.amp,
                    d3=args.d3)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        sys.exit(0)
