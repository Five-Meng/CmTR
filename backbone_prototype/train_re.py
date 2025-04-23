import sys
sys.path.append('/home/admin1/User/mxy/demo/')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from backbone_tabular_prototype.reconstruction_method import MLP_Embedding_R
from utils.utils import AverageMeter, accuracy
from utils.losses import ReconstructionLoss_MLP

from argparses.util.yaml_args import yaml_data
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--yaml_config', type=str, help='config args')
args = parser.parse_args()
opt_dict = yaml_data(args.yaml_config)


import torch
import random
import numpy as np
import pandas as pd
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
seed = opt_dict['train_config']['seed']
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



def create_log(opt_dict):
    log_train = None
    if not opt_dict['train_config']['find_epoch']:
        log_path = os.path.join(opt_dict['dataset_config']['it_result_path'], opt_dict['model_config']['net_v'], f"1024_{opt_dict['dataset_config']['dataname']}_{opt_dict['train_config']['epochs']}_{opt_dict['dataset_config']['missing_rate']}_{opt_dict['model_config']['model_name']}_{opt_dict['model_config']['net_v']}_{opt_dict['train_config']['lr_max']}_log_train.csv")
        print(log_path)
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        log_train = open(log_path, 'w')

    return log_train


def write_log(log_train, log_out, opt_dict):
    if not opt_dict['train_config']['find_epoch']:
        # print(log_out)
        log_train.write(log_out + '\n')
        log_train.flush()
    return log_train



def load_model_image(opt_dict):
    from backbone.efficientnet import efficientnet_b1
    model = efficientnet_b1(num_classes=opt_dict['train_config']['num_cls'])
    # net_pretrain_path = "/data/blood_dvm/data/result/end/encoder_image/classification_results/efficientnet-b1/h2t02_blood_135_crossentropy_efficientnet-b1_5e-5_best_model.pth"
    # net_pretrain_path = "/data/blood_dvm/data/result/end/encoder_image/classification_results/efficientnet-b1/dvm_300_crossentropy_efficientnet-b1_3e-3_best_model.pth"
    net_pretrain_path = "/data/blood_dvm/data/result/temp/img_re/00_feature_100_0.3266263020334844.pth"
    weights_dict = torch.load(net_pretrain_path, map_location='cpu')
    model.load_state_dict(weights_dict, strict=True)
    for name, param in model.named_parameters():
        param.requires_grad = True
    return model



def load_reconstruction_model(opt_dict): 
    if(opt_dict['dataset_config']['dataname'] == 'blood'):
        opt_dict['model_config']['hidden_size2'] = opt_dict['model_config']['efficientb1_outdim']
    else:
        opt_dict['model_config']['hidden_size3'] = opt_dict['model_config']['efficientb1_outdim']
    model = MLP_Embedding_R(opt_dict)

    net_pretrain_path = "/data/blood_dvm/data/result/temp/img_re/00_feature_100_0.3266263020334844.pth"
    weights_dict = torch.load(net_pretrain_path, map_location='cpu')
    model.load_state_dict(weights_dict, strict=True)
    for name, param in model.named_parameters():
        param.requires_grad = True
    return model



def train_img_re(model_feature, model_re, train_loader, test_loader, device, opt_dict):
    # import ipdb;ipdb.set_trace();
    pg_feature = [p for p in model_feature.parameters() if p.requires_grad]
    pg_re = [p for p in model_re.parameters() if p.requires_grad]
    pg = pg_feature + pg_re
    optimizer = optim.Adam(pg, lr=float(opt_dict['train_config']['lr_max']))
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt_dict['train_config']['epochs'])
    
    model_feature.to(device)
    model_re.to(device)
    
    if opt_dict['dataset_config']['dataname'] == 'blood':
        num_cat, num_con = 19, 22
    else:
        num_cat, num_con = 4, 13
    loss_function = ReconstructionLoss_MLP(num_cat, num_con, opt_dict)
    epochs = opt_dict['train_config']['epochs']
    print(f"epochs: {epochs}")
    log_train = create_log(opt_dict)

    train_losses_re = AverageMeter('Train_Loss_re', ':.4e')
    val_losses_re = AverageMeter('Val_loss_re', ':.4e')

    least_val_re = 1e9

    for epoch in range(epochs):
        train_losses_re.reset()

        model_feature.train()
        model_re.train()

        for step, data in enumerate(train_loader):
            # import ipdb;ipdb.set_trace();
            images, tables, masks, labels = data
            masks = torch.ones_like(masks, dtype=torch.bool)
            images, tables, masks, labels = images.to(device), tables.to(device), masks.to(device), labels.to(device)
            batch_size_cur = images.size(0)
            features = model_feature.forward_sne(images)
            results = model_re(features)
            loss = loss_function(results, tables, masks)
            loss_re = loss[0]

            train_losses_re.update(loss_re.item(), batch_size_cur)

            log_out = ('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                'Train_Loss_re {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                epoch, step, len(train_loader), lr=optimizer.param_groups[-1]['lr'],
                loss=train_losses_re
            ))
            log_train = write_log(log_train, log_out, opt_dict)
                
            optimizer.zero_grad()
            loss_re.backward()
            optimizer.step()

        schedule.step()

        model_feature.eval()
        model_re.eval()

        val_losses_re.reset()
        
        with torch.no_grad():
            for step, data in enumerate(test_loader):
                # import ipdb;ipdb.set_trace();
                images, tables, masks, labels = data
                masks = torch.ones_like(masks, dtype=torch.bool)
                images, tables, masks, labels = images.to(device), tables.to(device), masks.to(device), labels.to(device)
                batch_size_cur = images.size(0)
                features = model_feature.forward_sne(images)
                val_results = model_re(features)
                val_loss = loss_function(val_results, tables, masks)
                val_loss_re = val_loss[0]

                val_losses_re.update(val_loss_re.item(), batch_size_cur)

        # import ipdb;ipdb.set_trace();
        log_out = ('Val_Loss_re {loss.val:.4f} ({loss.avg:.4f})\t'.format(
            loss=val_losses_re
        ))
        log_train = write_log(log_train, log_out, opt_dict) 
        if(least_val_re > val_losses_re.avg):
            least_val_re = val_losses_re.avg
            log_best_re = f"Saved Least Loss_Re: {least_val_re}"
            log_train = write_log(log_train, log_best_re, opt_dict)
            if(val_losses_re.avg < 1.145):
                save_feature_path = f"/data/blood_dvm/data/result/temp/img_re/00_feature_{epochs}_{val_losses_re.avg}.pth"
                save_re_path = f"/data/blood_dvm/data/result/temp/img_re/00_re_{epochs}_{val_losses_re.avg}.pth"
                torch.save(model_feature.state_dict(), save_feature_path)
                torch.save(model_re.state_dict(), save_re_path)


    return least_val_re




def train_img_tab_r_epoch(train_loader, test_loader, device, opt_dict):
    least_val_re_list = []
    max_epochs = opt_dict['train_config']['max_epochs']
    for epochs in range(50, max_epochs, 5):
        opt_dict['train_config']['epochs'] = epochs
        model_feature = load_model_image(opt_dict)
        model_re = load_reconstruction_model(opt_dict)
        print(f"Epoch: {opt_dict['train_config']['epochs']}")
        least_val_re = train_img_re(model_feature, model_re, train_loader, test_loader, device, opt_dict)
        least_val_re_list.append(least_val_re)
        print(least_val_re_list)





if __name__ == '__main__':
    from build_dataset_unit import UnitDataset, UnitDataset_dvm
    from torchvision import transforms
    mask_version = opt_dict['dataset_config']['missing_rate']
    print(f"mask_version : {mask_version}")
    dataset_dir = opt_dict['dataset_config']['data_dir']

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    print(torch.cuda.is_available())

    transform_img_train = transforms.Compose([
        transforms.RandomResizedCrop(size=240, scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8)], p=0.8),     
        transforms.RandomGrayscale(p=0.2),     
        transforms.ToTensor(),
    ])
    transform_img_test = transforms.Compose([
        transforms.Resize([240, 240]),
        transforms.ToTensor(),
    ])

    if opt_dict['dataset_config']['dataname'] == 'blood':
        train_dataset = UnitDataset(dataset_dir, mode='train', dataset_type='image_tabular', mask_version=mask_version)
        test_dataset = UnitDataset(dataset_dir, mode='test', dataset_type='image_tabular', mask_version=mask_version)
    else:
        train_dataset = UnitDataset_dvm(opt_dict, mode='train', dataset_type='image_tabular', transform=transform_img_train)
        test_dataset = UnitDataset_dvm(opt_dict, mode='test', dataset_type='image_tabular', transform=transform_img_test)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64)
    model_feature = load_model_image(opt_dict)
    model_re = load_reconstruction_model(opt_dict)


    train_img_re(model_feature, model_re, train_loader, test_loader, device, opt_dict)

    # train_img_tab_r_epoch(train_loader, test_loader, device, opt_dict)
    

