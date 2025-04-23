import sys
sys.path.append('/home/admin1/User/mxy/demo/')

from backbone_tabular.TabularEncoder import TabularTransformerEncoder
from backbone_tabular.TabularEncoder2 import TabularEncoder
from utils.losses import ReconstructionLoss, ReconstructionLoss_MLP
from utils.utils import AverageMeter, accuracy
from reconstruction_method import Transformer_Embedding_R, MLP_Embedding_R
from build_dataset_tabular import tabular_dataset, tabular_dataset_dvm
from torch.utils.data import DataLoader
import torch.nn.functional as F
from utils.losses import FocalLoss

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

import torch.nn as nn
import torch.optim as optim
import os
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
plt.rcParams['axes.unicode_minus'] = False  


def create_log(opt_dict):
    log_train = None
    if not opt_dict['train_config']['find_epoch']:
        log_path = os.path.join(opt_dict['dataset_config']['tabular_result_path'], opt_dict['model_config']['net_v_tabular'], f"{opt_dict['dataset_config']['dataname']}_{opt_dict['train_config']['epochs']}_{opt_dict['dataset_config']['missing_rate']}_{opt_dict['model_config']['model_name']}_{opt_dict['model_config']['net_v_tabular']}_{opt_dict['train_config']['lr_max']}_log_train.csv")
        print(log_path)
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        log_train = open(log_path, 'w')

    return log_train


def write_log(log_train, log_out, opt_dict):
    if not opt_dict['train_config']['find_epoch']:
        print(log_out)
        log_train.write(log_out + '\n')
        log_train.flush()
    return log_train


def load_feature_model(opt_dict):
    if opt_dict['model_config']['net_v_tabular'] == 'encoder_mlp_block_prototype':
        model = TabularTransformerEncoder(opt_dict, has_fc=False)    
    elif opt_dict['model_config']['net_v_tabular'] == 'encoder_mlp_prototype':
        model = TabularEncoder(opt_dict, is_fc=False)

    return model


def load_reconstruction_model(opt_dict):
    if opt_dict['model_config']['net_v_tabular'] == 'encoder_mlp_block_prototype':
        model = Transformer_Embedding_R(opt_dict)
    elif opt_dict['model_config']['net_v_tabular'] == 'encoder_mlp_prototype': 
        model = MLP_Embedding_R(opt_dict)
    return model




def train_re(model_feature, model_re, train_loader, test_loader, device, opt_dict):
    
    num_cat, num_con, cat_offsets = model_feature.num_cat, model_feature.num_con, model_feature.cat_offsets.to(device)
    epochs = opt_dict['train_config']['epochs']
    log_train = create_log(opt_dict)

    train_losses_re = AverageMeter('Train_Loss_re', ':.4e')
    val_losses_re = AverageMeter('Val_loss', ':.4e')

    least_val_re = 1e9

    model_feature.to(device)
    model_re.to(device)
    if opt_dict['model_config']['net_v_tabular'] == 'encoder_mlp_block_prototype':
        print(opt_dict['model_config']['net_v_tabular'])
        loss_function = ReconstructionLoss(num_cat, num_con, cat_offsets)
    elif opt_dict['model_config']['net_v_tabular'] == 'encoder_mlp_prototype': 
        print(opt_dict['model_config']['net_v_tabular'])
        loss_function = ReconstructionLoss_MLP(num_cat, num_con, opt_dict)


    pg_feature = [p for p in model_feature.parameters() if p.requires_grad]
    pg_re = [p for p in model_re.parameters() if p.requires_grad]
    pg = pg_feature + pg_re

    optimizer = optim.Adam(pg, lr=float(opt_dict['train_config']['lr_max']))
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt_dict['train_config']['epochs'])

    for epoch in range(epochs):
        # import ipdb;ipdb.set_trace();
        train_losses_re.reset()

        model_feature.train()
        model_re.train()

        for step, data in enumerate(train_loader):
            tables, labels, masks = data
            tables, labels, masks = tables.to(device), labels.to(device), masks.to(device)

            batch_size_cur = tables.size(0)
        
            if opt_dict['model_config']['net_v_tabular'] == 'encoder_mlp_block_prototype':
                features, _ = model_feature(tables, masks, masks, has_fc=False)
                results = model_re(features)
                loss = loss_function(results, tables, masks)
            elif opt_dict['model_config']['net_v_tabular'] == 'encoder_mlp_prototype': 
                # import ipdb;ipdb.set_trace();
                features = model_feature(tables, masks, masks)
                results = model_re(features)
                loss = loss_function(results, tables, masks)

            loss_re = loss[0]
            loss_total = loss_re
            train_losses_re.update(loss_re.item(), batch_size_cur)
            log_out = ('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                'Train_Loss_re {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                epoch, step, len(train_loader), lr=optimizer.param_groups[-1]['lr'],
                loss=train_losses_re,
            ))
            log_train = write_log(log_train, log_out, opt_dict)
                
            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()

        schedule.step()

        model_feature.eval()
        model_re.eval()

        val_losses_re.reset()

        with torch.no_grad():
            for tables, labels, masks in test_loader:
                tables, labels, masks = tables.to(device), labels.to(device), masks.to(device)

                if opt_dict['model_config']['net_v_tabular'] == 'encoder_mlp_block_prototype':
                    val_features, _ = model_feature(tables, masks, masks, has_fc=False)
                    val_results = model_re(val_features)
                    val_loss = loss_function(val_results, tables, masks)
                elif opt_dict['model_config']['net_v_tabular'] == 'encoder_mlp_prototype': 
                    val_features = model_feature(tables, masks, masks)
                    val_results = model_re(val_features)
                    val_loss = loss_function(val_results, tables, masks)

                val_loss_re = val_loss[0]
                val_losses_re.update(val_loss_re.item(), tables.size(0))
                
            log_out_val = ('Val_Loss_re {loss_re.val:.4f} ({loss_re.avg:.4f})\n'.format(loss_re=val_losses_re))

            log_train = write_log(log_train, log_out_val, opt_dict)
            
            if(least_val_re > val_losses_re.avg):
                least_val_re = val_losses_re.avg
                log_best = f"Saved at {least_val_re}"
                log_train = write_log(log_train, log_best, opt_dict)
                if(val_losses_re.avg < 0.66):
                    print("have saved")
                    least_val_re = val_losses_re.avg
                    save_feature_path = f"/data/blood_dvm/data/result/temp/dvm/re052/05_feature_{epochs}_{opt_dict['train_config']['lr_max']}_{val_losses_re.avg}.pth"
                    save_re_path = f"/data/blood_dvm/data/result/temp/dvm/re052/05_re_{epochs}_{opt_dict['train_config']['lr_max']}_{val_losses_re.avg}.pth"
                    # import ipdb;ipdb.set_trace();
                    torch.save(model_feature.state_dict(), save_feature_path)
                    torch.save(model_re.state_dict(), save_re_path)


    print("Finish!")
    return least_val_re



def train_epoch(train_loader, test_loader, device, opt_dict):
    print(opt_dict['train_config']['mode'])
    print(opt_dict['model_config']['net_v_tabular'])
    update_losses_re = []

    max_epoch = opt_dict['train_config']['max_epochs']
    for epoch in range(100, max_epoch, 5):
        print(f"现在是Epoch={epoch}")
        opt_dict['train_config']['epochs'] = epoch
        print(f"opt_dict['train_config']['epoch']:{opt_dict['train_config']['epochs']}")
        model_feature = load_feature_model(opt_dict)
        model_re = load_reconstruction_model(opt_dict)
        update_loss_re = train_re(model_feature, model_re, train_loader, test_loader, device, opt_dict)
        update_losses_re.append(update_loss_re)
        print(f"update_losses_re:{update_losses_re}")

    print('Finished')
    return update_losses_re






if __name__ == '__main__':

    from build_dataset_unit import UnitDataset
    mask_version = opt_dict['dataset_config']['missing_rate']
    print(f"mask_version : {mask_version}")
    dataset_dir = opt_dict['dataset_config']['data_dir']
    batch_size = opt_dict['train_config']['batch_size']

    if opt_dict['dataset_config']['dataname'] == 'blood':
        train_dataset = UnitDataset(dataset_dir, mode='train', dataset_type='tabular', mask_version=mask_version)
        test_dataset = UnitDataset(dataset_dir, mode='test', dataset_type='tabular', mask_version=mask_version)
    else:
        # import ipdb;ipdb.set_trace();
        train_dataset = tabular_dataset_dvm(opt_dict, 'train')
        test_dataset = tabular_dataset_dvm(opt_dict, 'test')
        # train_dataset = UnitDataset(dataset_dir, mode='train', dataset_type='tabular', mask_version=mask_version)
        # test_dataset = UnitDataset(dataset_dir, mode='test', dataset_type='tabular', mask_version=mask_version)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    model_feature = load_feature_model(opt_dict)
    model_re = load_reconstruction_model(opt_dict)
    num_cat, num_con = model_feature.num_cat, model_feature.num_con

    # print(opt_dict['train_config']['lr_max'])
    train_epoch(train_loader, test_loader, device, opt_dict)
    import ipdb;ipdb.set_trace();
    # train_re(model_feature, model_re, train_loader, test_loader, device, opt_dict)

