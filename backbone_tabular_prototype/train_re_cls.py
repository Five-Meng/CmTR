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
# from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
plt.rcParams['axes.unicode_minus'] = False  


def create_log(opt_dict, lambda_re, lambda_cls):
    log_train = None
    if not opt_dict['train_config']['find_epoch']:
        log_path = os.path.join(opt_dict['dataset_config']['tabular_result_path'], opt_dict['model_config']['net_v_tabular'], f"{lambda_re}_{lambda_cls}_{opt_dict['dataset_config']['dataname']}_{opt_dict['train_config']['epochs']}_{opt_dict['dataset_config']['missing_rate']}_{opt_dict['model_config']['model_name']}_{opt_dict['model_config']['net_v_tabular']}_{opt_dict['train_config']['lr_max']}_log_train.csv")
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




def calculate_nonlinear(epoch, total_epochs, min_lambda_re=0.1, max_lambda_re=0.6, power=2):

    normalized_epoch = epoch / (total_epochs - 1)
    
    lambda_re = min_lambda_re + (max_lambda_re - min_lambda_re) * (normalized_epoch ** power)
    
    return lambda_re




def load_feature_model(opt_dict):
    model = None
    if opt_dict['model_config']['net_v_tabular'] == 'encoder_mlp_block_prototype':
        model = TabularTransformerEncoder(opt_dict, has_fc=False)    
    elif opt_dict['model_config']['net_v_tabular'] == 'encoder_mlp_prototype':
        model = TabularEncoder(opt_dict, is_fc=False)
    return model


def load_reconstruction_model(opt_dict):
    model = None
    if opt_dict['model_config']['net_v_tabular'] == 'encoder_mlp_block_prototype':
        model = Transformer_Embedding_R(opt_dict)
    elif opt_dict['model_config']['net_v_tabular'] == 'encoder_mlp_prototype': 
        model = MLP_Embedding_R(opt_dict)
    return model


def load_fc_model(opt_dict):
    if opt_dict['dataset_config']['dataname'] == 'dvm':
        input_size = opt_dict['model_config']['hidden_size3']
    else:
        input_size = opt_dict['model_config']['hidden_size2']
    model_fc3 = nn.Linear(input_size, opt_dict['train_config']['num_cls'])
    return model_fc3



def save_best_model_cls(lambda_re, lambda_cls, log_train, model_feature, model_re, model_fc, least_total_losses, val_losses, opt_dict):
    if(least_total_losses > val_losses.avg):
        if not opt_dict['train_config']['find_epoch']:
            save_model_path_feature = f"feature_{lambda_re}_{lambda_cls}_{opt_dict['dataset_config']['dataname']}_{opt_dict['train_config']['epochs']}_{opt_dict['dataset_config']['missing_rate']}_{opt_dict['model_config']['model_name']}_{opt_dict['train_config']['lossfunc']}_{opt_dict['model_config']['net_v_tabular']}_{opt_dict['train_config']['lr_max']}_best_model.pth"
            save_model_path_re = f"re_{lambda_re}_{lambda_cls}_{opt_dict['dataset_config']['dataname']}_{opt_dict['train_config']['epochs']}_{opt_dict['dataset_config']['missing_rate']}_{opt_dict['model_config']['model_name']}_{opt_dict['train_config']['lossfunc']}_{opt_dict['model_config']['net_v_tabular']}_{opt_dict['train_config']['lr_max']}_best_model.pth"            
            save_model_path_fc = f"fc_{lambda_re}_{lambda_cls}_{opt_dict['dataset_config']['dataname']}_{opt_dict['train_config']['epochs']}_{opt_dict['dataset_config']['missing_rate']}_{opt_dict['model_config']['model_name']}_{opt_dict['train_config']['lossfunc']}_{opt_dict['model_config']['net_v_tabular']}_{opt_dict['train_config']['lr_max']}_best_model.pth"            
            log_best_model = f'Saved best model with least_total_losses: {val_losses.avg:.4f}\n'
            log_train = write_log(log_train, log_best_model, opt_dict)
            torch.save(model_feature.state_dict(), os.path.join(opt_dict['dataset_config']['tabular_result_path'], opt_dict['model_config']['net_v_tabular'], save_model_path_feature))
            torch.save(model_re.state_dict(), os.path.join(opt_dict['dataset_config']['tabular_result_path'], opt_dict['model_config']['net_v_tabular'], save_model_path_re))
            torch.save(model_fc.state_dict(), os.path.join(opt_dict['dataset_config']['tabular_result_path'], opt_dict['model_config']['net_v_tabular'], save_model_path_fc))

    return log_train


def save_best_model_cls_acc(log_train, model_feature, model_re, model_fc, less_cls_acc, val_acc_v, opt_dict):
    if(less_cls_acc < val_acc_v.avg):
        less_cls_acc = val_acc_v.avg
        if not opt_dict['train_config']['find_epoch']:
            save_model_path_feature = f"feature_{opt_dict['dataset_config']['dataname']}_{opt_dict['train_config']['epochs']}_{opt_dict['dataset_config']['missing_rate']}_{opt_dict['model_config']['model_name']}_{opt_dict['train_config']['lossfunc']}_{opt_dict['model_config']['net_v_tabular']}_{opt_dict['train_config']['lr_max']}_best_model.pth"
            save_model_path_re = f"re_{opt_dict['dataset_config']['dataname']}_{opt_dict['train_config']['epochs']}_{opt_dict['dataset_config']['missing_rate']}_{opt_dict['model_config']['model_name']}_{opt_dict['train_config']['lossfunc']}_{opt_dict['model_config']['net_v_tabular']}_{opt_dict['train_config']['lr_max']}_best_model.pth"            
            save_model_path_fc = f"fc_{opt_dict['dataset_config']['dataname']}_{opt_dict['train_config']['epochs']}_{opt_dict['dataset_config']['missing_rate']}_{opt_dict['model_config']['model_name']}_{opt_dict['train_config']['lossfunc']}_{opt_dict['model_config']['net_v_tabular']}_{opt_dict['train_config']['lr_max']}_best_model.pth"            
            log_best_model = f'Saved best model with val_acc_v: {val_acc_v.avg:.4f}\n'
            log_train = write_log(log_train, log_best_model, opt_dict)
            torch.save(model_feature.state_dict(), os.path.join(opt_dict['dataset_config']['tabular_result_path'], opt_dict['model_config']['net_v_tabular'], save_model_path_feature))
            torch.save(model_re.state_dict(), os.path.join(opt_dict['dataset_config']['tabular_result_path'], opt_dict['model_config']['net_v_tabular'], save_model_path_re))
            torch.save(model_fc.state_dict(), os.path.join(opt_dict['dataset_config']['tabular_result_path'], opt_dict['model_config']['net_v_tabular'], save_model_path_fc))

    return less_cls_acc, log_train



def train_re_cls1(model_feature, model_re, model_fc, train_loader, test_loader, device, opt_dict):
    num_cat, num_con, cat_offsets = model_feature.num_cat, model_feature.num_con, model_feature.cat_offsets.to(device)
    lambda_re = 0.7
    lambda_cls = 0.3
    log_train = create_log(opt_dict, lambda_re, lambda_cls)

    train_acc_v = AverageMeter('Train_Acc@1', ':6.2f')
    val_acc_v = AverageMeter('Val_Acc@1', ':6.2f')
    train_losses = AverageMeter('Train_Loss', ':.4e')
    val_losses = AverageMeter('Val_loss', ':.4e')
    train_losses_re = AverageMeter('Val_loss', ':.4e')
    train_losses_cls = AverageMeter('Val_loss', ':.4e')
    val_losses_re = AverageMeter('Val_loss', ':.4e')
    val_losses_cls = AverageMeter('Val_loss', ':.4e')
    less_total_losses = 1e9
    less_cls_acc = 0
    update_re_losses = 0
    update_cls_losses = 0
    update_cls_acc = 0
    least_total_losses = 1e9

    model_feature.to(device)
    model_re.to(device)
    model_fc.to(device)

    if opt_dict['model_config']['net_v_tabular'] == 'encoder_mlp_block_prototype':
        print(opt_dict['model_config']['net_v_tabular'])
        loss_function_re = ReconstructionLoss(num_cat, num_con, cat_offsets)
    elif opt_dict['model_config']['net_v_tabular'] == 'encoder_mlp_prototype': 
        print(opt_dict['model_config']['net_v_tabular'])
        loss_function_re = ReconstructionLoss_MLP(num_cat, num_con, opt_dict)

    loss_function_cls = nn.CrossEntropyLoss()

    pg_feature = [p for p in model_feature.parameters() if p.requires_grad]
    pg_re = [p for p in model_re.parameters() if p.requires_grad]
    pg_fc = [p for p in model_fc.parameters() if p.requires_grad]
    pg = pg_feature + pg_re + pg_fc
    optimizer = optim.Adam(pg, lr=float(opt_dict['train_config']['lr_max']))

    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt_dict['train_config']['epochs'])
    epochs = opt_dict['train_config']['epochs']

    train_all_losses = []
    val_all_losses = []

    for epoch in range(epochs):
        # lambda_cls = calculate_nonlinear(epoch, epochs, 0.1, 0.9)
        # lambda_re = 1 - lambda_cls

        train_losses.reset()
        train_losses_re.reset()
        train_losses_cls.reset()
        train_acc_v.reset()

        model_feature.train()
        model_re.train()
        model_fc.train()
        # import ipdb;ipdb.set_trace();
        for step, data in enumerate(train_loader):
            tables, labels, masks = data
            tables, labels, masks = tables.to(device), labels.to(device), masks.to(device)
            batch_size_cur = tables.size(0)
            if opt_dict['model_config']['net_v_tabular'] == 'encoder_mlp_block_prototype':
                features = model_feature(tables, masks, masks, has_fc=False)
                results_re = model_re(features)
                B, N, D = features.shape
                features_fc = features.reshape(B, N * D)
                loss = loss_function_re(results_re, tables, masks)
            elif opt_dict['model_config']['net_v_tabular'] == 'encoder_mlp_prototype': 
                features = model_feature(tables, masks, masks)
                results = model_re(features)
                features_fc = features
                loss = loss_function_re(results, tables, masks)

            loss_re = loss[0]
            results_cls = model_fc(features_fc)
            loss_cls = loss_function_cls(results_cls, labels)
            loss_total = loss_cls * lambda_cls + loss_re * lambda_re

            train_losses.update(loss_total.item(), batch_size_cur)
            train_losses_re.update(loss_re.item(), batch_size_cur)
            train_losses_cls.update(loss_cls.item(), batch_size_cur)

            train_acc = accuracy(results_cls, labels)[0]
            train_acc_v.update(train_acc.item(), batch_size_cur)

            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()

            log_out = ('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                       'Lambda_re: {lambda_re:.4f}\t'
                       'Lambda_cls: {lambda_cls:.4f}\t'
                       'Train_Loss_Total {loss.val:.4f} ({loss.avg:.4f})\t'
                       'Train_Loss_Re {loss_re.val:.4f} ({loss_re.avg:.4f})\t'
                       'Train_Loss_Cls {loss_cls.val:.4f} ({loss_cls.avg:.4f})\t'
                       'Train Acc@1 {train_acc_v.val:.4f}({train_acc_v.avg:.4f})\t'.format(
                        epoch, step, len(train_loader), lr=optimizer.param_groups[-1]['lr'],
                        lambda_re=lambda_re, lambda_cls=lambda_cls,
                        loss=train_losses, loss_re=train_losses_re, loss_cls=train_losses_cls,
                        train_acc_v=train_acc_v
                       ))
            log_train = write_log(log_train, log_out, opt_dict)
        train_all_losses.append(train_losses.avg)
        schedule.step()

        model_feature.eval()
        model_fc.eval()
        model_re.eval()
        val_losses.reset()
        val_losses_re.reset()
        val_losses_cls.reset()
        val_acc_v.reset()

        with torch.no_grad():
            for tables, labels, masks in test_loader:
                tables, labels, masks = tables.to(device), labels.to(device), masks.to(device)

                if opt_dict['model_config']['net_v_tabular'] == 'encoder_mlp_block_prototype':
                    val_features = model_feature(tables, masks, masks, has_fc=False)
                    val_results_re = model_re(val_features)
                    B, N, D = val_features.shape
                    val_features_fc = val_features.reshape(B, N * D)
                    val_loss = loss_function_re(val_results_re, tables, masks)
                elif opt_dict['model_config']['net_v_tabular'] == 'encoder_mlp_prototype': 
                    val_features = model_feature(tables, masks, masks)
                    val_results = model_re(val_features)
                    val_features_fc = val_features
                    val_loss = loss_function_re(val_results, tables, masks)

                val_loss_re = val_loss[0]
                val_results_cls = model_fc(val_features_fc)
                val_loss_cls = loss_function_cls(val_results_cls, labels)
                val_loss_total = val_loss_cls * lambda_cls + val_loss_re * lambda_re

                val_losses.update(val_loss_total.item(), tables.size(0))
                val_losses_re.update(val_loss_re.item(), tables.size(0))
                val_losses_cls.update(val_loss_cls.item(), tables.size(0))

                val_acc = accuracy(val_results_cls, labels)[0]
                val_acc_v.update(val_acc.item(), tables.size(0))

            log_out_val = ('Val_Loss_total {loss.val:.4f} ({loss.avg:.4f})\t'
                           'Val_Loss_Re {loss_re.val:.4f} ({loss_re.avg:.4f})\t'
                           'Val_Loss_Cls {loss_cls.val:.4f} ({loss_cls.avg:.4f})\t'
                           'Val_Acc {val_acc_v.val:.4f}({val_acc_v.avg:.4f})'.format(
                               loss=val_losses, loss_re=val_losses_re, loss_cls=val_losses_cls, val_acc_v=val_acc_v)
                           )
            
            log_train = write_log(log_train, log_out_val, opt_dict)
            val_all_losses.append(val_losses.avg)
            log_train = save_best_model_cls(lambda_re, lambda_cls, log_train, model_feature, model_re, model_fc, less_total_losses, val_losses, opt_dict)
            # less_cls_acc, log_train = save_best_model_cls_acc(log_train, model_feature, model_re, model_fc, less_cls_acc, val_acc_v, opt_dict)
            if(less_total_losses > val_losses.avg):
                less_total_losses = val_losses.avg
                update_re_losses = val_losses_re.avg
                update_cls_losses = val_losses_cls.avg
                update_cls_acc = val_acc_v.avg

            if(val_acc_v.avg > 50):
                print("======")
                save_feature_path = f"/data/blood_dvm/data/result/temp/tabuar_re_cls/05_feature_{epochs}_{opt_dict['train_config']['lr_max']}_{val_losses.avg}_{val_acc_v.avg}.pth"
                save_re_path = f"/data/blood_dvm/data/result/temp/tabuar_re_cls/05_re_{epochs}_{opt_dict['train_config']['lr_max']}_{val_losses.avg}_{val_acc_v.avg}.pth"
                save_cls_path = f"/data/blood_dvm/data/result/temp/tabuar_re_cls/05_fc_{epochs}_{opt_dict['train_config']['lr_max']}_{val_losses.avg}_{val_acc_v.avg}.pth"
                torch.save(model_feature.state_dict(), save_feature_path)
                torch.save(model_re.state_dict(), save_re_path)
                torch.save(model_fc.state_dict(), save_cls_path)


    print(f"less_total_losses:{less_total_losses}")
    print(f"update_re_losses:{update_re_losses}")
    print(f"update_cls_losses:{update_cls_losses}")
    print(f"update_cls_acc:{update_cls_acc}")
    print("Finish!")
    return less_total_losses, update_re_losses, update_cls_losses, update_cls_acc



def train_epoch_cls(train_loader, test_loader, device, opt_dict, lambda_re=None, lambda_cls=None):
    print(opt_dict['train_config']['mode'])
    print(opt_dict['model_config']['net_v_tabular'])
    less_total_losses1 = [] 
    update_re_losses1 = []
    update_cls_losses1 = []
    update_cls_acc1 = []

    max_epoch = opt_dict['train_config']['max_epochs']
    for epoch in range(100, max_epoch, 5):
        print(f"现在是Epoch={epoch}")
        opt_dict['train_config']['epochs'] = epoch
        print(f"opt_dict['train_config']['epoch']:{opt_dict['train_config']['epochs']}")
        model_feature = load_feature_model(opt_dict)
        model_re = load_reconstruction_model(opt_dict)
        model_fc = load_fc_model(opt_dict)
        # less_total_losses, update_re_losses, update_cls_losses, update_cls_acc = train_re_cls(lambda_re, lambda_cls, model_transdim, model_feature, model_re, model_fc, train_loader, test_loader, device, opt_dict)
        less_total_losses, update_re_losses, update_cls_losses, update_cls_acc = train_re_cls1(model_feature, model_re, model_fc, train_loader, test_loader, device, opt_dict)
        # less_total_losses, update_re_losses, update_cls_losses, update_cls_acc = train_re_cls_gcn(lambda_re, lambda_cls, model_gcn, model_feature, model_re, model_fc, train_loader, test_loader, device, opt_dict)
        print(f"less_total_losses :{less_total_losses}")
        less_total_losses1.append(less_total_losses)
        update_re_losses1.append(update_re_losses)
        update_cls_losses1.append(update_cls_losses)
        update_cls_acc1.append(update_cls_acc)

    print('Finished')
    return less_total_losses1, update_re_losses1, update_cls_losses1, update_cls_acc1



if __name__ == '__main__':
    from build_dataset_unit import UnitDataset
    mask_version = opt_dict['dataset_config']['missing_rate']
    print(f"mask_version : {mask_version}")
    dataset_dir = opt_dict['dataset_config']['data_dir']

    if opt_dict['dataset_config']['dataname'] == 'blood':
        train_dataset = UnitDataset(dataset_dir, mode='train', dataset_type='tabular', mask_version=mask_version)
        test_dataset = UnitDataset(dataset_dir, mode='test', dataset_type='tabular', mask_version=mask_version)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=64)
    elif opt_dict['dataset_config']['dataname'] == 'dvm':
        train_dataset = tabular_dataset_dvm(opt_dict, 'train')
        test_dataset = tabular_dataset_dvm(opt_dict, 'test')
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=64)

    
    # model_feature = load_feature_model(opt_dict)
    # model_re = load_reconstruction_model(opt_dict)
    # model_fc = load_fc_model(opt_dict)

    # print(f"lambda_re:{lambda_re}, lambda_cls:{lambda_cls}")
    less_total_losses, update_re_losses, update_cls_losses, update_cls_acc = train_epoch_cls(train_loader, test_loader, device, opt_dict)
    # less_total_losses, update_re_losses, update_cls_losses, update_cls_acc = train_re_cls1(model_feature, model_re, model_fc, train_loader, test_loader, device, opt_dict)

