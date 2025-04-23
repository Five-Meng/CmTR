import sys
sys.path.append('/home/admin1/User/mxy/demo/')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.losses import ReconstructionLoss_MLP, KLLoss
from backbone_tabular_prototype.reconstruction_method import MLP_Embedding_R
from utils.utils import AverageMeter, accuracy
import torch.nn.functional as F
from crossattention import CrossAttention
from backbone_tabular.TabularEncoder2 import TabularEncoder
import os

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


def create_log(opt_dict, lambda_cls, lambda_align):
    log_path = os.path.join("/data/blood_dvm/data/result/temp/reclsp/", f"{lambda_cls}_{lambda_align}_{opt_dict['dataset_config']['dataname']}_{opt_dict['train_config']['epochs']}_{opt_dict['dataset_config']['missing_rate']}_{opt_dict['model_config']['model_name']}_{opt_dict['model_config']['net']}_{opt_dict['train_config']['lr_max']}_log_train.csv")
    print(log_path)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log_train = open(log_path, 'w')

    return log_train


def write_log(log_train, log_out, opt_dict):
    log_train.write(log_out + '\n')
    log_train.flush()
    return log_train


def load_feature_model(opt_dict, net_pretrain_path):
    model = TabularEncoder(opt_dict, is_fc=False)
    weights_dict = torch.load(net_pretrain_path, map_location='cpu')
    model.load_state_dict(weights_dict, strict=True)
    for name, param in model.named_parameters():
        param.requires_grad = True
    return model


def load_fc_model(opt_dict):
    input_size = opt_dict['model_config']['hidden_size3']
    model_fc3 = nn.Linear(input_size, opt_dict['train_config']['num_cls'])
    return model_fc3


def load_atten_model(opt_dict):
    model = CrossAttention(opt_dict['model_config']['hidden_size3'])
    return model


def img_tab_align(img_proj, tab_proj, model_atten):
    loss_function = KLLoss()
    loss_align = loss_function(tab_proj, img_proj)    # (input, target)
    features = model_atten(img_proj, tab_proj, tab_proj)
    # features = model_atten(tab_proj, img_proj, img_proj)
    return features, loss_align


def train(model_img, model_tab, model_fc, model_attn, train_loader, test_loader, device, opt_dict):
    lambda_cls, lambda_align = 0.7, 0.3
    num_cat, num_con, cat_offsets = model_img.num_cat, model_img.num_con, model_img.cat_offsets.to(device)
    epochs = opt_dict['train_config']['epochs']
    log_train = create_log(opt_dict, lambda_cls, lambda_align)

    train_acc = AverageMeter('train_acc', ':.4e')
    val_acc = AverageMeter('val_acc', ':.4e')
    train_align = AverageMeter('train_align', ':.4e')
    val_align = AverageMeter('val_align', ':.4e')

    best_val_acc = 0

    model_img.to(device)
    model_tab.to(device)
    model_fc.to(device)
    model_attn.to(device)

    loss_function_cls = nn.CrossEntropyLoss()

    pg_img = [p for p in model_img.parameters() if p.requires_grad]
    pg_tab = [p for p in model_tab.parameters() if p.requires_grad]
    pg_fc = [p for p in model_fc.parameters() if p.requires_grad]
    pg_attn = [p for p in model_attn.parameters() if p.requires_grad]
    pg = pg_img + pg_tab + pg_fc + pg_attn

    optimizer = optim.Adam(pg, lr=float(opt_dict['train_config']['lr_max']))
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt_dict['train_config']['epochs'])

    for epoch in range(epochs):
        # # import ipdb;ipdb.set_trace();
        train_acc.reset()
        train_align.reset()

        model_img.train()
        model_tab.train()
        model_fc.train()
        model_attn.train()
        # import ipdb;ipdb.set_trace();
        for step, data in enumerate(train_loader):
            tables_i, tables_t, labels, masks = data
            tables_i, tables_t, labels, masks = tables_i.to(device), tables_t.to(device), labels.to(device), masks.to(device)
            batch_size_cur = tables_i.size(0)
            masks = torch.zeros_like(masks, dtype=torch.bool)
            features_img = model_img(tables_i, masks, masks)
            features_tab = model_tab(tables_t, masks, masks)
            feature, loss_align = img_tab_align(features_img, features_tab, model_attn)
            results = model_fc(feature)
            loss_cls = loss_function_cls(results, labels)
            loss_total = lambda_cls * loss_cls + lambda_align * loss_align

            train_acc_o = accuracy(results, labels)[0]
            train_acc.update(train_acc_o.item(), batch_size_cur)
            train_align.update(loss_align.item(), batch_size_cur)

            log_out = ('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                'Train_Acc {acc.val:.4f} ({acc.avg:.4f})\t'
                'Train_Align {loss_align.val:.4f} ({loss_align.avg:.4f})\t'.format(
                epoch, step, len(train_loader), lr=optimizer.param_groups[-1]['lr'],
                acc=train_acc, loss_align=train_align
            ))
            log_train = write_log(log_train, log_out, opt_dict)
                
            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()

        schedule.step()

        model_img.eval()
        model_tab.eval()
        model_fc.eval()
        model_attn.eval()

        val_acc.reset()
        val_align.reset()

        # import ipdb;ipdb.set_trace();
        with torch.no_grad():
            for step, data in enumerate(test_loader):
                tables_i, tables_t, labels, masks = data
                tables_i, tables_t, labels, masks = tables_i.to(device), tables_t.to(device), labels.to(device), masks.to(device)
                masks = torch.zeros_like(masks, dtype=torch.bool)
                features_img = model_img(tables_i, masks, masks)
                features_tab = model_tab(tables_t, masks, masks)
                feature, loss_align = img_tab_align(features_img, features_tab, model_attn)
                results = model_fc(feature)
                loss_cls = loss_function_cls(results, labels)
                loss_total = lambda_cls * loss_cls + lambda_align * loss_align

                val_acc_o = accuracy(results, labels)[0]
                val_acc.update(val_acc_o.item(), batch_size_cur)
                val_align.update(loss_align.item(), batch_size_cur)

            log_out_val = ('Val_Acc {acc.val:.4f} ({acc.avg:.4f})\t'
                'Val_Align {loss_align.val:.4f} ({loss_align.avg:.4f})\t'.format(
                acc=val_acc, loss_align=val_align
            ))
            log_train = write_log(log_train, log_out_val, opt_dict)
            
            if(val_acc.avg > best_val_acc):
                best_val_acc = val_acc.avg
                log_best = f"Saved at {best_val_acc}"
                log_train = write_log(log_train, log_best, opt_dict)

            if(best_val_acc > 83):
                # save_feature_path = f"/data/blood_dvm/data/result/temp/reclsp/feature_{lambda_cls}_{lambda_align}_{opt_dict['dataset_config']['missing_rate']}_{epochs}_{opt_dict['train_config']['lr_max']}_{best_val_acc}.pth"
                # save_re_path = f"/data/blood_dvm/data/result/temp/reclsp/re_{lambda_cls}_{lambda_align}_{opt_dict['dataset_config']['missing_rate']}_{epochs}_{opt_dict['train_config']['lr_max']}_{best_val_acc}.pth"
                # torch.save(model_img.state_dict(), save_feature_path)
                # torch.save(model_tab.state_dict(), save_re_path)
                save_attn_path = f"/data/blood_dvm/data/result/temp/reclsp/DVM_attn_{lambda_cls}_{lambda_align}_{opt_dict['dataset_config']['missing_rate']}_{epochs}_{opt_dict['train_config']['lr_max']}_{best_val_acc}.pth"
                torch.save(model_attn.state_dict(), save_attn_path)


    print("Finish!")
    return best_val_acc


def train_epoch(train_loader, test_loader, device, opt_dict):
    max_epoch = opt_dict['train_config']['max_epochs']
    for epoch in range(200, max_epoch, 5):
        print(f"现在是Epoch={epoch}")
        opt_dict['train_config']['epochs'] = epoch
        print(f"opt_dict['train_config']['epoch']:{opt_dict['train_config']['epochs']}")
        # net_pretrain = "/data/blood_dvm/data/result/end/encoder_tabular/blood/reclsprototype/05_feature_140_78_1e-5_74.05857740585775_0.5890737486018629.pth"
        # net_pretrain = "/data/blood_dvm/data/result/end/encoder_tabular/blood/recls/feature_0.2_0.8_blood_140_0.3_None_crossentropy_encoder_mlp_prototype_5e-5_best_model.pth"
        # net_pretrain = "/data/blood_dvm/data/result/end/encoder_tabular/blood/recls/feature_0.2_0.8_blood_140_0.7_None_crossentropy_encoder_mlp_prototype_5e-5_best_model.pth"
        # net_pretrain = "/data/blood_dvm/data/result/end/encoder_tabular/blood/recls/feature_0.2_0.8_blood_140_0.5_None_crossentropy_encoder_mlp_prototype_5e-5_best_model.pth"
        net_pretrain = "/data/blood_dvm/data/result/end/encoder_tabular/dvm/re_cls_prototype/03_feature_100_5e-5_0.7530563086211822_69.85080718302194.pth"
        model_img = load_feature_model(opt_dict, net_pretrain)
        model_tab = load_feature_model(opt_dict, net_pretrain)
        model_fc = load_fc_model(opt_dict)
        model_attn = load_atten_model(opt_dict)
        best_acc = train(model_img, model_tab, model_fc, model_attn, train_loader, test_loader, device, opt_dict)
        print(f"best_acc: {best_acc}")

    print('Finished')
    return 



if __name__ == '__main__':

    from build_dataset_2tab import tabular_dataset_2tab, tabular_dataset_dvm_2tab
    mask_version = opt_dict['dataset_config']['missing_rate']
    print(f"mask_version : {mask_version}")
    dataset_dir = opt_dict['dataset_config']['data_dir']
    batch_size = opt_dict['train_config']['batch_size']

    if opt_dict['dataset_config']['dataname'] == 'blood':
        train_dataset = tabular_dataset_2tab(opt_dict, 'train')
        test_dataset = tabular_dataset_2tab(opt_dict, 'test')
    else:
        train_dataset = tabular_dataset_dvm_2tab(opt_dict, 'train')
        test_dataset = tabular_dataset_dvm_2tab(opt_dict, 'test')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # import ipdb;ipdb.set_trace();
    # net_pretrain = "/data/blood_dvm/data/result/end/encoder_tabular/blood/reclsprototype/05_feature_140_78_1e-5_74.05857740585775_0.5890737486018629.pth"
    # net_pretrain = "/data/blood_dvm/data/result/end/encoder_tabular/blood/recls/feature_0.2_0.8_blood_140_0.3_None_crossentropy_encoder_mlp_prototype_5e-5_best_model.pth"
    # model_img = load_feature_model(opt_dict, net_pretrain)
    # model_tab = load_feature_model(opt_dict, net_pretrain)
    # model_fc = load_fc_model(opt_dict)
    # model_attn = load_atten_model(opt_dict)
    # import ipdb;ipdb.set_trace();
    # train(model_img, model_tab, model_fc, model_attn, train_loader, test_loader, device, opt_dict)
    train_epoch(train_loader, test_loader, device, opt_dict)


