import sys
sys.path.append('/home/admin1/User/mxy/demo/')

import torch
import torch.nn as nn
import torch.optim as optim
import time
from torch.utils.data import DataLoader
from utils.losses import ReconstructionLoss_MLP, KLLoss
from backbone_tabular_prototype.reconstruction_method import MLP_Embedding_R
import os
from utils.utils import AverageMeter, accuracy

from argparses.yaml_args import yaml_data
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



class AlignmentModel(nn.Module):
    def __init__(self, opt_dict, img_dim=1280, tab_dim=128):
        super().__init__()
        latent_dim = opt_dict['model_config']['latent_dim']
        # re_dim = opt_dict['model_config']['re_dim']
        self.img_proj = nn.Sequential(
            nn.Linear(img_dim, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim)
        )
        
        self.tab_proj = nn.Sequential(
            nn.Linear(tab_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )

    def forward(self, img_feat, tab_feat):
        return self.img_proj(img_feat), self.tab_proj(tab_feat)

    


def create_log(opt_dict):
    log_train = None
    if not opt_dict['train_config']['find_epoch']:
        log_path = os.path.join(opt_dict['dataset_config']['it_result_path'], opt_dict['model_config']['net_v'], f"{opt_dict['dataset_config']['dataname']}_{opt_dict['train_config']['epochs']}_{opt_dict['dataset_config']['missing_rate']}_{opt_dict['model_config']['model_name']}_{opt_dict['model_config']['net_v']}_{opt_dict['train_config']['lr_max']}_log_train.csv")
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


def load_model_tabular(opt_dict):
    from backbone_tabular.TabularEncoder2 import TabularEncoder
    model = TabularEncoder(opt_dict, is_fc=False)
    net_pretrain_path = "/data/blood_dvm/data/result/end/encoder_mlp_prototype/feature_blood_195_0.7_crossentropy_encoder_mlp_prototype_1e-4_best_model.pth"
    # net_pretrain_path = "/data/blood_dvm/data/result/temp/tab/feature_70_0.8045396147662127.pth"
    weights_dict = torch.load(net_pretrain_path, map_location='cpu')
    model.load_state_dict(weights_dict, strict=True)
    for name, param in model.named_parameters():
        param.requires_grad = False
    return model


def load_reconstruction_model(opt_dict, hidden_size2, net_pretrain_path=None): 
    # opt_dict['model_config']['hidden_size2'] = 1280
    # opt_dict['model_config']['hidden_size2'] = 128
    # opt_dict['model_config']['hidden_size2'] = opt_dict['model_config']['latent_dim']
    opt_dict['model_config']['hidden_size2'] = hidden_size2
    model = MLP_Embedding_R(opt_dict)
    if(net_pretrain_path):
        weights_dict = torch.load(net_pretrain_path, map_location='cpu')
        model.load_state_dict(weights_dict, strict=True)
        for name, param in model.named_parameters():
            param.requires_grad = False
    return model



def load_model_image(opt_dict):
    from backbone.efficientnet import efficientnet_b1
    model = efficientnet_b1(num_classes=opt_dict['train_config']['num_cls'])
    net_pretrain_path = "/data/blood_dvm/data/result/temp/img/feature_120_1e-3_0.5603332085797311.pth"
    weights_dict = torch.load(net_pretrain_path, map_location='cpu')
    # import ipdb;ipdb.set_trace();
    model.load_state_dict(weights_dict, strict=True)
    for name, param in model.named_parameters():
        param.requires_grad = False
    return model



def load_model_align(opt_dict):
    model = AlignmentModel(opt_dict)
    return model



def img_tab_align(model_align, features_image, features_tabular, method='kl'):
    img_proj, tab_proj = model_align(features_image, features_tabular)
    if method == 'mse':
        loss_function = nn.MSELoss()   
        loss_align = loss_function(img_proj, tab_proj)   
        features = img_proj

    elif method == 'kl':
        loss_function = KLLoss()
        loss_align = loss_function(tab_proj, img_proj)    # (input, target)
        # features = img_proj
        features = tab_proj

    return features, loss_align



def train_tabular_re(model_feature, model_re, train_loader, test_loader, device, opt_dict):
    num_cat, num_con, cat_offsets = model_feature.num_cat, model_feature.num_con, model_feature.cat_offsets.to(device)
    epochs = opt_dict['train_config']['epochs']
    print(f"epochs: {epochs}")
    log_train = create_log(opt_dict)
    train_losses_re = AverageMeter('Train_Loss_re', ':.4e')
    val_losses_re = AverageMeter('Val_loss', ':.4e')
    least_val_re = 1e9

    model_feature.to(device)
    model_re.to(device)
    loss_function = ReconstructionLoss_MLP(num_cat, num_con, opt_dict)

    pg_feature = [p for p in model_feature.parameters() if p.requires_grad]
    pg_re = [p for p in model_re.parameters() if p.requires_grad]
    pg = pg_feature + pg_re

    optimizer = optim.Adam(pg, lr=float(opt_dict['train_config']['lr_max']))
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt_dict['train_config']['epochs'])

    for epoch in range(epochs):
        train_losses_re.reset()

        model_feature.train()
        model_re.train()

        for step, data in enumerate(train_loader):
            images, tables, masks, labels = data
            masks = torch.ones_like(masks, dtype=torch.bool)
            images, tables, masks, labels = images.to(device), tables.to(device), masks.to(device), labels.to(device)
            batch_size_cur = tables.size(0)
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
            for step, data in enumerate(test_loader):
                images, tables, masks, labels = data
                masks = torch.ones_like(masks, dtype=torch.bool)
                images, tables, masks, labels = images.to(device), tables.to(device), masks.to(device), labels.to(device)
                val_features = model_feature(tables, masks, masks)
                val_results = model_re(val_features)
                val_loss = loss_function(val_results, tables, masks)

                val_loss_re = val_loss[0]
                val_losses_re.update(val_loss_re.item(), tables.size(0))
                
            log_out_val = ('Val_Loss_re {loss_re.val:.4f} ({loss_re.avg:.4f})\n'.format(loss_re=val_losses_re))
            log_train = write_log(log_train, log_out_val, opt_dict)
        
        if(least_val_re > val_losses_re.avg):
            least_val_re = val_losses_re.avg
            if(least_val_re < 0.81):
                save_feature_path = f"/data/blood_dvm/data/result/temp/tab/07_feature_{epochs}_{least_val_re}.pth"
                save_re_path = f"/data/blood_dvm/data/result/temp/tab/07_re_{epochs}_{least_val_re}.pth"
                torch.save(model_feature.state_dict(), save_feature_path)
                torch.save(model_re.state_dict(), save_re_path)

    return least_val_re




def train_img_r(model_feature, model_re, train_loader, test_loader, device, opt_dict):
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
                loss=train_losses_re,
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
            loss=val_losses_re,
        ))
        log_train = write_log(log_train, log_out, opt_dict) 
        if(least_val_re > val_losses_re.avg):
            least_val_re = val_losses_re.avg
            log_best_re = f"Saved Least Loss_Re: {least_val_re}"
            log_train = write_log(log_train, log_best_re, opt_dict)
            if(least_val_re < 0.58):
                save_feature_path = f"/data/blood_dvm/data/result/temp/img/{opt_dict['dataset_config']['missing_rate']}_feature_{epochs}_{opt_dict['train_config']['lr_max']}_{least_val_re}.pth"
                save_re_path = f"/data/blood_dvm/data/result/temp/img/{opt_dict['dataset_config']['missing_rate']}_re_{epochs}_{opt_dict['train_config']['lr_max']}_{least_val_re}.pth"
                # import ipdb;ipdb.set_trace();
                torch.save(model_feature.state_dict(), save_feature_path)
                torch.save(model_re.state_dict(), save_re_path)

    return least_val_re




def train_img_tab_r(model_image, model_tabular, model_re, model_re_img, model_re_tab, model_align, train_loader, test_loader, device, opt_dict):
    lambda_re = 0.7
    lambda_align = 0.3           # 调整

    # lambda_re = 0.5
    # lambda_align = 0.5
    
    # lambda_re = 0.3
    # lambda_align = 0.7

    pg_img = [p for p in model_image.parameters() if p.requires_grad]
    pg_tab = [p for p in model_tabular.parameters() if p.requires_grad]
    pg_re = [p for p in model_re.parameters() if p.requires_grad]
    pg_re_img = [p for p in model_re_img.parameters() if p.requires_grad]
    pg_re_tab = [p for p in model_re_tab.parameters() if p.requires_grad]
    pg_align = [p for p in model_align.parameters() if p.requires_grad]
    pg = pg_img + pg_tab + pg_re + pg_align + pg_re_img + pg_re_tab
    optimizer = optim.Adam(pg, lr=float(opt_dict['train_config']['lr_max']))
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt_dict['train_config']['epochs'])
    
    model_image.to(device)
    model_tabular.to(device)
    model_re.to(device)
    model_re_img.to(device)
    model_re_tab.to(device)
    model_align.to(device)
    
    if opt_dict['dataset_config']['dataname'] == 'blood':
        num_cat, num_con = 19, 22
    loss_function = ReconstructionLoss_MLP(num_cat, num_con, opt_dict)
    epochs = opt_dict['train_config']['epochs']
    log_train = create_log(opt_dict)

    train_losses = AverageMeter('Train_Loss', ':.4e')
    val_losses = AverageMeter('Val_loss', ':.4e')
    train_losses_re = AverageMeter('Train_re', ':.4e')
    val_losses_re = AverageMeter('Val_re', ':.4e')
    train_losses_align = AverageMeter('Train_align', ':.4e')
    val_losses_align = AverageMeter('Val_align', ':.4e')
    train_losses_re_img = AverageMeter('Train_align', ':.4e')
    val_losses_re_img = AverageMeter('Val_align', ':.4e')
    train_losses_re_tab = AverageMeter('Train_align', ':.4e')
    val_losses_re_tab = AverageMeter('Val_align', ':.4e')
    
    least_val_re = 1e9

    for epoch in range(epochs):
        train_losses_re.reset()
        train_losses.reset()
        train_losses_align.reset()

        model_image.train()
        model_tabular.train()
        model_align.train()
        model_re.train()
        model_re_img.train()
        model_re_tab.train()

        for step, data in enumerate(train_loader):
            # import ipdb;ipdb.set_trace();
            images, tables, masks, labels = data
            images, tables, masks, labels = images.to(device), tables.to(device), masks.to(device), labels.to(device)
            batch_size_cur = images.size(0)
            features_image = model_image.forward_sne(images)
            features_tabular = model_tabular(tables, masks, masks)
            features, loss_align = img_tab_align(model_align, features_image, features_tabular)
            results_img = model_re_img(features_image)
            results_tab = model_re_tab(features_tabular)
            results = model_re(features)
            loss = loss_function(results, tables, masks)
            masks_i = torch.ones_like(masks, dtype=torch.bool)
            loss_img = loss_function(results_img, tables, masks_i)
            loss_tab = loss_function(results_tab, tables, masks)
            loss_re_total = loss[0]
            loss_re_img = loss_img[0]
            loss_re_tab = loss_tab[0]
            # loss_re = (loss_re_total + loss_re_img + loss_re_tab) / 3
            loss_re = loss_re_total
            loss_total = lambda_re * loss_re + lambda_align * loss_align

            train_losses_re.update(loss_re.item(), batch_size_cur)
            train_losses_align.update(loss_align.item(), batch_size_cur)
            train_losses.update(loss_total.item(), batch_size_cur)
            train_losses_re_img.update(loss_re_img.item(), batch_size_cur)
            train_losses_re_tab.update(loss_re_tab.item(), batch_size_cur)

            log_out = ('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                'Lambda_re {lambda_re}, Lambda_align {lambda_align}\t'
                'Train_Loss_re {loss.val:.4f} ({loss.avg:.4f})\t'
                'Train_losses_re_img {train_losses_re_img.val:.4f} ({train_losses_re_img.avg:.4f})\t'
                'Train_losses_re_tab {train_losses_re_tab.val:.4f} ({train_losses_re_tab.avg:.4f})\t'
                'Train_Loss_align {loss_align.val:.4f} ({loss_align.avg:.4f})\t'
                'Train_Loss_Total {loss_total.val: .4f} ({loss_total.avg:.4f})'.format(
                epoch, step, len(train_loader), lr=optimizer.param_groups[-1]['lr'],
                lambda_re=lambda_re, lambda_align=lambda_align, 
                loss=train_losses_re, train_losses_re_img=train_losses_re_img, train_losses_re_tab=train_losses_re_tab, 
                loss_align=train_losses_align, loss_total=train_losses
            ))
            log_train = write_log(log_train, log_out, opt_dict)
                
            optimizer.zero_grad()
            loss_re.backward()
            optimizer.step()

        schedule.step()

        model_image.eval()
        model_tabular.eval()
        model_re.eval()
        model_align.eval()
        model_re_img.eval()
        model_re_tab.eval()

        val_losses_re.reset()
        val_losses.reset()
        val_losses_align.reset()
        
        with torch.no_grad():
            for step, data in enumerate(test_loader):
                # import ipdb;ipdb.set_trace();
                images, tables, masks, labels = data
                images, tables, masks, labels = images.to(device), tables.to(device), masks.to(device), labels.to(device)
                batch_size_cur = images.size(0)
                val_features_image = model_image.forward_sne(images)
                val_features_tabular = model_tabular(tables, masks, masks)
                val_features, val_loss_align = img_tab_align(model_align, val_features_image, val_features_tabular)
                val_results = model_re(val_features)
                val_results_img = model_re_img(val_features_image)
                val_results_tab = model_re_tab(val_features_tabular)
                masks_i = torch.ones_like(masks, dtype=torch.bool)
                loss = loss_function(val_results, tables, masks)
                loss_img = loss_function(val_results_img, tables, masks_i)
                loss_tab = loss_function(val_results_tab, tables, masks)
                val_loss_re = loss[0]
                val_loss_re_img = loss_img[0]
                val_loss_re_tab = loss_tab[0]
                val_loss_total = lambda_re * val_loss_re + lambda_align * val_loss_align

                val_losses_re.update(val_loss_re.item(), batch_size_cur)
                val_losses.update(val_loss_total.item(), batch_size_cur)
                val_losses_align.update(val_loss_align.item(), batch_size_cur)
                val_losses_re_img.update(val_loss_re_img.item(), batch_size_cur)
                val_losses_re_tab.update(val_loss_re_tab.item(), batch_size_cur)

        log_out = ('Val_Loss_re {val_loss.val:.4f} ({val_loss.avg:.4f})\t'
            'Val_Loss_re_img {val_losses_re_img.val:.4f} ({val_losses_re_img.avg:.4f})\t'
            'Val_Loss_re_tab {val_losses_re_tab.val:.4f} ({val_losses_re_tab.avg:.4f})\n'
            'Val_Loss_align {val_loss_align.val:.4f} ({val_loss_align.avg:.4f})\t'
            'Val_Loss_Total {val_loss_total.val: .4f} ({val_loss_total.avg:.4f})'.format(
            val_loss=val_losses_re, val_losses_re_img=val_losses_re_img, val_losses_re_tab=val_losses_re_tab, 
            val_loss_align=val_losses_align, val_loss_total=val_losses
        ))
        log_train = write_log(log_train, log_out, opt_dict)
            
        if(least_val_re > val_losses_re.avg):
            least_val_re = val_losses_re.avg
            log_best_re = f"Saved Least Loss_Re: {least_val_re}"
            log_train = write_log(log_train, log_best_re, opt_dict)

    return least_val_re




if __name__ == '__main__':
    from build_dataset_unit import UnitDataset
    mask_version = opt_dict['dataset_config']['missing_rate']
    print(f"mask_version : {mask_version}")
    dataset_dir = opt_dict['dataset_config']['data_dir']

    if opt_dict['dataset_config']['dataname'] == 'blood':
        train_dataset = UnitDataset(dataset_dir, mode='train', dataset_type='image_tabular', mask_version=mask_version)
        test_dataset = UnitDataset(dataset_dir, mode='test', dataset_type='image_tabular', mask_version=mask_version)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=64)

    # re_loss_list = []
    # for epoch in range(30, 400, 5):
    #     # model_feature = load_model_image(opt_dict)
    #     model_feature = load_model_tabular(opt_dict)
    #     model_re = load_reconstruction_model(opt_dict)
    #     opt_dict['train_config']['epochs'] = epoch
    #     # least_val_re = train_img_r(model_feature, model_re, train_loader, test_loader, device, opt_dict)
    #     least_val_re = train_tabular_re(model_feature, model_re, train_loader, test_loader, device, opt_dict)
    #     print(f"epoch: {epoch}, least_val_re: {least_val_re}")
    #     re_loss_list.append(least_val_re)

    model_image = load_model_image(opt_dict)
    model_tabular = load_model_tabular(opt_dict)
    model_align = load_model_align(opt_dict)
    model_re = load_reconstruction_model(opt_dict, hidden_size2=opt_dict['model_config']['latent_dim'])
    model_re_img = load_reconstruction_model(opt_dict, hidden_size2=1280, net_pretrain_path="/data/blood_dvm/data/result/temp/img/re_120_1e-3_0.5603332085797311.pth")
    model_re_tab = load_reconstruction_model(opt_dict, hidden_size2=128, net_pretrain_path="/data/blood_dvm/data/result/end/encoder_mlp_prototype/re_blood_195_0.7_crossentropy_encoder_mlp_prototype_1e-4_best_model.pth")

    train_img_tab_r(model_image, model_tabular, model_re, model_re_img, model_re_tab, model_align, train_loader, test_loader, device, opt_dict)

    import ipdb;ipdb.set_trace();
