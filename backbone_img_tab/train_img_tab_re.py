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
from Alignment import AlignmentModel
from img_tab_util import create_log, write_log
from ContrastiveLoss import ContrastiveLoss


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



def load_model_tabular(opt_dict):
    from backbone_tabular.TabularEncoder2 import TabularEncoder
    model = TabularEncoder(opt_dict, is_fc=False)
    net_pretrain_path = "/data/blood_dvm/data/result/end/encoder_tabular/blood/re/feature_blood_195_0.7_crossentropy_encoder_mlp_prototype_1e-4_best_model.pth"
    # net_pretrain_path = "/data/blood_dvm/data/result/end/encoder_tabular/blood/re/feature_blood_195_0.5_crossentropy_encoder_mlp_prototype_1e-4_best_model.pth"

    weights_dict = torch.load(net_pretrain_path, map_location='cpu')
    model.load_state_dict(weights_dict, strict=True)
    for name, param in model.named_parameters():
        param.requires_grad = True
    return model


def load_reconstruction_model(opt_dict, hidden_size2): 
    opt_dict['model_config']['hidden_size2'] = hidden_size2
    model = MLP_Embedding_R(opt_dict)

    return model



def load_model_image(opt_dict):
    from backbone.efficientnet import efficientnet_b1
    model = efficientnet_b1(num_classes=opt_dict['train_config']['num_cls'])
    net_pretrain_path = "/data/blood_dvm/data/result/temp/img/feature_120_1e-3_0.5603332085797311.pth"
    weights_dict = torch.load(net_pretrain_path, map_location='cpu')
    model.load_state_dict(weights_dict, strict=True)
    for name, param in model.named_parameters():
        param.requires_grad = True
    return model



def load_model_align(opt_dict):
    model = AlignmentModel(opt_dict)
    return model



def load_model_atten(opt_dict):
    model = CrossAttention(opt_dict['model_config']['latent_dim'])
    return model



def img_tab_align(model_align, model_atten, tab_mask, features_image, features_tabular, is_crossatten=True, method='con'):
    # import ipdb;ipdb.set_trace();
    img_proj, tab_proj = model_align(features_image, features_tabular)
    if method == 'mse':
        loss_function = nn.MSELoss()   
        loss_align = loss_function(img_proj, tab_proj)   
        features = img_proj

    elif method == 'kl':
        loss_function = KLLoss()
        loss_align = loss_function(tab_proj, img_proj)    # (input, target)
        if(is_crossatten):
            # features = model_atten(img_proj, tab_proj, tab_proj)
            features = model_atten(tab_proj, img_proj, img_proj)
            return features, loss_align
        else:
            features_ = img_proj
            features = tab_proj
            return features_, features, loss_align

    elif method == 'con':
        print("ContrastiveLoss")
        loss_function = ContrastiveLoss()
        loss_align = loss_function(img_proj, tab_proj)
        return img_proj, tab_proj, loss_align
    


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



def train_img_tab_r_epoch(train_loader, test_loader, device, opt_dict):
    least_val_re_list = []
    max_epochs = opt_dict['train_config']['max_epochs']
    for epochs in range(50, max_epochs, 5):
        opt_dict['train_config']['epochs'] = epochs
        model_image = load_model_image(opt_dict)
        model_tabular = load_model_tabular(opt_dict)
        model_align = load_model_align(opt_dict)
        model_re = load_reconstruction_model(opt_dict, hidden_size2=opt_dict['model_config']['latent_dim'])
        model_atten = load_model_atten(opt_dict)
        # model_re_img = load_reconstruction_model(opt_dict, hidden_size2=opt_dict['model_config']['latent_dim'])
        print(f"Epoch: {opt_dict['train_config']['epochs']}")
        # least_val_re = train_img_tab_r(model_image, model_tabular, model_re, model_re_img, model_align, train_loader, test_loader, device, opt_dict)
        least_val_re = train_img_tab_r_atten(model_image, model_tabular, model_re, model_atten, model_align, train_loader, test_loader, device, opt_dict)
        least_val_re_list.append(least_val_re)
        print(least_val_re_list)



def train_img_tab_r(model_image, model_tabular, model_re, model_re_img, model_align, train_loader, test_loader, device, opt_dict):

    pg_img = [p for p in model_image.parameters() if p.requires_grad]
    pg_tab = [p for p in model_tabular.parameters() if p.requires_grad]
    pg_re = [p for p in model_re.parameters() if p.requires_grad]
    pg_re_img = [p for p in model_re_img.parameters() if p.requires_grad]
    pg_align = [p for p in model_align.parameters() if p.requires_grad]
    pg = pg_img + pg_tab + pg_re + pg_align + pg_re_img
    optimizer = optim.Adam(pg, lr=float(opt_dict['train_config']['lr_max']))
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt_dict['train_config']['epochs'])
    
    model_image.to(device)
    model_tabular.to(device)
    model_re.to(device)
    model_re_img.to(device)
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
    
    least_val_re = 1e9

    for epoch in range(epochs):
        # if(epoch * 5 < epochs * 2):
        #     lambda_re = 1
        #     lambda_align = 1
        # else:
        #     lambda_re = 1
        #     lambda_align = 10

        # lambda_re = 1
        # lambda_align = 1

        if(epoch * 5 < epochs * 2):
            lambda_re = 0.7
            lambda_align = 0.3
            for name, param in model_image.named_parameters():
                param.requires_grad = False
            for name, param in model_tabular.named_parameters():
                param.requires_grad = False
        else:
            lambda_re = 0.9
            lambda_align = 0.1
            for name, param in model_image.named_parameters():
                param.requires_grad = False
            for name, param in model_tabular.named_parameters():
                param.requires_grad = False

        train_losses_re.reset()
        train_losses.reset()
        train_losses_align.reset()
        train_losses_re_img.reset()

        model_image.train()
        model_tabular.train()
        model_align.train()
        model_re.train()
        model_re_img.train()

        for step, data in enumerate(train_loader):
            # import ipdb;ipdb.set_trace();
            images, tables, masks, labels = data
            images, tables, masks, labels = images.to(device), tables.to(device), masks.to(device), labels.to(device)
            batch_size_cur = images.size(0)
            features_image = model_image.forward_sne(images)
            features_tabular = model_tabular(tables, masks, masks)

            features_i, features_t, loss_align = img_tab_align(model_align, None, masks, features_image, features_tabular, False)

            results_img = model_re_img(features_i)
            results = model_re(features_t)

            loss = loss_function(results, tables, masks)
            masks_i = torch.ones_like(masks, dtype=torch.bool)
            loss_img = loss_function(results_img, tables, masks)

            loss_re = loss[0]
            loss_re_img = loss_img[0]
            # loss_re_total = (loss_re + 4 * loss_re_img) / 5
            loss_re_total = (loss_re + loss_re_img) / 2
            # loss_re_total = loss_re + loss_re_tab
            loss_total = lambda_re * loss_re_total + lambda_align * loss_align

            train_losses_re.update(loss_re.item(), batch_size_cur)
            train_losses_align.update(loss_align.item(), batch_size_cur)
            train_losses.update(loss_total.item(), batch_size_cur)
            train_losses_re_img.update(loss_re_img.item(), batch_size_cur)

            log_out = ('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                'Lambda_re {lambda_re}, Lambda_align {lambda_align}\t'
                'Train_Loss_re {loss.val:.4f} ({loss.avg:.4f})\t'
                'Train_losses_re_img {train_losses_re_img.val:.4f} ({train_losses_re_img.avg:.4f})\t'
                'Train_Loss_align {loss_align.val:.4f} ({loss_align.avg:.4f})\t'
                'Train_Loss_Total {loss_total.val: .4f} ({loss_total.avg:.4f})'.format(
                epoch, step, len(train_loader), lr=optimizer.param_groups[-1]['lr'],
                lambda_re=lambda_re, lambda_align=lambda_align, 
                loss=train_losses_re, train_losses_re_img=train_losses_re_img,
                loss_align=train_losses_align, loss_total=train_losses
            ))
            log_train = write_log(log_train, log_out, opt_dict)
                
            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()

        schedule.step()

        model_image.eval()
        model_tabular.eval()
        model_re.eval()
        model_align.eval()
        model_re_img.eval()

        val_losses_re.reset()
        val_losses.reset()
        val_losses_align.reset()
        val_losses_re_img.reset()
        
        with torch.no_grad():
            for step, data in enumerate(test_loader):
                # import ipdb;ipdb.set_trace();
                images, tables, masks, labels = data
                images, tables, masks, labels = images.to(device), tables.to(device), masks.to(device), labels.to(device)
                batch_size_cur = images.size(0)
                val_features_image = model_image.forward_sne(images)
                val_features_tabular = model_tabular(tables, masks, masks)

                val_features_i, val_features_t, val_loss_align = img_tab_align(model_align, None, masks, val_features_image, val_features_tabular, False)

                val_results = model_re(val_features_t)
                val_results_img = model_re_img(val_features_i)

                masks_i = torch.ones_like(masks, dtype=torch.bool)
                loss = loss_function(val_results, tables, masks)
                loss_img = loss_function(val_results_img, tables, masks)
                val_loss_re = loss[0]
                val_loss_re_img = loss_img[0]
                # val_loss_re_total = (val_loss_re + 4 * val_loss_re_img) / 5
                val_loss_re_total = (val_loss_re + val_loss_re_img) / 2
                # val_loss_re_total = val_loss_re + val_loss_re_tab
                val_loss_total = lambda_re * val_loss_re_total + lambda_align * val_loss_align

                val_losses_re.update(val_loss_re.item(), batch_size_cur)
                val_losses.update(val_loss_total.item(), batch_size_cur)
                val_losses_align.update(val_loss_align.item(), batch_size_cur)
                val_losses_re_img.update(val_loss_re_img.item(), batch_size_cur)

        log_out = ('Val_Loss_re {val_loss.val:.4f} ({val_loss.avg:.4f})\t'
            'Val_Loss_re_img {val_losses_re_img.val:.4f} ({val_losses_re_img.avg:.4f})\t'
            'Val_Loss_align {val_loss_align.val:.4f} ({val_loss_align.avg:.4f})\t'
            'Val_Loss_Total {val_loss_total.val: .4f} ({val_loss_total.avg:.4f})'.format(
            val_loss=val_losses_re, val_losses_re_img=val_losses_re_img, 
            val_loss_align=val_losses_align, val_loss_total=val_losses
        ))
        log_train = write_log(log_train, log_out, opt_dict)
            
        if(least_val_re > val_losses_re.avg):
            least_val_re = val_losses_re.avg
            log_best_re = f"Saved Least Loss_Re: {least_val_re}"
            log_train = write_log(log_train, log_best_re, opt_dict)
            # if(val_losses_re.avg < 0.61):
            #     print("have saved")
            #     save_image_path = f"/data/blood_dvm/data/result/temp/blood/b2/05_image_{epochs}_{opt_dict['train_config']['lr_max']}_{least_val_re}.pth"
            #     save_table_path = f"/data/blood_dvm/data/result/temp/blood/b2/05_table_{epochs}_{opt_dict['train_config']['lr_max']}_{least_val_re}.pth"
            #     save_imgre_path = f"/data/blood_dvm/data/result/temp/blood/b2/05_imgre_{epochs}_{opt_dict['train_config']['lr_max']}_{least_val_re}.pth"
            #     save_tabre_path = f"/data/blood_dvm/data/result/temp/blood/b2/05_tabre_{epochs}_{opt_dict['train_config']['lr_max']}_{least_val_re}.pth"
            #     save_align_path = f"/data/blood_dvm/data/result/temp/blood/b2/05_align_{epochs}_{opt_dict['train_config']['lr_max']}_{least_val_re}.pth"
            #     torch.save(model_image.state_dict(), save_image_path)
            #     torch.save(model_tabular.state_dict(), save_table_path)
            #     torch.save(model_re_img.state_dict(), save_imgre_path)
            #     torch.save(model_re.state_dict(), save_tabre_path)
            #     torch.save(model_align.state_dict(), save_align_path)

    return least_val_re
    


def train_img_tab_r_atten(model_image, model_tabular, model_re, model_atten, model_align, train_loader, test_loader, device, opt_dict):

    pg_img = [p for p in model_image.parameters() if p.requires_grad]
    pg_tab = [p for p in model_tabular.parameters() if p.requires_grad]
    pg_re = [p for p in model_re.parameters() if p.requires_grad]
    pg_atten = [p for p in model_atten.parameters() if p.requires_grad]
    pg_align = [p for p in model_align.parameters() if p.requires_grad]
    pg = pg_img + pg_tab + pg_re + pg_atten + pg_align
    optimizer = optim.Adam(pg, lr=float(opt_dict['train_config']['lr_max']))
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt_dict['train_config']['epochs'])
    
    model_image.to(device)
    model_tabular.to(device)
    model_re.to(device)
    model_atten.to(device)
    model_align.to(device)
    
    if opt_dict['dataset_config']['dataname'] == 'blood':
        num_cat, num_con = 19, 22
    loss_function = ReconstructionLoss_MLP(num_cat, num_con, opt_dict)
    epochs = opt_dict['train_config']['epochs']
    log_train = create_log(opt_dict)

    train_losses_re = AverageMeter('Train_re', ':.4e')
    val_losses_re = AverageMeter('Val_re', ':.4e')
    # train_losses_align = AverageMeter('Train_align', ':.4e')
    # val_losses_align = AverageMeter('Val_align', ':.4e')
    
    least_val_re = 1e9

    for epoch in range(epochs):
        # lambda_re = 0.1
        # lambda_align = 0.9

        train_losses_re.reset()
        # train_losses_align.reset()

        model_image.train()
        model_tabular.train()
        model_re.train()
        model_atten.train()
        model_align.train()

        for step, data in enumerate(train_loader):
            # import ipdb;ipdb.set_trace();
            images, tables, masks, labels = data
            images, tables, masks, labels = images.to(device), tables.to(device), masks.to(device), labels.to(device)
            batch_size_cur = images.size(0)
            features_image = model_image.forward_sne(images)
            features_tabular = model_tabular(tables, masks, masks)

            features, loss_align = img_tab_align(model_align, model_atten, masks, features_image, features_tabular, True)
            results = model_re(features)
            loss = loss_function(results, tables, masks)
            loss_re = loss[0]
            # loss_total = lambda_re * loss_re + lambda_align * loss_align

            train_losses_re.update(loss_re.item(), batch_size_cur)
            # train_losses_align.update(loss_align.item(), batch_size_cur)

            log_out = ('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                'Train_Loss_re {loss.val:.4f} ({loss.avg:.4f})\t'
                # 'Train_Loss_align {loss_align.val:.4f} ({loss_align.avg:.4f})'
                .format(
                epoch, step, len(train_loader), lr=optimizer.param_groups[-1]['lr'],
                loss=train_losses_re, 
                # loss_align=train_losses_align
            ))
            log_train = write_log(log_train, log_out, opt_dict)
                
            optimizer.zero_grad()
            # loss_total.backward()
            loss_re.backward()
            optimizer.step()

        schedule.step()

        model_image.eval()
        model_tabular.eval()
        model_re.eval()
        model_atten.eval()
        model_align.eval()

        val_losses_re.reset()
        # val_losses_align.reset()
        
        with torch.no_grad():
            for step, data in enumerate(test_loader):
                # import ipdb;ipdb.set_trace();
                images, tables, masks, labels = data
                images, tables, masks, labels = images.to(device), tables.to(device), masks.to(device), labels.to(device)
                batch_size_cur = images.size(0)
                val_features_image = model_image.forward_sne(images)
                val_features_tabular = model_tabular(tables, masks, masks)

                val_features, val_loss_align = img_tab_align(model_align, model_atten, masks, val_features_image, val_features_tabular, True)

                val_results = model_re(val_features)

                loss = loss_function(val_results, tables, masks)
                val_loss_re = loss[0]

                val_losses_re.update(val_loss_re.item(), batch_size_cur)
                # val_losses_align.update(val_loss_align.item(), batch_size_cur)

        log_out = ('Val_Loss_re {val_loss.val:.4f} ({val_loss.avg:.4f})\t'
                #    'Val_Loss_align {val_loss_align.val:.4f} ({val_loss_align.avg:.4f})'
                .format(
            val_loss=val_losses_re, 
            # val_loss_align=val_losses_align
        ))
        log_train = write_log(log_train, log_out, opt_dict)
            
        if(least_val_re > val_losses_re.avg):
            least_val_re = val_losses_re.avg
            log_best_re = f"Saved Least Loss_Re: {least_val_re}"
            log_train = write_log(log_train, log_best_re, opt_dict)
            # if(val_losses_re.avg < 0.65):
            #     print("have saved")
            #     save_image_path = f"/data/blood_dvm/data/result/temp/blood/b3_tabq/05_image_{epochs}_{opt_dict['train_config']['lr_max']}_{least_val_re}.pth"
            #     save_table_path = f"/data/blood_dvm/data/result/temp/blood/b3_tabq/05_table_{epochs}_{opt_dict['train_config']['lr_max']}_{least_val_re}.pth"
            #     # save_imgre_path = f"/data/blood_dvm/data/result/temp/blood/b3_tabq/07_imgre_{epochs}_{opt_dict['train_config']['lr_max']}_{least_val_re}.pth"
            #     save_tabre_path = f"/data/blood_dvm/data/result/temp/blood/b3_tabq/05_tabre_{epochs}_{opt_dict['train_config']['lr_max']}_{least_val_re}.pth"
            #     save_align_path = f"/data/blood_dvm/data/result/temp/blood/b3_tabq/05_align_{epochs}_{opt_dict['train_config']['lr_max']}_{least_val_re}.pth"
            #     save_atten_path = f"/data/blood_dvm/data/result/temp/blood/b3_tabq/05_atten_{epochs}_{opt_dict['train_config']['lr_max']}_{least_val_re}.pth"

            #     torch.save(model_image.state_dict(), save_image_path)
            #     torch.save(model_tabular.state_dict(), save_table_path)
            #     # torch.save(model_re_img.state_dict(), save_imgre_path)
            #     torch.save(model_re.state_dict(), save_tabre_path)
            #     torch.save(model_align.state_dict(), save_align_path)
            #     torch.save(model_atten.state_dict(), save_atten_path)

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

    model_image = load_model_image(opt_dict)
    model_tabular = load_model_tabular(opt_dict)
    model_align = load_model_align(opt_dict)
    model_re = load_reconstruction_model(opt_dict, hidden_size2=opt_dict['model_config']['latent_dim'])
    model_re_img = load_reconstruction_model(opt_dict, hidden_size2=opt_dict['model_config']['latent_dim'])
    # model_atten = load_model_atten(opt_dict)
    # train_img_tab_r_atten(model_image, model_tabular, model_re, model_atten, model_align, train_loader, test_loader, device, opt_dict)


    # model_re_img = load_reconstruction_model(opt_dict, hidden_size2=1280, net_pretrain_path="/data/blood_dvm/data/result/temp/img/re_120_1e-3_0.5603332085797311.pth")
    # model_re_tab = load_reconstruction_model(opt_dict, hidden_size2=128, net_pretrain_path="/data/blood_dvm/data/result/end/encoder_mlp_prototype/re_blood_195_0.7_crossentropy_encoder_mlp_prototype_1e-4_best_model.pth")
    # import ipdb;ipdb.set_trace();
    train_img_tab_r(model_image, model_tabular, model_re, model_re_img, model_align, train_loader, test_loader, device, opt_dict)
    
    # train_img_tab_r_atten(model_image, model_tabular, model_re, model_atten, model_align, train_loader, test_loader, device, opt_dict)

    # train_img_tab_r_epoch(train_loader, test_loader, device, opt_dict)

    import ipdb;ipdb.set_trace();
