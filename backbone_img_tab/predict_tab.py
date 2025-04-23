import sys
sys.path.append('/home/admin1/User/mxy/demo/')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.losses import ReconstructionLoss_MLP, KLLoss
from backbone_tabular_prototype.reconstruction_method import MLP_Embedding_R
from crossattention import CrossAttention
from train_img_tab_re import AlignmentModel
from build_dataset_tabular import tabular_dataset_dvm, tabular_dataset

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



def load_model_tabular(opt_dict):
    from backbone_tabular.TabularEncoder2 import TabularEncoder
    model = TabularEncoder(opt_dict, is_fc=False)
    return model


def load_reconstruction_model(opt_dict, hidden_size2, net_pretrain_path=None): 
    opt_dict['model_config']['hidden_size2'] = hidden_size2
    model = MLP_Embedding_R(opt_dict)  
    return model



def load_model_image(opt_dict):
    from backbone.efficientnet import efficientnet_b1
    model = efficientnet_b1(num_classes=opt_dict['train_config']['num_cls'])
    return model



def load_model_align(opt_dict):
    model = AlignmentModel(opt_dict)
    return model


def load_model_atten(opt_dict):
    model = CrossAttention(opt_dict['model_config']['latent_dim'])
    return model



def img_tab_align(model_align, model_atten, tab_mask, features_image, features_tabular, is_crossatten=True, method='kl'):
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


def load_checkpoint(model, checkpoint_path, strict=True):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))  # 或者使用'cuda'如果你有足够的GPU内存
    model.load_state_dict(checkpoint, strict=strict)
    print(f"Checkpoint {checkpoint_path} loaded successfully.")


def predict_table_r(opt_dict):
    checkpoint_feature_img = opt_dict['predict_config']['load_predict_feature_img']
    checkpoint_re_img = opt_dict['predict_config']['load_predict_re_img']
    checkpoint_feature_tab = opt_dict['predict_config']['load_predict_feature_tab']
    checkpoint_re_tab = opt_dict['predict_config']['load_predict_re_tab']
    checkpoint_align = opt_dict['predict_config']['load_predict_align']
    
    if opt_dict['dataset_config']['dataname'] == 'blood':
        train_dataset = UnitDataset(dataset_dir, mode='train', dataset_type='image_tabular', mask_version=mask_version)
        test_dataset = UnitDataset(dataset_dir, mode='test', dataset_type='image_tabular', mask_version=mask_version)
    else:
        train_dataset = tabular_dataset_dvm(opt_dict, 'train')
        test_dataset = tabular_dataset_dvm(opt_dict, 'test')

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model_image = load_model_image(opt_dict)
    model_tabular = load_model_tabular(opt_dict)
    model_re_tab = load_reconstruction_model(opt_dict, hidden_size2=opt_dict['model_config']['latent_dim'])
    model_re_img = load_reconstruction_model(opt_dict, hidden_size2=opt_dict['model_config']['latent_dim'])
    model_align = load_model_align(opt_dict)

    load_checkpoint(model_image, checkpoint_feature_img)
    load_checkpoint(model_tabular, checkpoint_feature_tab)
    load_checkpoint(model_re_img, checkpoint_re_img)
    load_checkpoint(model_re_tab, checkpoint_re_tab)
    load_checkpoint(model_align, checkpoint_align)

    num_cat, num_con, cat_offsets = model_tabular.num_cat, model_tabular.num_con, model_tabular.cat_offsets.to(device)
    loss_function = ReconstructionLoss_MLP(num_cat, num_con, opt_dict)
    pred_save = pd.DataFrame()

    model_image, model_tabular, model_re_tab, model_re_img, model_align = model_image.to(device), model_tabular.to(device), model_re_tab.to(device), model_re_img.to(device), model_align.to(device)
    model_image.eval()
    model_tabular.eval()
    model_re_tab.eval()
    model_re_img.eval()
    model_align.eval()

    with torch.no_grad():
        # import ipdb;ipdb.set_trace();
        for step, data in enumerate(train_loader):
            images, tables, masks, labels = data
            images, tables, masks, labels = images.to(device), tables.to(device), masks.to(device), labels.to(device)
            batch_size_cur = images.size(0)
            features_image = model_image.forward_sne(images)
            features_tabular = model_tabular(tables, masks, masks)
            features_i, features_t, loss_align = img_tab_align(model_align, None, masks, features_image, features_tabular, False)
            results_img = model_re_img(features_i)
            results = model_re_tab(features_t)

            losses = loss_function(results, tables, masks)
            loss, cat_labels, recon_con = losses
            recon_cat = torch.stack(cat_labels, dim=1)

            mask_cat = masks[:, :num_cat]
            mask_con = masks[:, num_cat:]
            target_cat = (tables[:, :num_cat].long())
            target_con = tables[:, num_cat:]
            pred_cat1 = torch.where(mask_cat == 1, recon_cat, target_cat)
            pred_con1 = torch.where(mask_con == 1, recon_con, target_con)

            pred_cat1 = pd.DataFrame(pred_cat1.cpu().numpy()).reset_index(drop=True)
            pred_con1 = pd.DataFrame(pred_con1.cpu().numpy()).reset_index(drop=True)
            labels1 = pd.DataFrame(labels.cpu().numpy(), columns=['target']).reset_index(drop=True)
            pred1 = pd.concat([pred_cat1, pred_con1, labels1], axis=1)
            pred1 = pred1.reset_index(drop=True)
            pred_save = pd.concat([pred_save, pred1], axis=0).reset_index(drop=True)

        # import ipdb;ipdb.set_trace();
        for step, data in enumerate(test_loader):
            images, tables, masks, labels = data
            images, tables, masks, labels = images.to(device), tables.to(device), masks.to(device), labels.to(device)
            batch_size_cur = images.size(0)
            val_features_image = model_image.forward_sne(images)
            val_features_tabular = model_tabular(tables, masks, masks)

            val_features_i, val_features_t, val_loss_align = img_tab_align(model_align, None, masks, val_features_image, val_features_tabular, False)

            results = model_re_tab(val_features_t)
            results_img = model_re_img(val_features_i)

            losses = loss_function(results, tables, masks)
            loss, cat_labels, recon_con = losses
            recon_cat = torch.stack(cat_labels, dim=1)

            mask_cat = masks[:, :num_cat]
            mask_con = masks[:, num_cat:]
            target_cat = (tables[:, :num_cat].long())
            target_con = tables[:, num_cat:]
            # import ipdb;ipdb.set_trace();
            pred_cat1 = torch.where(mask_cat == 1, recon_cat, target_cat)
            pred_con1 = torch.where(mask_con == 1, recon_con, target_con)

            pred_cat1 = pd.DataFrame(pred_cat1.cpu().numpy()).reset_index(drop=True)
            pred_con1 = pd.DataFrame(pred_con1.cpu().numpy()).reset_index(drop=True)
            labels1 = pd.DataFrame(labels.cpu().numpy(), columns=['target']).reset_index(drop=True)
            pred1 = pd.concat([pred_cat1, pred_con1, labels1], axis=1)
            pred1 = pred1.reset_index(drop=True)
            pred_save = pd.concat([pred_save, pred1], axis=0).reset_index(drop=True)

        # import ipdb;ipdb.set_trace();
        df_total = pd.read_csv(opt_dict['dataset_config']['dataset_tabular'])
        if(opt_dict['dataset_config']['dataname'] == 'blood'):
            column_name = ['WBC(10^9/L)', 'RBC(10^12/L)', 'HGB(g/L)', 'PLT(10^9/L)',
                            'MCV(fL)', 'RDW-CV(%)', 'MPV(fL)', 'P-LCR(%)', 'LYMPH#(10^9/L)',
                            'LYMPH%(%)', 'MONO#(10^9/L)', 'MONO%(%)', 'NEUT%(%)', 'EO%(%)',
                            'BASO%(%)', 'Q-Flag(Blasts/Abn Lympho?)', 'Q-Flag(Blasts?)',
                            'Q-Flag(Abn Lympho?)', 'Q-Flag(Atypical Lympho?)', 'PDW(fL)',
                            '[IG%(%)]', 'NRBC%(%)', '[HFLC%(%)]', '[NE-SSC(ch)]', '[NE-SFL(ch)]',
                            '[NE-FSC(ch)]', '[LY-X(ch)]', '[LY-Y(ch)]', '[LY-Z(ch)]', '[MO-X(ch)]',
                            '[MO-Y(ch)]', '[MO-Z(ch)]', '[NE-WX]', '[NE-WY]', '[NE-WZ]', '[LY-WX]',
                            '[LY-WY]', '[LY-WZ]', '[MO-WX]', '[MO-WY]', '[MO-WZ]', 'target']
        else:
            column_name = ['Color', 'Bodytype', 'Gearbox', 'Fuel_type', 'Adv_year', 'Adv_month',
                            'Reg_year', 'Runned_Miles', 'Price', 'Seat_num', 'Door_num',
                            'Entry_price', 'Engine_size', 'Wheelbase', 'Height', 'Width', 'Length', 'Genmodel_ID']
        pred_save.columns = column_name
        pred_save['train_val'] = df_total['train_val']
        if(opt_dict['dataset_config']['dataname'] == 'blood'):
            pred_save['barcode'] = df_total['barcode']
        else:
            pred_save['Adv_ID'] = df_total['Adv_ID']
        pred_save.to_csv(opt_dict['dataset_config']['save_csv'], index=False)
        print("save csv")



def predict_table_r_atten(opt_dict):
    checkpoint_feature_img = opt_dict['predict_config']['load_predict_feature_img']
    checkpoint_feature_tab = opt_dict['predict_config']['load_predict_feature_tab']
    checkpoint_re_tab = opt_dict['predict_config']['load_predict_re_tab']
    checkpoint_atten = opt_dict['predict_config']['load_predict_atten']
    if 'load_predict_atten' in opt_dict['predict_config']:
        checkpoint_align = opt_dict['predict_config']['load_predict_align']
    
    if opt_dict['dataset_config']['dataname'] == 'blood':
        train_dataset = UnitDataset(dataset_dir, mode='train', dataset_type='image_tabular', mask_version=mask_version)
        test_dataset = UnitDataset(dataset_dir, mode='test', dataset_type='image_tabular', mask_version=mask_version)
    else:
        train_dataset = tabular_dataset_dvm(opt_dict, 'train')
        test_dataset = tabular_dataset_dvm(opt_dict, 'test')

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model_image = load_model_image(opt_dict)
    model_tabular = load_model_tabular(opt_dict)
    model_re_tab = load_reconstruction_model(opt_dict, hidden_size2=opt_dict['model_config']['latent_dim'])
    model_atten = load_model_atten(opt_dict)
    model_align = load_model_align(opt_dict)

    load_checkpoint(model_image, checkpoint_feature_img)
    load_checkpoint(model_tabular, checkpoint_feature_tab)
    load_checkpoint(model_re_tab, checkpoint_re_tab)
    load_checkpoint(model_atten, checkpoint_atten)
    if 'load_predict_align' in opt_dict['predict_config']:
        load_checkpoint(model_align, checkpoint_align)


    model_image, model_tabular, model_re_tab, model_atten, model_align = model_image.to(device), model_tabular.to(device), model_re_tab.to(device), model_atten.to(device), model_align.to(device)
    num_cat, num_con, cat_offsets = model_tabular.num_cat, model_tabular.num_con, model_tabular.cat_offsets.to(device)
    loss_function = ReconstructionLoss_MLP(num_cat, num_con, opt_dict)
    pred_save = pd.DataFrame()

    model_image.eval()
    model_tabular.eval()
    model_re_tab.eval()
    model_atten.eval()
    model_align.eval()

    with torch.no_grad():
        # import ipdb;ipdb.set_trace();
        for step, data in enumerate(train_loader):
            images, tables, masks, labels = data
            images, tables, masks, labels = images.to(device), tables.to(device), masks.to(device), labels.to(device)
            batch_size_cur = images.size(0)
            features_image = model_image.forward_sne(images)
            features_tabular = model_tabular(tables, masks, masks)

            features, loss_align = img_tab_align(model_align, model_atten, masks, features_image, features_tabular, True)
            results = model_re_tab(features)

            losses = loss_function(results, tables, masks)
            loss, cat_labels, recon_con = losses
            recon_cat = torch.stack(cat_labels, dim=1)

            mask_cat = masks[:, :num_cat]
            mask_con = masks[:, num_cat:]
            target_cat = (tables[:, :num_cat].long())
            target_con = tables[:, num_cat:]
            pred_cat1 = torch.where(mask_cat == 1, recon_cat, target_cat)
            pred_con1 = torch.where(mask_con == 1, recon_con, target_con)

            pred_cat1 = pd.DataFrame(pred_cat1.cpu().numpy()).reset_index(drop=True)
            pred_con1 = pd.DataFrame(pred_con1.cpu().numpy()).reset_index(drop=True)
            labels1 = pd.DataFrame(labels.cpu().numpy(), columns=['target']).reset_index(drop=True)
            pred1 = pd.concat([pred_cat1, pred_con1, labels1], axis=1)
            pred1 = pred1.reset_index(drop=True)
            pred_save = pd.concat([pred_save, pred1], axis=0).reset_index(drop=True)

        # import ipdb;ipdb.set_trace();
        for step, data in enumerate(test_loader):
            images, tables, masks, labels = data
            images, tables, masks, labels = images.to(device), tables.to(device), masks.to(device), labels.to(device)
            batch_size_cur = images.size(0)
            features_image = model_image.forward_sne(images)
            features_tabular = model_tabular(tables, masks, masks)

            features, loss_align = img_tab_align(model_align, model_atten, masks, features_image, features_tabular, True)
            results = model_re_tab(features)
            losses = loss_function(results, tables, masks)
            loss, cat_labels, recon_con = losses
            recon_cat = torch.stack(cat_labels, dim=1)

            mask_cat = masks[:, :num_cat]
            mask_con = masks[:, num_cat:]
            target_cat = (tables[:, :num_cat].long())
            target_con = tables[:, num_cat:]
            # import ipdb;ipdb.set_trace();
            pred_cat1 = torch.where(mask_cat == 1, recon_cat, target_cat)
            pred_con1 = torch.where(mask_con == 1, recon_con, target_con)

            pred_cat1 = pd.DataFrame(pred_cat1.cpu().numpy()).reset_index(drop=True)
            pred_con1 = pd.DataFrame(pred_con1.cpu().numpy()).reset_index(drop=True)
            labels1 = pd.DataFrame(labels.cpu().numpy(), columns=['target']).reset_index(drop=True)
            pred1 = pd.concat([pred_cat1, pred_con1, labels1], axis=1)
            pred1 = pred1.reset_index(drop=True)
            pred_save = pd.concat([pred_save, pred1], axis=0).reset_index(drop=True)

        # import ipdb;ipdb.set_trace();
        df_total = pd.read_csv(opt_dict['dataset_config']['dataset_tabular'])
        if(opt_dict['dataset_config']['dataname'] == 'blood'):
            column_name = ['WBC(10^9/L)', 'RBC(10^12/L)', 'HGB(g/L)', 'PLT(10^9/L)',
                            'MCV(fL)', 'RDW-CV(%)', 'MPV(fL)', 'P-LCR(%)', 'LYMPH#(10^9/L)',
                            'LYMPH%(%)', 'MONO#(10^9/L)', 'MONO%(%)', 'NEUT%(%)', 'EO%(%)',
                            'BASO%(%)', 'Q-Flag(Blasts/Abn Lympho?)', 'Q-Flag(Blasts?)',
                            'Q-Flag(Abn Lympho?)', 'Q-Flag(Atypical Lympho?)', 'PDW(fL)',
                            '[IG%(%)]', 'NRBC%(%)', '[HFLC%(%)]', '[NE-SSC(ch)]', '[NE-SFL(ch)]',
                            '[NE-FSC(ch)]', '[LY-X(ch)]', '[LY-Y(ch)]', '[LY-Z(ch)]', '[MO-X(ch)]',
                            '[MO-Y(ch)]', '[MO-Z(ch)]', '[NE-WX]', '[NE-WY]', '[NE-WZ]', '[LY-WX]',
                            '[LY-WY]', '[LY-WZ]', '[MO-WX]', '[MO-WY]', '[MO-WZ]', 'target']
        else:
            column_name = ['Color', 'Bodytype', 'Gearbox', 'Fuel_type', 'Adv_year', 'Adv_month',
                            'Reg_year', 'Runned_Miles', 'Price', 'Seat_num', 'Door_num',
                            'Entry_price', 'Engine_size', 'Wheelbase', 'Height', 'Width', 'Length', 'Genmodel_ID']
        pred_save.columns = column_name
        pred_save['train_val'] = df_total['train_val']
        if(opt_dict['dataset_config']['dataname'] == 'blood'):
            pred_save['barcode'] = df_total['barcode']
        else:
            pred_save['Adv_ID'] = df_total['Adv_ID']
        pred_save.to_csv(opt_dict['dataset_config']['save_csv'], index=False)
        print("save csv")



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
    
    # model_feature = load_feature_model(opt_dict)
    # model_re = load_reconstruction_model(opt_dict)
    # num_cat, num_con = model_feature.num_cat, model_feature.num_con

    predict_table_r_atten(opt_dict)

