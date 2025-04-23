import sys
sys.path.append('/home/admin1/User/mxy/demo/')

from backbone_tabular.TabularEncoder import TabularTransformerEncoder
from backbone_tabular.TabularEncoder2 import TabularEncoder
from utils.losses import ReconstructionLoss, ReconstructionLoss_MLP
from utils.utils import AverageMeter, accuracy
from backbone_tabular_prototype.reconstruction_method import MLP_Embedding_R
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


def load_feature_model(opt_dict):
    from backbone.efficientnet import efficientnet_b1
    model = efficientnet_b1(num_classes=opt_dict['train_config']['num_cls'])
    return model


def load_reconstruction_model(opt_dict): 
    if(opt_dict['dataset_config']['dataname'] == 'blood'):
        opt_dict['model_config']['hidden_size2'] = opt_dict['model_config']['efficientb1_outdim']
    else:
        opt_dict['model_config']['hidden_size3'] = opt_dict['model_config']['efficientb1_outdim']
    model = MLP_Embedding_R(opt_dict)
    return model



def predict_table(opt_dict):
    from torchvision import transforms
    transform_img_train = transforms.Compose([
        transforms.RandomResizedCrop(size=240, scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8)], p=0.8),     
        transforms.RandomGrayscale(p=0.2),     
        transforms.Resize([240, 240]),
        transforms.ToTensor(),
    ])
    transform_img_test = transforms.Compose([
        transforms.Resize([240, 240]),
        transforms.ToTensor(),
    ])
    checkpoint_feature = opt_dict['predict_config']['load_predict_feature']
    checkpoint_re = opt_dict['predict_config']['load_predict_re']
    print(checkpoint_feature)
    print(checkpoint_re)
    from build_dataset_unit import UnitDataset, UnitDataset_dvm
    mask_version = opt_dict['dataset_config']['missing_rate']
    print(f"mask_version : {mask_version}")
    dataset_dir = opt_dict['dataset_config']['data_dir']

    if opt_dict['dataset_config']['dataname'] == 'blood':
        train_dataset = UnitDataset(dataset_dir, mode='train', dataset_type='image_tabular', mask_version=mask_version)
        test_dataset = UnitDataset(dataset_dir, mode='test', dataset_type='image_tabular', mask_version=mask_version)
    else:
        train_dataset = UnitDataset_dvm(opt_dict, mode='train', dataset_type='image_tabular', transform=transform_img_train)
        test_dataset = UnitDataset_dvm(opt_dict, mode='test', dataset_type='image_tabular', transform=transform_img_test)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model_feature = load_feature_model(opt_dict).to(device)
    model_re = load_reconstruction_model(opt_dict).to(device)
    state_dict_feature, state_dict_re = torch.load(checkpoint_feature), torch.load(checkpoint_re)
    # import ipdb;ipdb.set_trace();
    model_feature.load_state_dict(state_dict_feature, strict=True)
    model_re.load_state_dict(state_dict_re, strict=True)

    if opt_dict['dataset_config']['dataname'] == 'blood':
        num_cat, num_con = 19, 22
    else:
        num_cat, num_con = 4, 13
    loss_function = ReconstructionLoss_MLP(num_cat, num_con, opt_dict)
    pred_save = pd.DataFrame()

    model_feature.eval()
    model_re.eval()
    # import ipdb;ipdb.set_trace();
    with torch.no_grad():   
        # import ipdb;ipdb.set_trace();
        for step, data in enumerate(train_loader):
            # import ipdb;ipdb.set_trace();
            images, tables, masks, labels = data
            images, tables, masks, labels = images.to(device), tables.to(device), masks.to(device), labels.to(device)
            masks = torch.ones_like(masks, dtype=torch.bool)
            features = model_feature.forward_sne(images)
            results = model_re(features)
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
            # import ipdb;ipdb.set_trace();
            images, tables, masks, labels = data
            images, tables, masks, labels = images.to(device), tables.to(device), masks.to(device), labels.to(device)
            masks = torch.ones_like(masks, dtype=torch.bool)
            features = model_feature.forward_sne(images)
            results = model_re(features)
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

        import ipdb;ipdb.set_trace();
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
        df_total1 = df_total[df_total['train_val'] != 'v'].reset_index(drop=True)
        df_total = df_total1.reset_index(drop=True)
        pred_save = pred_save.reset_index(drop=True)
        pred_save['train_val'] = df_total['train_val']
        if(opt_dict['dataset_config']['dataname'] == 'blood'):
            pred_save['barcode'] = df_total['barcode']
        else:
            pred_save['Adv_ID'] = df_total['Adv_ID']
        import ipdb;ipdb.set_trace();
        pred_save.to_csv(opt_dict['dataset_config']['save_csv'], index=False)
        print("save csv")


if __name__ == '__main__':

    predict_table(opt_dict)