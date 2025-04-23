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



def predict_table(opt_dict):
    checkpoint_feature = opt_dict['predict_config']['load_predict_feature']
    checkpoint_re = opt_dict['predict_config']['load_predict_re']
    from build_dataset_unit import UnitDataset
    mask_version = opt_dict['dataset_config']['missing_rate']
    print(f"mask_version : {mask_version}")
    dataset_dir = opt_dict['dataset_config']['data_dir']

    if opt_dict['dataset_config']['dataname'] == 'blood':
        train_dataset = UnitDataset(dataset_dir, mode='train', dataset_type='tabular', mask_version=mask_version)
        test_dataset = UnitDataset(dataset_dir, mode='test', dataset_type='tabular', mask_version=mask_version)
    else:
        train_dataset = tabular_dataset_dvm(opt_dict, 'train')
        test_dataset = tabular_dataset_dvm(opt_dict, 'test')

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model_feature = load_feature_model(opt_dict).to(device)
    model_re = load_reconstruction_model(opt_dict).to(device)
    num_cat, num_con, cat_offsets = model_feature.num_cat, model_feature.num_con, model_feature.cat_offsets.to(device)
    state_dict_feature, state_dict_re = torch.load(checkpoint_feature), torch.load(checkpoint_re)
    # import ipdb;ipdb.set_trace();
    model_feature.load_state_dict(state_dict_feature, strict=True)
    model_re.load_state_dict(state_dict_re, strict=True)

    loss_function = ReconstructionLoss_MLP(num_cat, num_con, opt_dict)
    pred_save = pd.DataFrame()

    model_feature.eval()
    model_re.eval()

    with torch.no_grad():   
        # import ipdb;ipdb.set_trace();
        for tables, labels, masks in train_loader:
            tables, labels, masks = tables.to(device), labels.to(device), masks.to(device)
            features = model_feature(tables, masks, masks)
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
        for tables, labels, masks in test_loader:
            tables, labels, masks = tables.to(device), labels.to(device), masks.to(device)
            features = model_feature(tables, masks, masks)
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

    predict_table(opt_dict)