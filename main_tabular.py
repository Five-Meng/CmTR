# import sys
# sys.path.append('/home/admin1/User/mxy/demo/')

from train_tabular import train
from epoch_tabular import train_epoch
from backbone_tabular.model_tabular import build_model
from build_dataset_tabular import tabular_dataset, tabular_dataset_dvm
from build_dataset_unit import UnitDataset
import torch
import random
import numpy as np
from torch.utils.data import DataLoader

from argparses.util.yaml_args import yaml_data
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--yaml_config', type=str, help='config args')
args = parser.parse_args()
opt_dict = yaml_data(args.yaml_config)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
seed = opt_dict['train_config']['seed']
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# from prototype.method import get_tab_prototype

import warnings
warnings.filterwarnings("ignore")


##########################################################   Model   ###########################################################
model = build_model(opt_dict)

##########################################################   Dataset   #########################################################
mask_version = opt_dict['dataset_config']['missing_rate']
print(f"mask_version : {mask_version}")
# dataset_dir = opt_dict['dataset_config']['data_dir']

if opt_dict['dataset_config']['dataname'] == 'blood':
    # import ipdb;ipdb.set_trace();
    train_dataset = tabular_dataset(opt_dict, 'train')
    test_dataset = tabular_dataset(opt_dict, 'test')
    # train_dataset = UnitDataset(dataset_dir, mode='train', dataset_type='tabular', mask_version=mask_version)
    # test_dataset = UnitDataset(dataset_dir, mode='test', dataset_type='tabular', mask_version=mask_version)

elif opt_dict['dataset_config']['dataname'] == 'dvm':
    train_dataset = tabular_dataset_dvm(opt_dict, 'train')
    test_dataset = tabular_dataset_dvm(opt_dict, 'test')
    # train_dataset = UnitDataset(dataset_dir, mode='train', dataset_type='tabular', mask_version=mask_version)
    # test_dataset = UnitDataset(dataset_dir, mode='test', dataset_type='tabular', mask_version=mask_version)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)


##########################################################   Train   ###########################################################
if opt_dict['train_config']['find_epoch']:
    # import ipdb;ipdb.set_trace();
    print("find_epoch")
    train_epoch(model, train_loader, test_loader, device, opt_dict)
else:
    # import ipdb;ipdb.set_trace();
    train(model, train_loader, test_loader, device, opt_dict)


      







