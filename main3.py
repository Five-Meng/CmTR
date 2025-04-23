import sys
sys.path.append('/home/admin1/User/mxy/demo/')

from build_dataset3 import img_dataset3
import torch.utils.data
from torchvision import transforms
from train import train
from epoch import train_epoch
from backbone.model import build_model
import random
import numpy as np


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

################################################  Dataset  ####################################################

if opt_dict['model_config']['net_v'] in ['resnet34', 'resnet18', 'resnet50', 'vit16', 'vit16_in21k', 'vit32', 'vit32_in21k', 'efficientnet-b0', 'shufflenet']:
    transform_img_train = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
    ])
    transform_img_test = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
    ])
elif opt_dict['model_config']['net_v'] in ['efficientnet-b1']:
    transform_img_train = transforms.Compose([
        transforms.Resize([240, 240]),
        transforms.ToTensor(),
    ])
    transform_img_test = transforms.Compose([
        transforms.Resize([240, 240]),
        transforms.ToTensor(),
    ])
elif opt_dict['model_config']['net_v'] in ['efficientnet-b2']:
    if opt_dict['dataset_config']['dataname'] == 'blood':
        transform_img_train = transforms.Compose([
            transforms.Resize([260, 260]),
            transforms.ToTensor(),
        ])
        transform_img_test = transforms.Compose([
            transforms.Resize([260, 260]),
            transforms.ToTensor(),
        ])



dataset_train = img_dataset3(opt_dict, 'train', transform_img_train)
dataset_test = img_dataset3(opt_dict, 'test', transform_img_test)

train_loader = torch.utils.data.DataLoader(dataset_train,
                                               batch_size=opt_dict['train_config']['batch_size'],
                                               shuffle=True,
                                               num_workers=4)

test_loader = torch.utils.data.DataLoader(dataset_test,
                                               batch_size=opt_dict['train_config']['batch_size'],
                                               shuffle=False,
                                               num_workers=4)


################################################  Model  ####################################################
model = build_model(opt_dict)

################################################  Train Predict  ####################################################
if opt_dict['train_config']['find_epoch']:
    train_epoch(model, train_loader, test_loader, device, opt_dict)
else:
    print("train")
    train(model, train_loader, test_loader, device, opt_dict)















