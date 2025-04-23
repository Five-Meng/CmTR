import sys
sys.path.append('/home/admin1/User/mxy/demo/')

from build_dataset import img_dataset, img_dataset_H2T, img_dataset_dvm, img_dataset_H2T_dvm
from build_dataset3 import img_dataset3
from build_dataset_unit import UnitDataset
import torch.utils.data
from torchvision import transforms
from train import train, train_H2T
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
print(device)
seed = opt_dict['train_config']['seed']
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

################################################  Dataset  ####################################################

if opt_dict['model_config']['net_v'] in ['resnet34', 'resnet18', 'resnet50', 'vit16', 'vit16_in21k', 'vit32', 'vit32_in21k', 'efficientnet-b0', 'shufflenet']:
    if opt_dict['dataset_config']['dataname'] == 'blood':
        transform_img_train = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
        ])
        transform_img_test = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
        ])
    elif opt_dict['dataset_config']['dataname'] == 'dvm':
        print("================= dvm_transform =================")
        transform_img_train = transforms.Compose([
            transforms.RandomResizedCrop(size=224, scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8)], p=0.8),     
            transforms.RandomGrayscale(p=0.2),     
            transforms.ToTensor(),
        ])
        transform_img_test = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
        ])

elif opt_dict['model_config']['net_v'] in ['vit_new']:
    if opt_dict['dataset_config']['dataname'] == 'blood':
        transform_img_train = transforms.Compose([
            transforms.Resize([384, 384]),
            transforms.ToTensor(),
        ])
        transform_img_test = transforms.Compose([
            transforms.Resize([384, 384]),
            transforms.ToTensor(),
        ])
    elif opt_dict['dataset_config']['dataname'] == 'dvm':
        print("================= dvm_transform =================")
        transform_img_train = transforms.Compose([
            transforms.RandomResizedCrop(size=384, scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8)], p=0.8),     
            transforms.RandomGrayscale(p=0.2),     
            transforms.ToTensor(),
        ])
        transform_img_test = transforms.Compose([
            transforms.Resize([384, 384]),
            transforms.ToTensor(),
        ])


elif opt_dict['model_config']['net_v'] in ['efficientnet-b1']:
    if opt_dict['dataset_config']['dataname'] == 'blood':
        transform_img_train = transforms.Compose([
            transforms.Resize([240, 240]),
            transforms.ToTensor(),
        ])
        transform_img_test = transforms.Compose([
            transforms.Resize([240, 240]),
            transforms.ToTensor(),
        ])
    elif opt_dict['dataset_config']['dataname'] == 'dvm':
        print("================= dvm_transform =================")
        transform_img_train = transforms.Compose([
            transforms.RandomResizedCrop(size=240, scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8)], p=0.8),     
            transforms.RandomGrayscale(p=0.2),     
            # transforms.Lambda(lambda img: img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.1, 2.0))) if random.random() < 0.5 else img),
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
    elif opt_dict['dataset_config']['dataname'] == 'dvm':
        print("================= dvm_transform =================")
        transform_img_train = transforms.Compose([
            transforms.RandomResizedCrop(size=260, scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8)], p=0.8),     
            transforms.RandomGrayscale(p=0.2),     
            transforms.ToTensor(),
        ])
        transform_img_test = transforms.Compose([
            transforms.Resize([260, 260]),
            transforms.ToTensor(),
        ])
elif opt_dict['model_config']['net_v'] in ['efficientnet-b3']:
    transform_img_train = transforms.Compose([
        transforms.Resize([300, 300]),
        transforms.ToTensor(),
    ])
    transform_img_test = transforms.Compose([
        transforms.Resize([300, 300]),
        transforms.ToTensor(),
    ])
elif opt_dict['model_config']['net_v'] == 'efficientnetv2_s':
    transform_img_train = transforms.Compose([
        transforms.Resize([300, 300]),
        transforms.ToTensor(),
    ])
    transform_img_test = transforms.Compose([
        transforms.Resize([384, 384]),
        transforms.ToTensor(),
    ])

elif opt_dict['model_config']['net_v'] == 'efficientnetv2_m':
    transform_img_train = transforms.Compose([
        transforms.Resize([384, 384]),
        transforms.ToTensor(),
    ])
    transform_img_test = transforms.Compose([
        transforms.Resize([480, 480]),
        transforms.ToTensor(),
    ])

# dataset_dir = opt_dict['dataset_config']['data_dir']

# H2T
if opt_dict['H2T']['h2t']:
    if opt_dict['dataset_config']['dataname'] == 'blood':
        print("============== H2T_img_dataset_blood =================")
        dataset_train = img_dataset_H2T(opt_dict, 'train', transform_img_train)
        dataset_test = img_dataset(opt_dict, 'test', transform_img_test)
    elif opt_dict['dataset_config']['dataname'] == 'dvm':
        print("============== H2T_img_dataset_dvm =================")
        dataset_train = img_dataset_H2T_dvm(opt_dict, 'train', transform_img_train)
        dataset_test = img_dataset_dvm(opt_dict, 'test', transform_img_test)
# 正常的
else:
    if opt_dict['dataset_config']['dataname'] == 'blood':
        print("============== img_dataset_blood =================")
        dataset_train = img_dataset(opt_dict, 'train', transform_img_train)
        dataset_test = img_dataset(opt_dict, 'test', transform_img_test)
        # dataset_train = UnitDataset(dataset_dir, mode='train', dataset_type='image', mask_version=None)
        # dataset_test = UnitDataset(dataset_dir, mode='test', dataset_type='image', mask_version=None)
    elif opt_dict['dataset_config']['dataname'] == 'dvm':
        print("============== img_dataset_dvm =================")
        dataset_train = img_dataset_dvm(opt_dict, 'train', transform_img_train)
        dataset_test = img_dataset_dvm(opt_dict, 'test', transform_img_test)
        # dataset_train = UnitDataset(dataset_dir, mode='train', dataset_type='image', mask_version=None)
        # dataset_test = UnitDataset(dataset_dir, mode='test', dataset_type='image', mask_version=None)


train_loader = torch.utils.data.DataLoader(dataset_train,
                                               batch_size=opt_dict['train_config']['batch_size'],
                                               shuffle=True,
                                               num_workers=4)

test_loader = torch.utils.data.DataLoader(dataset_test,
                                               batch_size=opt_dict['train_config']['batch_size'],
                                               shuffle=False,
                                               num_workers=4)


################################################  Model  ####################################################
if opt_dict['H2T']['h2t']:
    print("============== H2T model start =================")
    model = build_model(opt_dict)
else:
    model = build_model(opt_dict)

################################################  Train Predict  ####################################################
if opt_dict['train_config']['find_epoch']:
    train_epoch(model, train_loader, test_loader, device, opt_dict)
else:
    if opt_dict['H2T']['h2t']:
        train_H2T(model, train_loader, test_loader, device, opt_dict)
    else:
        print("train")
        train(model, train_loader, test_loader, device, opt_dict)















