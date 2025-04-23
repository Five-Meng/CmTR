import sys
sys.path.append('/home/admin1/User/mxy/demo/')

import torch
import torch.nn as nn
from utils.losses import ReconstructionLoss_MLP, KLLoss
from backbone_tabular.TabularEncoder2 import TabularEncoder
from crossattention import CrossAttention
from torchvision import transforms
from torch.utils.data import DataLoader

from argparses.util.yaml_args import yaml_data
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--yaml_config', type=str, help='config args')
args = parser.parse_args()
opt_dict = yaml_data(args.yaml_config)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_model_image(opt_dict):
    from backbone.efficientnet import efficientnet_b1
    model = efficientnet_b1(num_classes=opt_dict['train_config']['num_cls'])
    # net_pretrain_path = "/data/blood_dvm/data/result/end/encoder_image/classification_results/efficientnet-b1/h2t02_blood_135_crossentropy_efficientnet-b1_5e-5_best_model.pth"
    net_pretrain_path = "/data/blood_dvm/data/result/end/encoder_image/classification_results/efficientnet-b1/dvm_300_crossentropy_efficientnet-b1_3e-3_best_model.pth"
    weights_dict = torch.load(net_pretrain_path, map_location='cpu')
    model.load_state_dict(weights_dict, strict=True)
    for name, param in model.named_parameters():
        param.requires_grad = False
    return model



# def load_atten_model(opt_dict):
#     model = CrossAttention(opt_dict['model_config']['embedding_dim'])
#     net_pretrain_path = "/data/blood_dvm/data/result/end/encoder_tabular/blood/branch/attn_0.3_0.7_0.5_100_5e-5_84.11458333333333.pth"
#     weights_dict = torch.load(net_pretrain_path, map_location='cpu')
#     model.load_state_dict(weights_dict, strict=True)
#     for name, param in model.named_parameters():
#         param.requires_grad = False
#     return model


def load_feature_model(opt_dict, net_pretrain_path):
    model = TabularEncoder(opt_dict, is_fc=False)
    weights_dict = torch.load(net_pretrain_path, map_location='cpu')
    model.load_state_dict(weights_dict, strict=True)
    for name, param in model.named_parameters():
        param.requires_grad = False
    return model



# def img_tab_align(img_proj, tab_proj, model_atten):
#     loss_function = KLLoss()
#     loss_align = loss_function(tab_proj, img_proj)    # (input, target)
#     features = model_atten(img_proj, tab_proj, tab_proj)
#     # features = model_atten(tab_proj, img_proj, img_proj)
#     return features, loss_align





model_image = load_model_image(opt_dict)
# net_pretrain = "/data/blood_dvm/data/result/end/encoder_tabular/blood/branch/feature_0.3_0.7_0.5_180_5e-5_84.24479166666667.pth"
# model_img = load_feature_model(opt_dict, net_pretrain)
net_pretrain = "/data/blood_dvm/data/result/end/encoder_tabular/dvm/re_cls_prototype/03_feature_100_5e-5_0.7530563086211822_69.85080718302194.pth"
model_tab = load_feature_model(opt_dict, net_pretrain)
# model_attn = load_atten_model(opt_dict)


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



from build_dataset_unit import UnitDataset_dvm
train_dataset = UnitDataset_dvm(opt_dict, 'train', dataset_type='image_tabular', transform=transform_img_train)
test_dataset = UnitDataset_dvm(opt_dict, 'test', dataset_type='image_tabular', transform=transform_img_test)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)



import numpy as np
import os
from tqdm import tqdm
import torch


def save_features_to_single_npy(opt_dict, model_image, model_tab, train_loader, test_loader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    save_dir = os.path.join("/home/admin1/User/mxy/demo/", "dvm_extracted_features")
    os.makedirs(save_dir, exist_ok=True)
    
    save_paths = {
        'train': os.path.join(save_dir, "train_data07.npy"),
        'test': os.path.join(save_dir, "test_data07.npy")
    }

    train_data = {
        'features_i': [],
        'features_t': [],
        'labels': []
    }
    
    test_data = {
        'features_i': [],
        'features_t': [],
        'labels': []
    }
    model_image, model_tab = model_image.to(device), model_tab.to(device)
    model_image.eval()
    model_tab.eval()
    import ipdb;ipdb.set_trace();
    with torch.no_grad():
        for step, data in tqdm(enumerate(train_loader), desc="Processing Train Set"):
            image, tables_t, masks, labels = data
            image = image.to(device)
            tables_t = tables_t.to(device)
            masks = torch.zeros_like(masks, dtype=torch.bool).to(device)
            
            features_i = model_image.forward_sne(image)
            features_t = model_tab(tables_t, masks, masks)
            features_t = features_t.cpu().numpy()
            features_i = features_i.cpu().numpy()
            train_data['features_i'].append(features_i)
            train_data['features_t'].append(features_t)
            train_data['labels'].append(labels.cpu().numpy())

        print("train")
        import ipdb;ipdb.set_trace();

        for step, data in tqdm(enumerate(test_loader), desc="Processing Test Set"):
            image, tables_t, masks, labels = data
            image = image.to(device)
            tables_t = tables_t.to(device)
            masks = torch.zeros_like(masks, dtype=torch.bool).to(device)
            
            features_i = model_image.forward_sne(image)
            features_t = model_tab(tables_t, masks, masks)
            features_t = features_t.cpu().numpy()
            features_i = features_i.cpu().numpy()
            test_data['features_i'].append(features_i)
            test_data['features_t'].append(features_t)
            test_data['labels'].append(labels.cpu().numpy())


    print("test")
    import ipdb;ipdb.set_trace();
    train_data['features_i'] = np.concatenate(train_data['features_i'], axis=0)
    train_data['features_t'] = np.concatenate(train_data['features_t'], axis=0)
    train_data['labels'] = np.concatenate(train_data['labels'], axis=0)
    
    np.save(save_paths['train'], train_data)

    test_data['features_i'] = np.concatenate(test_data['features_i'], axis=0)
    test_data['features_t'] = np.concatenate(test_data['features_t'], axis=0)
    test_data['labels'] = np.concatenate(test_data['labels'], axis=0)
    
    np.save(save_paths['test'], test_data)

    print(f"训练数据已保存到: {save_paths['train']}")
    print(f"特征维度 - features_i: {train_data['features_i'].shape}, features_t: {train_data['features_t'].shape}")
    print(f"标签数量: {train_data['labels'].shape[0]}")
    
    print(f"\n测试数据已保存到: {save_paths['test']}")
    print(f"特征维度 - features_i: {test_data['features_i'].shape}, features_t: {test_data['features_t'].shape}")
    print(f"标签数量: {test_data['labels'].shape[0]}")




save_features_to_single_npy(opt_dict, model_image, model_tab, train_loader, test_loader)

