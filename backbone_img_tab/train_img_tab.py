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



def get_long_tail_id(opt_dict):
    if opt_dict['dataset_config']['dataname'] == 'blood':
        class_longtail_ptpath = torch.load("/data/blood_dvm/data/blood/blood_longtail_target.pt")
    elif opt_dict['dataset_config']['dataname'] == 'dvm':
        class_longtail_ptpath = torch.load("/data/blood_dvm/data/dvm/dvm_longtail_500_id.pt")
    return class_longtail_ptpath


def create_log(opt_dict):
    log_path = os.path.join("/data/blood_dvm/data/result/temp/conbine/", f"DVM_{opt_dict['dataset_config']['dataname']}_{opt_dict['train_config']['epochs']}_{opt_dict['dataset_config']['missing_rate']}_{opt_dict['model_config']['model_name']}_{opt_dict['model_config']['net']}_{opt_dict['train_config']['lr_max']}_log_train.csv")
    print(log_path)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log_train = open(log_path, 'w')

    return log_train


def write_log(log_train, log_out, opt_dict):
    log_train.write(log_out + '\n')
    log_train.flush()
    return log_train



class ConcatLinearClassifier(nn.Module):
    def __init__(self, dim_fusion, dim_i, dim_t, num_classes):
        super().__init__()
        self.img_proj = nn.Sequential(
            nn.Linear(dim_i, dim_fusion),
            nn.ReLU(),
        )
        self.tab_proj = nn.Sequential(
            nn.Linear(dim_t, dim_fusion),
            nn.ReLU(),
        )
        self.norm_i = nn.LayerNorm(dim_fusion)
        self.norm_t = nn.LayerNorm(dim_fusion)

        self.classifier = nn.Linear(2 * dim_fusion, num_classes)

    def forward(self, x_i, x_t):
        x_i = self.img_proj(x_i)
        x_t = self.tab_proj(x_t)
        x_i = self.norm_i(x_i)
        x_t = self.norm_t(x_t)
        fused = torch.cat([x_i, x_t], dim=1)
        return self.classifier(fused)



def load_fusion_model(opt_dict):
    dim_fusion = opt_dict['model_config']['dim_fusion']
    dim_i = 1280
    dim_t = opt_dict['model_config']['hidden_size3']
    num_classes = opt_dict['train_config']['num_cls']
    model = ConcatLinearClassifier(dim_fusion, dim_i, dim_t, num_classes)
    return model




from sklearn.metrics import f1_score  

def train(model_fusion, train_loader, test_loader, device, opt_dict):
    epochs = opt_dict['train_config']['epochs']
    num_cls = opt_dict['train_config']['num_cls'] 
    class_longtail_ptpath = get_long_tail_id(opt_dict)
    tail_classes = class_longtail_ptpath[0]  
    long_classes = class_longtail_ptpath[1]  
    
    log_train = create_log(opt_dict)

    train_acc = AverageMeter('train_acc', ':.4e')
    val_acc = AverageMeter('val_acc', ':.4e')
    best_val_acc = 0

    model_fusion.to(device)
    loss_function_cls = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model_fusion.parameters(), lr=float(opt_dict['train_config']['lr_max']))
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(epochs):
        # Training phase
        train_acc.reset()
        model_fusion.train()
        
        for step, data in enumerate(train_loader):
            image, tables, labels = data
            image, tables, labels = image.to(device), tables.to(device), labels.to(device)
            
            results = model_fusion(image, tables)
            loss_cls = loss_function_cls(results, labels)
            
            batch_size_cur = tables.size(0)
            train_acc_o = accuracy(results, labels)[0]
            train_acc.update(train_acc_o.item(), batch_size_cur)

            optimizer.zero_grad()
            loss_cls.backward()
            optimizer.step()

            log_out = ('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                       'Train_Acc {acc.val:.4f} ({acc.avg:.4f})\t'.format(
                        epoch, step, len(train_loader), 
                        lr=optimizer.param_groups[-1]['lr'],
                        acc=train_acc))
            log_train = write_log(log_train, log_out, opt_dict)

        schedule.step()

        # Validation phase
        model_fusion.eval()
        val_acc.reset()
        
        correct_each_class = {i:0 for i in range(num_cls)}
        total_each_class = [0] * num_cls
        correct_tail, total_tail = 0, 0
        correct_long, total_long = 0, 0
        y_true = []
        y_pred = []

        with torch.no_grad():
            for step, data in enumerate(test_loader):
                image, tables, labels = data
                image, tables, labels = image.to(device), tables.to(device), labels.to(device)
                
                results = model_fusion(image, tables)
                loss_cls = loss_function_cls(results, labels)
                
                batch_size_cur = tables.size(0)
                val_acc_o = accuracy(results, labels)[0]
                val_acc.update(val_acc_o.item(), batch_size_cur)
                
                preds = results.argmax(dim=1)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())
                
                for true, pred in zip(labels.cpu(), preds.cpu()):
                    cls = true.item()
                    total_each_class[cls] += 1
                    if true == pred:
                        correct_each_class[cls] += 1
                    
                    if cls in long_classes:
                        total_long += 1
                        if true == pred:
                            correct_long += 1
                    elif cls in tail_classes:
                        total_tail += 1
                        if true == pred:
                            correct_tail += 1

        tail_acc = correct_tail / total_tail if total_tail > 0 else 0.0
        long_acc = correct_long / total_long if total_long > 0 else 0.0
        f1 = f1_score(y_true, y_pred, average='macro')
        
        log_out_val = (
            f"Val_Acc: {val_acc.avg:.4f}\t"
            f"Tail_Acc: {tail_acc:.4f}\t"
            f"Long_Acc: {long_acc:.4f}\t"
            f"F1-Score: {f1:.4f}\n"
        )
        log_train = write_log(log_train, log_out_val, opt_dict)

        for cls in range(num_cls):
            acc_cls = correct_each_class[cls] / total_each_class[cls] if total_each_class[cls] > 0 else 0.0
            log_cls = f"Class {cls}: Acc={acc_cls:.4f} ({correct_each_class[cls]}/{total_each_class[cls]})"
            log_train = write_log(log_train, log_cls, opt_dict)

        if val_acc.avg > best_val_acc:
            best_val_acc = val_acc.avg
            log_best = f"Best Model Saved at Acc: {best_val_acc:.4f}, F1: {f1:.4f}"
            log_train = write_log(log_train, log_best, opt_dict)
            

    print("Training Finished!")
    return best_val_acc


def train_epoch(train_loader, test_loader, device, opt_dict):
    max_epoch = opt_dict['train_config']['max_epochs']
    for epoch in range(50, max_epoch, 5):
        print(f"现在是Epoch={epoch}")
        opt_dict['train_config']['epochs'] = epoch
        print(f"opt_dict['train_config']['epoch']:{opt_dict['train_config']['epochs']}")

        model_fusion = load_fusion_model(opt_dict)
        # import ipdb;ipdb.set_trace();
        best_acc = train(model_fusion, train_loader, test_loader, device, opt_dict)
        print(f"best_acc: {best_acc}")

    print('Finished')
    return 



if __name__ == '__main__':
    from torchvision import transforms

    from build_dataset_2tab import tabular_dataset_2tab
    from build_dataset_unit import UnitDataset_blood_2tab
    mask_version = opt_dict['dataset_config']['missing_rate']
    print(f"mask_version : {mask_version}")
    dataset_dir = opt_dict['dataset_config']['data_dir']
    batch_size = opt_dict['train_config']['batch_size']

    from build_dataset_fusion import MultiFeatureDataset
    train_dataset = MultiFeatureDataset(
        data_path="/home/admin1/User/mxy/demo/dvm_extracted_features/train_data07.npy"
    )
    test_dataset = MultiFeatureDataset(
        data_path="/home/admin1/User/mxy/demo/dvm_extracted_features/test_data07.npy"
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    train_epoch(train_loader, test_loader, device, opt_dict)


