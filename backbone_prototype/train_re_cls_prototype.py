import sys
sys.path.append('/home/admin1/User/mxy/demo/')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from backbone_tabular_prototype.reconstruction_method import MLP_Embedding_R
from utils.utils import AverageMeter, accuracy
from utils.losses import ReconstructionLoss_MLP
from torch_geometric.nn import GATConv
import torch.nn.functional as F

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



def generate_prototype_dict(features_dict):
    prototype_dict = {}
    for label, features in features_dict.items():
        features_tensor = torch.stack(features)
        median = torch.median(features_tensor, dim=0).values        
        prototype_dict[label] = {
            'mean': median.detach(),
            'count': len(features)
        }
    return prototype_dict



def inital_prototype(train_loader, opt_dict, model_feature, collect_features_dict, prototype_tab):
    with torch.no_grad():
        for step, data in enumerate(train_loader):
            images, tables, masks, labels = data
            masks = torch.ones_like(masks, dtype=torch.bool)
            images, tables, masks, labels = images.to(device), tables.to(device), masks.to(device), labels.to(device)
            batch_size_cur = images.size(0)
            features = model_feature.forward_sne(images)
            for i in range(batch_size_cur):
                label = labels[i].item()  
                feature = features[i]  
                if label not in collect_features_dict:
                    collect_features_dict[label] = []
                
                collect_features_dict[label].append(feature)

    prototype_tab = generate_prototype_dict(collect_features_dict)
    return prototype_tab, collect_features_dict




def refresh_prototype(prototype_tab, collect_features_dict):
    new_prototypes = generate_prototype_dict(collect_features_dict)
    updated_prototype_tab = {}
    
    for label, old_proto in prototype_tab.items():
        current_proto = new_prototypes[label]
        n_samples = old_proto['count']
        similarity = F.cosine_similarity(old_proto['mean'], current_proto['mean'], dim=0)
        # import ipdb;ipdb.set_trace();
        gate = torch.sigmoid(similarity)  
        updated_mean = old_proto['mean'] + gate * (current_proto['mean'] - old_proto['mean'])
        # alpha = 0.1
        # updated_mean = (1 - alpha) * old_proto['mean'] + alpha * current_proto['mean']
        updated_prototype_tab[label] = {
            'mean': updated_mean.detach(),
            'count': n_samples
        }

    return updated_prototype_tab



def collect_features(collect_features_dict, features, labels):
    batch_size_cur = features.shape[0]
    for i in range(batch_size_cur):
        label = labels[i].item()  
        feature = features[i]  
        if label not in collect_features_dict:
            collect_features_dict[label] = []
        
        collect_features_dict[label].append(feature)
    return collect_features_dict




def add_prototype(graph_net, model_fc, prototype_tab, features, features_fc, loss_function_cls, labels, device):
    prototypes = torch.stack([p['mean'].to(device) for p in prototype_tab.values()])
    results_cls = model_fc(features_fc)
    loss_cls = loss_function_cls(results_cls, labels)
    train_acc = accuracy(results_cls, labels)[0]
    
    probs = F.softmax(results_cls, dim=1)
    # import ipdb;ipdb.set_trace();
    fused_features = features.clone()
    for i in range(features.size(0)):
        selected_protos = prototypes
        num_nodes = len(selected_protos) + 1 
        node_features = torch.cat([features[[i]], selected_protos], dim=0)
        
        edge_index = []
        for j in range(1, num_nodes):
            edge_index.append([0, j])  # 特征到原型
            edge_index.append([j, 0])  # 原型到特征
        edge_index = torch.tensor(edge_index, dtype=torch.long, device=device).t()
        # import ipdb;ipdb.set_trace();
        edge_weight = torch.cat([probs[i], probs[i]], dim=0).to(device) 

        refined_proto = graph_net(node_features, edge_index, edge_weight)[0]  # 取中心节点
        similarity = F.cosine_similarity(features[i], refined_proto, dim=0)
        gate = torch.sigmoid(similarity)  
       
        fused_features[i] = features[i] + gate * (refined_proto - features[i])
    
    return fused_features, loss_cls, train_acc



class ProtoGraph(nn.Module):
    def __init__(self, opt_dict):
        super().__init__()
        # feat_dim=128
        feat_dim = opt_dict['model_config']['embedding_dim']
        num_heads = opt_dict['model_config']['num_heads']
        latent_dim = opt_dict['model_config']['latent_dim']
        
        self.gat1 = GATConv(feat_dim, latent_dim, heads=num_heads)
        self.gat2 = GATConv(latent_dim * num_heads, feat_dim, heads=1)
        
    def forward(self, x, edge_index):
        x = F.elu(self.gat1(x, edge_index))
        return self.gat2(x, edge_index)


from torch_geometric.nn import GCNConv
class GCNModel(nn.Module):
    def __init__(self, opt_dict):
        super().__init__()
        feat_dim = opt_dict['model_config']['efficientb1_outdim']
        latent_dim = opt_dict['model_config']['latent_dim']
        
        self.gcn1 = GCNConv(feat_dim, latent_dim)
        self.gcn2 = GCNConv(latent_dim, feat_dim)
    
    def forward(self, x, edge_index, edge_weight):
        x = F.relu(self.gcn1(x, edge_index, edge_weight))
        return self.gcn2(x, edge_index, edge_weight)



def create_log(opt_dict, lambda_re, lambda_match):
    log_path = os.path.join(opt_dict['dataset_config']['it_result_path'], opt_dict['model_config']['net_v'], f"{lambda_re}_{lambda_match}_{opt_dict['dataset_config']['dataname']}_{opt_dict['train_config']['epochs']}_{opt_dict['dataset_config']['missing_rate']}_{opt_dict['model_config']['model_name']}_{opt_dict['model_config']['net_v']}_{opt_dict['train_config']['lr_max']}_log_train.csv")
    print(log_path)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log_train = open(log_path, 'w')

    return log_train


def write_log(log_train, log_out, opt_dict):
    # print(log_out)
    log_train.write(log_out + '\n')
    log_train.flush()
    return log_train



def calculate_nonlinear(epoch, total_epochs, min_lambda_re=0.1, max_lambda_re=0.6, power=2):
    normalized_epoch = epoch / (total_epochs - 1)
    lambda_re = min_lambda_re + (max_lambda_re - min_lambda_re) * (normalized_epoch ** power)
    return lambda_re




def load_model_image(opt_dict):
    from backbone.efficientnet import efficientnet_b1
    model = efficientnet_b1(num_classes=opt_dict['train_config']['num_cls'])
    net_pretrain_path = "/data/blood_dvm/data/result/end/encoder_image/img_re_cls/00_feature_100_0.37886158412475457_97.5478414656267.pth"
    weights_dict = torch.load(net_pretrain_path, map_location='cpu')
    model.load_state_dict(weights_dict, strict=True)
    for name, param in model.named_parameters():
        param.requires_grad = True
    return model



def load_reconstruction_model(opt_dict): 
    if(opt_dict['dataset_config']['dataname'] == 'blood'):
        opt_dict['model_config']['hidden_size2'] = opt_dict['model_config']['efficientb1_outdim']
    else:
        opt_dict['model_config']['hidden_size3'] = opt_dict['model_config']['efficientb1_outdim']
    model = MLP_Embedding_R(opt_dict)
    net_pretrain_path = "/data/blood_dvm/data/result/end/encoder_image/img_re_cls/00_re_100_0.37886158412475457_97.5478414656267.pth"
    weights_dict = torch.load(net_pretrain_path, map_location='cpu')
    model.load_state_dict(weights_dict, strict=True)
    for name, param in model.named_parameters():
        param.requires_grad = True
    return model



def load_graph_model(opt_dict):
    # model = ProtoGraph(opt_dict)
    model = GCNModel(opt_dict)
    net_pretrain_path = "/data/blood_dvm/data/result/temp/img_re_cls_prototype/00_graph_100_0.36349486799783287_97.12497732631961.pth"
    weights_dict = torch.load(net_pretrain_path, map_location='cpu')
    model.load_state_dict(weights_dict, strict=True)
    for name, param in model.named_parameters():
        param.requires_grad = True
    return model


def load_fc_model(opt_dict):
    input_size = opt_dict['model_config']['efficientb1_outdim']
    model_fc3 = nn.Linear(input_size, opt_dict['train_config']['num_cls'])
    net_pretrain_path = "/data/blood_dvm/data/result/end/encoder_image/img_re_cls/00_fc_100_0.37886158412475457_97.5478414656267.pth"
    weights_dict = torch.load(net_pretrain_path, map_location='cpu')
    model_fc3.load_state_dict(weights_dict, strict=True)
    for name, param in model_fc3.named_parameters():
        param.requires_grad = True
    return model_fc3



def train_re_cls_prototype(model_feature, model_re, model_fc, model_graph, train_loader, test_loader, device, opt_dict, lambda_re, lambda_cls):
    if opt_dict['dataset_config']['dataname'] == 'blood':
        num_cat, num_con = 19, 22
    else:
        num_cat, num_con = 4, 13
    log_train = create_log(opt_dict, lambda_re, lambda_cls)

    train_acc_v = AverageMeter('Train_Acc@1', ':6.2f')
    val_acc_v = AverageMeter('Val_Acc@1', ':6.2f')
    train_losses = AverageMeter('Train_Loss', ':.4e')
    val_losses = AverageMeter('Val_loss', ':.4e')
    train_losses_re = AverageMeter('Val_loss', ':.4e')
    train_losses_cls = AverageMeter('Val_loss', ':.4e')
    val_losses_re = AverageMeter('Val_loss', ':.4e')
    val_losses_cls = AverageMeter('Val_loss', ':.4e')
    less_total_losses = 1e9
    update_re_losses = 0
    update_cls_losses = 0
    update_cls_acc = 0

    model_feature.to(device)
    model_re.to(device)
    model_fc.to(device)
    model_graph.to(device)

    loss_function_re = ReconstructionLoss_MLP(num_cat, num_con, opt_dict)
    loss_function_cls = nn.CrossEntropyLoss()

    pg_feature = [p for p in model_feature.parameters() if p.requires_grad]
    pg_re = [p for p in model_re.parameters() if p.requires_grad]
    pg_fc = [p for p in model_fc.parameters() if p.requires_grad]
    pg_graph = [p for p in model_graph.parameters() if p.requires_grad]
    pg = pg_feature + pg_re + pg_fc + pg_graph
    optimizer = optim.Adam(pg, lr=float(opt_dict['train_config']['lr_max']))

    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt_dict['train_config']['epochs'])
    epochs = opt_dict['train_config']['epochs']

    train_all_losses = []
    val_all_losses = []

    prototype_tab = {}
    collect_features_dict = {}

    for epoch in range(epochs):

        if((len(collect_features_dict) == 0) and (len(prototype_tab) == 0)):
            print("collecting")
            prototype_tab, collect_features_dict = inital_prototype(train_loader, opt_dict, model_feature, collect_features_dict, prototype_tab)
            print("collected")
            # import ipdb;ipdb.set_trace();
            collect_features_dict = {}
        elif ((len(collect_features_dict) > 0) and (len(prototype_tab) > 0)):
            if(epoch % 2 == 0):
                print("refreshing")
                prototype_tab = refresh_prototype(prototype_tab, collect_features_dict)
                print("refreshed")
            else:
                print(f"Epoch: {epoch}, no need to refresh")
            # if(epoch == 40 or epoch == 20 or epoch == 80):
            #     t_sne_prototype(prototype_tab, collect_features_dict, opt_dict)
            collect_features_dict = {}
            # import ipdb;ipdb.set_trace();
        else:
            print("========================= None =========================")

        train_losses.reset()
        train_losses_re.reset()
        train_losses_cls.reset()
        train_acc_v.reset()

        model_feature.train()
        model_re.train()
        model_fc.train()
        model_graph.train()

        for step, data in enumerate(train_loader):
            # import ipdb;ipdb.set_trace();
            images, tables, masks, labels = data
            masks = torch.ones_like(masks, dtype=torch.bool)
            images, tables, masks, labels = images.to(device), tables.to(device), masks.to(device), labels.to(device)
            batch_size_cur = images.size(0)

            features = model_feature.forward_sne(images)
            features_fc = features
            collect_features_dict = collect_features(collect_features_dict, features, labels)
            features_prototype, loss_cls, train_acc = add_prototype(model_graph, model_fc, prototype_tab, features, features_fc, loss_function_cls, labels, device)
            results = model_re(features_prototype)
            loss = loss_function_re(results, tables, masks)
            loss_re = loss[0]
            loss_total = loss_cls * lambda_cls + loss_re * lambda_re

            train_losses.update(loss_total.item(), batch_size_cur)
            train_losses_re.update(loss_re.item(), batch_size_cur)
            train_losses_cls.update(loss_cls.item(), batch_size_cur)

            train_acc_v.update(train_acc.item(), batch_size_cur)

            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()

            log_out = ('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                       'Lambda_re: {lambda_re:.4f}\t'
                       'Lambda_cls: {lambda_cls:.4f}\t'
                       'Train_Loss_Total {loss.val:.4f} ({loss.avg:.4f})\t'
                       'Train_Loss_Re {loss_re.val:.4f} ({loss_re.avg:.4f})\t'
                       'Train_Loss_Cls {loss_cls.val:.4f} ({loss_cls.avg:.4f})\t'
                       'Train Acc@1 {train_acc_v.val:.4f}({train_acc_v.avg:.4f})\t'.format(
                        epoch, step, len(train_loader), lr=optimizer.param_groups[-1]['lr'],
                        lambda_re=lambda_re, lambda_cls=lambda_cls,
                        loss=train_losses, loss_re=train_losses_re, loss_cls=train_losses_cls,
                        train_acc_v=train_acc_v
                       ))
            log_train = write_log(log_train, log_out, opt_dict)
        train_all_losses.append(train_losses.avg)
        schedule.step()

        model_feature.eval()
        model_fc.eval()
        model_re.eval()
        model_graph.eval()
        val_losses.reset()
        val_losses_re.reset()
        val_losses_cls.reset()
        val_acc_v.reset()

        with torch.no_grad():
            for step, data in enumerate(test_loader):
                # import ipdb;ipdb.set_trace();
                images, tables, masks, labels = data
                masks = torch.ones_like(masks, dtype=torch.bool)
                images, tables, masks, labels = images.to(device), tables.to(device), masks.to(device), labels.to(device)
                batch_size_cur = images.size(0)

                val_features = model_feature.forward_sne(images)
                val_features_fc = val_features
                collect_features_dict = collect_features(collect_features_dict, val_features, labels)
                val_features_prototype, val_loss_cls, val_acc = add_prototype(model_graph, model_fc, prototype_tab, val_features, val_features_fc, loss_function_cls, labels, device)
                val_results = model_re(val_features_prototype)
                val_loss = loss_function_re(val_results, tables, masks)
                val_loss_re = val_loss[0]
                val_loss_total = val_loss_cls * lambda_cls + loss_re * val_loss_re

                val_losses.update(val_loss_total.item(), tables.size(0))
                val_losses_re.update(val_loss_re.item(), tables.size(0))
                val_losses_cls.update(val_loss_cls.item(), tables.size(0))

                val_acc_v.update(val_acc.item(), tables.size(0))

            log_out_val = ('Val_Loss_total {loss.val:.4f} ({loss.avg:.4f})\t'
                           'Val_Loss_Re {loss_re.val:.4f} ({loss_re.avg:.4f})\t'
                           'Val_Loss_Cls {loss_cls.val:.4f} ({loss_cls.avg:.4f})\t'
                           'Val_Acc {val_acc_v.val:.4f}({val_acc_v.avg:.4f})'.format(
                               loss=val_losses, loss_re=val_losses_re, loss_cls=val_losses_cls, val_acc_v=val_acc_v)
                           )
            val_losses_plus = val_losses_re.avg + val_losses_cls.avg
            log_train = write_log(log_train, log_out_val, opt_dict)
            val_all_losses.append(val_losses.avg)
            if(less_total_losses > val_losses_plus):
                less_total_losses = val_losses_plus
                update_re_losses = val_losses_re.avg
                update_cls_losses = val_losses_cls.avg
                update_cls_acc = val_acc_v.avg
                log_out_best = f"less_total_losses:{less_total_losses} \n update_re_losses:{update_re_losses} \n update_cls_losses:{update_cls_losses} \n update_cls_acc:{update_cls_acc}"
                log_train = write_log(log_train, log_out_best, opt_dict)

            save_feature_path = f"/data/blood_dvm/data/result/temp/img_re_cls_prototype/00_feature_{epochs}_{val_losses_re.avg}_{val_acc_v.avg}.pth"
            save_re_path = f"/data/blood_dvm/data/result/temp/img_re_cls_prototype/00_re_{epochs}_{val_losses_re.avg}_{val_acc_v.avg}.pth"
            save_fc_path = f"/data/blood_dvm/data/result/temp/img_re_cls_prototype/00_fc_{epochs}_{val_losses_re.avg}_{val_acc_v.avg}.pth"
            save_graph_path = f"/data/blood_dvm/data/result/temp/img_re_cls_prototype/00_graph_{epochs}_{val_losses_re.avg}_{val_acc_v.avg}.pth"

            torch.save(model_feature.state_dict(), save_feature_path)
            torch.save(model_re.state_dict(), save_re_path)
            torch.save(model_fc.state_dict(), save_fc_path)
            torch.save(model_graph.state_dict(), save_graph_path)


    print("Finish!")
    # return less_total_losses, update_re_losses, update_cls_losses, update_cls_acc



def train_img_tab_r_epoch(train_loader, test_loader, device, opt_dict):
    least_val_re_list = []
    max_epochs = opt_dict['train_config']['max_epochs']
    for epochs in range(50, max_epochs, 5):
        opt_dict['train_config']['epochs'] = epochs
        model_feature = load_model_image(opt_dict)
        model_re = load_reconstruction_model(opt_dict)
        model_fc = load_fc_model(opt_dict)
        model_graph = load_graph_model(opt_dict)
        print(f"Epoch: {opt_dict['train_config']['epochs']}")
        lambda_re, lambda_cls = 0.9, 0.1
        train_re_cls_prototype(model_feature, model_re, model_fc, model_graph, train_loader, test_loader, device, opt_dict, lambda_re, lambda_cls)




if __name__ == '__main__':
    from build_dataset_unit import UnitDataset, UnitDataset_dvm
    from torchvision import transforms
    mask_version = opt_dict['dataset_config']['missing_rate']
    print(f"mask_version : {mask_version}")
    dataset_dir = opt_dict['dataset_config']['data_dir']
    batch_size = opt_dict['train_config']['batch_size']
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    print(torch.cuda.is_available())

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

    if opt_dict['dataset_config']['dataname'] == 'blood':
        train_dataset = UnitDataset(dataset_dir, mode='train', dataset_type='image_tabular', mask_version=mask_version)
        test_dataset = UnitDataset(dataset_dir, mode='test', dataset_type='image_tabular', mask_version=mask_version)
    else:
        train_dataset = UnitDataset_dvm(opt_dict, mode='train', dataset_type='image_tabular', transform=transform_img_train)
        test_dataset = UnitDataset_dvm(opt_dict, mode='test', dataset_type='image_tabular', transform=transform_img_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    model_feature = load_model_image(opt_dict)
    model_re = load_reconstruction_model(opt_dict)
    model_fc = load_fc_model(opt_dict)
    model_graph = load_graph_model(opt_dict)
    
    # 0.3_0.7
    # 0.2, 0.8
    # 0.1, 0.9
    # 0.95, 0.05
    lambda_re, lambda_cls = 0.7, 0.3
    train_re_cls_prototype(model_feature, model_re, model_fc, model_graph, train_loader, test_loader, device, opt_dict, lambda_re, lambda_cls)
    
    # train_img_tab_r_epoch(train_loader, test_loader, device, opt_dict)
