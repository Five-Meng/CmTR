import sys
sys.path.append('/home/admin1/User/mxy/demo/')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import torch.nn.functional as F
from backbone_tabular_prototype.reconstruction_method import MLP_Embedding_R
from utils.utils import AverageMeter, accuracy
from utils.losses import ReconstructionLoss_MLP
from torch_geometric.nn import GATConv
from utils.t_sne import t_sne_prototype

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



def create_log(opt_dict, lambda_re, lambda_match):
    log_train = None
    if not opt_dict['train_config']['find_epoch']:
        log_path = os.path.join(opt_dict['dataset_config']['it_result_path'], opt_dict['model_config']['net_v'], f"05_{lambda_re}_{lambda_match}_{opt_dict['dataset_config']['dataname']}_{opt_dict['train_config']['epochs']}_{opt_dict['dataset_config']['missing_rate']}_{opt_dict['model_config']['model_name']}_{opt_dict['model_config']['net_v']}_{opt_dict['train_config']['lr_max']}_log_train.csv")
        print(log_path)
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        log_train = open(log_path, 'w')

    return log_train


def write_log(log_train, log_out, opt_dict):
    if not opt_dict['train_config']['find_epoch']:
        # print(log_out)
        log_train.write(log_out + '\n')
        log_train.flush()
    return log_train



def calculate_nonlinear(epoch, total_epochs, min_lambda_re=0.1, max_lambda_re=0.6, power=2):
    normalized_epoch = epoch / (total_epochs - 1)
    lambda_re = min_lambda_re + (max_lambda_re - min_lambda_re) * (normalized_epoch ** power)
    return lambda_re

    

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



def load_model_image(opt_dict):
    from backbone.efficientnet import efficientnet_b1
    model = efficientnet_b1(num_classes=opt_dict['train_config']['num_cls'])
    net_pretrain_path = "/data/blood_dvm/data/result/end/encoder_image/img_re/00_feature_115_0.5056565061440195.pth"
    weights_dict = torch.load(net_pretrain_path, map_location='cpu')
    model.load_state_dict(weights_dict, strict=True)
    for name, param in model.named_parameters():
        param.requires_grad = True
    return model



def load_reconstruction_model(opt_dict): 
    opt_dict['model_config']['hidden_size2'] = opt_dict['model_config']['efficientb1_outdim']
    model = MLP_Embedding_R(opt_dict)
    net_pretrain_path = "/data/blood_dvm/data/result/end/encoder_image/img_re/00_re_115_0.5056565061440195.pth"
    weights_dict = torch.load(net_pretrain_path, map_location='cpu')
    model.load_state_dict(weights_dict, strict=True)
    for name, param in model.named_parameters():
        param.requires_grad = True
    return model



def load_graph_model(opt_dict):
    opt_dict['model_config']['embedding_dim'] = opt_dict['model_config']['efficientb1_outdim']
    model = ProtoGraph(opt_dict)
    return model



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



def refresh_prototype(prototype_tab, collect_features_dict, epoch, total_epoch):
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




def lambda_decay(epoch, max_epoch):
    decay_factor = (epoch / max_epoch) ** 2  
    lambda_match = 1 - decay_factor
    
    return lambda_match



def add_prototype(graph_net, prototype_tab, features, true_labels, device, epoch, epochs):
    # import ipdb;ipdb.set_trace();
    prototypes = torch.stack([p['mean'].to(device) for p in prototype_tab.values()])
    prototypes_norm = F.normalize(prototypes, p=2, dim=1)
    features_norm = F.normalize(features, p=2, dim=1)
    sim_matrix = torch.mm(features_norm, prototypes_norm.T)
    
    fused_features = features.clone()
    batch_size = features.size(0)
    
    for i in range(batch_size):
        # sim_min = calculate_nonlinear(epoch, epochs, 0.90, 0.95)
        sim_min = calculate_nonlinear(epoch, epochs, 0.85, 0.95)
        proto_mask = sim_matrix[i] > sim_min
        selected_protos = prototypes[proto_mask]
        if len(selected_protos) < 1:
            continue
            
        num_nodes = len(selected_protos) + 1 
        node_features = torch.cat([features[[i]], selected_protos], dim=0)
        
        # 当前特征与所有选择了的原型连接
        edge_index = []
        for j in range(1, num_nodes):
            edge_index.append([0, j])  # 特征到原型
            edge_index.append([j, 0])  # 原型到特征
        edge_index = torch.tensor(edge_index, dtype=torch.long, device=device).t()
        
        refined_proto = graph_net(node_features, edge_index)[0]  # 取中心节点
        
        similarity = F.cosine_similarity(features[i], refined_proto, dim=0)
        gate = torch.sigmoid(similarity)  
        
        fused_features[i] = features[i] + gate * (refined_proto - features[i])
    
    pos_sim = sim_matrix[range(batch_size), true_labels]
    pos_loss = (1 - pos_sim).mean()
    neg_mask = torch.ones_like(sim_matrix, dtype=bool)
    neg_mask[range(batch_size), true_labels] = False
    neg_loss = (sim_matrix[neg_mask] - 0.5).clamp(min=0).mean()
    # total_loss = pos_loss + 0.8 * neg_loss
    # total_loss = pos_loss + 0.6 * neg_loss
    # total_loss = pos_loss + 0.7 * neg_loss
    total_loss = pos_loss + 0.8 * neg_loss
    return fused_features, total_loss


def train_re_prototype(model_feature, model_re, model_graph, train_loader, test_loader, device, lambda_re, lambda_match, opt_dict):
    
    epochs = opt_dict['train_config']['epochs']
    log_train = create_log(opt_dict, lambda_re, lambda_match)

    train_losses_re = AverageMeter('Train_Loss_re', ':.4e')
    train_losses_match = AverageMeter('Train_Loss_match', ':.4e')
    train_losses_total = AverageMeter('Train_Loss_total', ':.4e')
    val_losses_re = AverageMeter('Val_loss', ':.4e')
    val_losses_match = AverageMeter('Val_Loss_match', ':.4e')
    val_losses_total = AverageMeter('Val_Loss_total', ':.4e')

    least_val_total = 1e9
    least_val_re = 1e9

    model_feature.to(device)
    model_re.to(device)
    model_graph.to(device)

    if opt_dict['dataset_config']['dataname'] == 'blood':
        num_cat, num_con = 19, 22
    loss_function = ReconstructionLoss_MLP(num_cat, num_con, opt_dict)

    pg_feature = [p for p in model_feature.parameters() if p.requires_grad]
    pg_re = [p for p in model_re.parameters() if p.requires_grad]
    pg_graph = [p for p in model_graph.parameters() if p.requires_grad]
    pg = pg_feature + pg_re + pg_graph

    optimizer = optim.Adam(pg, lr=float(opt_dict['train_config']['lr_max']))
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt_dict['train_config']['epochs'])

    prototype_tab = {}
    collect_features_dict = {}

    for epoch in range(epochs):
        # lambda_re = calculate_nonlinear(epoch, epochs, 0.1, 0.5)

        if((len(collect_features_dict) == 0) and (len(prototype_tab) == 0)):
            prototype_tab, collect_features_dict = inital_prototype(train_loader, opt_dict, model_feature, collect_features_dict, prototype_tab)
            # t_sne_prototype(prototype_tab, collect_features_dict, opt_dict)
            # import ipdb;ipdb.set_trace();
            collect_features_dict = {}
        elif ((len(collect_features_dict) > 0) and (len(prototype_tab) > 0)):
            prototype_tab = refresh_prototype(prototype_tab, collect_features_dict, epoch, epochs)
            # if(epoch % 20 == 0):
            #     t_sne_prototype(prototype_tab, collect_features_dict, opt_dict)
            collect_features_dict = {}
            # import ipdb;ipdb.set_trace();
        else:
            print("========================= None =========================")

        train_losses_re.reset()
        train_losses_match.reset()
        train_losses_total.reset()

        model_feature.train()
        model_re.train()
        model_graph.train()

        for step, data in enumerate(train_loader):
            images, tables, masks, labels = data
            masks = torch.ones_like(masks, dtype=torch.bool)
            images, tables, masks, labels = images.to(device), tables.to(device), masks.to(device), labels.to(device)
            batch_size_cur = images.size(0)
            features = model_feature.forward_sne(images)

            collect_features_dict = collect_features(collect_features_dict, features, labels)
            features_prototype, loss_match = add_prototype(model_graph, prototype_tab, features, labels, device, epoch, epochs)
            results = model_re(features_prototype)
            loss = loss_function(results, tables, masks)
            # import ipdb;ipdb.set_trace();
            loss_re = loss[0]
            loss_total = loss_re * lambda_re + loss_match * lambda_match
            train_losses_re.update(loss_re.item(), batch_size_cur)
            train_losses_match.update(loss_match.item(), batch_size_cur)
            train_losses_total.update(loss_total.item(), batch_size_cur)
            log_out = ('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                    'lambda_re: {lambda_re}, lambda_match: {lambda_match}\t'
                    'Train_Loss_re: {loss_re.val:.4f} ({loss_re.avg:.4f})\t'
                    'Train_Loss_match: {loss_match.val:.4f} ({loss_match.avg:.4f})\t'
                    'Train_Loss_total: {loss_total.val:.4f} ({loss_total.avg:.4f})'.format(
                    epoch, step, len(train_loader), lr=optimizer.param_groups[-1]['lr'],
                    lambda_re=lambda_re, lambda_match=lambda_match,
                    loss_re=train_losses_re, loss_match=train_losses_match, loss_total=train_losses_total
                ))
            log_train = write_log(log_train, log_out, opt_dict)
                
            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()

        schedule.step()

        model_feature.eval()
        model_re.eval()
        model_graph.eval()

        val_losses_re.reset()
        val_losses_match.reset()
        val_losses_total.reset()

        with torch.no_grad():
            for step, data in enumerate(test_loader):
                images, tables, masks, labels = data
                masks = torch.ones_like(masks, dtype=torch.bool)
                images, tables, masks, labels = images.to(device), tables.to(device), masks.to(device), labels.to(device)
                batch_size_cur = images.size(0)
                val_features = model_feature.forward_sne(images)

                val_features_prototype, val_loss_match = add_prototype(model_graph, prototype_tab, val_features, labels, device, epoch, epochs)
                val_results = model_re(val_features_prototype)
                val_loss = loss_function(val_results, tables, masks)

                val_loss_re = val_loss[0]
                val_loss_total = val_loss_re * lambda_re + val_loss_match * lambda_match
                val_losses_re.update(val_loss_re.item(), tables.size(0))
                val_losses_match.update(val_loss_match.item(), tables.size(0))
                val_losses_total.update(val_loss_total.item(), tables.size(0))
                
            log_out_val = ('Val_Loss_re {loss_re.val:.4f} ({loss_re.avg:.4f})\t'
                        'Val Loss_match {loss_match.val:.4f} ({loss_match.avg:.4f})\t'
                        'Val Loss_total {loss_total.val:.4f} ({loss_total.avg:.4f})\n'.format(
                        loss_re=val_losses_re, loss_match=val_losses_match, loss_total=val_losses_total
                        ))
            val_losses_total_plus = val_losses_re.avg + val_losses_match.avg
            log_train = write_log(log_train, log_out_val, opt_dict)
            
            if(least_val_total > val_losses_total_plus):
                least_val_total = val_losses_total_plus
                log_out_least_total = f"Saved Val Loss_total : {least_val_total}"
                log_train = write_log(log_train, log_out_least_total, opt_dict)
                least_val_re = val_losses_re.avg
                if(val_losses_re.avg < 0.52):
                    save_feature_path = f"/data/blood_dvm/data/result/temp/img_re_prototype/00_feature_{epochs}_{val_losses_re.avg}.pth"
                    save_re_path = f"/data/blood_dvm/data/result/temp/img_re_prototype/00_re_{epochs}_{val_losses_re.avg}.pth"
                    torch.save(model_feature.state_dict(), save_feature_path)
                    torch.save(model_re.state_dict(), save_re_path)


    return least_val_re




def train_img_tab_r_epoch(train_loader, test_loader, device, opt_dict):
    least_val_re_list = []
    max_epochs = opt_dict['train_config']['max_epochs']
    lambda_re = 0.6
    lambda_match = 0.4
    for epochs in range(50, max_epochs, 5):
        opt_dict['train_config']['epochs'] = epochs
        model_feature = load_model_image(opt_dict)
        model_re = load_reconstruction_model(opt_dict)
        model_graph = load_graph_model(opt_dict)
        print(f"Epoch: {opt_dict['train_config']['epochs']}")
        least_val_re = train_re_prototype(model_feature, model_re, model_graph, train_loader, test_loader, device, lambda_re, lambda_match, opt_dict)
        least_val_re_list.append(least_val_re)
        print(least_val_re_list)




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
    
    model_feature = load_model_image(opt_dict)
    model_re = load_reconstruction_model(opt_dict)
    model_graph = load_graph_model(opt_dict)

    # 0.6_0.4【√】
    lambda_re = 0.6
    lambda_match = 0.4

    # train_re_prototype(model_feature, model_re, model_graph, train_loader, test_loader, device, lambda_re, lambda_match, opt_dict)
    train_img_tab_r_epoch(train_loader, test_loader, device, opt_dict)






