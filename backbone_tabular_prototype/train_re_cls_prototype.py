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
# from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
plt.rcParams['axes.unicode_minus'] = False  
from torch_geometric.nn import GATConv
from datetime import datetime



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
        for tables, labels, masks in train_loader:
            tables, labels, masks = tables.to(device), labels.to(device), masks.to(device)
            batch_size_cur = tables.shape[0]
            if opt_dict['model_config']['net_v_tabular'] == 'encoder_mlp_block_prototype':
                features, _ = model_feature(tables, masks, masks, has_fc=False)
            elif opt_dict['model_config']['net_v_tabular'] == 'encoder_mlp_prototype': 
                features = model_feature(tables, masks, masks)
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




# def add_prototype(graph_net, model_fc, prototype_tab, features, features_fc, loss_function_cls, labels, device, epoch, epochs):
#     prototypes = torch.stack([p['mean'].to(device) for p in prototype_tab.values()])
#     results_cls = model_fc(features_fc)
#     loss_cls = loss_function_cls(results_cls, labels)
#     train_acc = accuracy(results_cls, labels)[0]
    
#     probs = F.softmax(results_cls, dim=1)
#     # import ipdb;ipdb.set_trace();
#     fused_features = features.clone()
#     for i in range(features.size(0)):
#         prob_min = 0.05          # 0.1
#         proto_mask = probs[i] > prob_min
#         # print(f"proto_mask: {proto_mask.sum().item()}")
#         selected_protos = prototypes[proto_mask]
        
#         if len(selected_protos) < 1:
#             continue
            
#         num_nodes = len(selected_protos) + 1 
#         node_features = torch.cat([features[[i]], selected_protos], dim=0)
        
#         edge_index = []
#         for j in range(1, num_nodes):
#             edge_index.append([0, j])  # 特征到原型
#             edge_index.append([j, 0])  # 原型到特征
#         edge_index = torch.tensor(edge_index, dtype=torch.long, device=device).t()
        
#         refined_proto = graph_net(node_features, edge_index)[0]  # 取中心节点
#         similarity = F.cosine_similarity(features[i], refined_proto, dim=0)
#         gate = torch.sigmoid(similarity)  
#         # print(f"=====similarity: {similarity} ======= gate: {gate}")
        
#         fused_features[i] = features[i] + gate * (refined_proto - features[i])
    
#     return fused_features, loss_cls, train_acc



def add_prototype(graph_net, model_fc, prototype_tab, features, features_fc, loss_function_cls, labels, device, epoch, epochs):
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
        feat_dim = opt_dict['model_config']['hidden_size3']
        latent_dim = opt_dict['model_config']['latent_dim']
        
        self.gcn1 = GCNConv(feat_dim, latent_dim)
        self.gcn2 = GCNConv(latent_dim, feat_dim)
    
    def forward(self, x, edge_index, edge_weight):
        x = F.relu(self.gcn1(x, edge_index, edge_weight))
        return self.gcn2(x, edge_index, edge_weight)


def create_log(opt_dict, lambda_re, lambda_cls):
    log_train = None
    if not opt_dict['train_config']['find_epoch']:
        log_path = os.path.join(opt_dict['dataset_config']['tabular_result_path'], opt_dict['model_config']['net_v_tabular'], f"{lambda_re}_{lambda_cls}_{opt_dict['dataset_config']['dataname']}_{opt_dict['train_config']['epochs']}_{opt_dict['dataset_config']['missing_rate']}_{opt_dict['model_config']['model_name']}_{opt_dict['model_config']['net_v_tabular']}_{opt_dict['train_config']['lr_max']}_log_train.csv")
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


def load_graph_model(opt_dict):
    # model = ProtoGraph(opt_dict)
    model = GCNModel(opt_dict)
    return model


def load_feature_model(opt_dict):
    if opt_dict['model_config']['net_v_tabular'] == 'encoder_mlp_block_prototype':
        model = TabularTransformerEncoder(opt_dict, has_fc=False)    
    elif opt_dict['model_config']['net_v_tabular'] == 'encoder_mlp_prototype':
        model = TabularEncoder(opt_dict, is_fc=False)
        checkpoint_feature = opt_dict['dataset_config']['load_checkpoint_feature']
        checkpoint = torch.load(checkpoint_feature)
        model.load_state_dict(checkpoint,strict=True)
        print("load_feature_checkpoint")
    return model


def load_reconstruction_model(opt_dict):
    if opt_dict['model_config']['net_v_tabular'] == 'encoder_mlp_block_prototype':
        model = Transformer_Embedding_R(opt_dict)
    elif opt_dict['model_config']['net_v_tabular'] == 'encoder_mlp_prototype': 
        model = MLP_Embedding_R(opt_dict)
        checkpoint_re = opt_dict['dataset_config']['load_checkpoint_re']
        checkpoint = torch.load(checkpoint_re)
        model.load_state_dict(checkpoint,strict=True)
        print("load_re_checkpoint")
    return model


def load_fc_model(opt_dict, num_cat, num_con):
    model_fc3 = None
    if opt_dict['model_config']['net_v_tabular'] == 'encoder_mlp_block_prototype':
        input_size = opt_dict['model_config']['tabular_embedding_dim'] * (num_cat + num_con + 1)
        model_fc3 = nn.Linear(input_size, opt_dict['train_config']['num_cls'])
    elif opt_dict['model_config']['net_v_tabular'] == 'encoder_mlp_prototype': 
        input_size = opt_dict['model_config']['hidden_size3']
        model_fc3 = nn.Linear(input_size, opt_dict['train_config']['num_cls'])
        checkpoint_re = opt_dict['dataset_config']['load_checkpoint_fc']
        checkpoint = torch.load(checkpoint_re)
        model_fc3.load_state_dict(checkpoint,strict=True)
        print("load_fc_checkpoint")
    return model_fc3



def save_best_model_cls(lambda_re, lambda_cls, log_train, model_feature, model_re, model_fc, least_total_losses, val_losses, opt_dict):
    if(least_total_losses > val_losses.avg):
        if not opt_dict['train_config']['find_epoch']:
            save_model_path_feature = f"feature_{lambda_re}_{lambda_cls}_{opt_dict['dataset_config']['dataname']}_{opt_dict['train_config']['epochs']}_{opt_dict['dataset_config']['missing_rate']}_{opt_dict['model_config']['model_name']}_{opt_dict['train_config']['lossfunc']}_{opt_dict['model_config']['net_v_tabular']}_{opt_dict['train_config']['lr_max']}_best_model.pth"
            save_model_path_re = f"re_{lambda_re}_{lambda_cls}_{opt_dict['dataset_config']['dataname']}_{opt_dict['train_config']['epochs']}_{opt_dict['dataset_config']['missing_rate']}_{opt_dict['model_config']['model_name']}_{opt_dict['train_config']['lossfunc']}_{opt_dict['model_config']['net_v_tabular']}_{opt_dict['train_config']['lr_max']}_best_model.pth"            
            save_model_path_fc = f"fc_{lambda_re}_{lambda_cls}_{opt_dict['dataset_config']['dataname']}_{opt_dict['train_config']['epochs']}_{opt_dict['dataset_config']['missing_rate']}_{opt_dict['model_config']['model_name']}_{opt_dict['train_config']['lossfunc']}_{opt_dict['model_config']['net_v_tabular']}_{opt_dict['train_config']['lr_max']}_best_model.pth"            
            log_best_model = f'Saved best model with least_total_losses: {val_losses.avg:.4f}\n'
            log_train = write_log(log_train, log_best_model, opt_dict)
            torch.save(model_feature.state_dict(), os.path.join(opt_dict['dataset_config']['tabular_result_path'], opt_dict['model_config']['net_v_tabular'], save_model_path_feature))
            torch.save(model_re.state_dict(), os.path.join(opt_dict['dataset_config']['tabular_result_path'], opt_dict['model_config']['net_v_tabular'], save_model_path_re))
            torch.save(model_fc.state_dict(), os.path.join(opt_dict['dataset_config']['tabular_result_path'], opt_dict['model_config']['net_v_tabular'], save_model_path_fc))

    return log_train


def save_best_model_cls_acc(log_train, model_feature, model_re, model_fc, less_cls_acc, val_acc_v, opt_dict):
    if(less_cls_acc < val_acc_v.avg):
        less_cls_acc = val_acc_v.avg
        if not opt_dict['train_config']['find_epoch']:
            save_model_path_feature = f"feature_{opt_dict['dataset_config']['dataname']}_{opt_dict['train_config']['epochs']}_{opt_dict['dataset_config']['missing_rate']}_{opt_dict['model_config']['model_name']}_{opt_dict['train_config']['lossfunc']}_{opt_dict['model_config']['net_v_tabular']}_{opt_dict['train_config']['lr_max']}_best_model.pth"
            save_model_path_re = f"re_{opt_dict['dataset_config']['dataname']}_{opt_dict['train_config']['epochs']}_{opt_dict['dataset_config']['missing_rate']}_{opt_dict['model_config']['model_name']}_{opt_dict['train_config']['lossfunc']}_{opt_dict['model_config']['net_v_tabular']}_{opt_dict['train_config']['lr_max']}_best_model.pth"            
            save_model_path_fc = f"fc_{opt_dict['dataset_config']['dataname']}_{opt_dict['train_config']['epochs']}_{opt_dict['dataset_config']['missing_rate']}_{opt_dict['model_config']['model_name']}_{opt_dict['train_config']['lossfunc']}_{opt_dict['model_config']['net_v_tabular']}_{opt_dict['train_config']['lr_max']}_best_model.pth"            
            log_best_model = f'Saved best model with val_acc_v: {val_acc_v.avg:.4f}\n'
            log_train = write_log(log_train, log_best_model, opt_dict)
            torch.save(model_feature.state_dict(), os.path.join(opt_dict['dataset_config']['tabular_result_path'], opt_dict['model_config']['net_v_tabular'], save_model_path_feature))
            torch.save(model_re.state_dict(), os.path.join(opt_dict['dataset_config']['tabular_result_path'], opt_dict['model_config']['net_v_tabular'], save_model_path_re))
            torch.save(model_fc.state_dict(), os.path.join(opt_dict['dataset_config']['tabular_result_path'], opt_dict['model_config']['net_v_tabular'], save_model_path_fc))

    return less_cls_acc, log_train




def t_sne_prototype(prototype_dict, collect_features_dict, opt_dict, epoch, epochs):
    print("============= t_sne_prototype ======================")
    all_features = []
    all_labels = []
    
    for label, features in collect_features_dict.items():
        all_features.extend([f.detach().cpu().numpy() for f in features])  
        all_labels.extend([f"class_{label}"] * len(features))  
    
    proto_means = [proto['mean'].cpu().numpy() for proto in prototype_dict.values()]
    proto_labels = [f"proto_{label}" for label in prototype_dict.keys()]
    all_features.extend(proto_means)
    all_labels.extend(proto_labels)
    
    all_features = np.array(all_features)
    all_labels = np.array(all_labels)
    
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    embeddings = tsne.fit_transform(all_features)
    
    plt.figure(figsize=(15, 12))
    custom_colors = {
        '1': '#1f77b4',  # 蓝色
        '2': '#ff7f0e',  # 橙色
        '3': '#2ca02c',  # 绿色
        '4': '#d62728',  # 红色
        '5': '#9467bd',  # 紫色
        '6': '#8c564b',  # 棕色
        '7': '#e377c2',  # 粉色
        '8': '#7f7f7f',  # 灰色
        '9': '#bcbd22',  # 黄绿色
        '0': '#17becf'  # 蓝绿色
    }
    
    class_mask = [l.startswith('class') for l in all_labels]
    unique_classes = sorted(set(l.split('_')[1] for l in all_labels[class_mask]))
    class_colors = {cls: custom_colors[cls] for cls in unique_classes}
    
    for idx, cls in enumerate(unique_classes):
        mask = (all_labels == f"class_{cls}")
        plt.scatter(embeddings[mask, 0], embeddings[mask, 1],
                    color=class_colors[cls],
                    alpha=0.8, s=20, label=f'Class {cls}', marker='o')
    
    proto_sample_mask = [l.startswith('proto') and 'center' not in l for l in all_labels]
    proto_samples = embeddings[proto_sample_mask]
    for idx, (x, y) in enumerate(proto_samples):
        class_id = proto_labels[idx].split('_')[1]
        plt.scatter(x, y, s=400, marker='*',
                    edgecolors='black', linewidths=1.5,
                    color=class_colors[class_id],
                    label=f'Proto {class_id}')
        plt.text(x, y, class_id, ha='center', va='bottom', 
                 fontsize=9, weight='bold', color='darkred')
    
    
    plt.title("t-SNE Visualization (Features and Prototypes)", fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), frameon=False)
    plt.tight_layout()
    
    save_path = opt_dict['dataset_config']['t_sne_path']
    os.makedirs(save_path, exist_ok=True)
    save_file = f"{save_path}/{epochs}_tsne_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{epoch}.png"
    plt.savefig(save_file, dpi=200, bbox_inches='tight')
    print(f"Saved to {save_file}")
    plt.close()




def train_re_cls_prototype(model_feature, model_re, model_fc, model_graph, train_loader, test_loader, device, opt_dict):
    num_cat, num_con, cat_offsets = model_feature.num_cat, model_feature.num_con, model_feature.cat_offsets.to(device)
    # lambda_re = 0.7
    # lambda_cls = 0.3

    lambda_re = 0.4
    lambda_cls = 0.6

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
    less_cls_acc = 0
    update_re_losses = 0
    update_cls_losses = 0
    update_cls_acc = 0
    least_re_losses = 1e9

    model_feature.to(device)
    model_re.to(device)
    model_fc.to(device)
    model_graph.to(device)

    print(opt_dict['model_config']['net_v_tabular'])
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
        # lambda_re = calculate_nonlinear(epoch, epochs, 0.95, 0.95)   # lambda_re提升了
        # lambda_cls = 1 - lambda_re

        if((len(collect_features_dict) == 0) and (len(prototype_tab) == 0)):
            prototype_tab, collect_features_dict = inital_prototype(train_loader, opt_dict, model_feature, collect_features_dict, prototype_tab)
            # t_sne_prototype(prototype_tab, collect_features_dict, opt_dict, epoch, epochs)
            # import ipdb;ipdb.set_trace();
            collect_features_dict = {}
        elif ((len(collect_features_dict) > 0) and (len(prototype_tab) > 0)):
            prototype_tab = refresh_prototype(prototype_tab, collect_features_dict, epoch, epochs)
            # if(epoch == 40 or epoch == 20 or epoch == 80):
            #     t_sne_prototype(prototype_tab, collect_features_dict, opt_dict, epoch, epochs)
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
            tables, labels, masks = data
            tables, labels, masks = tables.to(device), labels.to(device), masks.to(device)
            batch_size_cur = tables.size(0)

            features = model_feature(tables, masks, masks)
            features_fc = features
            collect_features_dict = collect_features(collect_features_dict, features, labels)
            features_prototype, loss_cls, train_acc = add_prototype(model_graph, model_fc, prototype_tab, features, features_fc, loss_function_cls, labels, device, epoch, epochs)
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
            for tables, labels, masks in test_loader:
                tables, labels, masks = tables.to(device), labels.to(device), masks.to(device)

                val_features = model_feature(tables, masks, masks)
                val_features_fc = val_features
                collect_features_dict = collect_features(collect_features_dict, val_features, labels)
                val_features_prototype, val_loss_cls, val_acc = add_prototype(model_graph, model_fc, prototype_tab, val_features, val_features_fc, loss_function_cls, labels, device, epoch, epochs)
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
                # if(epoch > 60):
                #     t_sne_prototype(prototype_tab, collect_features_dict, opt_dict, epoch, epochs)
                less_total_losses = val_losses_plus
                update_re_losses = val_losses_re.avg
                update_cls_losses = val_losses_cls.avg
                update_cls_acc = val_acc_v.avg
                log_out_best = f"less_total_losses:{less_total_losses} \n update_re_losses:{update_re_losses} \n update_cls_losses:{update_cls_losses} \n update_cls_acc:{update_cls_acc}"
                log_train = write_log(log_train, log_out_best, opt_dict)
                
            if((val_acc_v.avg > 68) or (val_losses_re.avg < 1)):
                save_feature_path = f"/data/blood_dvm/data/result/temp/reclsprototype/03_feature_{epochs}_{epoch}_{opt_dict['train_config']['lr_max']}_{val_acc_v.avg}_{val_losses_re.avg}.pth"
                save_re_path = f"/data/blood_dvm/data/result/temp/reclsprototype/03_re_{epochs}_{epoch}_{opt_dict['train_config']['lr_max']}_{val_acc_v.avg}_{val_losses_re.avg}.pth"
                save_fc_path = f"/data/blood_dvm/data/result/temp/reclsprototype/03_fc_{epochs}_{epoch}_{opt_dict['train_config']['lr_max']}_{val_acc_v.avg}_{val_losses_re.avg}.pth"
                torch.save(model_feature.state_dict(), save_feature_path)
                torch.save(model_re.state_dict(), save_re_path)
                torch.save(model_fc.state_dict(), save_fc_path)
                



    print("Finish!")
    return less_total_losses, update_re_losses, update_cls_losses, update_cls_acc



def train_epoch_cls(train_loader, test_loader, device, opt_dict, lambda_re=None, lambda_cls=None):
    print(opt_dict['train_config']['mode'])
    print(opt_dict['model_config']['net_v_tabular'])
    less_total_losses1 = [] 
    update_re_losses1 = []
    update_cls_losses1 = []
    update_cls_acc1 = []

    max_epoch = opt_dict['train_config']['max_epochs']
    for epoch in range(100, max_epoch, 5):
        print(f"现在是Epoch={epoch}")
        opt_dict['train_config']['epochs'] = epoch
        print(f"opt_dict['train_config']['epoch']:{opt_dict['train_config']['epochs']}")
        model_feature = load_feature_model(opt_dict)
        model_re = load_reconstruction_model(opt_dict)
        num_cat, num_con = model_feature.num_cat, model_feature.num_con
        model_fc = load_fc_model(opt_dict, num_cat, num_con)
        model_graph = load_graph_model(opt_dict)
        less_total_losses, update_re_losses, update_cls_losses, update_cls_acc = train_re_cls_prototype(model_feature, model_re, model_fc, model_graph, train_loader, test_loader, device, opt_dict)
        log_each = f"less_total_losses :{less_total_losses} \n update_re_losses :{update_re_losses} \n update_cls_losses :{update_cls_losses} \n update_cls_acc :{update_cls_acc}"
        print(log_each)
        less_total_losses1.append(less_total_losses)
        update_re_losses1.append(update_re_losses)
        update_cls_losses1.append(update_cls_losses)
        update_cls_acc1.append(update_cls_acc)

    print('Finished')
    return less_total_losses1, update_re_losses1, update_cls_losses1, update_cls_acc1



if __name__ == '__main__':
    from build_dataset_unit import UnitDataset
    mask_version = opt_dict['dataset_config']['missing_rate']
    print(f"mask_version : {mask_version}")
    dataset_dir = opt_dict['dataset_config']['data_dir']

    if opt_dict['dataset_config']['dataname'] == 'blood':
        train_dataset = UnitDataset(dataset_dir, mode='train', dataset_type='tabular', mask_version=mask_version)
        test_dataset = UnitDataset(dataset_dir, mode='test', dataset_type='tabular', mask_version=mask_version)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=64)
    elif opt_dict['dataset_config']['dataname'] == 'dvm':
        train_dataset = tabular_dataset_dvm(opt_dict, 'train')
        test_dataset = tabular_dataset_dvm(opt_dict, 'test')
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=64)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    print(torch.cuda.is_available())

    model_feature = load_feature_model(opt_dict)
    model_re = load_reconstruction_model(opt_dict)
    num_cat, num_con = model_feature.num_cat, model_feature.num_con
    model_fc = load_fc_model(opt_dict, num_cat, num_con)
    model_graph = load_graph_model(opt_dict)
    lambda_re = 0.3
    lambda_cls = 0.7
    # less_total_losses, update_re_losses, update_cls_losses, update_cls_acc = train_epoch_cls(train_loader, test_loader, device, opt_dict, lambda_re, lambda_cls)
    less_total_losses, update_re_losses, update_cls_losses, update_cls_acc = train_re_cls_prototype(model_feature, model_re, model_fc, model_graph, train_loader, test_loader, device, opt_dict)

