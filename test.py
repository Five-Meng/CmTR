import torch
import torch.nn as nn
import torch.nn.functional as F
from argparses.util.yaml_args import yaml_data
import argparse
from build_dataset_unit import UnitDataset, UnitDataset_dvm
from build_dataset_tabular import tabular_dataset_dvm
from backbone.efficientnet import efficientnet_b1
from backbone_tabular.TabularEncoder2 import TabularEncoder
from backbone_img_tab.crossattn import CrossAttention
from backbone_img_tab.train_img_tab import ConcatLinearClassifier
from utils.losses import ReconstructionLoss_MLP, KLLoss
from utils.utils import accuracy, f1_c
from torch_geometric.nn import GCNConv
from backbone_tabular_prototype.train_re_cls_prototype import add_prototype

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


def test_integrated_model(opt_dict, device):
    # 加载图像模型
    model_img = efficientnet_b1(num_classes=opt_dict['train_config']['num_cls'])
    checkpoint_img_path = opt_dict['predict_config']['load_predict_feature_img']
    model_img.load_state_dict(torch.load(checkpoint_img_path, map_location=device))
    model_img.to(device)
    model_img.eval()

    # 加载表格模型
    model_tab = TabularEncoder(opt_dict, is_fc=False)
    checkpoint_tab_path = opt_dict['predict_config']['load_predict_feature_tab']
    model_tab.load_state_dict(torch.load(checkpoint_tab_path, map_location=device))
    model_tab.to(device)
    model_tab.eval()

    # 加载CrossAttention模型
    model_attn = CrossAttention(opt_dict['model_config']['hidden_size3'])
    checkpoint_attn_path = opt_dict['predict_config']['load_predict_attn']
    model_attn.load_state_dict(torch.load(checkpoint_attn_path, map_location=device))
    model_attn.to(device)
    model_attn.eval()

    # 加载FM部分的线性层和重建模块
    dim_fusion = opt_dict['model_config']['dim_fusion']
    dim_i = 1280
    dim_t = opt_dict['model_config']['hidden_size3']
    num_classes = opt_dict['train_config']['num_cls']
    model_fc = ConcatLinearClassifier(dim_fusion, dim_i, dim_t, num_classes)
    checkpoint_fc_path = opt_dict['predict_config']['load_predict_fc']
    model_fc.load_state_dict(torch.load(checkpoint_fc_path, map_location=device))
    model_fc.to(device)
    model_fc.eval()

    # 加载graph模型
    model_graph = GCNModel(opt_dict)
    checkpoint_graph_path = opt_dict['predict_config']['load_predict_graph']
    model_graph.load_state_dict(torch.load(checkpoint_graph_path, map_location=device))
    model_graph.to(device)
    model_graph.eval()

    prototype_dict = torch.load(opt_dict['predict_config']['load_prototype_img'])
    prototype_tab = torch.load(opt_dict['predict_config']['load_prototype_tab'])

    if opt_dict['dataset_config']['dataname'] == 'blood':
        dataset = UnitDataset(opt_dict, mode='test', dataset_type='image_tabular')
    else:
        dataset = UnitDataset_dvm(opt_dict, mode='test', dataset_type='image_tabular')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)

    loss_function_cls = nn.CrossEntropyLoss()
    num_cat, num_con = 4, 13
    loss_function_recon = ReconstructionLoss_MLP(num_cat, num_con, opt_dict)
    total_loss_cls = 0
    total_loss_recon = 0
    total_acc = 0
    num_samples = 0

    with torch.no_grad():
        for images, tables, labels, masks in dataloader:
            images, tables, labels, masks = images.to(device), tables.to(device), labels.to(device), masks.to(device)
            # VDM模块
            features_img = model_img.forward(images)
            features_tab = model_tab(tables, masks, masks)

            # 分别处理图像和表格特征
            features_img_fc = features_img
            features_tab_fc = features_tab

            # 分别使用 add_prototype 函数
            features_img_prototype, loss_cls_img, train_acc_img = add_prototype(model_graph, model_fc, prototype_tab, features_img, features_img_fc, loss_function_cls, labels, device, 0, 1)
            features_tab_prototype, loss_cls_tab, train_acc_tab = add_prototype(model_graph, model_fc, prototype_tab, features_tab, features_tab_fc, loss_function_cls, labels, device, 0, 1)

            features_attn = model_attn(features_img_prototype, features_tab_prototype)

            # FM模块
            combined_features = torch.cat([features_img, features_attn], dim=1)
            logits = model_fc(combined_features)
            recon_output = model_fc.reconstruct(combined_features)
            loss_cls = loss_function_cls(logits, labels)
            loss_recon = loss_function_recon(recon_output, tables, masks)[0]
            acc = accuracy(logits, labels)[0]

            total_loss_cls += loss_cls.item() * images.size(0)
            total_loss_recon += loss_recon.item() * images.size(0)
            total_acc += acc.item() * images.size(0)
            num_samples += images.size(0)
    avg_loss_cls = total_loss_cls / num_samples
    avg_loss_recon = total_loss_recon / num_samples
    avg_acc = total_acc / num_samples
    print(f"Classification Loss: {avg_loss_cls}, Reconstruction Loss: {avg_loss_recon}, Test Acc: {avg_acc}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml_config', type=str, help='config args')
    args = parser.parse_args()
    opt_dict = yaml_data(args.yaml_config)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    test_integrated_model(opt_dict, device)
