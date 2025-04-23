

import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset

class MultiFeatureDataset(Dataset):

    def __init__(self, data_path):
        super().__init__()
        self.data = np.load(data_path, allow_pickle=True).item()
        
        assert 'features_i' in self.data, "缺少features_i数据"
        assert 'features_t' in self.data, "缺少features_t数据"
        assert 'labels' in self.data, "缺少labels数据"
        assert len(self.data['features_i']) == len(self.data['labels']), "特征与标签数量不匹配"
        assert len(self.data['features_t']) == len(self.data['labels']), "特征与标签数量不匹配"

    def __len__(self):
        return len(self.data['labels'])

    def __getitem__(self, idx):
        feature_i = torch.FloatTensor(self.data['features_i'][idx])
        feature_t = torch.FloatTensor(self.data['features_t'][idx])
        label = torch.LongTensor([self.data['labels'][idx]]).squeeze()
        return feature_i, feature_t, label

    def get_feature_dim(self):
        return {
            'feature_i_dim': self.data['features_i'].shape[1],
            'feature_t_dim': self.data['features_t'].shape[1]
        }

    def get_class_distribution(self):
        unique, counts = np.unique(self.data['labels'], return_counts=True)
        return dict(zip(unique, counts))




# train_dataset = MultiFeatureDataset(
#     data_path="path/to/extracted_features/train_data.npy"
# )

# test_dataset = MultiFeatureDataset(
#     data_path="path/to/extracted_features/test_data.npy"
# )

# # 获取数据集信息
# print(f"训练集大小: {len(train_dataset)}")
# print(f"特征维度: {train_dataset.get_feature_dim()}")
# print(f"类别分布: {train_dataset.get_class_distribution()}") 