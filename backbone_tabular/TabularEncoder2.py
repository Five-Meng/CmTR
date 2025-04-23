import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init



# class TabularEncoder(nn.Module):
#     def __init__(self, opt_dict, is_fc=False):
#         super(TabularEncoder, self).__init__()
#         self.is_fc = is_fc
#         self.num_classes = opt_dict['train_config']['num_cls']
#         self.field_lengths_tabular = torch.load(opt_dict['dataset_config']['field_lengths_tabular'])
#         self.field_lengths_tabular = [int(x) for x in self.field_lengths_tabular]
#         self.hidden_size1 = opt_dict['model_config']['hidden_size1']
#         self.hidden_size2 = opt_dict['model_config']['hidden_size2']
#         self.hidden_size3 = opt_dict['model_config']['hidden_size3']
#         self.embedding_dim = opt_dict['model_config']['embedding_dim']
#         self.hidden_size_importance = opt_dict['model_config']['hidden_size1']

#         self.cat_lengths_tabular = []
#         self.con_lengths_tabular = []
#         for x in self.field_lengths_tabular:
#             if x == 1:
#                 self.con_lengths_tabular.append(x)
#             else:
#                 self.cat_lengths_tabular.append(x)

#         self.num_cat = len(self.cat_lengths_tabular)
#         self.num_con = len(self.con_lengths_tabular)
#         self.num_unique_cat = sum(self.cat_lengths_tabular)

#         # 离散特征 Embedding
#         cat_offsets = torch.tensor([0] + self.cat_lengths_tabular[:-1]).cumsum(0)
#         self.register_buffer('cat_offsets', cat_offsets, persistent=False)
#         self.cat_embedding = nn.Embedding(self.num_unique_cat, self.embedding_dim)
#         # 连续特征线性变换
#         self.con_proj = nn.Linear(1, self.embedding_dim)
#         # 缺失值
#         self.mask_special_token = nn.Parameter(torch.zeros(1, 1, self.embedding_dim))


#         self.encoder_mlp = nn.Sequential(
#             nn.Linear((self.num_cat+self.num_con) * self.embedding_dim, self.hidden_size1),
#             nn.ReLU(),
#             nn.Dropout(p=0.2),
#             nn.Linear(self.hidden_size1, self.hidden_size2),
#             nn.ReLU(),
#             nn.Dropout(p=0.2),
#             nn.Linear(self.hidden_size2, self.hidden_size3),
#             nn.ReLU(),
#         )

#         self.norm = nn.LayerNorm(opt_dict['model_config']['embedding_dim'])
#         self.dropout = nn.Dropout(opt_dict['model_config']['embedding_dropout']) if opt_dict['model_config']['embedding_dropout'] > 0 else nn.Identity()
#         self.fc = nn.Linear(self.hidden_size3, self.num_classes)

#     def forward(self, x, mask=None, mask_special=None):
#         B, N = x.shape
#         x_cat_emb = self.cat_embedding(x[:, :self.num_cat].long() + self.cat_offsets)
#         x_cont = self.con_proj(x[:, self.num_cat:].unsqueeze(-1))
#         x_combined = torch.cat([x_cat_emb, x_cont], dim=1)
#         # x_combined = self.norm(x_combined)

#         if mask_special is not None:
#             mask_special = mask_special.unsqueeze(-1)
#             mask_special_tokens = self.mask_special_token.expand(
#                 x_combined.shape[0], x_combined.shape[1], -1
#             )
#             x_combined = mask_special * mask_special_tokens + (~mask_special) * x_combined


#         x_fused = x_combined.reshape(B, -1)
#         z = self.encoder_mlp(x_fused)
#         if(self.is_fc == False):
#             return z
#         else:
#             output = self.fc(z)
#             return output



class TabularEncoder(nn.Module):
    def __init__(self, opt_dict, is_fc=True):
        super(TabularEncoder, self).__init__()
        self.num_classes = opt_dict['train_config']['num_cls']
        self.field_lengths_tabular = torch.load(opt_dict['dataset_config']['field_lengths_tabular'])
        self.field_lengths_tabular = [int(x) for x in self.field_lengths_tabular]
        self.hidden_size1 = opt_dict['model_config']['hidden_size1']
        self.hidden_size2 = opt_dict['model_config']['hidden_size2']
        if(opt_dict['dataset_config']['dataname'] == 'dvm'):
            self.hidden_size3 = opt_dict['model_config']['hidden_size3']

        self.embedding_dim = opt_dict['model_config']['embedding_dim']
        self.hidden_size_importance = opt_dict['model_config']['hidden_size1']
        self.is_fc = is_fc
        
        self.cat_lengths_tabular = []
        self.con_lengths_tabular = []
        for x in self.field_lengths_tabular:
            if x == 1:
                self.con_lengths_tabular.append(x)
            else:
                self.cat_lengths_tabular.append(x)

        self.num_cat = len(self.cat_lengths_tabular)
        self.num_con = len(self.con_lengths_tabular)
        self.num_unique_cat = sum(self.cat_lengths_tabular)

        # 离散特征 Embedding
        cat_offsets = torch.tensor([0] + self.cat_lengths_tabular[:-1]).cumsum(0)
        self.register_buffer('cat_offsets', cat_offsets, persistent=False)
        self.cat_embedding = nn.Embedding(self.num_unique_cat, self.embedding_dim)
        # 连续特征线性变换
        self.con_proj = nn.Linear(1, self.embedding_dim)
        # 缺失值
        self.mask_special_token = nn.Parameter(torch.zeros(1, 1, self.embedding_dim))

        if(opt_dict['dataset_config']['dataname'] == 'blood'):
            self.encoder_mlp = nn.Sequential(
                nn.Linear((self.num_cat+self.num_con) * self.embedding_dim, self.hidden_size1),
                nn.ReLU(),
                nn.Dropout(p=0.4),
                nn.Linear(self.hidden_size1, self.hidden_size2),
                nn.ReLU(),
            )
            self.fc = nn.Linear(self.hidden_size2, self.num_classes)
        elif(opt_dict['dataset_config']['dataname'] == 'dvm'):
            self.encoder_mlp = nn.Sequential(
                nn.Linear((self.num_cat+self.num_con) * self.embedding_dim, self.hidden_size1),
                nn.ReLU(),
                nn.Dropout(p=0.2),
                nn.Linear(self.hidden_size1, self.hidden_size2),
                nn.ReLU(),
                nn.Dropout(p=0.2),
                nn.Linear(self.hidden_size2, self.hidden_size3),
                nn.ReLU(),
            )
            self.fc = nn.Linear(self.hidden_size3, self.num_classes)

        self.norm = nn.LayerNorm(opt_dict['model_config']['embedding_dim'])
        self.dropout = nn.Dropout(opt_dict['model_config']['embedding_dropout']) if opt_dict['model_config']['embedding_dropout'] > 0 else nn.Identity()

        self._initialize_weights()

    def forward(self, x, mask=None, mask_special=None):
        # import ipdb;ipdb.set_trace();
        B, N = x.shape
        x_cat_emb = self.cat_embedding(x[:, :self.num_cat].long() + self.cat_offsets)
        # import ipdb;ipdb.set_trace();
        x_cont = self.con_proj(x[:, self.num_cat:].unsqueeze(-1))
        x_combined = torch.cat([x_cat_emb, x_cont], dim=1)

        if mask_special is not None:
            mask_special = mask_special.unsqueeze(-1)
            mask_special_tokens = self.mask_special_token.expand(
                x_combined.shape[0], x_combined.shape[1], -1
            )
            x_combined = mask_special * mask_special_tokens + (~mask_special) * x_combined

        x_fused = x_combined.reshape(B, -1)
        z = self.encoder_mlp(x_fused)
        # import ipdb;ipdb.set_trace();
        if self.is_fc:
            output = self.fc(z)
            return output
        else:
            return z
    

    def _initialize_weights(self):
        nn.init.xavier_normal_(self.cat_embedding.weight)
        nn.init.xavier_normal_(self.con_proj.weight)
        if self.con_proj.bias is not None:
            nn.init.zeros_(self.con_proj.bias)
        
        for module in self.encoder_mlp:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(
                    module.weight, 
                    mode='fan_in', 
                    nonlinearity='relu'
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        nn.init.xavier_normal_(self.fc.weight)
        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)
        
        nn.init.normal_(self.mask_special_token, mean=0.0, std=0.02)
 