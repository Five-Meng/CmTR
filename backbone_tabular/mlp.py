import torch.nn as nn


# 只有MLP
class MLP(nn.Module):
    def __init__(self, input_size, opt_dict, dropout_prob=0.5):
        super(MLP, self).__init__()
        hidden_size1 = opt_dict['model_config']['hidden_size1']
        hidden_size2 = opt_dict['model_config']['hidden_size2']
        num_classes = opt_dict['train_config']['num_cls']

        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.bn1 = nn.BatchNorm1d(hidden_size1)
        self.dropout1 = nn.Dropout(p=dropout_prob)
        self.relu = nn.ReLU()

        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.bn2 = nn.BatchNorm1d(hidden_size2)
        self.dropout2 = nn.Dropout(p=dropout_prob)

        self.fc3 = nn.Linear(hidden_size2, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout1(out)

        out = self.fc2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout2(out)

        out = self.fc3(out)
        return out

# class MLP(nn.Module):
#     def __init__(self, input_size, hidden_size1, hidden_size2, num_classes, dropout_prob=0.5):
#         super(MLP, self).__init__()
#         # print(f"dropout_prob:{dropout_prob}")
#         self.hidden_size3 = 2048
#         self.hidden_size4 = 4096

#         self.fc1 = nn.Linear(input_size, hidden_size1)
#         self.bn1 = nn.BatchNorm1d(hidden_size1)
#         self.dropout1 = nn.Dropout(p=dropout_prob)
#         self.relu = nn.ReLU()

#         self.fc2 = nn.Linear(hidden_size1, hidden_size2)
#         self.bn2 = nn.BatchNorm1d(hidden_size2)
#         self.dropout2 = nn.Dropout(p=dropout_prob)

#         self.fc3 = nn.Linear(hidden_size2, self.hidden_size3)
#         self.bn3 = nn.BatchNorm1d(self.hidden_size3)
#         self.dropout3 = nn.Dropout(p=dropout_prob)

#         self.fc4 = nn.Linear(self.hidden_size3, self.hidden_size4)
#         self.bn4 = nn.BatchNorm1d(self.hidden_size4)
#         self.dropout4 = nn.Dropout(p=dropout_prob)

#         self.fc5 = nn.Linear(self.hidden_size4, num_classes)

#     def forward(self, x):
#         out = self.fc1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#         out = self.dropout1(out)

#         out = self.fc2(out)
#         out = self.bn2(out)
#         out = self.relu(out)
#         out = self.dropout2(out)

#         out = self.fc3(out)
#         out = self.bn3(out)
#         out = self.relu(out)
#         out = self.dropout3(out)

#         out = self.fc4(out)
#         out = self.bn4(out)
#         out = self.relu(out)
#         out = self.dropout4(out)

#         out = self.fc5(out)
#         return out




# 【只有embedding】，没有transformer的MLP
# class Encoder_MLP_Embedding(nn.Module):
#     def __init__(self, opt_dict, hidden_size1, hidden_size2, num_classes, dropout_prob=0.5):
#         super(Encoder_MLP_Embedding, self).__init__()

#         # 加载特征长度
#         # [1, 1, 1, 1, 2, 3, 2, 2]
#         self.field_lengths_tabular = torch.load(opt_dict['dataset_config']['field_lengths_tabular'])
#         self.cat_lengths_tabular = []
#         self.con_lengths_tabular = []
#         for x in self.field_lengths_tabular:
#             if x == 1:
#                 self.con_lengths_tabular.append(x)
#             else:
#                 self.cat_lengths_tabular.append(x)
#         self.num_con = len(self.con_lengths_tabular)
#         self.num_cat = len(self.cat_lengths_tabular)

#         self.tabular_encoder = TabularEmbeddingEncoder(opt_dict, self.cat_lengths_tabular, self.con_lengths_tabular)
#         print('Using Encoder_MLP_Embedding tabular!')

#         self.input_size = opt.tabular_embedding_dim * (self.num_con + self.num_cat + 1)

#         self.fc1 = nn.Linear(self.input_size, hidden_size1)
#         self.bn1 = nn.BatchNorm1d(hidden_size1)
#         self.dropout1 = nn.Dropout(p=dropout_prob)
#         self.relu = nn.ReLU()

#         self.fc2 = nn.Linear(hidden_size1, hidden_size2)
#         self.bn2 = nn.BatchNorm1d(hidden_size2)
#         self.dropout2 = nn.Dropout(p=dropout_prob)

#         self.fc3 = nn.Linear(hidden_size2, num_classes)


#     def forward(self, x):
#         x_tab = self.tabular_encoder(x)
#         x_tab = x_tab.view(x_tab.size(0), -1)

#         out = self.fc1(x_tab)
#         out = self.bn1(out)
#         out = self.relu(out)
#         out = self.dropout1(out)

#         out = self.fc2(out)
#         out = self.bn2(out)
#         out = self.relu(out)
#         out = self.dropout2(out)

#         out = self.fc3(out)
#         return out


# #有embedding和离散加连续transformer的MLP
# class Encoder_MLP_WithBlock(nn.Module):
#     def __init__(self, opt_dict, hidden_size1, hidden_size2, num_classes, dropout_prob=0.5):
#         super(Encoder_MLP_WithBlock, self).__init__()

#         # 加载特征长度
#         self.field_lengths_tabular = torch.load(opt_dict['dataset_config']['field_lengths_tabular'])
#         self.cat_lengths_tabular = []
#         self.con_lengths_tabular = []
#         for x in self.field_lengths_tabular:
#             if x == 1:
#                 self.con_lengths_tabular.append(x)
#             else:
#                 self.cat_lengths_tabular.append(x)
#         self.num_con = len(self.con_lengths_tabular)
#         self.num_cat = len(self.cat_lengths_tabular)

#         # print(f"self.cat_lengths_tabular : {self.cat_lengths_tabular}")
#         # print(f"self.con_lengths_tabular : {self.con_lengths_tabular}")

#         self.tabular_encoder = TabularTransformerEncoder(opt_dict, self.cat_lengths_tabular, self.con_lengths_tabular)
#         print('Using Encoder_MLP_WithBlock tabular!')

#         # 定义全连接层
#         self.fc1 = nn.Linear(opt.tabular_embedding_dim, hidden_size2)
#         self.bn1 = nn.BatchNorm1d(hidden_size2)
#         self.dropout1 = nn.Dropout(p=dropout_prob)
#         self.relu = nn.ReLU()

#         self.fc3 = nn.Linear(hidden_size2, num_classes)

#     def get_input_size(self) -> int:
#         """
#         Returns the number of fields in the table.
#         Used to set the input number of nodes in the MLP
#         """
#         return len(self.field_lengths)


#     def forward(self, x):
#         # 取出CLS_Token
#         x_tab = self.tabular_encoder(x)[:, 0, :]

#         # 通过第一个全连接层
#         out = self.fc1(x_tab)
#         out = self.bn1(out)
#         out = self.relu(out)
#         out = self.dropout1(out)

#         if opt.prototype:
#             return out
#         else:
#             out = self.fc3(out)
#             return out


# #有embedding和离散transformer的MLP
# class Encoder_MLP_WithCatBlock(nn.Module):
#     def __init__(self, opt_dict, hidden_size1, hidden_size2, num_classes, dropout_prob=0.5):
#         super(Encoder_MLP_WithCatBlock, self).__init__()

#         self.field_lengths_tabular = torch.load(opt_dict['dataset_config']['field_lengths_tabular'])
#         self.cat_lengths_tabular = []
#         self.con_lengths_tabular = []
#         for x in self.field_lengths_tabular:
#             if x == 1:
#                 self.con_lengths_tabular.append(x)
#             else:
#                 self.cat_lengths_tabular.append(x)
#         self.num_con = len(self.con_lengths_tabular)
#         self.num_cat = len(self.cat_lengths_tabular)

#         self.tabular_encoder = TabularCatTransformerEncoder(opt_dict, self.cat_lengths_tabular, self.con_lengths_tabular)
#         print('Using Encoder_MLP_WithCatBlock tabular!')

#         self.fc1 = nn.Linear(opt.tabular_embedding_dim + self.num_con, hidden_size2)
#         self.bn1 = nn.BatchNorm1d(hidden_size2)
#         self.dropout1 = nn.Dropout(p=dropout_prob)
#         self.relu = nn.ReLU()

#         self.fc3 = nn.Linear(hidden_size2, num_classes)
#         self.norm = nn.LayerNorm(len(self.con_lengths_tabular))

#     def forward(self, x):
#         if(len(x.shape)>2):
#             x = x.squeeze(1)

#         x_cat = self.tabular_encoder(x)[:, 0, :]

#         con_x = x[:, self.num_cat:]
#         con_x = self.norm(con_x)

#         x_combined = torch.cat((x_cat, con_x), dim=1)

#         out = self.fc1(x_combined)
#         out = self.bn1(out)
#         out = self.relu(out)
#         out = self.dropout1(out)

#         if not opt.net_v_imgtab == 'img_tab_align' and not opt.net_v_imgtab == 'img_tab_distill':
#             out = self.fc3(out)
            
#         return out