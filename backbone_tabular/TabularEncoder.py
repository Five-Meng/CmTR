from typing import Dict, List, Optional
import torch
import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# class Mlp(nn.Module):
#     def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
#         super().__init__()
#         out_features = out_features or in_features
#         hidden_features = hidden_features or in_features
#         self.fc1 = nn.Linear(in_features, hidden_features)
#         self.act = act_layer()
#         self.fc2 = nn.Linear(hidden_features, 1024)
#         self.fc3 = nn.Linear(1024, out_features)
#         self.drop = nn.Dropout(drop)

#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.act(x)
#         x = self.drop(x)
#         x = self.fc2(x)
#         x = self.act(x)
#         x = self.drop(x)
#         x = self.fc3(x)
#         x = self.drop(x)
#         return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=False, qk_scale=None, attn_drop=0.3, proj_drop=0.3, with_qkv=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.with_qkv = with_qkv
        if self.with_qkv:
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
            self.proj = nn.Linear(dim, dim)
            self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.save_attention = False
        self.save_gradients = False

    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients

    def get_attn_gradients(self):
        return self.attn_gradients

    def save_attention_map(self, attention_map):
        self.attention_map = attention_map

    def get_attention_map(self):
        return self.attention_map

    def forward(self, x, mask=None, visualize=True):
        B, N, C = x.shape
        if self.with_qkv:
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
        else:
            qkv = x.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            q, k, v = qkv, qkv, qkv

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if mask is not None:
            # import ipdb;ipdb.set_trace();
            attn = attn + mask

        attn = attn.softmax(dim=-1)
        if self.save_attention:
            self.save_attention_map(attn)
        if self.save_gradients:
            attn.register_hook(self.save_attn_gradients)
        attn = self.attn_drop(attn)
        # print(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        if self.with_qkv:
            x = self.proj(x)
            x = self.proj_drop(x)
        if visualize == False:
            return x
        else:
            return x, attn


class CrossAttention(nn.Module):
    def __init__(self, q_dim, k_dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 with_qkv=True):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        head_dim = k_dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.with_qkv = with_qkv
        self.kv_proj = nn.Linear(k_dim, k_dim * 2, bias=qkv_bias)
        self.q_proj = nn.Linear(q_dim, k_dim)
        self.proj = nn.Linear(k_dim, k_dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.save_attention = False
        self.save_gradients = False

    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients

    def get_attn_gradients(self):
        return self.attn_gradients

    def save_attention_map(self, attention_map):
        self.attention_map = attention_map

    def get_attention_map(self):
        return self.attention_map

    def forward(self, q, k, visualize=True):
        B, N_k, K = k.shape
        _, N_q, _ = q.shape
        kv = self.kv_proj(k).reshape(B, N_k, 2, self.num_heads, K // self.num_heads).permute(2, 0, 3, 1, 4)  #
        k, v = kv[0], kv[1]  # (B,H,N,C)
        q = self.q_proj(q).reshape(B, N_q, self.num_heads, K // self.num_heads).permute(0, 2, 1, 3)  # (B,H,N,C)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        if self.save_attention:
            self.save_attention_map(attn)
        if self.save_gradients:
            attn.register_hook(self.save_attn_gradients)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N_q, K)
        out = self.proj(out)
        out = self.proj_drop(out)
        if visualize == False:
            return out
        else:
            return out, attn


class Block(nn.Module):
    def __init__(self, dim, num_heads, is_cross_attention=False, encoder_dim=None, mlp_ratio=4., qkv_bias=False,
                 qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.scale = 0.5
        self.norm1 = norm_layer(dim)
        self.is_cross_attention = is_cross_attention
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        if self.is_cross_attention:
            self.cross_attn = CrossAttention(
                q_dim=dim, k_dim=encoder_dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                attn_drop=attn_drop, proj_drop=drop)
            self.cross_norm = norm_layer(dim)

        ## drop path
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, encoder_hidden_states=None, mask=None, visualize=True):
        if visualize == False:
            # self attention
            x = x + self.drop_path(self.attn(self.norm1(x), mask=mask))
            # cross attention
            if self.is_cross_attention:
                assert encoder_hidden_states is not None
                x = x + self.drop_path(self.cross_attn(self.cross_norm(x), encoder_hidden_states))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x
        else:
            # import ipdb;ipdb.set_trace();
            tmp, self_attn = self.attn(self.norm1(x), mask=mask, visualize=visualize)
            x = x + self.drop_path(tmp)
            if self.is_cross_attention:
                assert encoder_hidden_states is not None
                tmp, cross_attn = self.cross_attn(self.cross_norm(x), encoder_hidden_states, visualize=visualize)
                x = x + self.drop_path(tmp)
            # import ipdb;ipdb.set_trace();
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x, {'self_attn': self_attn, 'cross_attn': cross_attn if self.is_cross_attention else None}



class TabularTransformerEncoder(nn.Module):
    '''
    Tabular Transformer Encoder based on BERT
    cat_lengths_tabular: categorical feature length list, e.g., [5,4,2]
    con_lengths_tabular: continuous feature length list, e.g., [1,1]
    '''
    def __init__(self, opt_dict, has_fc=True) -> None:
        super(TabularTransformerEncoder, self).__init__()
        self.num_classes = opt_dict['train_config']['num_cls']
        self.field_lengths_tabular = torch.load(opt_dict['dataset_config']['field_lengths_tabular'])
        self.field_lengths_tabular = [int(x) for x in self.field_lengths_tabular]
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
        self.batch_size = opt_dict['train_config']['batch_size']
        self.tabular_embedding_dim = opt_dict['model_config']['tabular_embedding_dim']

        # Categorical embedding
        cat_offsets = torch.tensor([0] + self.cat_lengths_tabular[:-1]).cumsum(0)
        self.register_buffer('cat_offsets', cat_offsets, persistent=False)
        self.cat_embedding = nn.Embedding(self.num_unique_cat, opt_dict['model_config']['tabular_embedding_dim'])
        
        # Continuous embedding
        self.con_proj = nn.Linear(1, opt_dict['model_config']['tabular_embedding_dim'])

        # Class token
        # self.cls_token = nn.Parameter(torch.zeros(1, 1, opt_dict['model_config']['tabular_embedding_dim']))
        # nn.init.xavier_uniform_(self.cls_token)

        self.mask_special_token = nn.Parameter(torch.zeros(1, 1, opt_dict['model_config']['tabular_embedding_dim']))

        # Column embedding
        pos_ids = torch.arange(self.num_cat + self.num_con).expand(1, -1)
        self.register_buffer('pos_ids', pos_ids, persistent=False)
        self.column_embedding = nn.Embedding(self.num_cat + self.num_con, opt_dict['model_config']['tabular_embedding_dim'])

        self.norm = nn.LayerNorm(opt_dict['model_config']['tabular_embedding_dim'])
        self.dropout = nn.Dropout(opt_dict['model_config']['embedding_dropout']) if opt_dict['model_config']['embedding_dropout'] > 0 else nn.Identity()

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            Block(dim=opt_dict['model_config']['tabular_embedding_dim'], num_heads=opt_dict['model_config']['num_heads'], drop=opt_dict['model_config']['drop_rate'], is_cross_attention=False, mlp_ratio=opt_dict['model_config']['mlp_radio'], drop_path=opt_dict['model_config']['drop_path'])
            for _ in range(opt_dict['model_config']['tabular_transformer_num_layers'])
        ])

        self.input_size = opt_dict['model_config']['tabular_embedding_dim'] * (self.num_cat + self.num_con)
        self.hidden2 = opt_dict['model_config']['tab_proj_dim']
        self.proj = nn.Sequential(
            nn.Linear(self.input_size, self.hidden2),
            nn.ReLU(),
            nn.Dropout(p=opt_dict['model_config']['tab_proj_dropout']),
        )

        if has_fc:
            self.fc3 = nn.Linear(self.hidden2, self.num_classes)

    def embedding(self, x, mask_special=None):
        # categorical embedding
        cat_x = self.cat_embedding(x[:, :self.num_cat].long()+self.cat_offsets)
        # continuous embedding
        con_x = self.con_proj(x[:, self.num_cat:].unsqueeze(-1))
        x = torch.cat([cat_x, con_x], dim=1)
        # mask special token
        if mask_special is not None:
            mask_special = mask_special.unsqueeze(-1)
            mask_special_tokens = self.mask_special_token.expand(x.shape[0], x.shape[1], -1)
            x = mask_special*mask_special_tokens + (~mask_special)*x
        # concat
        # cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        # x = torch.cat([cls_tokens, x], dim=1)

        column_embed = self.column_embedding(self.pos_ids)
        x = x+column_embed
        x = self.norm(x)
        x = self.dropout(x)
        return x
    
    def forward(self, x, mask=None, mask_special=None, has_fc=True) -> torch.Tensor:
        x = self.embedding(x, mask_special=mask_special)
        # create attention mask
        if mask is not None:
            B, N = mask.shape
            # cls_mask = torch.zeros(B, 1).bool().to(mask.device)
            # mask = torch.cat([cls_mask, mask], dim=1)
            mask = mask[:,None,:].repeat(1, N, 1)
            mask_eye = ~torch.eye(N).bool().to(mask.device)
            mask_eye = mask_eye[None,:,:]
            mask = mask * mask_eye
            mask = mask[:,None,:,:]
            mask = mask*(-1e9)
            assert x.shape[1] == mask.shape[2]
        for transformer_block in self.transformer_blocks:
            x, _ = transformer_block(x, mask=mask)

        if has_fc:
            # import ipdb;ipdb.set_trace();
            x = self.proj(x.reshape(x.shape[0], -1))  # (B, 16)
            # x = x.reshape(x.shape[0], x.shape[1] * x.shape[2])
            out = self.fc3(x)
        else:
            out = x
        return out, _
    
    # def forward(self, x, mask=None, mask_special=None, has_fc=True, visualize=True) -> torch.Tensor:
    #     x = self.embedding(x, mask_special=mask_special)
    #     attention_weights = []  # 用于保存注意力权重
    #     if mask is not None:
    #         B, N = mask.shape
    #         cls_mask = torch.zeros(B, 1).bool().to(mask.device)
    #         mask = torch.cat([cls_mask, mask], dim=1)
    #         mask = mask[:,None,:].repeat(1, N, 1)
    #         mask_eye = ~torch.eye(N).bool().to(mask.device)
    #         mask_eye = mask_eye[None,:,:]
    #         mask = mask * mask_eye
    #         mask = mask[:,None,:,:]
    #         mask = mask*(-1e9)
    #         assert x.shape[1] == mask.shape[2]

    #     for transformer_block in self.transformer_blocks:
    #         if visualize:
    #             # 获取注意力权重
    #             x, attn_weights = transformer_block(x, mask=mask, visualize=True)
    #             attention_weights.append(attn_weights)
    #         else:
    #             x = transformer_block(x, mask=mask)
        
    #     if has_fc:
    #         x = x.reshape(x.shape[0], x.shape[1] * x.shape[2])
    #         out = self.fc3(x)
    #     else:
    #         out = x
        
    #     if visualize:
    #         return out, attention_weights
    #     else:
    #         return out


class TabularEmbeddingEncoder(nn.Module):
    '''
    Tabular Embedding Encoder based on BERT
    cat_lengths_tabular: categorical feature length list, e.g., [5,4,2]
    con_lengths_tabular: continuous feature length list, e.g., [1,1]
    '''
    def __init__(self, opt_dict, dropout_prob=0.5) -> None:
        super(TabularEmbeddingEncoder, self).__init__()
        self.hidden_size1 = opt_dict['model_config']['hidden_size1']
        self.hidden_size2 = opt_dict['model_config']['hidden_size2']
        self.num_classes = opt_dict['train_config']['num_cls']
        self.field_lengths_tabular = torch.load(opt_dict['dataset_config']['field_lengths_tabular'])
        self.field_lengths_tabular = [int(x) for x in self.field_lengths_tabular]
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
        self.batch_size = opt_dict['train_config']['batch_size']
        self.tabular_embedding_dim = opt_dict['model_config']['tabular_embedding_dim']

        # Categorical embedding
        cat_offsets = torch.tensor([0] + self.cat_lengths_tabular[:-1]).cumsum(0)
        self.register_buffer('cat_offsets', cat_offsets, persistent=False)
        self.cat_embedding = nn.Embedding(self.num_unique_cat, opt_dict['model_config']['tabular_embedding_dim'])

        # Continuous embedding
        self.con_proj = nn.Linear(1, opt_dict['model_config']['tabular_embedding_dim'])

        self.norm = nn.LayerNorm(opt_dict['model_config']['tabular_embedding_dim'])
        self.dropout = nn.Dropout(opt_dict['model_config']['embedding_dropout']) if opt_dict['model_config']['embedding_dropout'] > 0 else nn.Identity()

        self.input_size = opt_dict['model_config']['tabular_embedding_dim'] * (self.num_con + self.num_cat)

        self.fc1 = nn.Linear(self.input_size, self.hidden_size1)
        self.bn1 = nn.BatchNorm1d(self.hidden_size1)
        self.dropout1 = nn.Dropout(p=dropout_prob)
        self.relu = nn.ReLU()

        self.fc2 = nn.Linear(self.hidden_size1, self.hidden_size2)
        self.bn2 = nn.BatchNorm1d(self.hidden_size2)
        self.dropout2 = nn.Dropout(p=dropout_prob)

        self.fc3 = nn.Linear(self.hidden_size2, self.num_classes)

    def embedding(self, x):
        cat_x = self.cat_embedding(x[:, :self.num_cat].long() + self.cat_offsets)

        con_x_input = x[:, self.num_cat:].unsqueeze(-1)
        input_dim = con_x_input.shape[0]

        if not con_x_input.is_contiguous():
            con_x_input = con_x_input.clone().contiguous()
        con_x_input = con_x_input.view(-1, 1)

        con_x = self.con_proj(con_x_input)

        con_x = con_x.view(input_dim, self.num_con, self.tabular_embedding_dim)

        # Concatenate categorical and continuous embeddings
        x = torch.cat([cat_x, con_x], dim=1)

        # Normalize and apply dropout
        x = self.norm(x)
        x = self.dropout(x)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # import ipdb;ipdb.set_trace();
        x = self.embedding(x)
        x_tab = x.view(x.size(0), -1)

        out = self.fc1(x_tab)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout1(out)

        out = self.fc2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout2(out)

        out = self.fc3(out)

        return out


class TabularEmbeddingEncoder2(nn.Module):
    '''
    Tabular Embedding Encoder based on BERT
    cat_lengths_tabular: categorical feature length list, e.g., [5,4,2]
    con_lengths_tabular: continuous feature length list, e.g., [1,1]
    '''
    def __init__(self, opt_dict, dropout_prob=0.5) -> None:
        super(TabularEmbeddingEncoder2, self).__init__()
        self.hidden_size1 = opt_dict['model_config']['hidden_size1']
        self.hidden_size2 = opt_dict['model_config']['hidden_size2']
        self.num_classes = opt_dict['train_config']['num_cls']
        self.field_lengths_tabular = torch.load(opt_dict['dataset_config']['field_lengths_tabular'])
        self.field_lengths_tabular = [int(x) for x in self.field_lengths_tabular]
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
        self.batch_size = opt_dict['train_config']['batch_size']
        self.tabular_embedding_dim = opt_dict['model_config']['tabular_embedding_dim']

        # Embedding
        self.cat_embedding = nn.Embedding(self.num_unique_cat, opt_dict['model_config']['tabular_embedding_dim'])
        self.con_proj = nn.Linear(1, opt_dict['model_config']['tabular_embedding_dim'])

        self.norm = nn.LayerNorm(opt_dict['model_config']['tabular_embedding_dim'])
        self.dropout = nn.Dropout(opt_dict['model_config']['embedding_dropout']) if opt_dict['model_config']['embedding_dropout'] > 0 else nn.Identity()

        self.input_size = opt_dict['model_config']['tabular_embedding_dim'] * (self.num_con + self.num_cat)

        self.fc1 = nn.Linear(self.input_size, self.hidden_size1)
        self.bn1 = nn.BatchNorm1d(self.hidden_size1)
        self.dropout1 = nn.Dropout(p=dropout_prob)
        self.relu = nn.ReLU()

        self.fc2 = nn.Linear(self.hidden_size1, self.hidden_size2)
        self.bn2 = nn.BatchNorm1d(self.hidden_size2)
        self.dropout2 = nn.Dropout(p=dropout_prob)

        self.fc3 = nn.Linear(2048, self.num_classes)

    def embedding(self, x):
        cat_x = self.cat_embedding(x[:, :self.num_cat].long())

        con_x_input = x[:, self.num_cat:].unsqueeze(-1)
        input_dim = con_x_input.shape[0]

        if not con_x_input.is_contiguous():
            con_x_input = con_x_input.clone().contiguous()
        con_x_input = con_x_input.view(-1, 1)

        con_x = self.con_proj(con_x_input)

        con_x = con_x.view(input_dim, self.num_con, self.tabular_embedding_dim)

        # Concatenate categorical and continuous embeddings
        x = torch.cat([cat_x, con_x], dim=1)

        # Normalize and apply dropout
        x = self.norm(x)
        x = self.dropout(x)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        import ipdb;ipdb.set_trace();
        x = self.embedding(x)
        x_tab = x.view(x.size(0), -1)

        out = self.fc1(x_tab)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout1(out)

        out = self.fc2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout2(out)

        out = self.fc4(out)
        out = self.bn4(out)
        out = self.relu(out)
        out = self.dropout4(out)

        out = self.fc3(out)

        return out
    

class MLPEncoder(nn.Module):
    def __init__(self, input_size, opt_dict, dropout_prob=0.5):
        super(MLPEncoder, self).__init__()
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
    


if __name__ == '__main__':
    pass
