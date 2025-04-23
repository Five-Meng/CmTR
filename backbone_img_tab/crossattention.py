import torch.nn as nn
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
# import seaborn as sns
    

class CrossAttention(nn.Module):
    def __init__(self, dim, drop_rate=0.1):
        super(CrossAttention, self).__init__()
        self.query_proj = nn.Linear(dim, dim)
        self.key_proj = nn.Linear(dim, dim)
        self.value_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(drop_rate)
        self.scale = dim ** -0.5

    def forward(self, query, key, value):
        Q = self.query_proj(query)  
        K = self.key_proj(key)      
        V = self.value_proj(value)  

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        attn_out = torch.matmul(attn_probs, V) 
        out = self.out_proj(attn_out)
        return out + query
        # return out




    

