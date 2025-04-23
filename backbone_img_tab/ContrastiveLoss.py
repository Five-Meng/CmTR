import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1, pos_margin=0.0, neg_margin=1.0):
        super().__init__()
        self.temperature = temperature
        self.pos_margin = pos_margin
        self.neg_margin = neg_margin
        
    def forward(self, img_emb, tab_emb):
        img_emb = F.normalize(img_emb, p=2, dim=-1).detach()
        tab_emb = F.normalize(tab_emb, p=2, dim=-1)
        
        sim_matrix = torch.matmul(tab_emb, img_emb.T) / self.temperature
        
        labels = torch.arange(img_emb.size(0)).to(img_emb.device)
        
        loss = F.cross_entropy(sim_matrix, labels)
        
        pos_sim = torch.diag(sim_matrix)
        loss += F.relu(self.pos_margin - pos_sim).mean()
        
        neg_mask = ~torch.eye(len(sim_matrix), dtype=bool).to(img_emb.device)
        neg_sim = sim_matrix.masked_select(neg_mask)
        loss += F.relu(neg_sim - self.neg_margin).mean()
        
        return loss
    
