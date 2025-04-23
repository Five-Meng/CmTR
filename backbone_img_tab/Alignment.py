import torch.nn as nn

class AlignmentModel(nn.Module):
    def __init__(self, opt_dict, img_dim=1280, tab_dim=128):
        super().__init__()
        latent_dim = opt_dict['model_config']['latent_dim']
        # re_dim = opt_dict['model_config']['re_dim']
        self.img_proj = nn.Sequential(
            nn.Linear(img_dim, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim),
            nn.ReLU(),
        )
        
        self.tab_proj = nn.Sequential(
            nn.Linear(tab_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
            nn.ReLU(),
        )

    def forward(self, img_feat, tab_feat):
        return self.img_proj(img_feat), self.tab_proj(tab_feat)