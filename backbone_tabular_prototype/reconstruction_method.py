import torch
import torch.nn as nn
import torch.nn.functional as F


class Transformer_Embedding_R(nn.Module):
    '''Masked Tabular Reconstruction'''
    def __init__(self, opt_dict) -> None:
        super(Transformer_Embedding_R, self).__init__()
        self.opt_dict = opt_dict
        self.field_lengths_tabular = torch.load(opt_dict['dataset_config']['field_lengths_tabular'])
        self.field_lengths_tabular = [int(x) for x in self.field_lengths_tabular]
        self.cat_lengths_tabular = []
        self.con_lengths_tabular = []
        self.hidden_size1 = opt_dict['model_config']['hidden_size1']
        self.hidden_size2 = opt_dict['model_config']['hidden_size2']
        for x in self.field_lengths_tabular:
            if x == 1:
                self.con_lengths_tabular.append(x)
            else:
                self.cat_lengths_tabular.append(x)

        self.num_cat = len(self.cat_lengths_tabular) 
        self.num_con = len(self.con_lengths_tabular)  
        self.num_unique_cat = sum(self.cat_lengths_tabular)  
        
        # categorical classifier
        self.cat_classifier = nn.Linear(opt_dict['model_config']['tabular_embedding_dim'], self.num_unique_cat, bias=True)
        # continuous regessor
        self.con_regressor = nn.Linear(opt_dict['model_config']['tabular_embedding_dim'], 1, bias=True)

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module, init_gain=0.02) -> None:
        """
        Initializes the weights of the neural network layers according to the desired strategy.
        Args:
            m (nn.Module): The module (layer) to initialize.
            init_gain (float): The gain parameter used for initialization (default is 0.02).
        """
        if isinstance(m, nn.Linear):
            if self.opt_dict['model_config']['init_strat'] == 'normal':
                nn.init.normal_(m.weight.data, mean=0, std=0.001)
            elif self.opt_dict['model_config']['init_strat'] == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif self.opt_dict['model_config']['init_strat'] == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif self.opt_dict['model_config']['init_strat'] == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=init_gain)

            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # remove clstoken
        x = x[:,1:,:]
        # categorical classifier
        cat_x = self.cat_classifier(x[:, :self.num_cat])
        # continuous regessor
        con_x = self.con_regressor(x[:, self.num_cat:])
        return (cat_x, con_x)    


class MLP_Embedding_R(nn.Module):
    def __init__(self, opt_dict):
        super(MLP_Embedding_R, self).__init__()
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
        if('hidden_size3' in opt_dict['model_config']):
            self.hidden_size = opt_dict['model_config']['hidden_size3']
        else:
            self.hidden_size = opt_dict['model_config']['hidden_size2']

        # if(opt_dict['dataset_config']['dataname'] == 'blood'):
        #     # blood, 22, 41
        #     self.decoder_cont_mlp = nn.Sequential(
        #         nn.Linear(self.hidden_size, self.num_con)  
        #     )

        #     self.decoder_cat_mlp = nn.Sequential(
        #         nn.Linear(self.hidden_size, sum(self.cat_lengths_tabular))  
        #     )
        # else:
        #     # lr=5e-5
        #     dropout_rate = 0.2     
        #     # dim = 512
        #     dim = 1024     
        #     self.decoder_cont_mlp = nn.Sequential(
        #         nn.Linear(self.hidden_size, dim),
        #         nn.ReLU(),
        #         nn.Dropout(dropout_rate),
        #         nn.Linear(dim, self.num_con), 
        #     )
        #     self.decoder_cat_mlp = nn.Sequential(
        #         nn.Linear(self.hidden_size, dim),
        #         nn.ReLU(),
        #         nn.Dropout(dropout_rate),
        #         nn.Linear(dim, sum(self.cat_lengths_tabular)),
        #     )
        self.decoder_cont_mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.num_con)  
        )

        self.decoder_cat_mlp = nn.Sequential(
            nn.Linear(self.hidden_size, sum(self.cat_lengths_tabular))  
        )
        self._init_weights()


    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        nn.init.normal_(self.decoder_cont_mlp[-1].weight, std=0.01) 
        nn.init.normal_(self.decoder_cat_mlp[-1].weight, std=0.1)   


    def forward(self, z):
        # import ipdb;ipdb.set_trace();
        shared_rep = z
        x_cont_recon = self.decoder_cont_mlp(shared_rep)
        cat_logits = self.decoder_cat_mlp(shared_rep)
        x_cat_recon = cat_logits

        return x_cont_recon, x_cat_recon
    
    
    
