from build_dataset_tabular import tabular_dataset, tabular_dataset_dvm
from backbone_tabular.TabularEncoder import MLPEncoder, TabularTransformerEncoder, TabularEmbeddingEncoder
from backbone_tabular.TabularEncoder2 import TabularEncoder
import torch

def build_model(opt_dict):

    model = None
    if opt_dict['model_config']['net_v_tabular'] == 'mlp':
        print("================= mlp =================")
        input_size = 41
        model = MLPEncoder(input_size, opt_dict)

    elif opt_dict['model_config']['net_v_tabular'] == 'encoder_mlp_embedding':
        print("================= encoder_mlp_embedding =================")
        model = TabularEmbeddingEncoder(opt_dict)
        
    elif opt_dict['model_config']['net_v_tabular'] == 'encoder_mlp_block':
        print("================= encoder_mlp_block =================")
        model = TabularTransformerEncoder(opt_dict)

    elif opt_dict['model_config']['net_v_tabular'] == 'encoder_newmlp':
        print("================= encoder_newmlp =================")
        model = TabularEncoder(opt_dict)
        checkpoint_feature = opt_dict['dataset_config']['load_checkpoint_feature']
        checkpoint_fc = opt_dict['dataset_config']['load_checkpoint_fc']

        if opt_dict['dataset_config']['load_checkpoint_feature']:
            # import ipdb;ipdb.set_trace();
            print(checkpoint_feature)
            checkpoint_feature = torch.load(checkpoint_feature)
            # new_checkpoint_feature = {k: v for k, v in checkpoint_feature.items() if 'fc' not in k}
            model.load_state_dict(checkpoint_feature, strict=True)
            if opt_dict['dataset_config']['load_checkpoint_fc']:
                print(checkpoint_fc)
                checkpoint_fc = torch.load(checkpoint_fc)
                model.fc.load_state_dict(checkpoint_fc, strict=True)
            
            for name, param in model.named_parameters():
                if 'fc' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = True

    if model == None:
        print("输入的tabular模型有问题")
        
    return model



