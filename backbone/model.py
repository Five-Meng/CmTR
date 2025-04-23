from utils.losses import MetaLinear_Norm
from .resnet import resnet34,resnet18, resnet50
from .vit import vit_base_patch16_224, vit_base_patch16_224_in21k, vit_base_patch32_224, vit_base_patch32_224_in21k
from .efficientnet import efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3
from .efficientnetV2 import efficientnetv2_s
import torch
import torch.nn as nn
from .convnext import convnext_base
from .shufflenet import shufflenet_v2_x1_0, shufflenet_v2_x1_5, shufflenet_v2_x2_0
from .vit2 import VisionTransformer

def build_model(opt_dict):
    model = None
    print(opt_dict['model_config']['net_v'])

    if opt_dict['model_config']['net_v'] == 'resnet34':
        model_class = resnet34
        net_pretrain_path = '/data/blood_dvm/data/pretrain/resnet34-333f7ec4.pth'
    elif opt_dict['model_config']['net_v'] == 'resnet18':
        model_class = resnet18
        net_pretrain_path = '/data/blood_dvm/data/pretrain/resnet18-5c106cde.pth'
    elif opt_dict['model_config']['net_v'] == 'resnet50':
        model_class = resnet50
        net_pretrain_path = '/data/blood_dvm/data/pretrain/resnet50-19c8e357.pth'

    elif opt_dict['model_config']['net_v'] == 'vit16':
        model_class = vit_base_patch16_224
        net_pretrain_path = '/data/blood_dvm/data/pretrain/vit_base_patch16_224.pth'
    elif opt_dict['model_config']['net_v'] == 'vit16_in21k':
        model_class = vit_base_patch16_224_in21k
        net_pretrain_path = '/data/blood_dvm/data/pretrain/jx_vit_base_patch16_224_in21k-e5005f0a.pth'
    elif opt_dict['model_config']['net_v'] == 'vit32':
        model_class = vit_base_patch32_224
        net_pretrain_path = '/data/blood_dvm/data/pretrain/vit_base_patch32_224.pth'
    elif opt_dict['model_config']['net_v'] == 'vit32_in21k':
        model_class = vit_base_patch32_224_in21k
        net_pretrain_path = '/data/blood_dvm/data/pretrain/jx_vit_base_patch32_224_in21k-8db57226.pth'
    elif opt_dict['model_config']['net_v'] == 'vit_new':
        model_class = VisionTransformer
        net_pretrain_path = '/data/blood_dvm/data/pretrain/imagenet21k+imagenet2012_ViT-B_32.pth'
    

    elif opt_dict['model_config']['net_v'] == 'efficientnet-b0':
        model_class = efficientnet_b0
        net_pretrain_path = '/data/blood_dvm/data/pretrain/efficientnet-b0-355c32eb.pth'
    elif opt_dict['model_config']['net_v'] == 'efficientnet-b1':
        model_class = efficientnet_b1
        net_pretrain_path = '/data/blood_dvm/data/pretrain/efficientnet-b1.pth'
    elif opt_dict['model_config']['net_v'] == 'efficientnet-b2':
        model_class = efficientnet_b2
        net_pretrain_path = '/data/blood_dvm/data/pretrain/efficientnet-b2.pth'
    elif opt_dict['model_config']['net_v'] == 'efficientnet-b3':
        model_class = efficientnet_b3
        net_pretrain_path = '/data/blood_dvm/data/pretrain/efficientnet-b3.pth'

    elif opt_dict['model_config']['net_v'] == 'efficientnetv2_s':
        model_class = efficientnetv2_s
        net_pretrain_path = '/data/blood_dvm/data/pretrain/pre_efficientnetv2-s.pth'
    elif opt_dict['model_config']['net_v'] == 'efficientnetv2_m':
        model_class = efficientnetv2_s
        net_pretrain_path = '/data/blood_dvm/data/pretrain/pre_efficientnetv2-m.pth'

    elif opt_dict['model_config']['net_v'] == 'convnext_base':
        model_class = convnext_base
        net_pretrain_path = '/data/blood_dvm/data/pretrain/convnext_base_1k_224_ema.pth'

    elif opt_dict['model_config']['net_v'] == 'shufflenet':
        # model_class = shufflenet_v2_x1_0
        model_class = shufflenet_v2_x1_5
        # net_pretrain_path = '/data/blood_dvm/data/pretrain/shufflenetv2_x1-5666bf0f80.pth'
        net_pretrain_path = '/data/blood_dvm/data/pretrain/shufflenetv2_x1_5-3c479a10.pth'



    if opt_dict['train_config']['mode'] == 'train':
        if opt_dict['model_config']['net_v'] in ['resnet18', 'resnet34', 'resnet50']:
            model = model_class()
            model.load_state_dict(torch.load(net_pretrain_path, map_location='cpu'))
            if opt_dict['train_config']['lossfunc'] == 'ldamloss':
                print(f"loss is {opt_dict['train_config']['lossfunc']}")
                model.fc = MetaLinear_Norm(model.fc.in_features, opt_dict['train_config']['num_cls'])
            else:
                model.fc = nn.Linear(model.fc.in_features, opt_dict['train_config']['num_cls'])


        elif opt_dict['model_config']['net_v'] in ['vit16', 'vit16_in21k', 'vit32', 'vit32_in21k']:
            model = model_class(num_classes=opt_dict['train_config']['num_cls']) if opt_dict['model_config']['net_v'] in ['vit16', 'vit32'] else model_class(
                num_classes=opt_dict['train_config']['num_cls'], has_logits=False)
            # weights_dict = torch.load(net_pretrain_path, map_location='cpu')

            # if opt_dict['model_config']['freeze_layers']:
            #     del_keys = ['head.weight', 'head.bias']

            #     if opt_dict['model_config']['net_v'] == 'vit16_in21k' or opt_dict['model_config']['net_v'] == 'vit32_in21k':
            #         del_keys += ['pre_logits.fc.weight', 'pre_logits.fc.bias']

            #     for k in del_keys: 
            #         del weights_dict[k]

            #     # print(model.load_state_dict(weights_dict, strict=False))
            #     print(model.load_state_dict(weights_dict, strict=True))

            #     for name, para in model.named_parameters():
            #         if "head" not in name and ("pre_logits" not in name if model_class == 'vit16_in21k' else True):
            #             para.requires_grad_(False)
            #         else:
            #             print(f"Training {name}")
        
        elif opt_dict['model_config']['net_v'] in ['vit_new']:
            model = VisionTransformer(num_classes=opt_dict['train_config']['num_cls'], patch_size=(32, 32), num_heads=12, num_layers=12)
            # import ipdb;ipdb.set_trace();
            weights_dict = torch.load(net_pretrain_path, map_location='cpu')['state_dict']
            model_dict = model.state_dict()
            shared_dict = {k: v for k, v in weights_dict.items() if k in model_dict}
            model_dict.update(shared_dict)
            model.load_state_dict(model_dict, strict=False)
            model.classifier = nn.Linear(model.emb_dim, opt_dict['train_config']['num_cls'])


        elif opt_dict['model_config']['net_v'] in ['efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2', 'efficientnet-b3']:
            model = model_class(num_classes=opt_dict['train_config']['num_cls'])
            if not opt_dict['H2T']['h2t']:
                weights_dict = torch.load(net_pretrain_path, map_location='cpu')
                load_weights_dict = {k: v for k, v in weights_dict.items() if
                                    k in model.state_dict() and model.state_dict()[k].numel() == v.numel()}
                print(model.load_state_dict(load_weights_dict, strict=False))
                


        elif opt_dict['model_config']['net_v'] in ['efficientnetv2_s', 'efficientnetv2_m']:
            model = model_class(num_classes=opt_dict['train_config']['num_cls'])
            weights_dict = torch.load(net_pretrain_path, map_location='cpu')
            load_weights_dict = {k: v for k, v in weights_dict.items() if model.state_dict()[k].numel() == v.numel()}
            print(model.load_state_dict(load_weights_dict, strict=False))


        elif opt_dict['model_config']['net_v'] in ['convnext_base']:
            model = model_class(num_classes=opt_dict['train_config']['num_cls'])
            weights_dict = torch.load(net_pretrain_path, map_location='cpu')["model"]
            for k in list(weights_dict.keys()):
                if "head" in k:
                    del weights_dict[k]
            print(model.load_state_dict(weights_dict, strict=False))


        elif opt_dict['model_config']['net_v'] == 'shufflenet':
            model = model_class(num_classes=opt_dict['train_config']['num_cls'])
        #     weights_dict = torch.load(net_pretrain_path, map_location='cpu')
        #     load_weights_dict = {k: v for k, v in weights_dict.items()
        #                          if model.state_dict()[k].numel() == v.numel()}
        #     # print(model.load_state_dict(load_weights_dict, strict=False))

        #     for name, para in model.named_parameters():
        #         if "fc" not in name:
        #             para.requires_grad_(False)


        # print(opt_dict['train_config']['mode'])

    if model is None:
        raise ValueError("模型未成功初始化")

    return model


def build_model_h2t(opt_dict):

    if opt_dict['model_config']['net_v'] == 'resnet18':
        print("resnet18_h2t")
        model = resnet18()
        net_pretrain_path = '/data/blood_dvm/data/pretrain/resnet18-5c106cde.pth'
        model.load_state_dict(torch.load(net_pretrain_path, map_location='cpu'), strict=False)
        model.fc = nn.Linear(model.fc.in_features, opt_dict['train_config']['num_cls'])

    if model is None:
        raise ValueError("模型未成功初始化")

    return model



