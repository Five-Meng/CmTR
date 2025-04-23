import numpy as np
import torch
import torch.nn as nn

#------- Mixup -------
def mixup_data(x, y, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    lam = np.random.beta(1, 1)

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam, extra_info=None):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

#------- H2T -------
def H2T(x1, x2, rho = 0.3):          
    if len(x1.shape) == 4:
        fea_num = x1.shape[1]
        index = torch.randperm(fea_num).cuda()
        slt_num = int(rho*fea_num)
        index = index[:slt_num]
        x1[:,index,:,:] = x2[:,index,:,:] 
                          #torch.rand(x2[:,index,:,:].shape).to(x2.device) #torch.zeros(x2[:,index,:,:].shape).to(x2.device) 
                          #x2[:,index,:,:]
    else:
        for i in range(len(x1)):
            fea_num = x1[i].shape[1]
            index = torch.randperm(fea_num).cuda()
            slt_num = int(rho*fea_num)
            index = index[:slt_num]
            x1[i][:,index,:,:] = x2[i][:,index,:,:]    
    return x1



#------- CMO -------
def get_weighted_sampler(cls_num_list, weighted_alpha, labels):
    cls_weight = 1.0 / (np.array(cls_num_list) ** weighted_alpha)
    cls_weight = cls_weight / np.sum(cls_weight) * len(cls_num_list)
    samples_weight = np.array([cls_weight[t] for t in labels])
    samples_weight = torch.from_numpy(samples_weight)
    samples_weight = samples_weight.double()
    sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(labels), replacement=True)
    return sampler

def cmo_build_loader(dataset_train, cls_num_list, args):
    weighted_sampler = get_weighted_sampler(cls_num_list, args.cmo_weighted_alpha, dataset_train.targets)
    weighted_train_loader = torch.utils.data.DataLoader(
            dataset_train, batch_size=args.batch_size,
            num_workers=2, pin_memory=True, sampler=weighted_sampler)
    return weighted_train_loader

def cmo_condition(args, epoch):
    if args.data_aug == 'cmo' and args.start_data_aug < epoch < (args.epochs - args.end_data_aug):
        return True
    else:
        return False

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def rand_bbox_withcenter(size, lam, cx, cy):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def cmo_mix_sample(args, input1, input2):
    # generate mixed sample
    lam = np.random.beta(1, 1)
    bbx1, bby1, bbx2, bby2 = rand_bbox(input1.size(), lam)
    input1[:, :, bbx1:bbx2, bby1:bby2] = input2[:, :, bbx1:bbx2, bby1:bby2]
    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input1.size()[-1] * input1.size()[-2]))
    return input1, lam
    
