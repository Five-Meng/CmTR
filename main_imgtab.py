from train_imgtab import train, train_align, train_distill
from epoch_imgtab import train_epoch
from model_imgtab import img_tab_baseline, img_tab_align, img_tab_distill
from build_dataset_imgtab import ImgTabDataset
import torch
import random
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms
import opt
opt = opt.opt_algorithm()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import warnings
warnings.filterwarnings("ignore")


seed = 2024
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


transform_img_train = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
])
transform_img_test = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
])

##########################################################   Model   ###########################################################
if opt.net_v_imgtab == 'img_tab_base':
    model = img_tab_baseline(opt)
elif opt.net_v_imgtab == 'img_tab_align':
    model = img_tab_align(opt)
elif opt.net_v_imgtab == 'img_tab_distill':
    model = img_tab_distill(opt)

##########################################################   Dataset   #########################################################
if opt.mode == 'train':
    dataset_train = ImgTabDataset(opt_dict, 'train', transform_img_train)
    dataset_test = ImgTabDataset(opt_dict, 'test', transform_img_test)


train_loader = torch.utils.data.DataLoader(dataset_train,
                                               batch_size=opt.batch_size,
                                               shuffle=True,
                                               num_workers=4)

test_loader = torch.utils.data.DataLoader(dataset_test,
                                               batch_size=opt.batch_size,
                                               shuffle=False,
                                               num_workers=4)

##########################################################   Train   ###########################################################
if opt.mode == 'train':
    if opt_dict['train_config']['find_epoch']:
        train_epoch(model, train_loader, test_loader, device, opt)
    else:
        if opt.net_v_imgtab == 'img_tab_base':
            train(model, train_loader, test_loader, device, opt)
        elif opt.net_v_imgtab == 'img_tab_align':
            train_align(model, train_loader, test_loader, device, opt)
        elif opt.net_v_imgtab == 'img_tab_distill':
            train_distill(model, train_loader, test_loader, device, opt)












