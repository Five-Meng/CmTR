from build_dataset import get_barcode_img_dict
import pandas as pd
from PIL import Image
import torch
import os
from torch.utils.data import Dataset
from torchvision import transforms


def barcode_imgtab_dict(opt_dict):
    import ipdb; ipdb.set_trace();
    train_barcode_dict, test_barcode_dict = get_barcode_img_dict(opt_dict['dataset_config']['dataset'])
    df = pd.read_csv(opt_dict['dataset_config']['dataset_tabular'],  encoding='utf-8')
    df_train = df[df['train_val'] == 'j'].drop(columns=['train_val', 'Unnamed: 0'])
    df_test = df[df['train_val'] == 'y'].drop(columns=['train_val', 'Unnamed: 0'])
    for i in range(len(df_train)):
        row = df_train.iloc[i]
        barcode = row['barcode']
        target = row['target']
        X = row[0: -2]
        X = torch.tensor(X, dtype=torch.float32)
        if barcode in train_barcode_dict.keys():
            train_barcode_dict[barcode].append(X)
            train_barcode_dict[barcode].append(target)
        else:
            print(f"barcode: {barcode}不存在!")

    for i in range(len(df_test)):
        row = df_test.iloc[i]
        barcode = row['barcode']
        target = row['target']
        X = row[1: -2]
        X = torch.tensor(X, dtype=torch.float32)
        if barcode in test_barcode_dict.keys():
            test_barcode_dict[barcode].append(X)
            test_barcode_dict[barcode].append(target)
        else:
            print(f"barcode: {barcode}不存在!")

    return train_barcode_dict, test_barcode_dict


def barcode_imgtab_dict_dvm(opt_dict):
    train_dict = {}
    test_dict = {}

    df_dvm = pd.read_csv(opt_dict['dataset_config']['dataset_tabular'], encoding='utf-8')
    df_dvm = df_dvm.drop(columns=['Unnamed: 0', 'index'])
    # '/data/dvm/dvm_img/'
    img_base_path = opt_dict['dataset_config']['dataset']

    df_train_dvm = df_dvm[df_dvm['train_val'] == 'j']
    df_test_dvm = df_dvm[df_dvm['train_val'] == 'y']

    need_col = ['Adv_year',
            'Adv_month',
            'Reg_year',
            'Runned_Miles',
            'Price',
            'Seat_num',
            'Door_num',
            'Entry_price', 
            'Engine_size',
            'Color',
            'Bodytype',
            'Gearbox',
            'Fuel_type',
            'Genmodel_ID',
            'Wheelbase',
            'Length',
            'Width',
            'Height']

    for idx, row in df_train_dvm.iterrows():
        if(idx % 1000 == 0):
            print(idx)
        adv_id = row['Adv_ID']
        img_name = row['Image_name']
        genmodel_id = row['Genmodel_ID']
        img_path = os.path.join(img_base_path, 'train', str(genmodel_id) + '_' + img_name)
        tab_data = row[need_col]
        tab_data = torch.tensor(tab_data, dtype=torch.float32)
        train_dict[adv_id] = [img_path, tab_data, genmodel_id]

    for idx, row in df_test_dvm.iterrows():
        if(idx % 1000 == 0):
            print(idx)
        adv_id = row['Adv_ID']
        img_name = row['Image_name']
        genmodel_id = row['Genmodel_ID']
        img_path = os.path.join(img_base_path, 'test', str(genmodel_id) + '_' + img_name)
        tab_data = row[need_col]
        tab_data = torch.tensor(tab_data, dtype=torch.float32)
        test_dict[adv_id] = [img_path, tab_data, genmodel_id]

    return train_dict, test_dict

# class ImgTabDataset(Dataset):
#     def __init__(self, opt_dict, mode, transform=None):

#         self.mode = mode
#         self.transform = transform

#         if opt_dict['dataset_config']['dataname'] == 'blood':
#             self.train_dict, self.test_dict = barcode_imgtab_dict(opt)
#         elif opt_dict['dataset_config']['dataname'] == 'dvm':
#             self.train_dict, self.test_dict = barcode_imgtab_dict_dvm(opt)

#         self.data_dict = self.train_dict if self.mode == 'train' else self.test_dict

#         self.img_pic = []
#         self.img_tabular_data = []
#         self.labels = []

#         for barcode, data in self.data_dict.items():
#             img_path = data[0]  
#             tabular_data = data[1]  
#             label = data[2]  

#             # 加载图像
#             image_pic = self.loader(img_path)

#             if self.transform is not None:
#                 image_pic = self.transform(image_pic)
#             if isinstance(image_pic, Image.Image): 
#                 image_pic = transforms.ToTensor()(image_pic)

#             self.img_pic.append(image_pic)

#             self.img_tabular_data.append(tabular_data)

#             self.labels.append(int(label))

#         self.labels_tensor = torch.tensor(self.labels, dtype=torch.long)

#     def loader(self, image_path):
#         return Image.open(image_path).convert('RGB')

#     def __getitem__(self, index):
#         return (self.img_pic[index], self.img_tabular_data[index], self.labels_tensor[index])

#     def __len__(self):
#         return len(self.labels_tensor)


class ImgTabDataset(Dataset):
    def __init__(self, opt_dict, mode, transform=None):

        self.mode = mode
        self.transform = transform

        if opt_dict['dataset_config']['dataname'] == 'blood':
            self.train_dict, self.test_dict = barcode_imgtab_dict(opt_dict)
        elif opt_dict['dataset_config']['dataname'] == 'dvm':
            self.train_dict, self.test_dict = barcode_imgtab_dict_dvm(opt_dict)

        self.data_dict = self.train_dict if self.mode == 'train' else self.test_dict

        self.barcodes = list(self.data_dict.keys())

    def loader(self, image_path):
        return Image.open(image_path).convert('RGB')

    def __getitem__(self, index):

        barcode = self.barcodes[index]
        img_path, tabular_data, label = self.data_dict[barcode]

        image_pic = self.loader(img_path)
        if self.transform is not None:
            image_pic = self.transform(image_pic)
        if isinstance(image_pic, Image.Image): 
            image_pic = transforms.ToTensor()(image_pic)

        tabular_tensor = torch.tensor(tabular_data, dtype=torch.float32)
        label_tensor = torch.tensor(int(label), dtype=torch.long)

        return image_pic, tabular_tensor, label_tensor

    def __len__(self):
        return len(self.barcodes)



if __name__ == '__main__':
    from argparses.yaml_args import yaml_data
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml_config', type=str, help='config args')
    args = parser.parse_args()
    opt_dict = yaml_data(args.yaml_config)
    
    train_dict, test_dict = barcode_imgtab_dict(opt_dict)
    import ipdb; ipdb.set_trace();

    transform_img_train = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
    ])
    transform_img_test = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
    ])

    dataset_train = ImgTabDataset(opt_dict, 'train', transform_img_train)
    import ipdb; ipdb.set_trace();
    dataset_test = ImgTabDataset(opt_dict, 'test', transform_img_test)

    train_loader = torch.utils.data.DataLoader(dataset_train,
                                               batch_size=opt_dict['train_config']['batch_size'],
                                               shuffle=True,
                                               num_workers=4)

    test_loader = torch.utils.data.DataLoader(dataset_test,
                                               batch_size=opt_dict['train_config']['batch_size'],
                                               shuffle=False,
                                               num_workers=4)
    
    for idx, (image_pic, tabular_tensor, label_tensor) in enumerate(train_loader):
        import ipdb; ipdb.set_trace();
