import os
import numpy as np
import torch
import torch.utils.data
from build_dataset_tabular import tabular_dataset_dvm
from build_dataset import img_dataset_dvm, img_dataset, get_barcode_img_dict
from build_dataset_2tab import tabular_dataset_2tab


class UnitDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, mode='train', dataset_type='image', mask_version=None):
        if dataset_type == 'image' or dataset_type == 'image_tabular':
            self.images = torch.load(os.path.join(data_dir, f'{mode}_images.pt'))

        self.tabular = torch.load(os.path.join(data_dir, f'{mode}_tabular.pt'))
        self.masks = None
        if dataset_type == 'tabular' or dataset_type == 'image_tabular':
            mask_path = os.path.join(data_dir, f'mask_{mask_version}_{mode}.npy')
            if os.path.exists(mask_path):
                self.masks = np.load(mask_path)
            else:
                raise FileNotFoundError(f"Mask file {mask_path} not found")

        self.barcodes = list(self.tabular.keys())
        self.dataset_type = dataset_type
        self.mask_version = mask_version

    def __len__(self):
        return len(self.barcodes)

    def __getitem__(self, idx):
        barcode = self.barcodes[idx]
        
        if self.dataset_type == 'image':
            img_data = self.images.get(barcode)
            tab_data = self.tabular[barcode]
            return img_data['image'], tab_data['label']
        elif self.dataset_type == 'tabular':
            tab_data = self.tabular[barcode]
            return (
                tab_data['features'],
                tab_data['label'],
                self.masks[idx] 
            )
        elif self.dataset_type == 'image_tabular':
            img_data = self.images.get(barcode)
            tab_data = self.tabular[barcode]
            return (
                img_data['image'],
                tab_data['features'],
                self.masks[idx],
                tab_data['label']
            )

    def get_by_barcode(self, barcode):
        if self.dataset_type == 'image':
            return self.images.get(barcode)
        else:
            data = self.tabular.get(barcode)
            if data and self.masks is not None:
                idx = self.barcodes.index(barcode)
                data['mask'] = self.masks[idx]
            return data


import pandas as pd
def barcode_imgtab_dict_dvm(opt_dict):
    train_dict = {}
    test_dict = {}

    df_dvm = pd.read_csv(opt_dict['dataset_config']['dataset_tabular'], encoding='utf-8')
    # df_dvm = df_dvm.drop(columns=['Unnamed: 0', 'index'])
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
        # if(idx % 1000 == 0):
        #     print(idx)
        adv_id = row['Adv_ID']
        img_name = row['Image_name']
        genmodel_id = row['Genmodel_ID']
        img_path = os.path.join(img_base_path, 'train', str(genmodel_id) + '_' + img_name)
        tab_data = row[need_col]
        tab_data = torch.tensor(tab_data, dtype=torch.float32)
        train_dict[adv_id] = [img_path, tab_data, genmodel_id]

    for idx, row in df_test_dvm.iterrows():
        # if(idx % 1000 == 0):
        #     print(idx)
        adv_id = row['Adv_ID']
        img_name = row['Image_name']
        genmodel_id = row['Genmodel_ID']
        img_path = os.path.join(img_base_path, 'test', str(genmodel_id) + '_' + img_name)
        tab_data = row[need_col]
        tab_data = torch.tensor(tab_data, dtype=torch.float32)
        test_dict[adv_id] = [img_path, tab_data, genmodel_id]

    return train_dict, test_dict




class UnitDataset_dvm(torch.utils.data.Dataset):
    def __init__(self, opt_dict, mode, dataset_type='image_tabular', transform=None):
        self.mode = mode
        self.dataset_type = dataset_type
        self.transform = transform
        if self.dataset_type == 'image':
            self.img_dataset = img_dataset_dvm(opt_dict, mode, transform)
        elif self.dataset_type == 'tabular':
            self.tabular_dataset = tabular_dataset_dvm(opt_dict, mode)
        elif self.dataset_type == 'image_tabular':
            self.img_dataset = img_dataset_dvm(opt_dict, mode, transform)
            self.tabular_dataset = tabular_dataset_dvm(opt_dict, mode)
            train_dict, test_dict = barcode_imgtab_dict_dvm(opt_dict)
            if self.mode in ['train', 'val']:
                self.adv_to_img_tab = train_dict
            else:
                self.adv_to_img_tab = test_dict

    def __len__(self):
        return len(self.tabular_dataset)

    def __getitem__(self, idx): 
        if self.dataset_type == 'image':
            img_data = self.img_dataset[idx]
            tab_data = self.tabular_dataset[idx]
            return img_data[0], tab_data[1]  # image, label

        elif self.dataset_type == 'tabular':
            tab_data = self.tabular_dataset[idx]
            return (
                tab_data[0],  # features
                tab_data[1],  # label
                tab_data[2]   # mask
            )

        elif self.dataset_type == 'image_tabular':
            # import ipdb;ipdb.set_trace();
            tab_data = self.tabular_dataset[idx]
            adv_id = self.tabular_dataset.df.iloc[idx]['Adv_ID']
            if adv_id not in self.adv_to_img_tab:
                raise KeyError(f"Adv_ID {adv_id} not found in image dictionary")
            img_path = self.adv_to_img_tab[adv_id][0]
            from PIL import Image
            image = Image.open(img_path).convert('RGB')
            if self.transform is not None:
                image = self.transform(image)
            return (
                image,         # 图像数据
                tab_data[0],   # features
                tab_data[2],   # mask
                tab_data[1]    # label
            )

        else:
            raise ValueError("Invalid dataset_type. Choose from 'image', 'tabular', or 'image_tabular'.")




import pandas as pd
import os
import torch
from PIL import Image

def barcode_imgtab_dict_blood(opt_dict):
    train_dict = {}
    test_dict = {}
    df1 = pd.read_csv(opt_dict['dataset_config']['dataset_tabular_i'], encoding='utf-8')
    df2 = pd.read_csv(opt_dict['dataset_config']['dataset_tabular_t'], encoding='utf-8')
    
    df_merged = pd.merge(
        df1, 
        df2, 
        on='barcode', 
        suffixes=('_1', '_2'),
        validate='one_to_one'  
    )
    df_merged['target_1'] = df_merged['target_1'].apply(lambda x: 6 if x == 7 else x)

    df_train = df_merged[df_merged['train_val_1'] == 'j'].copy()
    df_test = df_merged[df_merged['train_val_1'] == 'y'].copy()

    train_img_dict, test_img_dict = get_barcode_img_dict(opt_dict['dataset_config']['dataset'])

    def get_valid_features(source_cols, prefix=''):
        exclude_cols = {'train_val', 'target', 'barcode', 'Unnamed: 0'}
        return [
            col for col in source_cols 
            if col.replace(prefix, '') not in exclude_cols
            and not col.endswith(('_1', '_2'))  
        ]

    tab1_cols = get_valid_features(df1.columns)
    tab2_cols = get_valid_features(df2.columns)

    for barcode, img_paths in train_img_dict.items():
        if barcode in df_train['barcode'].values:
            row = df_train[df_train['barcode'] == barcode].iloc[0]
            tab1_data = row[[f"{col}_1" for col in tab1_cols]].values.astype(float)
            tab2_data = row[[f"{col}_2" for col in tab2_cols]].values.astype(float)
            
            train_dict[barcode] = [
                img_paths[0],
                torch.tensor(tab1_data, dtype=torch.float32),
                torch.tensor(tab2_data, dtype=torch.float32),
                torch.tensor(row['target_1'], dtype=torch.long)
            ]

    for barcode, img_paths in test_img_dict.items():
        if barcode in df_test['barcode'].values:
            row = df_test[df_test['barcode'] == barcode].iloc[0]
            
            tab1_data = row[[f"{col}_1" for col in tab1_cols]].values.astype(float)
            tab2_data = row[[f"{col}_2" for col in tab2_cols]].values.astype(float)
            
            test_dict[barcode] = [
                img_paths[0],
                torch.tensor(tab1_data, dtype=torch.float32),
                torch.tensor(tab2_data, dtype=torch.float32),
                torch.tensor(row['target_1'], dtype=torch.long)
            ]
    assert len(train_dict) > 0, "训练集未找到匹配的barcode"
    assert len(test_dict) > 0, "测试集未找到匹配的barcode"
    
    print(f"成功加载 {len(train_dict)} 个训练样本, {len(test_dict)} 个测试样本")
    return train_dict, test_dict



class UnitDataset_blood_2tab(torch.utils.data.Dataset):
    def __init__(self, opt_dict, mode, transform=None):
        self.mode = mode
        self.transform = transform
        # import ipdb;ipdb.set_trace();
        self.tabular_dataset = tabular_dataset_2tab(opt_dict, mode)
        self.img_dataset = img_dataset(opt_dict, mode, transform)
        self.barcode_dict_train, self.barcode_dict_test = barcode_imgtab_dict_blood(opt_dict)
        # import ipdb;ipdb.set_trace();
        # self.valid_indices = [
        #     i for i in range(len(self.tabular_dataset)) 
        #     if str(self.tabular_dataset.df.iloc[i]['barcode']) in self.barcode_dict
        # ]
        if mode == 'train':
            self.barcode_dict = self.barcode_dict_train
        else:
            self.barcode_dict = self.barcode_dict_test

        self.valid_indices = []
        for i in range(len(self.tabular_dataset)):
            if str(self.tabular_dataset.df.iloc[i]['barcode']) in self.barcode_dict:
                self.valid_indices.append(i)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        # import ipdb;ipdb.set_trace();
        real_idx = self.valid_indices[idx]
        barcode = str(self.tabular_dataset.df.iloc[real_idx]['barcode'])
        tab1, tab2, label, mask = self.tabular_dataset[real_idx]
        
        img_path = self.barcode_dict[barcode][0]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, tab1, tab2, mask, label






if __name__ == '__main__':
    from argparses.util.yaml_args import yaml_data
    opt_dict = yaml_data("/home/admin1/User/mxy/demo/argparses/blood_mlp.yaml")

    dataset_dir = opt_dict['dataset_config']['data_dir']

    import ipdb;ipdb.set_trace();
    # img_train = UnitDataset(dataset_dir, mode='train', dataset_type='image', mask_version=None)

    # train_loader = torch.utils.data.DataLoader(img_train,
    #                                            batch_size=opt_dict['train_config']['batch_size'],
    #                                            shuffle=True,
    #                                            num_workers=4)
    
    data_train = UnitDataset(dataset_dir, mode='train', dataset_type='image_tabular', mask_version="0.3")

    train_loader = torch.utils.data.DataLoader(data_train,
                                               batch_size=opt_dict['train_config']['batch_size'],
                                               shuffle=True,
                                               num_workers=4)
    for idx, data in enumerate(train_loader):
        import ipdb;ipdb.set_trace();
        print(idx)
