import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np


class tabular_dataset(Dataset):
    def __init__(self, opt_dict, mode):
        self.excel_path = opt_dict['dataset_config']['dataset_tabular']
        print(f"tabular_path: {self.excel_path}")
        self.df = pd.read_csv(self.excel_path, encoding='utf-8')
        self.opt_dict = opt_dict

        if mode == 'train':
            self.df = self.df[self.df['train_val'] == 'j']
            if opt_dict['dataset_config']['missing_mask_train']:
                self.mask_special = np.load(opt_dict['dataset_config']['missing_mask_train'])
        else:
            self.df = self.df[self.df['train_val'] == 'y']
            if opt_dict['dataset_config']['missing_mask_test']:
                self.mask_special = np.load(opt_dict['dataset_config']['missing_mask_test'])

        self.df_y = self.df['target']
        self.df_X = self.df.drop(columns=['train_val', 'target'])
        if 'Unnamed: 0' in self.df_X.columns:
            self.df_X = self.df_X.drop(columns=['Unnamed: 0'])
        if 'barcode' in self.df_X.columns:
            self.df_X = self.df_X.drop(columns=['barcode'])

        self.input_size = len(self.df_X.columns)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df_X.iloc[index]
        X = row.values
        y = self.df_y.iloc[index]
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)
        # import ipdb;ipdb.set_trace();
        return X_tensor, y_tensor, self.mask_special[index]


class tabular_dataset_dvm(Dataset):
    def __init__(self, opt_dict, mode):
        self.dataset_path = opt_dict['dataset_config']['dataset_tabular']
        self.df = pd.read_csv(self.dataset_path)
        
        if mode == 'train':
            self.df = self.df[self.df['train_val'] == 'j'].drop(columns=['train_val'])
            if opt_dict['dataset_config']['missing_mask_train']:
                self.mask_special = np.load(opt_dict['dataset_config']['missing_mask_train'])
        elif mode == 'val':
            self.df = self.df[self.df['train_val'] == 'v'].drop(columns=['train_val'])
            if opt_dict['dataset_config']['missing_mask_val']:
                self.mask_special = np.load(opt_dict['dataset_config']['missing_mask_val'])
        elif mode == 'test':
            self.df = self.df[self.df['train_val'] == 'y'].drop(columns=['train_val'])
            if opt_dict['dataset_config']['missing_mask_test']:
                self.mask_special = np.load(opt_dict['dataset_config']['missing_mask_test'])

        self.df_y = self.df['Genmodel_ID']
        self.df_X = self.df.drop(columns=['Genmodel_ID'])
            
        if 'target' in self.df_X.columns:
            self.df_X = self.df_X.drop(columns=['target'])
        if 'Adv_ID' in self.df_X.columns:
            self.df_X = self.df_X.drop(columns=['Adv_ID']) 
        if 'Image_name' in self.df_X.columns:
            self.df_X = self.df_X.drop(columns=['Image_name'])
        if  'Unnamed: 0' in self.df_X.columns:
            self.df_X = self.df_X.drop(columns=['Unnamed: 0'])
        self.input_size = len(self.df_X.columns)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # import ipdb;ipdb.set_trace();
        row = self.df_X.iloc[index]
        X = row.values
        y = self.df_y.iloc[index]
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)
        return X_tensor, y_tensor, self.mask_special[index]





if __name__ == '__main__':
    pass
    # import ipdb; ipdb.set_trace()
    # img_dataset = tabular_dataset(opt_dict, 'train')
    # import ipdb; ipdb.set_trace()
    # print(img_dataset.__getitem__(1))
    # print(img_dataset.__len__())