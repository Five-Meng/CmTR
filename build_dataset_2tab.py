import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np


class tabular_dataset_2tab(Dataset):
    def __init__(self, opt_dict, mode):
        self.excel_path1 = opt_dict['dataset_config']['dataset_tabular_i']  
        self.excel_path2 = opt_dict['dataset_config']['dataset_tabular_t'] 
        print(f"tabular_path1: {self.excel_path1}")
        print(f"tabular_path2: {self.excel_path2}")
        
        self.df1 = pd.read_csv(self.excel_path1, encoding='utf-8')
        self.df2 = pd.read_csv(self.excel_path2, encoding='utf-8')
        
        self.df = pd.merge(self.df1, self.df2, on='barcode', suffixes=('_1', '_2'))
        self.opt_dict = opt_dict

        if mode == 'train':
            self.df = self.df[self.df['train_val_1'] == 'j'] 
            if opt_dict['dataset_config']['missing_mask_train']:
                self.mask_special = np.load(opt_dict['dataset_config']['missing_mask_train'])
        else:
            self.df = self.df[self.df['train_val_1'] == 'y'] 
            if opt_dict['dataset_config']['missing_mask_test']:
                self.mask_special = np.load(opt_dict['dataset_config']['missing_mask_test'])

        self.df_y = self.df['target_1']

        df_X1 = self.df[[col for col in self.df.columns if col.endswith('_1')]]
        df_X1.columns = [col[:-2] for col in df_X1.columns] 
        
        df_X2 = self.df[[col for col in self.df.columns if col.endswith('_2')]]
        df_X2.columns = [col[:-2] for col in df_X2.columns] 
        self.df_X1 = df_X1.drop(columns=['train_val', 'target', 'Unnamed: 0'] if 'Unnamed: 0' in df_X1.columns else ['train_val', 'target'])
        self.df_X2 = df_X2.drop(columns=['train_val', 'target', 'Unnamed: 0'] if 'Unnamed: 0' in df_X2.columns else ['train_val', 'target'])
        
        if 'barcode' in self.df_X1.columns:
            self.df_X1 = self.df_X1.drop(columns=['barcode'])
        if 'barcode' in self.df_X2.columns:
            self.df_X2 = self.df_X2.drop(columns=['barcode'])

        self.input_size1 = len(self.df_X1.columns)
        self.input_size2 = len(self.df_X2.columns)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row1 = self.df_X1.iloc[index]
        X1 = row1.values
        
        row2 = self.df_X2.iloc[index]
        X2 = row2.values

        y = self.df_y.iloc[index]
        X_tensor1 = torch.tensor(X1, dtype=torch.float32)
        X_tensor2 = torch.tensor(X2, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)
        
        # x_i, x_t, y, m
        return X_tensor1, X_tensor2, y_tensor, self.mask_special[index]



class tabular_dataset_dvm_2tab(Dataset):
    def __init__(self, opt_dict, mode):
        self.excel_path1 = opt_dict['dataset_config']['dataset_tabular_i']  
        self.excel_path2 = opt_dict['dataset_config']['dataset_tabular_t'] 
        print(f"tabular_path1: {self.excel_path1}")
        print(f"tabular_path2: {self.excel_path2}")
        
        self.df1 = pd.read_csv(self.excel_path1, encoding='utf-8')
        self.df2 = pd.read_csv(self.excel_path2, encoding='utf-8')
        
        self.df = pd.merge(self.df1, self.df2, on='Adv_ID', suffixes=('_1', '_2'))
        self.opt_dict = opt_dict

        if mode == 'train':
            self.df = self.df[self.df['train_val_1'] == 'j']
            if opt_dict['dataset_config']['missing_mask_train']:
                self.mask_special = np.load(opt_dict['dataset_config']['missing_mask_train'])
        elif mode == 'val':
            self.df = self.df[self.df['train_val_1'] == 'v']
            if opt_dict['dataset_config']['missing_mask_val']:
                self.mask_special = np.load(opt_dict['dataset_config']['missing_mask_val'])
        elif mode == 'test':
            self.df = self.df[self.df['train_val_1'] == 'y']
            if opt_dict['dataset_config']['missing_mask_test']:
                self.mask_special = np.load(opt_dict['dataset_config']['missing_mask_test'])

        self.df_y = self.df['Genmodel_ID_1']

        df_X1 = self.df[[col for col in self.df.columns if col.endswith('_1')]]
        df_X1.columns = [col[:-2] for col in df_X1.columns] 
        
        df_X2 = self.df[[col for col in self.df.columns if col.endswith('_2')]]
        df_X2.columns = [col[:-2] for col in df_X2.columns] 
        self.df_X1 = df_X1.drop(columns=['train_val', 'Genmodel_ID', 'Unnamed: 0'] if 'Unnamed: 0' in df_X1.columns else ['train_val', 'Genmodel_ID'])
        self.df_X2 = df_X2.drop(columns=['train_val', 'Genmodel_ID', 'Unnamed: 0'] if 'Unnamed: 0' in df_X2.columns else ['train_val', 'Genmodel_ID'])
        
        if 'barcode' in self.df_X1.columns:
            self.df_X1 = self.df_X1.drop(columns=['barcode'])
        if 'barcode' in self.df_X2.columns:
            self.df_X2 = self.df_X2.drop(columns=['barcode'])

        self.input_size1 = len(self.df_X1.columns)
        self.input_size2 = len(self.df_X2.columns)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row1 = self.df_X1.iloc[index]
        X1 = row1.values
        
        row2 = self.df_X2.iloc[index]
        X2 = row2.values

        y = self.df_y.iloc[index]
        X_tensor1 = torch.tensor(X1, dtype=torch.float32)
        X_tensor2 = torch.tensor(X2, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)
        
        # x_i, x_t, y, m
        return X_tensor1, X_tensor2, y_tensor, self.mask_special[index]