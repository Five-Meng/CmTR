import io
import os
from PIL import Image
import numpy as np
import torch
import torch.utils.data
import random
from torchvision import transforms


class img_dataset(torch.utils.data.Dataset):
    def __init__(self, opt_dict, mode, transform=None):
        self.mode = mode
        # /home/admin1/users/mxy/数据集/图像数据/
        self.dataset = opt_dict['dataset_config']['dataset']
        self.img_pic = []
        self.img_label = []
        self.transform = transform

        if self.mode == 'train':
            self.img_dir = self.dataset + '建模组'
        elif self.mode == 'test':
            self.img_dir = self.dataset + '验证组'

        for subdir_cls in os.listdir(self.img_dir):
            img_path = self.img_dir + '/' + subdir_cls
            for barcode in os.listdir(img_path):
                barcode_path = img_path + '/' + barcode
                for wdf in os.listdir(barcode_path):
                    if wdf == 'WDF.png':
                        wdf_path = barcode_path + '/' + wdf
                        image_pic = self.loader(wdf_path)
                        if self.transform is not None:
                            image_pic = self.transform(image_pic)
                        self.img_pic.append(image_pic) 
                        if int(subdir_cls) == 7:
                            self.img_label.append(6)
                        else:
                            self.img_label.append(int(subdir_cls))

    def loader(self, image_path):
        return Image.open(image_path).convert('RGB')

    def __getitem__(self, index):
        return self.img_pic[index], self.img_label[index]

    def __len__(self):
        return len(self.img_label)
    

class img_dataset_dvm(torch.utils.data.Dataset):
    def __init__(self, opt_dict, mode, transform=None):
        self.mode = mode
        self.dataset = opt_dict['dataset_config']['dataset']
        self.img_label = []
        self.transform = transform
        
        if self.mode == 'train':
            self.img_dir = os.path.join(self.dataset, 'train')
        elif self.mode == 'test':
            self.img_dir = os.path.join(self.dataset, 'test')
        elif self.mode == 'val':
            self.img_dir = os.path.join(self.dataset, 'val')

        self.img_paths = []
        for img_name in os.listdir(self.img_dir):
            img_cls = img_name.split('_')[0]
            self.img_label.append(int(img_cls))
            img_path = os.path.join(self.img_dir, img_name)
            self.img_paths.append(img_path)

    
    def loader(self, image_path):
        return Image.open(image_path).convert('RGB')

    def resorter(self):
        # import ipdb;ipdb.set_trace();
        unique_labels = sorted(set(self.img_label))
        label_mapping = {old_val: new_val for new_val, old_val in enumerate(unique_labels)}
        new_label = [label_mapping[val] for val in self.img_label]
        return new_label

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img_cls = self.img_label[index]
        image_pic = self.loader(img_path)
        # import ipdb;ipdb.set_trace();
        if self.transform is not None:
            image_pic = self.transform(image_pic)

        img_cls = torch.tensor(int(img_cls), dtype=torch.long)
        return image_pic, img_cls

    def __len__(self):
        return len(self.img_label)


class img_dataset_H2T(torch.utils.data.Dataset):
    def __init__(self, opt_dict, mode, transform=None):
        self.mode = mode
        self.dataset = opt_dict['dataset_config']['dataset']
        self.img_label = []
        self.transform = transform
        self.cls_num = opt_dict['train_config']['num_cls']
        
        if opt_dict['dataset_config']['dataname'] == 'blood':
            if self.mode == 'train':
                self.img_dir = os.path.join(self.dataset, '建模组')
            elif self.mode == 'test':
                self.img_dir = os.path.join(self.dataset, '验证组')

            # 存储图像路径和标签
            self.img_paths = []
            self.class_dict = {}

            for subdir_cls in os.listdir(self.img_dir):
                img_path = os.path.join(self.img_dir, subdir_cls)
                for barcode in os.listdir(img_path):
                    barcode_path = os.path.join(img_path, barcode)
                    for wdf in os.listdir(barcode_path):
                        if wdf == 'WDF.png':
                            wdf_path = os.path.join(barcode_path, wdf)

                            self.img_paths.append(wdf_path)
                            
                            if int(subdir_cls) == 7:
                                self.img_label.append(6)
                            else:
                                self.img_label.append(int(subdir_cls))

        elif opt_dict['dataset_config']['dataname'] == 'dvm':
            if self.mode == 'train':
                self.img_dir = os.path.join(self.dataset, 'train')
            elif self.mode == 'test':
                self.img_dir = os.path.join(self.dataset, 'test')

            self.img_paths = []
            for img_name in os.listdir(self.img_dir):
                img_cls = img_name.split('_')[0]  
                self.img_label.append(img_cls)
                img_path = os.path.join(self.img_dir, img_name)
                self.img_paths.append(img_path)

        self.class_dict = self._get_class_dict()

    def loader(self, image_path):
        return Image.open(image_path).convert('RGB')

    def _get_class_dict(self):
        class_dict = {}
        for idx, label in enumerate(self.img_label):
            label = int(label)
            if label not in class_dict:
                class_dict[label] = []
            class_dict[label].append(idx)
        return class_dict

    def __getitem__(self, index):
        meta = dict()
        
        img_path = self.img_paths[index]
        img = self.loader(img_path)
        label = self.img_label[index]

        if self.transform is not None:
            img = self.transform(img)
        if isinstance(img, Image.Image):
            img = transforms.ToTensor()(img) 

        sample_class = random.randint(0, self.cls_num - 1)
        sample_indexes = self.class_dict[sample_class]
        random_index = random.choice(sample_indexes)
        sample_img_path = self.img_paths[random_index]
        sample_img = self.loader(sample_img_path)
        sample_label = self.img_label[random_index]

        if self.transform is not None:
            sample_img = self.transform(sample_img)
        if isinstance(sample_img, Image.Image): 
            sample_img = transforms.ToTensor()(sample_img)

        label = torch.tensor(int(label), dtype=torch.long)
        sample_label = torch.tensor(int(sample_label), dtype=torch.long)

        meta['sample_image'] = img
        meta['sample_label'] = label

        return sample_img, sample_label, meta

    def __len__(self):
        return len(self.img_paths)
    

class img_dataset_H2T_dvm(torch.utils.data.Dataset):
    def __init__(self, opt_dict, mode, transform=None):
        self.mode = mode
        self.dataset = opt_dict['dataset_config']['dataset']
        self.img_label = []
        self.transform = transform
        self.cls_num = opt_dict['train_config']['num_cls']

        if self.mode == 'train':
            self.img_dir = os.path.join(self.dataset, 'train')
        elif self.mode == 'test':
            self.img_dir = os.path.join(self.dataset, 'test')

        self.img_paths = []
        for img_name in os.listdir(self.img_dir):
            img_cls = img_name.split('_')[0]  
            self.img_label.append(img_cls)
            img_path = os.path.join(self.img_dir, img_name)
            self.img_paths.append(img_path)

        # import ipdb;ipdb.set_trace();
        self.class_dict = self._get_class_dict()

    def loader(self, image_path):
        return Image.open(image_path).convert('RGB')

    def _get_class_dict(self):
        class_dict = {}
        for idx, label in enumerate(self.img_label):
            label = int(label)
            if label not in class_dict:
                class_dict[label] = []
            class_dict[label].append(idx)
        return class_dict


    def __getitem__(self, index):
        # import ipdb;ipdb.set_trace();
        meta = dict()
        
        img_path = self.img_paths[index]
        img = self.loader(img_path)
        label = self.img_label[index]

        if self.transform is not None:
            img = self.transform(img)
        if isinstance(img, Image.Image):
            img = transforms.ToTensor()(img) 

        sample_class = random.randint(0, self.cls_num - 1)
        sample_indexes = self.class_dict[sample_class]
        random_index = random.choice(sample_indexes)
        sample_img_path = self.img_paths[random_index]
        sample_img = self.loader(sample_img_path)
        sample_label = self.img_label[random_index]

        if self.transform is not None:
            sample_img = self.transform(sample_img)
        if isinstance(sample_img, Image.Image): 
            sample_img = transforms.ToTensor()(sample_img)

        label = torch.tensor(int(label), dtype=torch.long)
        sample_label = torch.tensor(int(sample_label), dtype=torch.long)

        meta['sample_image'] = img
        meta['sample_label'] = label

        return sample_img, sample_label, meta
    

    def __len__(self):
        return len(self.img_label)


def get_barcode_img_dict(dataset_dir):
    train_barcode_dict, test_barcode_dict = {}, {}

    for mode in ['建模组', '验证组']:
        img_dir = os.path.join(dataset_dir, mode)

        for subdir_cls in os.listdir(img_dir):
            img_path = os.path.join(img_dir, subdir_cls)
            for barcode in os.listdir(img_path):
                barcode_path = os.path.join(img_path, barcode)
                wdf_path = os.path.join(barcode_path, 'WDF.png')
                
                if os.path.exists(wdf_path):  # 检查是否有WDF.png
                    target_dict = train_barcode_dict if mode == '建模组' else test_barcode_dict
                    target_dict.setdefault(barcode, []).append(wdf_path)

    return train_barcode_dict, test_barcode_dict



if __name__ == '__main__':
    from argparses.util.yaml_args import yaml_data
    opt_dict = yaml_data("./argparses/blood_resnet.yaml")

    dataset_dir = opt_dict['dataset_config']['dataset']
    train_barcode_dict, test_barcode_dict = get_barcode_img_dict(dataset_dir)
    import ipdb; ipdb.set_trace()

    print("Train Barcode Dictionary:", len(train_barcode_dict))
    print("Test Barcode Dictionary:", len(test_barcode_dict))
