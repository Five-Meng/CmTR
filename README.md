# CmTR
### python环境配置

```markdown
Python >= 3.8
Cuda >= 11.3
matplotlib == 3.7.5 
numpy == 1.24.4 
numpy == 1.24.3 
opt == 0.1 
pandas == 2.0.3 
Pillow == 11.2.1
PyYAML == 6.0.2 
PyYAML == 6.0.2 
scikit_learn == 1.3.2 
scipy == 1.15.2 
seaborn == 0.13.2 
tensorflow == 2.11.0 
timm == 1.0.15 
torch == 1.12.1+cu113 
torch_geometric == 2.6.1 
torchvision == 0.13.1+cu113
tqdm == 4.67.1
```

### 数据集

使用图像-表格对作为数据集，如图。这些数据集中的图像和表格数据相互关联，用于训练和测试多模态融合模型。图像数据主要包含各类视觉信息，表格数据则提供了结构化的数据特征，表格数据带有缺失。

![image-20250510190307821]()

### 运行方法

![image-20250510190332740]()



### 训练流程

该训练过程分为三个主要阶段，采用分步训练的方式，逐步构建并优化模型，以实现多模态数据（图像与表格数据）的有效处理和利用，以及对缺失数据的补充。

#### 参数配置

#### dataset_config

| 参数名              | 说明                                         |
| ------------------- | -------------------------------------------- |
| dataname            | 指定数据集名称（如 dvm），用于区分不同数据集 |
| dataset             | 图像数据集路径，模型读取图像数据的来源       |
| dataset_tabular     | 表格数据集（CSV 文件）路径，存储结构化数据   |
| image_result_path   | 图像相关结果的保存路径                       |
| tabular_result_path | 表格数据结果的保存路径                       |
| it_result_path      | 图像与表格融合结果的保存路径                 |
| num_cls             | 分类任务的类别数量                           |
| data_dir            | 数据集根目录                                 |
| missing_rate        | 表格数据的缺失率                             |
| missing_mask_train  | 训练集缺失掩码文件路径                       |
| missing_mask_val    | 验证集缺失掩码文件路径                       |
| missing_mask_test   | 测试集缺失掩码文件路径                       |

#### train_config

| 参数名          | 说明                                       |
| --------------- | ------------------------------------------ |
| mode            | 训练模式（通常设为 train）                 |
| epochs          | 训练轮数                                   |
| max_epochs      | 最大训练轮数                               |
| lr_max          | 最大学习率，控制参数更新步长               |
| lr_min          | 最小学习率，学习率衰减下限                 |
| batch_size      | 单次训练的数据样本数量                     |
| seed            | 随机种子，确保实验可复现                   |
| lossfunc        | 损失函数名称（如 focalloss、crossentropy） |
| checkpoint_path | 模型检查点保存路径                         |

#### H2T

| 参数名  | 说明                          |
| ------- | ----------------------------- |
| h2t     | 布尔值，控制是否启用 H2T 方法 |
| rho_h2t | H2T 方法中的系数              |

#### model_config

| 参数名            | 说明                                   |
| ----------------- | -------------------------------------- |
| net_v             | 图像模型网络结构（如 efficientnet-b1） |
| net_v_tabular     | 表格数据模型网络结构                   |
| net               | 跨模态 / 融合模型网络结构              |
| model_name        | 模型名称                               |
| freeze_layers     | 布尔值，控制是否冻结模型层             |
| embedding_dropout | 嵌入层丢弃率，防止过拟合               |
| embedding_dim     | 嵌入层维度                             |
| hidden_size1      | 隐藏层 1 维度                          |
| hidden_size2      | 隐藏层 2 维度                          |
| hidden_size3      | 隐藏层 3 维度                          |

#### **predict_config**

| 参数名               | 说明                           |
| -------------------- | ------------------------------ |
| load_predict_feature | 预测时加载特征提取模型参数路径 |
| load_predict_re      | 预测时加载重建模型参数路径     |

###  VDM

```bash
python main.py --yaml_config argparses/efficientnet.yaml
```

### DBRM

此阶段又细分为图像分支和表格分支的处理，以及跨模态信息补充。

图像数据处理训练命令：

```bash
python backbone_prototype/train_re_cls.py --yaml_config argparses/efficientnet_re_cls.yaml
python backbone_prototype/train_re_cls_prototype.py --yaml_config argparses/efficientnet_re_cls_prototype.yaml
python backbone_prototype/predict.py --yaml_config argparses/efficientnet_re_cls.yaml
```

表格数据处理训练命令：

```bash
python backbone_tabular_prototype/train_re_cls.py --yaml_config argparses/mlp_re_cls.yaml
python backbone_tabular_prototype/train_re_cls_prototype.py --yaml_config argparses/mlp_re_cls_prototype.yaml
python backbone_tabular_prototype/predict_tab.py --yaml_config argparses/mlp_re_cls.yaml
```

跨模态融合：

```bash
python backbone_img_tab/train_crossattn.py --yaml_config argparses/crossattn.yaml
```

### FM

```bash
python backbone_img_tab/train_multimodel.py --yaml_config argparses/multimodel.yaml
```
