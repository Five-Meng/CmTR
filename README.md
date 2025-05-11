# CmTR

<iframe src="https://docs.google.com/viewer?url=https://github.com/Five-Meng/CmTR/raw/main/method.pdf&embedded=true" style="width:100%; height:600px;" frameborder="0"></iframe>


### Environment

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

### Dataset

The dataset consists of image-table pairs, as illustrated in the figure. Within these datasets, images and tabular data are interrelated and jointly used for training and evaluating multimodal fusion models. The image data captures diverse visual information, whereas the tabular data offers structured features, some of which contain missing values.

#### Available 

The BSD dataset will be made available upon reasonable request. 

The DVM dataset is publicly accessible at https://deepvisualmarketing.github.io/.

<iframe src="https://docs.google.com/viewer?url=https://github.com/Five-Meng/CmTR/raw/main/dataset.pdf&embedded=true" style="width:100%; height:600px;" frameborder="0"></iframe>


### Run

### Train

The training process is divided into three main stages, using a step-by-step training approach to progressively build and optimize the model for effective processing and utilization of multimodal data (image and tabular data), as well as the completion of missing data.

#### Parameter 

#### dataset_config

| Parameter             | Description                                                    |
| --------------------- | -------------------------------------------------------------- |
| dataname              | Specify the dataset name (e.g., dvm) to distinguish datasets   |
| dataset               | Path to the image dataset, source for loading image data       |
| dataset\_tabular      | Path to the tabular dataset (CSV file) storing structured data |
| image\_result\_path   | Path to save image-related results                             |
| tabular\_result\_path | Path to save tabular data results                              |
| it\_result\_path      | Path to save image-table fusion results                        |
| num\_cls              | Number of classes for classification task                      |
| data\_dir             | Root directory of the dataset                                  |
| missing\_rate         | Missing rate of the tabular data                               |
| missing\_mask\_train  | Path to the missing mask file for the training set             |
| missing\_mask\_val    | Path to the missing mask file for the validation set           |
| missing\_mask\_test   | Path to the missing mask file for the test set                 |


#### train_config

| Parameter        | Description                                               |
| ---------------- | --------------------------------------------------------- |
| mode             | Training mode (usually set to "train")                    |
| epochs           | Number of training epochs                                 |
| max\_epochs      | Maximum number of training epochs                         |
| lr\_max          | Maximum learning rate, controls step size for updates     |
| lr\_min          | Minimum learning rate, lower bound for decay              |
| batch\_size      | Number of samples per training batch                      |
| seed             | Random seed to ensure reproducibility                     |
| lossfunc         | Name of the loss function (e.g., focalloss, crossentropy) |
| checkpoint\_path | Path to save model checkpoints                            |


#### H2T

| Parameter | Description                                        |
| --------- | -------------------------------------------------- |
| h2t       | Boolean value to control whether to use H2T method |
| rho\_h2t  | Coefficient used in the H2T method                 |


#### model_config

| Parameter          | Description                                                |
| ------------------ | ---------------------------------------------------------- |
| net\_v             | Image model architecture (e.g., efficientnet-b1)           |
| net\_v\_tabular    | Tabular data model architecture                            |
| net                | Cross-modal / fusion model architecture                    |
| model\_name        | Name of the model                                          |
| freeze\_layers     | Boolean value to control whether to freeze layers          |
| embedding\_dropout | Dropout rate in the embedding layer to prevent overfitting |
| embedding\_dim     | Dimension of the embedding layer                           |
| hidden\_size1      | Dimension of hidden layer 1                                |
| hidden\_size2      | Dimension of hidden layer 2                                |
| hidden\_size3      | Dimension of hidden layer 3                                |


#### predict_config

| Parameter              | Description                                                            |
| ---------------------- | ---------------------------------------------------------------------- |
| load\_predict\_feature | Path to load parameters for feature extraction model during prediction |
| load\_predict\_re      | Path to load parameters for reconstruction model during prediction     |


###  VDM

```bash
python main.py --yaml_config argparses/efficientnet.yaml
```

### DBRM

This stage is further divided into processing of the image branch and the tabular branch, as well as cross-modal information augmentation.

Training command for image data processing:

```bash
python backbone_prototype/train_re_cls.py --yaml_config argparses/efficientnet_re_cls.yaml
python backbone_prototype/train_re_cls_prototype.py --yaml_config argparses/efficientnet_re_cls_prototype.yaml
python backbone_prototype/predict.py --yaml_config argparses/efficientnet_re_cls.yaml
```

Training command for Tabular data processing:

```bash
python backbone_tabular_prototype/train_re_cls.py --yaml_config argparses/mlp_re_cls.yaml
python backbone_tabular_prototype/train_re_cls_prototype.py --yaml_config argparses/mlp_re_cls_prototype.yaml
python backbone_tabular_prototype/predict_tab.py --yaml_config argparses/mlp_re_cls.yaml
```

Cross-modal fusion:

```bash
python backbone_img_tab/train_crossattn.py --yaml_config argparses/crossattn.yaml
```


## Test

```bash
python test.py -–yaml_config argparses/test.yaml
```

### FM

```bash
python backbone_img_tab/train_multimodel.py --yaml_config argparses/multimodel.yaml
```
