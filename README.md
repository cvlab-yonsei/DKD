# Decomposed Knowledge Distillation for Class-Incremental Semantic Segmentation

This is an official implementation of the paper "Decomposed Knowledge Distillation for Class-Incremental Semantic Segmentation", accepted to NeurIPS 2022.

For more information, please checkout the project site [[website](https://cvlab.yonsei.ac.kr/projects/DKD/)] and our paper [[arXiv](http://arxiv.org/abs/2210.05941)].

## Pre-requisites
This repository uses the following libraries:
* Python (3.6)
* Pytorch (1.8.1)

## Getting Started

### Datasets
#### PASCAL VOC 2012
We use augmented 10,582 training samples and 1,449 validation samples for PASCAL VOC 2012. To train our model with augmented samples, please download labels of augmented samples (['SegmentationClassAug'](https://www.dropbox.com/s/oeu149j8qtbs1x0/SegmentationClassAug.zip)) and file names (['train_aug.txt'](https://github.com/cvlab-yonsei/DKD/releases/download/v1.0/train_aug.txt)). The structure of data path should be organized as follows:
```bash
└── /dataset/VOC2012
    ├── Annotations
    ├── ImageSets
    │   └── Segmentation
    │       ├── train_aug.txt
    │       └── val.txt
    ├── JPEGImages
    ├── SegmentationClass
    └── SegmentationClassAug
```

#### ADE20K
The structure of data path should be organized as follows:
```bash
└── /dataset/ADEChallengeData2016
    ├── annotations
    ├── images
    ├── objectInfo150.txt
    └── sceneCategories.txt
```

### Training
#### PASCAL VOC 2012
```Shell
# An example srcipt for 15-5 overlapped setting of PASCAL VOC

GPU=0,1,2,3
BS=8  # Total 32
SAVEDIR='saved_voc'

TASKSETTING='overlap'  # or 'disjoint'
TASKNAME='15-5' # or ['15-1', '19-1', '10-1', '5-3']
EPOCH=60
INIT_LR=0.001
LR=0.0001
INIT_POSWEIGHT=2
MEMORY_SIZE=0  # 100 for DKD-M

NAME='DKD'
python train_voc.py -c configs/config_voc.json \
-d ${GPU} --multiprocessing_distributed --save_dir ${SAVEDIR} --name ${NAME} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 0 --lr ${INIT_LR} --bs ${BS} --pos_weight ${INIT_POSWEIGHT}

python train_voc.py -c configs/config_voc.json \
-d ${GPU} --multiprocessing_distributed --save_dir ${SAVEDIR} --name ${NAME} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 1 --lr ${LR} --bs ${BS} --freeze_bn --mem_size ${MEMORY_SIZE}
```

<!-- All numbers are obtained by averaging results over five runs. 
|  Method   | VOC 19-1 (2 steps)<br>$\text{hIoU}$ / $\text{mIoU}_{\text{all}}$ | VOC 15-5 (2 steps)<br>$\text{hIoU}$ / $\text{mIoU}_{\text{all}}$ | VOC 15-1 (6 steps)<br>$\text{hIoU}$ / $\text{mIoU}_{\text{all}}$ | VOC 10-1 (11 steps)<br>$\text{hIoU}$ / $\text{mIoU}_{\text{all}}$ | VOC 5-3 (6 steps)<br>$\text{hIoU}$ / $\text{mIoU}_{\text{all}}$ |
| :-        | :------------: | :--------------:   | :---------------: | :----------------: | :----------------: |
| MiB       | 33.62   / 67.80   | 59.72   / 69.00   | 19.50   / 29.70   | 12.66   / 29.70   | 48.77   / 46.71   |
| SDR       | 44.30   / 67.40   | 61.97   / 69.90   | 29.31   / 39.20   | -                 | -                 |
| PLOP      | 49.94   / 73.54   | 61.46   / 70.09   | 31.88   / 54.64   | 57.66   / 29.70   | 18.28   / 18.68   |
| SSUL      | 42.96   / 75.44   | 60.96   / 71.22   | 49.67   / 67.61   | **57.66** / 29.70 | 59.63   / 56.89   |
| **DKD**   | **54.03 / 76.03** | **66.98 / 73.93** | **55.21 / 69.67** | 56.84 / **60.44** | **60.49 / 58.10** |
| SSUL-M    | 60.71   / 76.49   | 65.20   / 73.02   | 60.30   / 71.37   | 61.93   / 64.12   | 60.93   / 58.37   |
| **DKD-M** | **66.27 / 77.01** | **68.63 / 74.72** | **62.91 / 72.53** | **64.19 / 65.77** | **64.63 / 62.94** | -->


#### ADE20K
```Shell
# An example srcipt for 50-50 overlapped setting of ADE20K

GPU=0,1,2,3
BS=6  # Total 24
SAVEDIR='saved_ade'

TASKSETTING='overlap'
TASKNAME='50-50' # or ['100-10', '100-50']
EPOCH=100
INIT_LR=0.0025
LR=0.00025
MEMORY_SIZE=0 # 300 for DKD-M

NAME='DKD'
python train_ade.py -c configs/config_ade.json \
-d ${GPU} --multiprocessing_distributed --save_dir ${SAVEDIR} --name ${NAME} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 0 --lr ${INIT_LR} --bs ${BS}

python train_ade.py -c configs/config_ade.json \
-d ${GPU} --multiprocessing_distributed --save_dir ${SAVEDIR} --name ${NAME} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 1 --lr ${LR} --bs ${BS} --freeze_bn --mem_size ${MEMORY_SIZE}

python train_ade.py -c configs/config_ade.json \
-d ${GPU} --multiprocessing_distributed --save_dir ${SAVEDIR} --name ${NAME} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 2 --lr ${LR} --bs ${BS} --freeze_bn --mem_size ${MEMORY_SIZE}
```

<!-- All numbers are obtained by averaging results over five runs.
|  Method     | ADE 100-50 (2 steps)<br>$\text{hIoU}$ / $\text{mIoU}_{\text{all}}$ | ADE 100-10 (6 steps)<br>$\text{hIoU}$ / $\text{mIoU}_{\text{all}}$ | ADE 50-50 (3 steps)<br>$\text{hIoU}$ / $\text{mIoU}_{\text{all}}$ |
| :--------:  | :-----------------: | :--------------:   | :----------------: |
| MiB       | 24.12 / 32.79     | 17.23 / 29.24     | 28.76 / 29.31     |
| PLOP      | 21.97 / 32.94     | 20.37 / 31.59     | 29.36 / 30.40     |
| SSUL      | 25.09 / 33.58     | 25.57 / 33.10     | 28.45 / 29.56     |
| **DKD**   | **29.74 / 35.95** | **26.55 / 34.26** | **34.17 / 33.90** |
| SSUL-M    | 24.88 / 34.37     | 25.01 / 34.46     | 28.53 / 29.77     |
| **DKD-M** | **29.79 / 35.98** | **27.14 / 34.58** | **34.19 / 33.92** | -->


### Testing
#### PASCAL VOC 2012
```Shell
python eval_voc.py -d 0 -r path/to/weight.pth
```
We provide pretrained weights and configuration files. The results should be:
|  Method<br>(Overlapped)   | VOC 19-1<br>(2 steps)<br>$\text{hIoU}$ / $\text{mIoU}_{\text{all}}$ | VOC 15-5<br>(2 steps)<br>$\text{hIoU}$ / $\text{mIoU}_{\text{all}}$ | VOC 15-1<br>(6 steps)<br>$\text{hIoU}$ / $\text{mIoU}_{\text{all}}$ | VOC 10-1<br>(11 steps)<br>$\text{hIoU}$ / $\text{mIoU}_{\text{all}}$ | VOC 5-3<br>(6 steps)<br>$\text{hIoU}$ / $\text{mIoU}_{\text{all}}$ |
| :-    | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: |
| DKD   | [56.95 / 76.13](https://github.com/cvlab-yonsei/DKD/releases/download/v1.0/voc_overlapped_19-1.zip) | [67.17 / 73.95](https://github.com/cvlab-yonsei/DKD/releases/download/v1.0/voc_overlapped_15-1.zip) | [57.46 / 70.50](https://github.com/cvlab-yonsei/DKD/releases/download/v1.0/voc_overlapped_15-1.zip) | [57.21 / 60.43](https://github.com/cvlab-yonsei/DKD/releases/download/v1.0/voc_overlapped_10-1.zip) | [61.32 / 58.98](https://github.com/cvlab-yonsei/DKD/releases/download/v1.0/voc_overlapped_5-3.zip) |
| DKD-M | [68.14 / 77.04](https://github.com/cvlab-yonsei/DKD/releases/download/v1.0/voc_overlapped_19-1_memory.zip) | [68.96 / 74.84](https://github.com/cvlab-yonsei/DKD/releases/download/v1.0/voc_overlapped_15-5_memory.zip) | [64.09 / 72.95](https://github.com/cvlab-yonsei/DKD/releases/download/v1.0/voc_overlapped_15-1_memory.zip) | [64.89 / 66.20](https://github.com/cvlab-yonsei/DKD/releases/download/v1.0/voc_overlapped_10-1_memory.zip) | [65.02 / 63.32](https://github.com/cvlab-yonsei/DKD/releases/download/v1.0/voc_overlapped_5_3_memory.zip) |

|  Method<br>(Disjoint)   | VOC 19-1<br>(2 steps)<br>$\text{hIoU}$ / $\text{mIoU}_{\text{all}}$ | VOC 15-5<br>(2 steps)<br>$\text{hIoU}$ / $\text{mIoU}_{\text{all}}$ | VOC 15-1<br>(6 steps)<br>$\text{hIoU}$ / $\text{mIoU}_{\text{all}}$ | 
| :-    | :-----------: | :-----------: | :-----------: |
| DKD   | [57.98 / 75.90](https://github.com/cvlab-yonsei/DKD/releases/download/v1.0/voc_disjoint_19-1.zip) | [64.35 / 72.21](https://github.com/cvlab-yonsei/DKD/releases/download/v1.0/voc_disjoint_15-1.zip) | [54.21 / 68.48](https://github.com/cvlab-yonsei/DKD/releases/download/v1.0/voc_disjoint_15-1.zip) |
| DKD-M | [67.12 / 76.84](https://github.com/cvlab-yonsei/DKD/releases/download/v1.0/voc_disjoint_19-1_memory.zip) | [65.04 / 72.53](https://github.com/cvlab-yonsei/DKD/releases/download/v1.0/voc_disjoint_15-5_memory.zip) | [60.32 / 70.78](https://github.com/cvlab-yonsei/DKD/releases/download/v1.0/voc_disjoint_15-1_memory.zip) |

#### ADE20K
```Shell
python eval_ade.py -d 0 -r path/to/weight.pth
```
We provide pretrained weights and configuration files. The results should be:
|  Method     | ADE 100-50<br>(2 steps)<br>$\text{hIoU}$ / $\text{mIoU}_{\text{all}}$ | ADE 100-10<br>(6 steps)<br>$\text{hIoU}$ / $\text{mIoU}_{\text{all}}$ | ADE 50-50<br>(3 steps)<br>$\text{hIoU}$ / $\text{mIoU}_{\text{all}}$ |
| :---- | :-----------: | :-----------: | :-----------: |
| DKD   | [29.42 / 35.55](https://github.com/cvlab-yonsei/DKD/releases/download/v1.0/ade_overlapped_100-50.zip) | [26.79 / 34.53](https://github.com/cvlab-yonsei/DKD/releases/download/v1.0/ade_overlapped_100-10.zip) | [34.77 / 34.39](https://github.com/cvlab-yonsei/DKD/releases/download/v1.0/ade_overlapped_50-50.zip) |
| DKD-M | [29.45 / 35.58](https://github.com/cvlab-yonsei/DKD/releases/download/v1.0/ade_overlapped_100-50_memory.zip) | [27.44 / 34.83](https://github.com/cvlab-yonsei/DKD/releases/download/v1.0/ade_overlapped_100-10.zip) | [34.78 / 34.39](https://github.com/cvlab-yonsei/DKD/releases/download/v1.0/ade_overlapped_50-50.zip) |


## Acknowledgements
* This template is borrowed from [pytorch-template](https://github.com/victoresque/pytorch-template).
