# Swin3D: A Pretrained Transformer Backbone for 3D Indoor Scene Understanding
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/swin3d-a-pretrained-transformer-backbone-for/semantic-segmentation-on-scannet)](https://paperswithcode.com/sota/semantic-segmentation-on-scannet?p=swin3d-a-pretrained-transformer-backbone-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/swin3d-a-pretrained-transformer-backbone-for/semantic-segmentation-on-s3dis-area5)](https://paperswithcode.com/sota/semantic-segmentation-on-s3dis-area5?p=swin3d-a-pretrained-transformer-backbone-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/swin3d-a-pretrained-transformer-backbone-for/semantic-segmentation-on-s3dis)](https://paperswithcode.com/sota/semantic-segmentation-on-s3dis?p=swin3d-a-pretrained-transformer-backbone-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/swin3d-a-pretrained-transformer-backbone-for/3d-object-detection-on-scannetv2)](https://paperswithcode.com/sota/3d-object-detection-on-scannetv2?p=swin3d-a-pretrained-transformer-backbone-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/swin3d-a-pretrained-transformer-backbone-for/3d-object-detection-on-s3dis)](https://paperswithcode.com/sota/3d-object-detection-on-s3dis?p=swin3d-a-pretrained-transformer-backbone-for)

## Introduction
We present a pretrained 3D backbone, named Swin3D, that first-time outperforms all state-of-the-art methods on downstream 3D indoor scene understanding tasks. Our backbone network is based on a 3D Swin transformer and carefully designed for efficiently conducting self-attention on sparse voxels with a linear memory complexity and capturing the irregularity of point signals via generalized contextual relative positional embedding. Based on this backbone design, we pretrained a large Swin3D model on a synthetic Structed3D dataset that is 10 times larger than the ScanNet dataset and fine-tuned the pretrained model on various downstream real-world indoor scene understanding tasks.

## Overview
- [Data Preparation](#data-preparation)
- [Quick Start](#quick-start)
- [Results and models](#results-and-models)
- [Citation](#citation)

## Data Preparation
We pretrained our Swin3D on Structured3D, please to refer to this [link](https://github.com/yuxiaoguo/Uni3DScenes) to prepare the data.

## Quick Start
Install the package using 

    python setup.py install

Build models and load our pretrained weight, Then you can finetune your model in various task.

    import torch
    from Swin3D.models import Swin3DUNet
    from easydict import EasyDict
    args = EasyDict({
        'in_channels': 9,
        'num_layers': 5,
        'depths': [2, 4, 9, 4, 4],
        'channels': [48, 96, 192, 384, 384] ,
        'num_heads': [6, 6, 12, 24, 24],
        'window_sizes': [5, 7, 7, 7, 7],
        'quant_size': 4,
        'down_stride': 3,
        'knn_down': True,
        'stem_transformer': True,
        'upsample': 'linear_attn',
        'up_k': 3,
        'drop_path_rate': 0.3,
        'num_classes': 20,
        'ignore_label': -100,
        'base_lr': 0.001,
        'transformer_lr_scale': 0.1,
        'weight_decay': 0.0001,
    })
    model = Swin3DUNet(args.depths, args.channels, args.num_heads, \
            args.window_sizes, args.quant_size, up_k=args.up_k, drop_path_rate=args.drop_path_rate, num_classes=args.num_classes, \
            num_layers=args.num_layers, stem_transformer=args.stem_transformer, upsample=args.upsample, first_down_stride=args.down_stride,
            knn_down=args.knn_down, in_channels=args.in_channels, cRSE='XYZ_RGB_NORM', fp16_mode=2)
    model.load_pretrained_model(ckpt_path)

## Results and models
To reproduce our result, please follow the code in this [repo](https://github.com/Yukichiii/Swin3D_Task).

### ScanNet Segmentation

|          | Pretrained | mIoU(Val) | mIoU(Test) |   Model   |    Log    |
| :------- | :--------: | :-------: | :--------: | :-------: | :-------: |
| Swin3D-S |   &cross;   |   75.2    |     -      | [model]() | [log]() |
| Swin3D-S |   &check;   |   75.7    |     -      | [model]() | [log]() |
| Swin3D-L |   &check;   |   77.5    |    77.9    | [model]() | [log]() |


### S3DIS Segmentation

|          | Pretrained | Area 5 mIoU | 6-fold mIoU |   Model   |    Log    |
| :------- | :--------: | :---------: | :---------: | :-------: | :-------: |
| Swin3D-S |   &cross;   |    72.5     |    76.9     | [model]() | [log]() |
| Swin3D-S |   &check;   |    73.0     |    78.2     | [model]() | [log]() |
| Swin3D-L |   &check;   |    74.5     |    79.8     | [model]() | [log]() |


### ScanNet 3D Detection


|                    | Pretrained | mAP@0.25 | mAP@0.50 |   Model   |    Log    |
| :----------------- | :--------: | :------: | :------: | :-------: | :-------: |
| Swin3D-S+FCAF3D    |   &check;   |   74.2   |   59.5   | [model]() | [log]() |
| Swin3D-L+FCAF3D    |   &check;   |   74.2   |   58.6   | [model]() | [log]() |
| Swin3D-S+CAGroup3D |   &check;   |   76.4   |   62.7   | [model]() | [log]() |
| Swin3D-L+CAGroup3D |   &check;   |   76.4   |   63.2   | [model]() | [log]() |

### S3DIS 3D Detection

|                 | Pretrained | mAP@0.25 | mAP@0.50 |   Model   |    Log    |
| :-------------- | :--------: | :------: | :------: | :-------: | :-------: |
| Swin3D-S+FCAF3D |   &check;   |   69.9   |   50.2   | [model]() | [log]() |
| Swin3D-L+FCAF3D |   &check;   |   72.1   |   54.0   | [model]() | [log]() |


## Citation
If you find Swin3D useful to your research, please cite our work:
```
@misc{yang2023swin3d,
      title={Swin3D: A Pretrained Transformer Backbone for 3D Indoor Scene Understanding}, 
      author={Yu-Qi Yang and Yu-Xiao Guo and Jian-Yu Xiong and Yang Liu and Hao Pan and Peng-Shuai Wang and Xin Tong and Baining Guo},
      year={2023},
      eprint={2304.06906},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```