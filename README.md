# Swin3D
Created by <a href="https://yukichiii.github.io/" target="_blank">Yu-Qi Yang</a>
## Introduction
We present a pretrained 3D backbone, named Swin3D, that first-time outperforms all state-of-the-art methods on downstream 3D indoor scene understanding tasks. Our backbone network is based on a 3D Swin transformer and carefully designed for efficiently conducting self-attention on sparse voxels with a linear memory complexity and capturing the irregularity of point signals via generalized contextual relative positional embedding. Based on this backbone design, we pretrained a large Swin3D model on a synthetic Structed3D dataset that is 10 times larger than the ScanNet dataset and fine-tuned the pretrained model on various downstream real-world indoor scene understanding tasks.

## Overview

## Data Preparation

## Quick Start

Install the package using
`python setup.py install`

Build models and load our pretrained weight, Then you can finetune your model in various task.

    import torch
    from Swin3D.models import Swin3DUNet
    from easydict import EasyDict
    args = EasyDict({
        'in_channels': 6,
        'num_layers': 5,
        'depths': [2, 4, 9, 4, 4],
        'channels': [48, 96, 192, 384, 384] ,
        'num_heads': [6, 6, 12, 24, 24],
        'window_sizes': [5, 7, 7, 7, 7],
        'quant_sizes': [4, 4, 4, 4, 4],
        'down_stride': 3,
        'knn_down': True,
        'stem_transformer': True,
        'upsample': 'linear_attn',
        'up_k': 3,
        'drop_path_rate': 0.3,
        'num_classes': 13,
    })
    model = Swin3DUNet(args.depths, args.channels, args.num_heads, \
            args.window_sizes, args.quant_sizes, up_k=args.up_k, drop_path_rate=args.drop_path_rate, num_classes=args.num_classes, \
            num_layers=args.num_layers, stem_transformer=args.stem_transformer, upsample=args.upsample, first_down_stride=args.down_stride,
            knn_down=args.knn_down, in_channels=args.in_channels, cRSE='XYZ_RGB_NORM', fp16_mode=0)
    model.load_pretrained_model(ckpt_path)

## Results and models
### ScanNet Segmentation

|          | Pretrained | mIoU(Val) | mIoU(Test) |   Model   |    Log    |
| :------- | :--------: | :-------: | :--------: | :-------: | :-------: |
| Swin3D-S |   &cross;   |   75.2    |     -      | will soon | will soon |
| Swin3D-S |   &check;   |   75.7    |     -      | will soon | will soon |
| Swin3D-L |   &check;   |   77.5    |    77.9    | will soon | will soon |


### S3DIS Segmentation

|          | Pretrained | Area 5 mIoU | 6-fold mIoU |   Model   |    Log    |
| :------- | :--------: | :---------: | :---------: | :-------: | :-------: |
| Swin3D-S |   &cross;   |    72.5     |    76.9     | will soon | will soon |
| Swin3D-S |   &check;   |    73.0     |    78.2     | will soon | will soon |
| Swin3D-L |   &check;   |    74.5     |    80.3     | will soon | will soon |


### ScanNet 3D Detection


|                    | Pretrained | mAP@0.25 | mAP@0.50 |   Model   |    Log    |
| :----------------- | :--------: | :------: | :------: | :-------: | :-------: |
| Swin3D-S+FCAF3D    |   &check;   |   74.2   |   59.5   | will soon | will soon |
| Swin3D-L+FCAF3D    |   &check;   |   74.2   |   58.6   | will soon | will soon |
| Swin3D-S+CAGroup3D |   &check;   |   76.4   |   62.7   | will soon | will soon |
| Swin3D-L+CAGroup3D |   &check;   |   76.4   |   63.2   | will soon | will soon |

### S3DIS 3D Detection

|                 | Pretrained | mAP@0.25 | mAP@0.50 |   Model   |    Log    |
| :-------------- | :--------: | :------: | :------: | :-------: | :-------: |
| Swin3D-S+FCAF3D |   &check;   |   69.9   |   50.2   | will soon | will soon |
| Swin3D-L+FCAF3D |   &check;   |   72.1   |   54.0   | will soon | will soon |
