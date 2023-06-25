"""
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
import numpy as np
import torch
import torch.nn as nn
from Swin3D.models import Swin3DUNet
from easydict import EasyDict
from MinkowskiEngine import SparseTensor

args = EasyDict({
    'in_channels': 9,
    'num_layers': 5,
    'depths': [2, 2, 2, 2, 2],
    'channels': [16, 16, 32, 64, 64] ,
    'num_heads': [2, 2, 4, 8, 8],
    'window_sizes': [5, 7, 7, 7, 7],
    'quant_size': 4,
    'down_stride': 3,
    'knn_down': True,
    'stem_transformer': True,
    'upsample': 'linear',
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
print(model)
print('#Model parameters: {}'.format(sum([x.nelement() for x in model.parameters()])))
model = model.cuda()
criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label).cuda()
param_dicts = [
    {"params": [p for n, p in model.named_parameters() if "blocks" not in n and p.requires_grad]},
    {
        "params": [p for n, p in model.named_parameters() if "blocks" in n and p.requires_grad],
        "lr": args.base_lr * args.transformer_lr_scale,
    },
]
optimizer = torch.optim.AdamW(param_dicts, lr=args.base_lr, weight_decay=args.weight_decay)

data = np.load("examples/input.npz")
feat, xyz, batch, target = data["feat"], data["xyz"], data["batch"], data["target"]
# feats: [N, 6], RGB, Normal
# xyz: [N, 3],
# batch: [N],
# target: [N],
feat, xyz, batch, target = torch.from_numpy(feat).cuda(), torch.from_numpy(xyz).cuda(), torch.from_numpy(batch).cuda(), torch.from_numpy(target).cuda()
coords = torch.cat([batch.unsqueeze(-1), xyz], dim=-1)
feat = torch.cat([feat, xyz], dim=1)
sp = SparseTensor(feat.float(), torch.floor(coords).int(), device=feat.device)
colors = feat[:, 0:3]
normals = feat[:, 3:6]
coords_sp = SparseTensor(features=torch.cat([coords, colors, normals], dim=1), coordinate_map_key=sp.coordinate_map_key, 
coordinate_manager=sp.coordinate_manager)

use_amp = True
scaler = torch.cuda.amp.GradScaler()
with torch.cuda.amp.autocast(enabled=use_amp):
    output = model(sp, coords_sp)
loss = criterion(output, target)
optimizer.zero_grad()

if use_amp:
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
else:
    loss.backward()
    optimizer.step()
print("FINISHED!")

