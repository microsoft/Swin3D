"""
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
import os
import torch
import torch.nn as nn
from Swin3D.modules.mink_layers import MinkConvBNRelu, MinkResBlock
from Swin3D.modules.swin3d_layers import GridDownsample, GridKNNDownsample, BasicLayer, Upsample
from timm.models.layers import trunc_normal_

def load_state_with_same_shape(model, weights, skip_first_conv=False, verbose=True):
    if list(weights.keys())[0].startswith('module.'):
        if verbose:
            print("Loading multigpu weights with module. prefix...")
        weights = {k.partition('module.')[2]:weights[k] for k in weights.keys()}
    model_state = model.state_dict()
    
    droped_weights = [k for k in weights.keys() if ('upsamples' in k or 'classifier' in k)]
    weights = {k: v for k, v in weights.items() if 'upsamples' not in k}
    weights = {k: v for k, v in weights.items() if 'classifier' not in k}
    if skip_first_conv:
        droped_weights += [k for k in weights.keys() if 'stem_layer' in k]
        weights = {k: v for k, v in weights.items() if 'stem_layer' not in k}
    filtered_weights = {
        k: v for k, v in weights.items() if k in model_state and v.size() == model_state[k].size() 
    }
    diff_shape_weights = [
        k for k, v in weights.items() if k in model_state  and v.size() != model_state[k].size() 
    ]
    not_found_weights = [
        k for k in model_state.keys() if k not in weights.keys()
    ]
    droped_weights = droped_weights + diff_shape_weights
    if verbose:
        print("Not Loading weights: " + ', '.join(droped_weights))
        print("="*100)
        print("Not found weights: " + ', '.join(not_found_weights))
        print("="*100)
    return filtered_weights

class Swin3DUNet(nn.Module):
    def __init__(self, depths, channels, num_heads, window_sizes, \
            quant_size, drop_path_rate=0.2, up_k=3, \
            num_layers=5, num_classes=13, stem_transformer=True, first_down_stride=2, upsample='linear', knn_down=True, \
            in_channels=6, cRSE='XYZ_RGB', fp16_mode=0):
        super().__init__()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        if knn_down:
            downsample = GridKNNDownsample
        else:
            downsample = GridDownsample

        self.cRSE = cRSE
        if stem_transformer:
            self.stem_layer = MinkConvBNRelu(
                    in_channels=in_channels,
                    out_channels=channels[0],
                    kernel_size=3,
                    stride=1,
                )
            self.layer_start = 0
        else:
            self.stem_layer = nn.Sequential(              
                MinkConvBNRelu(
                    in_channels=in_channels,
                    out_channels=channels[0],
                    kernel_size=3,
                    stride=1,
                ),
                MinkResBlock(
                    in_channels=channels[0],
                    out_channels=channels[0]
                )
            )
            self.downsample = downsample(
                        channels[0],
                        channels[1],
                        kernel_size=first_down_stride,
                        stride=first_down_stride
            )
            self.layer_start = 1
        self.layers = nn.ModuleList([
            BasicLayer(
                dim=channels[i], 
                depth=depths[i], 
                num_heads=num_heads[i], 
                window_size=window_sizes[i],
                quant_size=quant_size,
                drop_path=dpr[sum(depths[:i]):sum(depths[:i+1])], 
                downsample=downsample if i < num_layers-1 else None,
                down_stride=first_down_stride if i==0 else 2,
                out_channels=channels[i+1] if i < num_layers-1 else None,
                cRSE=cRSE,
                fp16_mode=fp16_mode) for i in range(self.layer_start, num_layers)])

        if 'attn' in upsample:
            up_attn = True
        else:
            up_attn = False
        
        self.upsamples = nn.ModuleList([
                Upsample(channels[i], channels[i-1], num_heads[i-1], window_sizes[i-1], quant_size, attn=up_attn, \
                    up_k=up_k, cRSE=cRSE, fp16_mode=fp16_mode) 
            for i in range(num_layers-1, 0, -1)])

        self.classifier = nn.Sequential(
            nn.Linear(channels[0], channels[0]), 
            nn.BatchNorm1d(channels[0]), 
            nn.ReLU(inplace=True), 
            nn.Linear(channels[0], num_classes)
        )
        self.num_classes = num_classes
        self.init_weights()

    def forward(self, sp, coords_sp):
        # sp: MinkowskiEngine SparseTensor for feature input
        # sp.F: input features,         NxC
        # sp.C: input coordinates,      Nx4
        # coords_sp: MinkowskiEngine SparseTensor for position and feature embedding
        # coords_sp.F: embedding
        #       Batch: 0,...0,1,...1,2,...2,...,B,...B
        #       XYZ:   in Voxel Scale
        #       RGB:   in [-1,1]
        #       NORM:  in [-1,1]
        #       Batch, XYZ:             Nx4
        #       Batch, XYZ, RGB:        Nx7
        #       Batch, XYZ, RGB, NORM:  Nx10
        # coords_sp.C: input coordinates: Nx4
        sp_stack = []
        coords_sp_stack = []
        sp = self.stem_layer(sp)
        if self.layer_start > 0:
            sp_stack.append(sp)
            coords_sp_stack.append(coords_sp)
            sp, coords_sp = self.downsample(sp, coords_sp)

        for i, layer in enumerate(self.layers):
            coords_sp_stack.append(coords_sp)
            sp, sp_down, coords_sp = layer(sp, coords_sp)
            sp_stack.append(sp)
            assert (coords_sp.C == sp_down.C).all()
            sp = sp_down

        sp = sp_stack.pop()
        coords_sp = coords_sp_stack.pop()
        for i, upsample in enumerate(self.upsamples):
            sp_i = sp_stack.pop()
            coords_sp_i = coords_sp_stack.pop()
            sp = upsample(sp, coords_sp, sp_i, coords_sp_i)
            coords_sp = coords_sp_i

        output = self.classifier(sp.F)
        return output
    

    def init_weights(self):
        """Initialize the weights in backbone.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        self.apply(_init_weights)


    def load_pretrained_model(self, ckpt, skip_first_conv=True, verbose=True):
        if os.path.isfile(ckpt):
            checkpoint = torch.load(ckpt)
            weights = checkpoint['state_dict']
            matched_weights = load_state_with_same_shape(self, weights, skip_first_conv=skip_first_conv, verbose=verbose)
            self.load_state_dict(matched_weights, strict=False)
            if verbose:
                print("=> loaded weight '{}'".format(ckpt))
        else:
            if verbose:
                print("=> no weight found at '{}'".format(ckpt))

class Swin3DEncoder(nn.Module):
    def __init__(self, depths, channels, num_heads, window_sizes, quant_size, \
            drop_path_rate=0.2, num_layers=5, stem_transformer=True, first_down_stride=2, knn_down=True, \
            in_channels=6, cRSE='XYZ_RGB', fp16_mode=0):
        super().__init__()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        if knn_down:
            downsample = GridKNNDownsample
        else:
            downsample = GridDownsample

        self.cRSE = cRSE
        if stem_transformer:
            self.stem_layer = MinkConvBNRelu(
                    in_channels=in_channels,
                    out_channels=channels[0],
                    kernel_size=3,
                    stride=1,
                )
            self.layer_start = 0
        else:
            self.stem_layer = nn.Sequential(              
                MinkConvBNRelu(
                    in_channels=in_channels,
                    out_channels=channels[0],
                    kernel_size=3,
                    stride=1,
                ),
                MinkResBlock(
                    in_channels=channels[0],
                    out_channels=channels[0]
                )
            )
            self.downsample = downsample(
                        channels[0],
                        channels[1],
                        kernel_size=first_down_stride,
                        stride=first_down_stride
            )
            self.layer_start = 1
        self.layers = nn.ModuleList([
            BasicLayer(
                dim=channels[i], 
                depth=depths[i], 
                num_heads=num_heads[i], 
                window_size=window_sizes[i],
                quant_size=quant_size,
                drop_path=dpr[sum(depths[:i]):sum(depths[:i+1])], 
                downsample=downsample if i < num_layers-1 else None,
                down_stride=first_down_stride if i==0 else 2,
                out_channels=channels[i+1] if i < num_layers-1 else None,
                cRSE=cRSE,
                fp16_mode=fp16_mode) for i in range(self.layer_start, num_layers)])

        self.init_weights()

    def forward(self, sp, coords_sp):
        # sp: MinkowskiEngine SparseTensor for feature input
        # sp.F: input features,         NxC
        # sp.C: input coordinates,      Nx4
        # coords_sp: MinkowskiEngine SparseTensor for position and feature embedding
        # coords_sp.F: embedding
        #       Batch: 0,...0,1,...1,2,...2,...,B,...B
        #       XYZ:   in Voxel Scale
        #       RGB:   in [-1,1]
        #       NORM:  in [-1,1]
        #       Batch, XYZ:             Nx4
        #       Batch, XYZ, RGB:        Nx7
        #       Batch, XYZ, RGB, NORM:  Nx10
        # coords_sp.C: input coordinates: Nx4
        sp_stack = []
        coords_sp_stack = []
        sp = self.stem_layer(sp)
        if self.layer_start > 0:
            sp_stack.append(sp)
            coords_sp_stack.append(coords_sp)
            sp, coords_sp = self.downsample(sp, coords_sp)

        for i, layer in enumerate(self.layers):
            coords_sp_stack.append(coords_sp)
            sp, sp_down, coords_sp = layer(sp, coords_sp)
            sp_stack.append(sp)
            assert (coords_sp.C == sp_down.C).all()
            sp = sp_down
        return sp_stack
    

    def init_weights(self):
        """Initialize the weights in backbone.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        self.apply(_init_weights)

    def load_pretrained_model(self, ckpt, skip_first_conv=True, verbose=True):
        if os.path.isfile(ckpt):
            checkpoint = torch.load(ckpt)
            weights = checkpoint['state_dict']
            matched_weights = load_state_with_same_shape(self, weights, skip_first_conv=skip_first_conv, verbose=verbose)
            self.load_state_dict(matched_weights, strict=False)
            if verbose:
                print("=> loaded weight '{}'".format(ckpt))
        else:
            if verbose:
                print("=> no weight found at '{}'".format(ckpt))