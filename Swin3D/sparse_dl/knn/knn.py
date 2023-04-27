"""
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
import torch
from torch import Tensor
from torch.autograd.function import Function
import Swin3D.sparse_dl.knn_cuda as knn_cuda
class KNN(Function):
    @staticmethod
    def forward(ctx, K, src_xyz, query_xyz, src_offset, query_offset):
        assert src_xyz.is_contiguous() and query_xyz.is_contiguous()
        N = query_xyz.shape[0]
        ret_idx = torch.cuda.IntTensor(N, K).zero_()
        ret_dist = torch.cuda.FloatTensor(N, K).zero_() + 1e10
        knn_cuda.knn_cuda(
                K, 
                src_xyz, 
                query_xyz, 
                src_offset, 
                query_offset, 
                ret_idx, 
                ret_dist
        )
        ret_dist = torch.sqrt(ret_dist)
        return ret_idx, ret_dist