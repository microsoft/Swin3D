"""
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
# pylint: disable=arguments-differ,abstract-method,no-member
import enum
from typing import Any, List

import torch
from torch import Tensor
from torch.autograd.function import Function
import Swin3D.sparse_dl.attn_cuda as attn_module

# try:
#     from cuda import attn_module
# except ModuleNotFoundError:
#     from ..cuda import attn_module


class PosEmb(enum.Enum):
    NONE = enum.auto()
    LINEAR = enum.auto()
    SEPARATE = enum.auto()


class TableDims(enum.Enum):
    WHCD = enum.auto()
    DWHC = enum.auto()
    D0 = enum.auto()


class IndexMode(enum.Enum):
    DIRECT = enum.auto()
    INDIRECT = enum.auto()


class PrecisionMode(enum.Enum):
    HALF_ALL = enum.auto()
    HALF_FORWARD = enum.auto()
    HALF_NONE = enum.auto()


class AttnBaseFunction(Function):
    @staticmethod
    def auto_cast_layout(table: Tensor, table_dims: TableDims, reverse: bool = False):
        """
        Cast layout to native supported one
        """
        if table_dims == TableDims.WHCD and not reverse:
            table = torch.permute(table, [3, 0, 1, 2]).contiguous()
        elif table_dims == TableDims.WHCD and reverse:
            table = torch.permute(table, [1, 2, 3, 0]).contiguous()
        return table

    @staticmethod
    def auto_cast_type(src_tensor: Tensor, dst_tensor: Tensor):
        """
        Cast type to native supported one
        """
        if src_tensor.type() == dst_tensor.type():
            return src_tensor
        else:
            return src_tensor.type(dst_tensor.type())

    @staticmethod
    def cast_layout_and_type(tensor: Tensor, dst_tensor: Tensor, table_dims: 
        TableDims, reverse: bool = False):
        """
        Cast the layout and type
        """
        tensor = __class__.auto_cast_type(tensor, dst_tensor)
        tensor = __class__.auto_cast_layout(tensor, table_dims, reverse)
        return tensor

    @staticmethod
    def padding_out_grads(grads: List[Tensor], num_inputs: int):
        padding_grads = [None] * (num_inputs - len(grads))
        return (*grads, *padding_grads)


class AttnCalCoffFunction(AttnBaseFunction):
    @staticmethod
    def forward(ctx, raw_query_feats: Tensor, raw_key_feats: Tensor, query_table: 
        Tensor, key_table: Tensor, m2w_indices: Tensor, w_elems: Tensor, w2m_indices: Tensor,
        w2n_indices: Tensor, n2n_indices: Tensor, n_coords: Tensor, pe=PosEmb.SEPARATE, 
        table_dim=TableDims.WHCD) -> Tensor:
        """
        """
        ctx.save_for_backward(raw_query_feats, raw_key_feats, query_table, key_table,\
            m2w_indices, w_elems, w2m_indices, w2n_indices, n2n_indices, n_coords)
        setattr(ctx, 'table_dim', table_dim)
        setattr(ctx, 'pe', pe)
        query_table = __class__.cast_layout_and_type(query_table, raw_query_feats, table_dim)
        key_table = __class__.cast_layout_and_type(key_table, raw_key_feats, table_dim)
        coff, = attn_module.self_attn_cal_coff_indir_forward(
            raw_query_feats, raw_key_feats, query_table, key_table, m2w_indices, w_elems,
            w2m_indices, w2n_indices, n2n_indices, n_coords, pe.value)
        return coff

    @staticmethod
    def backward(ctx, *grad_outputs: Any) -> Any:
        raw_query_feats, raw_key_feats, query_table, key_table, m2w_indices, w_elems, w2m_indices,\
            w2n_indices, n2n_indices, n_coords = ctx.saved_tensors
        table_dim: TableDims = getattr(ctx, 'table_dim')
        pe: PosEmb = getattr(ctx, "pe")
        coff_grads, = grad_outputs
        query_table_fwd = __class__.cast_layout_and_type(query_table, raw_query_feats, table_dim)
        key_table_fwd = __class__.cast_layout_and_type(key_table, raw_key_feats, table_dim)
        raw_query_grads, raw_key_grads, query_table_grads, key_table_grads = \
            attn_module.self_attn_cal_coff_indir_backward(
                coff_grads, raw_query_feats, raw_key_feats, query_table_fwd, key_table_fwd, 
                m2w_indices, w_elems, w2m_indices, w2n_indices, n2n_indices, n_coords, pe.value
            )
        query_table_grads = __class__.cast_layout_and_type(query_table_grads, query_table, table_dim, True)
        key_table_grads = __class__.cast_layout_and_type(key_table_grads, key_table, table_dim, True)

        return raw_query_grads, raw_key_grads, query_table_grads, key_table_grads, None, None, None,\
            None, None, None


class AttnApplyCoffFunction(AttnBaseFunction):
    @staticmethod
    def forward(ctx, raw_value_feats: Tensor, coff_norm: Tensor, value_table: 
        Tensor, m2w_indices: Tensor, w_elems: Tensor, w2m_indices: Tensor, w2n_indices: Tensor,
        n2n_indices: Tensor, n_coords: Tensor, pe=PosEmb.SEPARATE, table_dim=TableDims.WHCD)\
            -> Tensor:
        ctx.save_for_backward(raw_value_feats, coff_norm, value_table, m2w_indices, w_elems,
            w2m_indices, w2n_indices, n2n_indices, n_coords)
        setattr(ctx, 'table_dim', table_dim)
        setattr(ctx, 'pe', pe)
        value_table = __class__.cast_layout_and_type(value_table, raw_value_feats, table_dim)
        updated_value_feats, = attn_module.self_attn_apply_coff_indir_forward(raw_value_feats, 
            coff_norm, value_table, m2w_indices, w_elems, w2m_indices, w2n_indices, n2n_indices,
            n_coords, pe.value)
        return updated_value_feats

    @staticmethod
    def backward(ctx, *grad_outputs: Any) -> Any:
        raw_value_feats, coff_norm, value_table, m2w_indices, w_elems, w2m_indices, w2n_indices,\
            n2n_indices, n_coords = ctx.saved_tensors
        table_dim: TableDims = getattr(ctx, 'table_dim')
        pe: PosEmb = getattr(ctx, "pe")
        updated_value_grads, = grad_outputs
        value_table_fwd = __class__.cast_layout_and_type(value_table, raw_value_feats, table_dim)
        raw_value_grads, coff_norm_grads, value_table_grads = \
            attn_module.self_attn_apply_coff_indir_backward(
                updated_value_grads, raw_value_feats, coff_norm, value_table_fwd, m2w_indices,
                w_elems, w2m_indices, w2n_indices, n2n_indices, n_coords, pe.value
            )
        value_table_grads = __class__.cast_layout_and_type(value_table_grads, value_table, table_dim,
            True)
        return raw_value_grads, coff_norm_grads, value_table_grads, None, None, None, None, None, None


class SelfAttnAIOFunction(AttnBaseFunction):
    """
    Function of all-in-one self-attention
    """
    @staticmethod
    def forward(ctx, raw_query_feats, raw_key_feats, raw_value_feats, query_table, key_table,
        value_table, table_offsets: Tensor, indices: List[Tensor], pe: PosEmb,
        table_dim: TableDims, mode: IndexMode, precision: PrecisionMode):
        assert table_dim == TableDims.D0
        setattr(ctx, 'table_dim', table_dim)
        setattr(ctx, 'pe', pe)
        setattr(ctx, 'mode', mode)
        setattr(ctx, 'precision', precision)

        qkv_feats = [raw_query_feats, raw_key_feats, raw_value_feats]
        qkv_tables = [query_table, key_table, value_table]

        if torch.is_autocast_enabled() and precision != PrecisionMode.HALF_NONE:
            c_dtype = torch.get_autocast_gpu_dtype()
        else:
            c_dtype = torch.float32

        c_qkv_feats = [_f.type(c_dtype) for _f in qkv_feats]
        c_qkv_tables = [_t.type(c_dtype) for _t in qkv_tables]

        coff_rmax = attn_module.cal_max_coffs(c_qkv_feats, c_qkv_tables, table_offsets, \
            indices, mode.value, pe.value)
        raw_attn_feats, sum_coffs = attn_module.self_attn_forward(c_qkv_feats, c_qkv_tables, \
            coff_rmax, table_offsets, indices, mode.value, pe.value)
        norm_attn_feats = raw_attn_feats / sum_coffs

        bp_feats = [sum_coffs, coff_rmax, norm_attn_feats]
        backward_tensors = [*bp_feats, *qkv_feats, *qkv_tables, table_offsets, *indices]
        ctx.save_for_backward(*backward_tensors)
        return norm_attn_feats

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        norm_attn_grads, = grad_outputs
        sum_coffs, coff_rmax, norm_attn_feats = ctx.saved_tensors[:3]
        raw_query_feats, raw_key_feats, raw_value_feats = ctx.saved_tensors[3:6]
        query_table, key_table, value_table = ctx.saved_tensors[6:9]
        table_offsets = ctx.saved_tensors[9]
        indices = ctx.saved_tensors[10:]

        pos_emb: PosEmb = getattr(ctx, "pe")
        mode: IndexMode = getattr(ctx, 'mode')
        precision: PrecisionMode = getattr(ctx, 'precision')

        qkv_feats = [raw_query_feats, raw_key_feats, raw_value_feats]
        qkv_tables = [query_table, key_table, value_table]

        if precision == PrecisionMode.HALF_ALL:
            c_dtype = norm_attn_grads.dtype
        else:
            c_dtype = torch.float32
            r_dtype = norm_attn_grads.dtype
            norm_attn_feats = norm_attn_feats.type(torch.float32)
            sum_coffs = sum_coffs.type(torch.float32)
            norm_attn_grads = norm_attn_grads.type(torch.float32)

        c_qkv_feats = [_f.type(c_dtype) for _f in qkv_feats]
        c_qkv_tables = [_t.type(c_dtype) for _t in qkv_tables]

        raw_attn_feats = norm_attn_feats * sum_coffs
        exp_sum_grads = attn_module.cal_exp_sum_grads(norm_attn_grads, raw_attn_feats, sum_coffs,
            c_qkv_feats[-1], value_table)
        grads = attn_module.self_attn_indir_backward(norm_attn_grads, exp_sum_grads, raw_attn_feats,
            sum_coffs, coff_rmax, *c_qkv_feats, *c_qkv_tables, table_offsets, indices, mode.value,
            pos_emb.value)

        for t_grad, t_in in zip(grads[3:6], qkv_tables):
            t_grad = t_grad.type(t_in.type())

        if precision == PrecisionMode.HALF_FORWARD:
            for t_grad in grads[:3]:
                t_grad = t_grad.type(r_dtype)
        return __class__.padding_out_grads(grads, 12)


class SelfAttnAIOModule(torch.nn.Module):
    def __init__(self, pe: PosEmb, table_dim: TableDims, mode: IndexMode) -> None:
        super().__init__()
        self.pe = pe
        self.table_dim = table_dim
        self.mode = mode

    def forward(self, raw_query_feats, raw_key_feats, raw_value_feats, query_table, key_table, 
        value_table, indices):
        # n_coords = n_coords[n2n_indices]
        norm_attn_feats = SelfAttnAIOFunction.apply(
            raw_query_feats, raw_key_feats, raw_value_feats, query_table, key_table, 
            value_table, indices, self.pe, self.table_dim, self.mode)
        return norm_attn_feats
