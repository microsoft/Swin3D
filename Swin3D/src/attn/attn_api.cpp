/*
Copyright (c) Microsoft Corporation.
Licensed under the MIT License.
*/
#include <torch/extension.h>

#include "self_attn_cal_coff_indir_forward.h"
#include "self_attn_cal_coff_indir_backward.h"
#include "self_attn_apply_coff_indir_forward.h"
#include "self_attn_apply_coff_indir_backward.h"
#include "self_attn_aio_fwd.h"
#include "self_attn_aio_bwd.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("self_attn_cal_coff_indir_forward", &self_attn_cal_coff_cuda_forward_indir, "Calculate indirect self-attention forward (CUDA)");
  m.def("self_attn_cal_coff_indir_backward", &self_attn_cal_coff_cuda_backward_indir, "Calculate indirect self-attention backward (CUDA)");
  m.def("self_attn_apply_coff_indir_forward", &self_attn_apply_coff_cuda_forward_indir, "Apply indirect self-attention forward (CUDA)");
  m.def("self_attn_apply_coff_indir_backward", &self_attn_apply_coff_cuda_backward_indir, "Apply indirect self-attention backward (CUDA)");

  m.def("self_attn_forward", &self_attn_cuda_forward, "All-in-one self-attention indirect forward (CUDA)");
  m.def("self_attn_indir_backward", &self_attn_cuda_backward, "All-in-one self-attention indirect backward (CUDA)");

  m.def("cal_max_coffs", &cal_max_coffs, "Calculate the maximum coffs for numerical stableness");
  m.def("cal_exp_sum_grads", &cal_exp_sum_grads, "Calculate the gradient of exp sum");
}
