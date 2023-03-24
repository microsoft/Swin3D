#include <torch/serialize/tensor.h>
#include <torch/extension.h>

#include "knnquery/knnquery_cuda_kernel.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("knnquery_cuda", &knnquery_cuda, "knnquery_cuda");
    }
