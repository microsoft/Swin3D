#include <torch/extension.h>
#include "knn_cuda_kernel.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("knn_cuda", &knn_cuda, "knn_cuda");
}
