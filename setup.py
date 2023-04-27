"""
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
import os
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

src_dir = 'Swin3D/src/attn'
src_files = [os.path.join(src_dir, _f) for _f in os.listdir(src_dir) if 
    os.path.splitext(_f)[1] in ['.cu', '.cpp']]

setup(
    name='Swin3D',
    packages=find_packages(exclude=[]),
    ext_modules=[
        CUDAExtension(
            name='Swin3D.sparse_dl.attn_cuda',
            sources=src_files,
            extra_compile_args={'cxx': ['-g'],
                                'nvcc': ['-O2']}),
        CUDAExtension(
            name='Swin3D.sparse_dl.knn_cuda',
            sources=[
                'Swin3D/src/knn/knn_api.cpp',
                'Swin3D/src/knn/knn_cuda_kernel.cu'],
            extra_compile_args={'cxx': ['-g'],
                                'nvcc': ['-O2']}),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })