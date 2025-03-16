import os.path as osp

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT = osp.dirname(osp.abspath(__file__))


setup(
    ext_modules=[
        CUDAExtension(
            "cuda_corr",
            sources=[
                "src/dpvo/altcorr/correlation.cpp",
                "src/dpvo/altcorr/correlation_kernel.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3"],
            },
        ),
        CUDAExtension(
            "cuda_ba",
            sources=[
                "src/dpvo/fastba/ba.cpp",
                "src/dpvo/fastba/ba_cuda.cu",
                "src/dpvo/fastba/block_e.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3"],
            },
            include_dirs=[osp.join(ROOT, "src/")],
        ),
        CUDAExtension(
            "lietorch_backends",
            include_dirs=[
                osp.join(ROOT, "src/dpvo/lietorch/include"),
                osp.join(ROOT, "src/"),
            ],
            sources=[
                "src/dpvo/lietorch/src/lietorch.cpp",
                "src/dpvo/lietorch/src/lietorch_gpu.cu",
                "src/dpvo/lietorch/src/lietorch_cpu.cpp",
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3"],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
