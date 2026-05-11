import os
from glob import glob
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

__author__ = 'NeRFICG'
__description__ = 'CUDA backend scaffold for Faster2DGS surfel rasterization.'

ENABLE_FASTMATH = True
ENABLE_NVCC_LINEINFO = False

module_root = Path(__file__).parent.absolute()
extension_name = module_root.name
extension_root = module_root / extension_name
cuda_modules = [d.name for d in Path(extension_root).iterdir() if d.is_dir() and d.name not in ['utils', 'torch_bindings', '__pycache__']]
fastergs_root = module_root.parent.parent / 'FasterGS' / 'FasterGSCudaBackend' / 'FasterGSCudaBackend'

sources = [str(extension_root / 'torch_bindings' / 'bindings.cpp')]
for module in cuda_modules:
    sources += glob(str(extension_root / module / 'src' / '**' / '*.cpp'), recursive=True)
    sources += glob(str(extension_root / module / 'src' / '**' / '*.cu'), recursive=True)

# FasterGS headers first so `#include "rasterization_api.h"` in embedded sources resolves to FasterGS.
include_dirs = [
    str(fastergs_root / 'rasterization' / 'include'),
    str(fastergs_root / 'utils'),
]
for module in cuda_modules:
    include_dirs.append(str(extension_root / module / 'include'))

cxx_flags = ['/std:c++17' if os.name == 'nt' else '-std=c++17']
nvcc_flags = ['-std=c++17']
if ENABLE_FASTMATH:
    cxx_flags.append('-O3')
    nvcc_flags.append('-O3')
    nvcc_flags.append('-use_fast_math')
if ENABLE_NVCC_LINEINFO:
    nvcc_flags.append('-lineinfo')

extension = CUDAExtension(
    name=f'{extension_name}._C',
    sources=sources,
    include_dirs=include_dirs,
    extra_compile_args={
        'cxx': cxx_flags,
        'nvcc': nvcc_flags
    }
)

setup(
    name=extension_name,
    author=__author__,
    packages=[f'{extension_name}.torch_bindings'],
    ext_modules=[extension],
    description=__description__,
    cmdclass={'build_ext': BuildExtension}
)
