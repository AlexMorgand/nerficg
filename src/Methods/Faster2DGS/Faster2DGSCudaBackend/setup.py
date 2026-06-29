import os
from pathlib import Path

from setuptools import setup

try:
    from torch.utils.cpp_extension import CUDAExtension, BuildExtension
except ImportError as exc:
    raise SystemExit(
        'Faster2DGSCudaBackend must be built with the same Python environment that has PyTorch installed.\n'
        'Activate your nerficg training conda/venv, verify with:\n'
        '  python -c "import torch; print(torch.__version__)"\n'
        'then reinstall using:\n'
        '  python -m pip install --no-build-isolation --no-cache-dir ./src/Methods/Faster2DGS/Faster2DGSCudaBackend\n'
        f'({exc})'
    ) from exc

__author__ = 'NeRFICG'
__description__ = 'Native 2DGS surfel rasterization CUDA backend for Faster2DGS.'

ENABLE_FASTMATH = True
ENABLE_NVCC_LINEINFO = False

module_root = Path(__file__).parent.absolute()
extension_name = module_root.name
extension_root = module_root / extension_name
surfel_root = extension_root / 'surfel_rasterization'
repo_root = module_root.parents[3]
glm_root = repo_root / 'submodules' / 'diff-surfel-rasterization' / 'third_party' / 'glm'
if not (glm_root / 'glm' / 'glm.hpp').is_file():
    raise SystemExit(
        'GLM headers not found for Faster2DGSCudaBackend surfel build.\n'
        f'Expected: {glm_root / "glm" / "glm.hpp"}\n'
        'Run: git submodule update --init submodules/diff-surfel-rasterization/third_party/glm'
    )

sources = [
    str(extension_root / 'torch_bindings' / 'bindings.cpp'),
    str(surfel_root / 'surfel_rasterize_points.cu'),
    str(surfel_root / 'cuda_rasterizer' / 'forward.cu'),
    str(surfel_root / 'cuda_rasterizer' / 'backward.cu'),
    str(surfel_root / 'cuda_rasterizer' / 'rasterizer_impl.cu'),
]

include_dirs = [
    str(surfel_root),
    str(surfel_root / 'cuda_rasterizer'),
    str(glm_root),
]

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
        'nvcc': nvcc_flags,
    },
)

setup(
    name=extension_name,
    author=__author__,
    packages=[f'{extension_name}.torch_bindings'],
    ext_modules=[extension],
    description=__description__,
    cmdclass={'build_ext': BuildExtension},
)
