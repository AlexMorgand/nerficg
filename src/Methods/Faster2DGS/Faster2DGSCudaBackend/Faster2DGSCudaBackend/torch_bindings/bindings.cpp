#include <torch/extension.h>
#include "faster2dgs_rasterization_api.h"

namespace rasterization_api = faster2dgs::rasterization;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &rasterization_api::forward_wrapper);
    m.def("backward", &rasterization_api::backward_wrapper);
    m.def("inference", &rasterization_api::inference_wrapper);
}
