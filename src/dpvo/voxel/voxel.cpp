#include <torch/extension.h>

// CUDA forward declaration
void voxel_cuda_forward(torch::Tensor voxel, torch::Tensor xs, torch::Tensor ys,
                        torch::Tensor ts, torch::Tensor ps, int Nbins);

void voxel_forward(torch::Tensor voxel, torch::Tensor xs, torch::Tensor ys,
                   torch::Tensor ts, torch::Tensor ps, int Nbins) {
  voxel_cuda_forward(voxel, xs, ys, ts, ps, Nbins);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &voxel_forward, "Voxel grid forward (in-place)");
}
