#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#define THREADS 512

__global__ void trilinear_voxel_kernel(
    float *__restrict__ voxel, const float *__restrict__ xs,
    const float *__restrict__ ys, const float *__restrict__ ts,
    const float *__restrict__ ps, const int N, const int W, const int H,
    const float t0, const float alpha, const int Nbins) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N)
    return;

  const float x = xs[i];
  const float y = ys[i];
  float t = (ts[i] - t0) * alpha;
  const float p = 2.0f * ps[i] - 1.0f; // 0 -> -1, 1 -> +1

  const int lim_x = __float2int_rd(x);
  const int lim_y = __float2int_rd(y);
  const int lim_t = __float2int_rd(t);

  if (lim_x < 0 || lim_y < 0 || lim_t < 0 || lim_x >= W - 1 || lim_y >= H - 1 ||
      lim_t >= Nbins)
    return;

  const float dx = (float)lim_x - x;
  const float dy = (float)lim_y - y;
  const float dt = (float)lim_t - t;

  const float dxy = dx * dy;
  const float dyt = dy * dt;
  const float dxt = dt * dx;

  const float v000 = dxy * dt;
  const float v001 = dxy + v000;
  const float v010 = dxt + v000;
  const float v100 = dyt + v000;
  const float v011 = v001 + dx + dxt;
  const float tt = dy + dyt;
  const float v101 = v001 + tt;
  const float v110 = v010 + dt + dyt;
  const float v111 = v011 + 1.0f + dt + tt;

  const int HW = H * W;
  const int idx = lim_t * HW + lim_y * W + lim_x;

  atomicAdd(voxel + idx, p * v111);
  atomicAdd(voxel + idx + 1, -p * v011);
  atomicAdd(voxel + idx + W, -p * v101);
  atomicAdd(voxel + idx + W + 1, p * v001);
  atomicAdd(voxel + idx + HW, -p * v110);
  atomicAdd(voxel + idx + HW + 1, p * v010);
  atomicAdd(voxel + idx + HW + W, p * v100);
  atomicAdd(voxel + idx + HW + W + 1, -p * v000);
}

void voxel_cuda_forward(torch::Tensor voxel, torch::Tensor xs, torch::Tensor ys,
                        torch::Tensor ts, torch::Tensor ps, int Nbins) {
  const int N = xs.size(0);
  const int H = voxel.size(1);
  const int W = voxel.size(2);

  // Zero the buffer
  voxel.zero_();

  if (N == 0)
    return;

  const float t0 = ts[0].item<float>();
  const float t_end = ts[N - 1].item<float>();
  const float duration = t_end - t0;
  if (duration <= 0.0f)
    return;

  const float alpha = (float)(Nbins - 1) / duration;

  const int blocks = (N + THREADS - 1) / THREADS;
  trilinear_voxel_kernel<<<blocks, THREADS>>>(
      voxel.data_ptr<float>(), xs.data_ptr<float>(), ys.data_ptr<float>(),
      ts.data_ptr<float>(), ps.data_ptr<float>(), N, W, H, t0, alpha, Nbins);
}
