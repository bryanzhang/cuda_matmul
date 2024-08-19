#include <cstdio>
#include <cstdlib>
#include <cmath>

#include <iostream>

#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <cublas_v2.h>

#define CUDA_CALL(func) \
    do { \
      cudaError_t err = (func); \
      if(err != cudaSuccess) { \
        printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
      } \
    } while(0)


// 每个thread block内线程为BLOCKSIZE * BLOCKSIZE
constexpr int BS_XY = 32;  // thread block x & y
constexpr int BS_TILE = BS_XY * 2;

template <int M, int N, int K>
__global__ void MatMulKernel_sharedMemory(float* A, float* B, float* C) {
  int row_block_start = blockIdx.y * BS_TILE;
  int col_block_start = blockIdx.x * BS_TILE;
  float values[2][2] = { 0.0, };

#pragma unroll
  for (int i = 0; i < K / BS_TILE; ++i) {
          __shared__ float asub[BS_TILE][BS_TILE];
          __shared__ float bsub[BS_TILE][BS_TILE];

          // hard code.
          float2 a0 = *(float2*)(A + (row_block_start + 2 * threadIdx.y) * K + i * BS_TILE + 2 * threadIdx.x);
          asub[2 * threadIdx.y][2 * threadIdx.x] = a0.x;
          asub[2 * threadIdx.y][2 * threadIdx.x + 1] = a0.y;

          float2 a1 = *(float2*)(A + (row_block_start + 2 * threadIdx.y + 1) * K + i * BS_TILE + 2 * threadIdx.x);
          asub[2 * threadIdx.y + 1][2 * threadIdx.x] = a1.x;
          asub[2 * threadIdx.y + 1][2 * threadIdx.x + 1] = a1.y;

          float2 b0 = *(float2*)(B + (i * BS_TILE + 2 * threadIdx.y) * N + col_block_start + 2 * threadIdx.x);
          bsub[2 * threadIdx.y][2 * threadIdx.x] = b0.x;
          bsub[2 * threadIdx.y][2 * threadIdx.x + 1] = b0.y;

          float2 b1 = *(float2*)(B + (i * BS_TILE + 2 * threadIdx.y + 1) * N + col_block_start + 2 * threadIdx.x);
          bsub[2 * threadIdx.y + 1][2 * threadIdx.x] = b1.x;
          bsub[2 * threadIdx.y + 1][2 * threadIdx.x + 1] = b1.y;

          __syncthreads();
          // transpose(original_bsub, bsub);

          for (int j = 0; j < BS_TILE; ++j) {
                values[0][0] += asub[2 * threadIdx.y][j] * bsub[j][2 * threadIdx.x];
                values[0][1] += asub[2 * threadIdx.y][j] * bsub[j][2 * threadIdx.x + 1];
                values[1][0] += asub[2 * threadIdx.y + 1][j] * bsub[j][2 * threadIdx.x];
                values[1][1] += asub[2 * threadIdx.y + 1][j] * bsub[j][2 * threadIdx.x + 1];
          }
  }
  *(float2*)(C + (row_block_start + 2 * threadIdx.y) * K + col_block_start + 2 * threadIdx.x) = *(float2*)values;
  *(float2*)(C + (row_block_start + 2 * threadIdx.y + 1) * K + col_block_start + 2 * threadIdx.x) = *(float2*)(values + 2);
}

template <int M, int K, int N, bool enableProfiler>
float testMatMul(bool checkResult) {
  float* h_A, *h_B, *h_C;
  float* d_A, *d_B, *d_C;

  // 分配主机内存.
  h_A = (float*)malloc(M * K * sizeof(float));
  h_B = (float*)malloc(K * N * sizeof(float));
  h_C = (float*)malloc(M * N * sizeof(float));

  // 分配设备内存
  CUDA_CALL(cudaMalloc((void**)&d_A, M * K * sizeof(float)));
  CUDA_CALL(cudaMalloc((void**)&d_B, K * N * sizeof(float)));
  CUDA_CALL(cudaMalloc((void**)&d_C, M * N * sizeof(float)));

  // 初始化主机内存
  for (int i = 0; i < M * K; ++i) {
    h_A[i] = rand() / (float)RAND_MAX;
  }
  for (int i = 0; i < K * N; ++i) {
    h_B[i] = rand() / (float)RAND_MAX;
  }

  // 将数据从主机内存拷贝到设备内存
  CUDA_CALL(cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice));

  // 创建CUDA事件
  cudaEvent_t start, stop;
  CUDA_CALL(cudaEventCreate(&start));
  CUDA_CALL(cudaEventCreate(&stop));

  dim3 dimBlock(BS_XY, BS_XY);
  dim3 dimGrid(N / BS_TILE, M / BS_TILE);
  CUDA_CALL(cudaEventRecord(start));
  if (enableProfiler) {
    cudaProfilerStart();
  }
  MatMulKernel_sharedMemory<M, N, K><<<dimGrid, dimBlock>>>(d_A, d_B, d_C);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
      printf("Error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err));
      exit(EXIT_FAILURE);
  }

  CUDA_CALL(cudaDeviceSynchronize());
  if (enableProfiler) {
    cudaProfilerStop();
  }
  CUDA_CALL(cudaEventRecord(stop));
  CUDA_CALL(cudaEventSynchronize(stop));

  float milliseconds = 0;
  CUDA_CALL(cudaEventElapsedTime(&milliseconds, start, stop));

  // 清理
  free(h_A);
  free(h_B);
  free(h_C);
  CUDA_CALL(cudaFree(d_A));
  CUDA_CALL(cudaFree(d_B));
  CUDA_CALL(cudaFree(d_C));
  CUDA_CALL(cudaEventDestroy(start));
  CUDA_CALL(cudaEventDestroy(stop));
  return milliseconds;
}

int main() {
  cudaDeviceProp prop;
  int count;
  cudaGetDeviceCount(&count);
  for (int i = 0; i < count; i++) {
    cudaGetDeviceProperties(&prop, i);
    std::cout << "Device " << i << ":\n";
    std::cout << "Maximum threads per block: " << prop.maxThreadsPerBlock << "\n";
    std::cout << "Maximum dimension size of a thread block: (" << prop.maxThreadsDim[0] << ", " << prop.maxThreadsDim[1] << ", " << prop.maxThreadsDim[2] << ")\n";
    std::cout << "Maximum dimension size of a grid size: (" << prop.maxGridSize[0] << ", " << prop.maxGridSize[1] << ", " << prop.maxGridSize[2] << ")\n";
  }

  constexpr int times = 30;
  constexpr int M = 3840, K = 2880, N = 3840;
  constexpr bool checkResult = false, enableProfiler = false;
  float accMillis = 0.0;
  for (int i = 0; i <  times; ++i) {
    accMillis += testMatMul<M, K, N, enableProfiler>(checkResult);
    if (((i + 1) % 10) == 0) {
      printf("Testing process: %d / %d\n", (i + 1), times);
    }
  }
  printf("M=%d, K=%d, N=%d, bs_m=%d, bs_n=%d, bs_k=%d, enableProfiler=%d, MatMul: Totally elapsed time in GPU was %.2f ms, %.2f ms per operation\n",
                  M, K, N, BS_XY, BS_XY, BS_TILE, enableProfiler ? 1: 0, accMillis, accMillis / times);
}
