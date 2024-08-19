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
constexpr int BS_XY = 16;  // thread block x & y
constexpr int BS_TILE = BS_XY * 4;

template <int M, int N, int K>
__global__ void MatMulKernel_sharedMemory(float* A, float* B, float* C) {
  int row_block_start = blockIdx.y * BS_TILE;
  int col_block_start = blockIdx.x * BS_TILE * 2;
  float values[4][8] = { 0.0, };
  extern __shared__ char sm[];
  float* asub = (float*)sm;   // [BS_TILE][BS_TILE];
  float* bsub = (float*)sm + BS_TILE * BS_TILE;  // [BS_TILE][BS_TILE];

#pragma unroll
  for (int i = 0; i < K / (2 * BS_TILE); ++i) {
          // part1
          // hard code.
#pragma unroll
          for (int j = 0; j < 4; ++j) {
#pragma unroll
                 for (int k = 0; k < 4; ++k) {  // 加载时完成转置
                        asub[(4 * threadIdx.x + j) * BS_TILE + 4 * threadIdx.y + k] = A[(row_block_start + 4 * threadIdx.y + k) * K + i * 2 * BS_TILE + 4 * threadIdx.x + j];
                 }
          }
          *(float4*)(&bsub[(4 * threadIdx.y) * BS_TILE + 4 * threadIdx.x]) = *(float4*)(B + (i * 2 * BS_TILE + 4 * threadIdx.y) * N + col_block_start + (blockIdx.y % 2) * BS_TILE + 4 * threadIdx.x);
          *(float4*)(&bsub[(4 * threadIdx.y + 1) * BS_TILE + 4 * threadIdx.x]) = *(float4*)(B + (i * 2 * BS_TILE + 4 * threadIdx.y + 1) * N + col_block_start + (blockIdx.y % 2) * BS_TILE + 4 * threadIdx.x);
          *(float4*)(&bsub[(4 * threadIdx.y + 2) * BS_TILE + 4 * threadIdx.x]) = *(float4*)(B + (i * 2 * BS_TILE + 4 * threadIdx.y + 2) * N + col_block_start + (blockIdx.y % 2) * BS_TILE + 4 * threadIdx.x);
          *(float4*)(&bsub[(4 * threadIdx.y + 3) * BS_TILE + 4 * threadIdx.x]) = *(float4*)(B + (i * 2 * BS_TILE + 4 * threadIdx.y + 3) * N + col_block_start + (blockIdx.y % 2) * BS_TILE + 4 * threadIdx.x);

          __syncthreads();

#pragma unroll
          for (int j = 0; j < BS_TILE; ++j) {
                float4 b0 = *(float4*)(&bsub[j * BS_TILE + 4 * threadIdx.x]);
                float4 a0 = *(float4*)(&asub[j * BS_TILE + 4 * threadIdx.y]);
                values[0][0] += a0.x * b0.x;
                values[0][1] += a0.x * b0.y;
                values[0][2] += a0.x * b0.z;
                values[0][3] += a0.x * b0.w;

                values[1][0] += a0.y * b0.x;
                values[1][1] += a0.y * b0.y;
                values[1][2] += a0.y * b0.z;
                values[1][3] += a0.y * b0.w;

                values[2][0] += a0.z * b0.x;
                values[2][1] += a0.z * b0.y;
                values[2][2] += a0.z * b0.z;
                values[2][3] += a0.z * b0.w;

                values[3][0] += a0.w * b0.x;
                values[3][1] += a0.w * b0.y;
                values[3][2] += a0.w * b0.z;
                values[3][3] += a0.w * b0.w;
          }

          // part2
          // hard code.
#pragma unroll
          for (int j = 0; j < 4; ++j) {
#pragma unroll
                 for (int k = 0; k < 4; ++k) {  // 加载时完成转置
                        asub[(4 * threadIdx.x + j) * BS_TILE + 4 * threadIdx.y + k] = A[(row_block_start + 4 * threadIdx.y + k) * K + i * 2 * BS_TILE + BS_TILE + 4 * threadIdx.x + j];
                 }
          }
          *(float4*)(&bsub[(4 * threadIdx.y) * BS_TILE + 4 * threadIdx.x]) = *(float4*)(B + (i * 2 * BS_TILE + 4 * threadIdx.y + BS_TILE) * N + col_block_start + (blockIdx.y % 2) * BS_TILE + 4 * threadIdx.x);
          *(float4*)(&bsub[(4 * threadIdx.y + 1) * BS_TILE + 4 * threadIdx.x]) = *(float4*)(B + (i * 2 * BS_TILE + 4 * threadIdx.y + BS_TILE + 1) * N + col_block_start + (blockIdx.y % 2) * BS_TILE + 4 * threadIdx.x);
          *(float4*)(&bsub[(4 * threadIdx.y + 2) * BS_TILE + 4 * threadIdx.x]) = *(float4*)(B + (i * 2 * BS_TILE + 4 * threadIdx.y + BS_TILE + 2) * N + col_block_start + (blockIdx.y % 2) * BS_TILE + 4 * threadIdx.x);
          *(float4*)(&bsub[(4 * threadIdx.y + 3) * BS_TILE + 4 * threadIdx.x]) = *(float4*)(B + (i * 2 * BS_TILE + 4 * threadIdx.y + BS_TILE + 3) * N + col_block_start + (blockIdx.y % 2) * BS_TILE + 4 * threadIdx.x);

          __syncthreads();

#pragma unroll
          for (int j = 0; j < BS_TILE; ++j) {
                float4 b0 = *(float4*)(&bsub[j * BS_TILE + 4 * threadIdx.x]);
                float4 a0 = *(float4*)(&asub[j * BS_TILE + 4 * threadIdx.y]);
                values[0][4] += a0.x * b0.x;
                values[0][5] += a0.x * b0.y;
                values[0][6] += a0.x * b0.z;
                values[0][7] += a0.x * b0.w;

                values[1][4] += a0.y * b0.x;
                values[1][5] += a0.y * b0.y;
                values[1][6] += a0.y * b0.z;
                values[1][7] += a0.y * b0.w;

                values[2][4] += a0.z * b0.x;
                values[2][5] += a0.z * b0.y;
                values[2][6] += a0.z * b0.z;
                values[2][7] += a0.z * b0.w;

                values[3][4] += a0.w * b0.x;
                values[3][5] += a0.w * b0.y;
                values[3][6] += a0.w * b0.z;
                values[3][7] += a0.w * b0.w;
          }
  }

  *(float4*)(C + (row_block_start + 4 * threadIdx.y) * K + col_block_start + 8 * threadIdx.x) = *(float4*)values;
  *(float4*)(C + (row_block_start + 4 * threadIdx.y + 1) * K + col_block_start + 8 * threadIdx.x) = *(float4*)(&values[1][0]);
  *(float4*)(C + (row_block_start + 4 * threadIdx.y + 2) * K + col_block_start + 8 * threadIdx.x) = *(float4*)(&values[2][0]);
  *(float4*)(C + (row_block_start + 4 * threadIdx.y + 3) * K + col_block_start + 8 * threadIdx.x) = *(float4*)(&values[3][0]);

  *(float4*)(C + (row_block_start + 4 * threadIdx.y) * K + col_block_start + 8 * threadIdx.x + 4) = *(float4*)(&values[0][4]);
  *(float4*)(C + (row_block_start + 4 * threadIdx.y + 1) * K + col_block_start + 8 * threadIdx.x + 4) = *(float4*)(&values[1][4]);
  *(float4*)(C + (row_block_start + 4 * threadIdx.y + 2) * K + col_block_start + 8 * threadIdx.x + 4) = *(float4*)(&values[2][4]);
  *(float4*)(C + (row_block_start + 4 * threadIdx.y + 3) * K + col_block_start + 8 * threadIdx.x + 4) = *(float4*)(&values[3][4]);
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
  dim3 dimGrid(N / BS_TILE / 2, M / BS_TILE);
  CUDA_CALL(cudaEventRecord(start));
  if (enableProfiler) {
    cudaProfilerStart();
  }
  cudaDeviceSetCacheConfig(cudaFuncCachePreferNone);
  MatMulKernel_sharedMemory<M, N, K><<<dimGrid, dimBlock, 2 * sizeof(float) * BS_TILE * BS_TILE>>>(d_A, d_B, d_C);
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

constexpr int M = 3840, K = 2880, N = 3840;
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
