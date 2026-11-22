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

#include "check_result.h"

// change the threads per block to 16x32, each thread calcs 8x8 outputs,so the tile is 128x256, which is the same as v10.
constexpr int BS_X = 32;  // thread block x & y
constexpr int BS_Y = 16;
// constexpr int TILE_BASE = BS_XY * 8;
constexpr int TILE_M = BS_Y * 8;
constexpr int TILE_N = BS_X * 8;
constexpr int TILE_K = 16;

template <int M, int N, int K>
__global__ void MatMulKernel_sharedMemory(float* _A, float* _B, float* _C) {
    int row_block_start = blockIdx.y * TILE_M;
    int col_block_start = blockIdx.x * TILE_N;
    float values[8][8] = { 0.0, };
    extern __shared__ __align__(16) char sm[];
    float (&A)[M][K] = *reinterpret_cast<float(*)[M][K]>(_A);
    float (&B)[K][N] = *reinterpret_cast<float(*)[K][N]>(_B);
    float (&C)[M][N] = *reinterpret_cast<float(*)[M][N]>(_C);
    float (&asub)[2][TILE_K][TILE_M + 2] = *reinterpret_cast<float(*)[2][TILE_K][TILE_M + 2]>(sm);  // 8320 bytes per buffer, +2 for interleave
    float (&bsub)[2][TILE_K][TILE_N] = *reinterpret_cast<float(*)[2][TILE_K][TILE_N]>((float*)sm + 2 * TILE_K * (TILE_M + 2));  // 16K per buffer

    auto loadTile = [&](int i, int buf) {
	auto logic_xidx = (threadIdx.x % 16); // for asub loading, [0,15)
	auto logic_yidx = 2 * threadIdx.y + threadIdx.x / 16;  // [0, 31)
        asub[buf][logic_xidx][logic_yidx] = A[row_block_start + logic_yidx][i * TILE_K + logic_xidx];  // transpose when load, interleave
        asub[buf][logic_xidx][32 + logic_yidx] = A[row_block_start + 32 + logic_yidx][i * TILE_K + logic_xidx];
        asub[buf][logic_xidx][64 + logic_yidx] = A[row_block_start + 64 + logic_yidx][i * TILE_K + logic_xidx];
        asub[buf][logic_xidx][96 + logic_yidx] = A[row_block_start + 96 + logic_yidx][i * TILE_K + logic_xidx];

	*(float4*)(&bsub[buf][threadIdx.y][4 * threadIdx.x]) = *(float4*)(&B[i * TILE_K + threadIdx.y][col_block_start + 4 * threadIdx.x]);
	*(float4*)(&bsub[buf][threadIdx.y][TILE_N / 2 + 4 * threadIdx.x]) = *(float4*)(&B[i * TILE_K + threadIdx.y][col_block_start + TILE_N / 2 + 4 * threadIdx.x]);  // interleave

    };

    loadTile(0, 0);
    __syncthreads();

#pragma unroll
    for (int i = 0; i < K / TILE_K; ++i) {
        int cur = i & 1;
        int next = cur ^ 1;
        if (i + 1 < K / TILE_K) {
            loadTile(i + 1, next);
        }

#pragma unroll
        for (int j = 0; j < TILE_K; ++j) {
	    float astripe[8], bstripe[8];
	    #pragma unroll
	    for (int k = 0; k < 8; ++k) {
	        astripe[k] = asub[cur][j][8 * threadIdx.y + k];  // broadcast
	    }
            #pragma unroll
	    for (int k = 0; k < 8; ++k) {
		bstripe[k] = bsub[cur][j][k * 32 + threadIdx.x];  // interleave to elimate 4-way conflict!
	    }
#pragma unroll
	    for (int k = 0; k < 8; ++k) {
#pragma unroll
		for (int l = 0; l < 8; ++l) {
		    values[k][l] += astripe[k] * bstripe[l];
		}
	    }
	}
	__syncthreads();
    }

#pragma unroll
    for (int i = 0; i < 8; ++i) {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
	    C[row_block_start + 8 * threadIdx.y + i][col_block_start + j * 32 + threadIdx.x] = values[i][j];
	}
    }
}

template <int M, int K, int N, bool enableProfiler>
float testMatMul(bool checkResult, bool& resultCorrect) {
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

  dim3 dimBlock(BS_X, BS_Y);
  dim3 dimGrid(N / TILE_N, M / TILE_M);
  constexpr int sharedMemoryBytes = 2 * sizeof(float) * (TILE_M + TILE_N + 2) * TILE_K;
  CUDA_CALL(cudaFuncSetAttribute(
      MatMulKernel_sharedMemory<M, N, K>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      sharedMemoryBytes));
  CUDA_CALL(cudaEventRecord(start));
  if (enableProfiler) {
    cudaProfilerStart();
  }
  cudaDeviceSetCacheConfig(cudaFuncCachePreferNone);
  MatMulKernel_sharedMemory<M, N, K><<<dimGrid, dimBlock, sharedMemoryBytes>>>(d_A, d_B, d_C);
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

  if (checkResult) {
    resultCorrect = checkResultAgainstCublas(d_A, d_B, d_C, M, K, N) && resultCorrect;
  }

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

  constexpr int times = 100;
  constexpr bool checkResult = true, enableProfiler = false;
  float accMillis = 0.0;
  bool resultCorrect = true;
  for (int i = 0; i <  times; ++i) {
    accMillis += testMatMul<M, K, N, enableProfiler>(checkResult && i == times - 1, resultCorrect);
    if (((i + 1) % 10) == 0) {
      printf("Testing process: %d / %d\n", (i + 1), times);
    }
  }
  printf("M=%d, K=%d, N=%d, bs_m=%d, bs_n=%d, tile_m=%d, tile_k=%d, tile_n=%d, enableProfiler=%d, MatMul: Totally elapsed time in GPU was %.2f ms, %.2f ms per operation\n",
                  M, K, N, BS_Y, BS_Y, TILE_M, TILE_K, TILE_N, enableProfiler ? 1: 0, accMillis, accMillis / times);
  printf("Final result check: %s\n", resultCorrect ? "PASS" : "FAIL");
  return resultCorrect ? 0 : 1;
}

