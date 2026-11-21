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

// 每个thread block内线程为BLOCKSIZE * BLOCKSIZE
constexpr int BS_XY = 16;  // thread block x & y
constexpr int TILE_BASE = BS_XY * 4;
constexpr int TILE_M = TILE_BASE * 2;
constexpr int TILE_N = TILE_M;
constexpr int TILE_K = TILE_BASE;

template <int M, int N, int K>
__global__ void MatMulKernel_sharedMemory(float* _A, float* _B, float* _C) {
    int row_block_start = blockIdx.y * TILE_M;
    int col_block_start = blockIdx.x * TILE_N;
    float values[8][8] = { 0.0, };
    extern __shared__ __align__(16) char sm[];
    float (&A)[M][K] = *reinterpret_cast<float(*)[M][K]>(_A);
    float (&B)[K][N] = *reinterpret_cast<float(*)[K][N]>(_B);
    float (&C)[M][N] = *reinterpret_cast<float(*)[M][N]>(_C);
    float (&asub)[TILE_BASE][TILE_K] = *reinterpret_cast<float(*)[TILE_BASE][TILE_K]>(sm);  // compute reuse
    float (&bsub)[TILE_K][TILE_BASE] = *reinterpret_cast<float(*)[TILE_K][TILE_BASE]>((float*)sm + TILE_K * TILE_BASE);  // io reuse

#pragma unroll
    for (int i = 0; i < K / TILE_K; ++i) {
        // part1
	*(float4*)(&asub[4 * threadIdx.y][4 * threadIdx.x]) = *(float4*)(&A[row_block_start + 4 * threadIdx.y][i * TILE_K + 4 * threadIdx.x]);
	*(float4*)(&asub[4 * threadIdx.y + 1][4 * threadIdx.x]) = *(float4*)(&A[row_block_start + 4 * threadIdx.y + 1][i * TILE_K + 4 * threadIdx.x]);
	*(float4*)(&asub[4 * threadIdx.y + 2][4 * threadIdx.x]) = *(float4*)(&A[row_block_start + 4 * threadIdx.y + 2][i * TILE_K + 4 * threadIdx.x]);
	*(float4*)(&asub[4 * threadIdx.y + 3][4 * threadIdx.x]) = *(float4*)(&A[row_block_start + 4 * threadIdx.y + 3][i * TILE_K + 4 * threadIdx.x]);

	*(float4*)(&bsub[4 * threadIdx.y][4 * threadIdx.x]) = *(float4*)(&B[i * TILE_K + 4 * threadIdx.y][col_block_start + 4 * threadIdx.x]);
	*(float4*)(&bsub[4 * threadIdx.y + 1][4 * threadIdx.x]) = *(float4*)(&B[i * TILE_K + 4 * threadIdx.y + 1][col_block_start + 4 * threadIdx.x]);
	*(float4*)(&bsub[4 * threadIdx.y + 2][4 * threadIdx.x]) = *(float4*)(&B[i * TILE_K + 4 * threadIdx.y + 2][col_block_start + 4 * threadIdx.x]);
	*(float4*)(&bsub[4 * threadIdx.y + 3][4 * threadIdx.x]) = *(float4*)(&B[i * TILE_K + 4 * threadIdx.y + 3][col_block_start + 4 * threadIdx.x]);

        __syncthreads();
#pragma unroll
        for (int j = 0; j < TILE_K; ++j) {
	    float4 b0 = *(float4*)(&bsub[j][4 * threadIdx.x]);
	    values[0][0] += asub[4 * threadIdx.y][j] * b0.x;
	    values[0][1] += asub[4 * threadIdx.y][j] * b0.y;
	    values[0][2] += asub[4 * threadIdx.y][j] * b0.z;
	    values[0][3] += asub[4 * threadIdx.y][j] * b0.w;

	    values[1][0] += asub[4 * threadIdx.y + 1][j] * b0.x;
	    values[1][1] += asub[4 * threadIdx.y + 1][j] * b0.y;
	    values[1][2] += asub[4 * threadIdx.y + 1][j] * b0.z;
	    values[1][3] += asub[4 * threadIdx.y + 1][j] * b0.w;

	    values[2][0] += asub[4 * threadIdx.y + 2][j] * b0.x;
	    values[2][1] += asub[4 * threadIdx.y + 2][j] * b0.y;
	    values[2][2] += asub[4 * threadIdx.y + 2][j] * b0.z;
	    values[2][3] += asub[4 * threadIdx.y + 2][j] * b0.w;

	    values[3][0] += asub[4 * threadIdx.y + 3][j] * b0.x;
	    values[3][1] += asub[4 * threadIdx.y + 3][j] * b0.y;
	    values[3][2] += asub[4 * threadIdx.y + 3][j] * b0.z;
	    values[3][3] += asub[4 * threadIdx.y + 3][j] * b0.w;
	}
	__syncthreads();

	// part2
	*(float4*)(&bsub[4 * threadIdx.y][4 * threadIdx.x]) = *(float4*)(&B[i * TILE_K + 4 * threadIdx.y][col_block_start + TILE_BASE+ 4 * threadIdx.x]);
	*(float4*)(&bsub[4 * threadIdx.y + 1][4 * threadIdx.x]) = *(float4*)(&B[i * TILE_K + 4 * threadIdx.y + 1][col_block_start + TILE_BASE + 4 * threadIdx.x]);
	*(float4*)(&bsub[4 * threadIdx.y + 2][4 * threadIdx.x]) = *(float4*)(&B[i * TILE_K + 4 * threadIdx.y + 2][col_block_start + TILE_BASE + 4 * threadIdx.x]);
	*(float4*)(&bsub[4 * threadIdx.y + 3][4 * threadIdx.x]) = *(float4*)(&B[i * TILE_K + 4 * threadIdx.y + 3][col_block_start + TILE_BASE + 4 * threadIdx.x]);
	__syncthreads();
#pragma unroll
        for (int j = 0; j < TILE_K; ++j) {
	    float4 b0 = *(float4*)(&bsub[j][4 * threadIdx.x]);
	    values[0][4] += asub[4 * threadIdx.y][j] * b0.x;
	    values[0][5] += asub[4 * threadIdx.y][j] * b0.y;
	    values[0][6] += asub[4 * threadIdx.y][j] * b0.z;
	    values[0][7] += asub[4 * threadIdx.y][j] * b0.w;

	    values[1][4] += asub[4 * threadIdx.y + 1][j] * b0.x;
	    values[1][5] += asub[4 * threadIdx.y + 1][j] * b0.y;
	    values[1][6] += asub[4 * threadIdx.y + 1][j] * b0.z;
	    values[1][7] += asub[4 * threadIdx.y + 1][j] * b0.w;

	    values[2][4] += asub[4 * threadIdx.y + 2][j] * b0.x;
	    values[2][5] += asub[4 * threadIdx.y + 2][j] * b0.y;
	    values[2][6] += asub[4 * threadIdx.y + 2][j] * b0.z;
	    values[2][7] += asub[4 * threadIdx.y + 2][j] * b0.w;

	    values[3][4] += asub[4 * threadIdx.y + 3][j] * b0.x;
	    values[3][5] += asub[4 * threadIdx.y + 3][j] * b0.y;
	    values[3][6] += asub[4 * threadIdx.y + 3][j] * b0.z;
	    values[3][7] += asub[4 * threadIdx.y + 3][j] * b0.w;
	}
	__syncthreads();

        // part3
	*(float4*)(&asub[4 * threadIdx.y][4 * threadIdx.x]) = *(float4*)(&A[row_block_start + TILE_BASE + 4 * threadIdx.y][i * TILE_K + 4 * threadIdx.x]);
	*(float4*)(&asub[4 * threadIdx.y + 1][4 * threadIdx.x]) = *(float4*)(&A[row_block_start + TILE_BASE + 4 * threadIdx.y + 1][i * TILE_K + 4 * threadIdx.x]);
	*(float4*)(&asub[4 * threadIdx.y + 2][4 * threadIdx.x]) = *(float4*)(&A[row_block_start + TILE_BASE + 4 * threadIdx.y + 2][i * TILE_K + 4 * threadIdx.x]);
	*(float4*)(&asub[4 * threadIdx.y + 3][4 * threadIdx.x]) = *(float4*)(&A[row_block_start + TILE_BASE + 4 * threadIdx.y + 3][i * TILE_K + 4 * threadIdx.x]);

	*(float4*)(&bsub[4 * threadIdx.y][4 * threadIdx.x]) = *(float4*)(&B[i * TILE_K + 4 * threadIdx.y][col_block_start + 4 * threadIdx.x]);
	*(float4*)(&bsub[4 * threadIdx.y + 1][4 * threadIdx.x]) = *(float4*)(&B[i * TILE_K + 4 * threadIdx.y + 1][col_block_start + 4 * threadIdx.x]);
	*(float4*)(&bsub[4 * threadIdx.y + 2][4 * threadIdx.x]) = *(float4*)(&B[i * TILE_K + 4 * threadIdx.y + 2][col_block_start + 4 * threadIdx.x]);
	*(float4*)(&bsub[4 * threadIdx.y + 3][4 * threadIdx.x]) = *(float4*)(&B[i * TILE_K + 4 * threadIdx.y + 3][col_block_start + 4 * threadIdx.x]);

        __syncthreads();
#pragma unroll
        for (int j = 0; j < TILE_K; ++j) {
	    float4 b0 = *(float4*)(&bsub[j][4 * threadIdx.x]);
	    values[4][0] += asub[4 * threadIdx.y][j] * b0.x;
	    values[4][1] += asub[4 * threadIdx.y][j] * b0.y;
	    values[4][2] += asub[4 * threadIdx.y][j] * b0.z;
	    values[4][3] += asub[4 * threadIdx.y][j] * b0.w;

	    values[5][0] += asub[4 * threadIdx.y + 1][j] * b0.x;
	    values[5][1] += asub[4 * threadIdx.y + 1][j] * b0.y;
	    values[5][2] += asub[4 * threadIdx.y + 1][j] * b0.z;
	    values[5][3] += asub[4 * threadIdx.y + 1][j] * b0.w;

	    values[6][0] += asub[4 * threadIdx.y + 2][j] * b0.x;
	    values[6][1] += asub[4 * threadIdx.y + 2][j] * b0.y;
	    values[6][2] += asub[4 * threadIdx.y + 2][j] * b0.z;
	    values[6][3] += asub[4 * threadIdx.y + 2][j] * b0.w;

	    values[7][0] += asub[4 * threadIdx.y + 3][j] * b0.x;
	    values[7][1] += asub[4 * threadIdx.y + 3][j] * b0.y;
	    values[7][2] += asub[4 * threadIdx.y + 3][j] * b0.z;
	    values[7][3] += asub[4 * threadIdx.y + 3][j] * b0.w;
	}
	__syncthreads();

	// part4
	*(float4*)(&bsub[4 * threadIdx.y][4 * threadIdx.x]) = *(float4*)(&B[i * TILE_K + 4 * threadIdx.y][col_block_start + TILE_BASE + 4 * threadIdx.x]);
	*(float4*)(&bsub[4 * threadIdx.y + 1][4 * threadIdx.x]) = *(float4*)(&B[i * TILE_K + 4 * threadIdx.y + 1][col_block_start + TILE_BASE + 4 * threadIdx.x]);
	*(float4*)(&bsub[4 * threadIdx.y + 2][4 * threadIdx.x]) = *(float4*)(&B[i * TILE_K + 4 * threadIdx.y + 2][col_block_start + TILE_BASE + 4 * threadIdx.x]);
	*(float4*)(&bsub[4 * threadIdx.y + 3][4 * threadIdx.x]) = *(float4*)(&B[i * TILE_K + 4 * threadIdx.y + 3][col_block_start + TILE_BASE + 4 * threadIdx.x]);
	__syncthreads();
#pragma unroll
        for (int j = 0; j < TILE_K; ++j) {
	    float4 b0 = *(float4*)(&bsub[j][4 * threadIdx.x]);
	    values[4][4] += asub[4 * threadIdx.y][j] * b0.x;
	    values[4][5] += asub[4 * threadIdx.y][j] * b0.y;
	    values[4][6] += asub[4 * threadIdx.y][j] * b0.z;
	    values[4][7] += asub[4 * threadIdx.y][j] * b0.w;

	    values[5][4] += asub[4 * threadIdx.y + 1][j] * b0.x;
	    values[5][5] += asub[4 * threadIdx.y + 1][j] * b0.y;
	    values[5][6] += asub[4 * threadIdx.y + 1][j] * b0.z;
	    values[5][7] += asub[4 * threadIdx.y + 1][j] * b0.w;

	    values[6][4] += asub[4 * threadIdx.y + 2][j] * b0.x;
	    values[6][5] += asub[4 * threadIdx.y + 2][j] * b0.y;
	    values[6][6] += asub[4 * threadIdx.y + 2][j] * b0.z;
	    values[6][7] += asub[4 * threadIdx.y + 2][j] * b0.w;

	    values[7][4] += asub[4 * threadIdx.y + 3][j] * b0.x;
	    values[7][5] += asub[4 * threadIdx.y + 3][j] * b0.y;
	    values[7][6] += asub[4 * threadIdx.y + 3][j] * b0.z;
	    values[7][7] += asub[4 * threadIdx.y + 3][j] * b0.w;
	}
	__syncthreads();
}

    *(float4*)(&C[row_block_start + 4 * threadIdx.y][col_block_start + 4 * threadIdx.x]) = *(float4*)(&values[0][0]);
    *(float4*)(&C[row_block_start + 4 * threadIdx.y + 1][col_block_start + 4 * threadIdx.x]) = *(float4*)(&values[1][0]);
    *(float4*)(&C[row_block_start + 4 * threadIdx.y + 2][col_block_start + 4 * threadIdx.x]) = *(float4*)(&values[2][0]);
    *(float4*)(&C[row_block_start + 4 * threadIdx.y + 3][col_block_start + 4 * threadIdx.x]) = *(float4*)(&values[3][0]);
    *(float4*)(&C[row_block_start + 4 * threadIdx.y][col_block_start + TILE_BASE + 4 * threadIdx.x]) = *(float4*)(&values[0][4]);
    *(float4*)(&C[row_block_start + 4 * threadIdx.y + 1][col_block_start + TILE_BASE + 4 * threadIdx.x]) = *(float4*)(&values[1][4]);
    *(float4*)(&C[row_block_start + 4 * threadIdx.y + 2][col_block_start + TILE_BASE + 4 * threadIdx.x]) = *(float4*)(&values[2][4]);
    *(float4*)(&C[row_block_start + 4 * threadIdx.y + 3][col_block_start + TILE_BASE + 4 * threadIdx.x]) = *(float4*)(&values[3][4]);

    *(float4*)(&C[row_block_start + TILE_BASE + 4 * threadIdx.y][col_block_start + 4 * threadIdx.x]) = *(float4*)(&values[4][0]);
    *(float4*)(&C[row_block_start + TILE_BASE + 4 * threadIdx.y + 1][col_block_start + 4 * threadIdx.x]) = *(float4*)(&values[5][0]);
    *(float4*)(&C[row_block_start + TILE_BASE + 4 * threadIdx.y + 2][col_block_start + 4 * threadIdx.x]) = *(float4*)(&values[6][0]);
    *(float4*)(&C[row_block_start + TILE_BASE + 4 * threadIdx.y + 3][col_block_start + 4 * threadIdx.x]) = *(float4*)(&values[7][0]);
    *(float4*)(&C[row_block_start + TILE_BASE + 4 * threadIdx.y][col_block_start + TILE_BASE + 4 * threadIdx.x]) = *(float4*)(&values[4][4]);
    *(float4*)(&C[row_block_start + TILE_BASE + 4 * threadIdx.y + 1][col_block_start + TILE_BASE + 4 * threadIdx.x]) = *(float4*)(&values[5][4]);
    *(float4*)(&C[row_block_start + TILE_BASE + 4 * threadIdx.y + 2][col_block_start + TILE_BASE + 4 * threadIdx.x]) = *(float4*)(&values[6][4]);
    *(float4*)(&C[row_block_start + TILE_BASE + 4 * threadIdx.y + 3][col_block_start + TILE_BASE + 4 * threadIdx.x]) = *(float4*)(&values[7][4]);
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

  dim3 dimBlock(BS_XY, BS_XY);
  dim3 dimGrid(N / TILE_N, M / TILE_M);
  CUDA_CALL(cudaEventRecord(start));
  if (enableProfiler) {
    cudaProfilerStart();
  }
  cudaDeviceSetCacheConfig(cudaFuncCachePreferNone);
  MatMulKernel_sharedMemory<M, N, K><<<dimGrid, dimBlock, 2 * sizeof(float) * TILE_BASE * TILE_K>>>(d_A, d_B, d_C);
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
                  M, K, N, BS_XY, BS_XY, TILE_M, TILE_K, TILE_N, enableProfiler ? 1: 0, accMillis, accMillis / times);
  printf("Final result check: %s\n", resultCorrect ? "PASS" : "FAIL");
  return resultCorrect ? 0 : 1;
}
