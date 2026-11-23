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
// v16: based on v14. Rows 8..15 of asub store column ly at ly ^ 8, which removes the 2-way bank conflicts
// when writing asub. The float4 load from A and the float4 reads from asub are unchanged.
// v17: based on v16.
//   1. bsub is read with two LDS.128 per k (columns [4tx, 4tx+4) and [128+4tx, 128+4tx+4)) instead of 8 LDS.32,
//      and C is written with float4 stores.
//   2. Wave quantization: 450 tiles on 40 SMs is 11.25 waves. The 440 tiles of the full waves run as before;
//      each of the 10 tail tiles is split into 4 K-chunks (40 blocks) that atomicAdd into a zeroed C.
//   3. The K loop now has a runtime trip count, so unroll it by 4 explicitly and cap registers at 128
//      with __launch_bounds__(512, 1).
constexpr int BS_X = 32;  // thread block x & y
constexpr int BS_Y = 16;
// constexpr int TILE_BASE = BS_XY * 8;
constexpr int TILE_M = BS_Y * 8;
constexpr int TILE_N = BS_X * 8;
constexpr int TILE_K = 16;

template <int M, int N, int K>
__global__ void __launch_bounds__(512, 1) MatMulKernel_sharedMemory(float* _A, float* _B, float* _C, int full_tiles, int split) {
    // Wave-quantization fix: tiles [0, full_tiles) are computed whole; each remaining tail tile is split
    // into `split` K-chunks handled by separate blocks that accumulate into C with atomicAdd.
    constexpr int TILES_X = N / TILE_N;
    constexpr int K_TILES = K / TILE_K;
    const bool is_split = blockIdx.x >= full_tiles;
    int tile = blockIdx.x, k_tile_begin = 0, k_tiles = K_TILES;
    if (is_split) {
        const int t = blockIdx.x - full_tiles;
        tile = full_tiles + t / split;
        k_tiles = K_TILES / split;
        k_tile_begin = (t % split) * k_tiles;
    }
    const int row_block_start = (tile / TILES_X) * TILE_M;
    const int col_block_start = (tile % TILES_X) * TILE_N;
    float values[8][8] = { 0.0, };
    extern __shared__ __align__(16) char sm[];
    float (&A)[M][K] = *reinterpret_cast<float(*)[M][K]>(_A);
    float (&B)[K][N] = *reinterpret_cast<float(*)[K][N]>(_B);
    float (&C)[M][N] = *reinterpret_cast<float(*)[M][N]>(_C);
    float (&asub)[2][TILE_K][TILE_M + 4] = *reinterpret_cast<float(*)[2][TILE_K][TILE_M + 4]>(sm);  // 8400 bytes per buffer, +4 for interleave
    float (&bsub)[2][TILE_K][TILE_N] = *reinterpret_cast<float(*)[2][TILE_K][TILE_N]>((float*)sm + 2 * TILE_K * (TILE_M + 4));  // 16K per buffer

    const int logic_xidx = (threadIdx.x % 4); // for asub loading, [0,4)
    const int logic_yidx = 8 * threadIdx.y + threadIdx.x / 4;  // [0, 128)
    // Rows 8..15 of asub (written by lx = 2, 3) store column ly at ly ^ 8, which moves them 8 banks away
    // from rows 0..7 (lx = 0, 1): bank = (16 * lx + 4 * c + (ly ^ 8)) % 32 -> conflict-free.
    const int store_col = logic_yidx ^ ((logic_xidx >> 1) << 3);
    auto loadTile = [&](int i, int buf) {
	float4 astripe = *(float4*)(&A[row_block_start + logic_yidx][i * TILE_K + logic_xidx * 4]);  // transponse when laod.
	asub[buf][4 * logic_xidx + 0][store_col] = astripe.x;
	asub[buf][4 * logic_xidx + 1][store_col] = astripe.y;
	asub[buf][4 * logic_xidx + 2][store_col] = astripe.z;
	asub[buf][4 * logic_xidx + 3][store_col] = astripe.w;

	*(float4*)(&bsub[buf][threadIdx.y][4 * threadIdx.x]) = *(float4*)(&B[i * TILE_K + threadIdx.y][col_block_start + 4 * threadIdx.x]);
	*(float4*)(&bsub[buf][threadIdx.y][TILE_N / 2 + 4 * threadIdx.x]) = *(float4*)(&B[i * TILE_K + threadIdx.y][col_block_start + TILE_N / 2 + 4 * threadIdx.x]);  // interleave

    };

    loadTile(k_tile_begin, 0);
    __syncthreads();

#pragma unroll 4
    for (int i = 0; i < k_tiles; ++i) {
        const int cur = i & 1;
        const int next = cur ^ 1;
        const bool has_next = i + 1 < k_tiles;
        const int next_k = (k_tile_begin + i + 1) * TILE_K;

        auto computeRange = [&](int j_begin, int j_end) {
#pragma unroll
            for (int j = j_begin; j < j_end; ++j) {
   	        float bstripe[8];
		const int acol = (j < 8) ? 8 * threadIdx.y : 8 * (threadIdx.y ^ 1);  // undo the ^8 swizzle of rows 8..15
		float4 a0 = *(const float4*)(
				    &asub[cur][j][acol]);
		float4 a1 = *(const float4*)(
				    &asub[cur][j][acol + 4]);
		// two LDS.128: columns [4tx, 4tx+4) and [128+4tx, 128+4tx+4); a quarter warp reads 128 contiguous bytes -> conflict-free
		const float4 b0 = *(const float4*)(&bsub[cur][j][4 * threadIdx.x]);
		const float4 b1 = *(const float4*)(&bsub[cur][j][TILE_N / 2 + 4 * threadIdx.x]);
		bstripe[0] = b0.x; bstripe[1] = b0.y; bstripe[2] = b0.z; bstripe[3] = b0.w;
		bstripe[4] = b1.x; bstripe[5] = b1.y; bstripe[6] = b1.z; bstripe[7] = b1.w;
#pragma unroll
		for (int l = 0; l < 8; ++l) {
		    values[0][l] += a0.x * bstripe[l];
		    values[1][l] += a0.y * bstripe[l];
		    values[2][l] += a0.z * bstripe[l];
		    values[3][l] += a0.w * bstripe[l];
		    values[4][l] += a1.x * bstripe[l];
		    values[5][l] += a1.y * bstripe[l];
		    values[6][l] += a1.z * bstripe[l];
		    values[7][l] += a1.w * bstripe[l];
	        }
	    }
        };

        {
            float4 next_a;
            if (has_next) {
	        next_a = *(float4*)(&A[row_block_start + logic_yidx][next_k + logic_xidx * 4]);  // transponse when laod.
            }
            computeRange(0, 8);
            if (has_next) {
	        asub[next][4 * logic_xidx + 0][store_col] = next_a.x;
	        asub[next][4 * logic_xidx + 1][store_col] = next_a.y;
	        asub[next][4 * logic_xidx + 2][store_col] = next_a.z;
	        asub[next][4 * logic_xidx + 3][store_col] = next_a.w;
            }
        }

        {
            float4 next_b0;
            if (has_next) {
                next_b0 = *(const float4*)(&B[next_k + threadIdx.y][col_block_start + 4 * threadIdx.x]);
            }
            computeRange(8, 12);
            if (has_next) {
                *(float4*)(&bsub[next][threadIdx.y][4 * threadIdx.x]) = next_b0;
            }
        }

        {
            float4 next_b1;
            if (has_next) {
                next_b1 = *(const float4*)(&B[next_k + threadIdx.y][col_block_start + TILE_N / 2 + 4 * threadIdx.x]);
            }
            computeRange(12, 16);
            if (has_next) {
                *(float4*)(&bsub[next][threadIdx.y][TILE_N / 2 + 4 * threadIdx.x]) = next_b1;
            }
        }

        __syncthreads();
    }

    if (is_split) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                atomicAdd(&C[row_block_start + 8 * threadIdx.y + i][col_block_start + (j < 4 ? 4 * threadIdx.x + j : TILE_N / 2 + 4 * threadIdx.x + j - 4)], values[i][j]);
            }
        }
        return;
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        *(float4*)(&C[row_block_start + 8 * threadIdx.y + i][col_block_start + 4 * threadIdx.x]) = make_float4(values[i][0], values[i][1], values[i][2], values[i][3]);
	*(float4*)(&C[row_block_start + 8 * threadIdx.y + i][col_block_start + TILE_N / 2 + 4 * threadIdx.x]) = make_float4(values[i][4], values[i][5], values[i][6], values[i][7]);
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
  int sms = 0;
  CUDA_CALL(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
  constexpr int tiles = (M / TILE_M) * (N / TILE_N);
  int full_tiles = tiles / sms * sms, split = 1;
  if (full_tiles < tiles) {
    split = sms / (tiles - full_tiles);
    while (split > 1 && (K / TILE_K) % split != 0) --split;
  }
  if (split <= 1) { full_tiles = tiles; split = 1; }
  dim3 dimGrid(full_tiles + (tiles - full_tiles) * split);
  constexpr int sharedMemoryBytes = 2 * sizeof(float) * (TILE_M + TILE_N + 4) * TILE_K;
  CUDA_CALL(cudaFuncSetAttribute(
      MatMulKernel_sharedMemory<M, N, K>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      sharedMemoryBytes));
  CUDA_CALL(cudaEventRecord(start));
  // split tail tiles accumulate with atomicAdd, so zero their part of C first (one memset per tile row)
  for (int t = full_tiles; t < tiles;) {
    const int tr = t / (N / TILE_N), tc = t % (N / TILE_N);
    const int cols = (N / TILE_N - tc) < (tiles - t) ? (N / TILE_N - tc) : (tiles - t);
    CUDA_CALL(cudaMemset2DAsync(d_C + (size_t)tr * TILE_M * N + tc * TILE_N, N * sizeof(float), 0,
                                cols * TILE_N * sizeof(float), TILE_M));
    t += cols;
  }
  if (enableProfiler) {
    cudaProfilerStart();
  }
  cudaDeviceSetCacheConfig(cudaFuncCachePreferNone);
  MatMulKernel_sharedMemory<M, N, K><<<dimGrid, dimBlock, sharedMemoryBytes>>>(d_A, d_B, d_C, full_tiles, split);
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

