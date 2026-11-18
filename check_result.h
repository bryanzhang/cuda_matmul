#pragma once

#include <cmath>
#include <cstdio>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CUBLAS_CALL(func) \
    do { \
      cublasStatus_t status = (func); \
      if(status != CUBLAS_STATUS_SUCCESS) { \
        printf("cuBLAS error at %s:%d: %d\n", __FILE__, __LINE__, static_cast<int>(status)); \
        exit(EXIT_FAILURE); \
      } \
    } while(0)

struct ResultCheckStats {
  int errorCount;
  int firstErrorIndex;
  float maxAbsError;
};

__global__ void CompareResultsKernel(
    const float* actual,
    const float* expected,
    int size,
    ResultCheckStats* stats) {
  int localErrorCount = 0;
  int localFirstErrorIndex = size;
  float localMaxAbsError = 0.0f;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
    float actualValue = actual[i];
    float expectedValue = expected[i];
    float absError = fabsf(actualValue - expectedValue);
    localMaxAbsError = fmaxf(localMaxAbsError, absError);
    float tolerance = 1e-3f + 1e-4f * fabsf(expectedValue);
    if (!isfinite(actualValue) || !isfinite(expectedValue) || !isfinite(absError) || absError > tolerance) {
      ++localErrorCount;
      localFirstErrorIndex = min(localFirstErrorIndex, i);
    }
  }
  if (localErrorCount > 0) {
    atomicAdd(&stats->errorCount, localErrorCount);
    atomicMin(&stats->firstErrorIndex, localFirstErrorIndex);
  }
  atomicMax(
      reinterpret_cast<unsigned int*>(&stats->maxAbsError),
      __float_as_uint(localMaxAbsError));
}

bool compareResults(const float* d_C, const float* d_reference, int M, int N) {
  int size = M * N;
  ResultCheckStats h_stats = {0, size, 0.0f};
  ResultCheckStats* d_stats;
  CUDA_CALL(cudaMalloc((void**)&d_stats, sizeof(ResultCheckStats)));
  CUDA_CALL(cudaMemcpy(d_stats, &h_stats, sizeof(ResultCheckStats), cudaMemcpyHostToDevice));
  CompareResultsKernel<<<256, 256>>>(d_C, d_reference, size, d_stats);
  CUDA_CALL(cudaGetLastError());
  CUDA_CALL(cudaDeviceSynchronize());
  CUDA_CALL(cudaMemcpy(&h_stats, d_stats, sizeof(ResultCheckStats), cudaMemcpyDeviceToHost));
  CUDA_CALL(cudaFree(d_stats));

  bool passed = h_stats.errorCount == 0;
  printf(
      "Result check: %s, max abs error=%.9g, mismatches=%d/%d",
      passed ? "PASS" : "FAIL",
      h_stats.maxAbsError,
      h_stats.errorCount,
      size);
  if (!passed && h_stats.firstErrorIndex < size) {
    printf(
        ", first mismatch: row=%d, col=%d, index=%d",
        h_stats.firstErrorIndex / N,
        h_stats.firstErrorIndex % N,
        h_stats.firstErrorIndex);
  }
  printf("\n");
  return passed;
}

bool checkResultAgainstCublas(
    const float* d_A,
    const float* d_B,
    const float* d_C,
    int M,
    int K,
    int N) {
  float* d_reference;
  CUDA_CALL(cudaMalloc((void**)&d_reference, M * N * sizeof(float)));

  cublasHandle_t handle;
  CUBLAS_CALL(cublasCreate(&handle));
  float alpha = 1.0f;
  float beta = 0.0f;
  CUBLAS_CALL(cublasSgemm(
      handle,
      CUBLAS_OP_N,
      CUBLAS_OP_N,
      N,
      M,
      K,
      &alpha,
      d_B,
      N,
      d_A,
      K,
      &beta,
      d_reference,
      N));
  CUDA_CALL(cudaDeviceSynchronize());
  CUBLAS_CALL(cublasDestroy(handle));

  bool passed = compareResults(d_C, d_reference, M, N);
  CUDA_CALL(cudaFree(d_reference));
  return passed;
}
