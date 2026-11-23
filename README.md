# cuda_matmul

在 Tesla T4 上手写 FP32 矩阵乘法（SGEMM），固定形状 M = N = 3840，K = 2880。从 naive 版开始逐步优化，每个版本的输出都与 cuBLAS 逐元素校验。

- **当前最好的版本 v17**（`17_float4_b_tail_splitk.cu`）：约 29.5 ms/次；同一组交替测试中 cuBLAS 约 29.0 ms/次，手写版慢约 2%。
- **起点 v0**（`0_naive.cu`）：119.06 ms/次。v17 的吞吐约为起点的 4 倍。
- **校验**：全部 14,745,600 个输出与 `cublasSgemm` 逐元素比较，拒绝非有限值，容差为 `1e-3 + 1e-4 × |参考值|`（见 `check_result.h`）。

> **更正**：本仓库早期 README 写的“T4 卡特定 shape 上比 cuBLAS 快 15%”，以及 2024 年博客中“超过 cuBLAS 25%”的结论，都来自没有校验输出的旧代码，已经撤回。

完整过程和逐版数据：<https://bryanzhang.github.io/posts/cuda-matmul-optimization-v2/>

## 运行

```bash
./run.sh 17_float4_b_tail_splitk.cu   # 编译（sm_75）并运行指定版本
./run_cublas.sh                       # cuBLAS 基线
```

两个脚本最后一行会用 Nsight Compute（`nv-nsight-cu-cli`）采集指标，需要 GPU 性能计数器权限；没有权限时可以忽略这一步的报错。

---

Hand-written FP32 SGEMM on a Tesla T4 (M = N = 3840, K = 2880), optimized step by step from a naive kernel, with every version's output verified element-wise against cuBLAS. The best version, v17 (`17_float4_b_tail_splitk.cu`), runs at about 29.5 ms per call versus about 29.0 ms for cuBLAS in the same interleaved runs, about 2% slower. The naive kernel takes 119.06 ms.

**Correction:** earlier claims of beating cuBLAS (15% in this README, 25% in a 2024 blog post) came from code that did not verify its output, and have been retracted.

Write-up: <https://bryanzhang.github.io/en/posts/cuda-matmul-optimization-v2/>
