# BF16 GEMM Implementation Plan

## Microkernel Requirements
- Goto's 5-loop algorithm with packing
- SIMD intrinsics for AVX2
- Edge case handling
- Row/column storage compatible
- Parallelism support

## Implementation Steps
1. Implement GEMM locally on the lab machine
2. Test and benchmark
3. Open PR into BLIS codebase

## Testing

### Accuracy
- **Naive vs Goto**: Should produce the same outputs with 0 error.
- **bf16 vs fp32**: Convert bf16 result to fp32, check relative error < `n * 0.008`

### Performance
- Measure GFLOPS and compare to fp32 BLIS baseline and bf16 naive.