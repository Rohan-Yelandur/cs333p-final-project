#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <float.h>
#include <stdint.h>
#include <immintrin.h>
#include <math.h>
#include "blis.h"
#include <limits.h>

#define max(x, y) (((x) > (y)) ? (x) : (y))

int get_args( int argc, char **argv, int *nrepeats, int *first, int *last, int *inc );

void microkernel_bf16(int kc, __bfloat16 *A, __bfloat16 *B, __bfloat16 *C, int rsC, int csC);

void gemm_bf16(int m, int n, int k,
          __bfloat16 *A, int rsA, int csA,
          __bfloat16 *B, int rsB, int csB,
          __bfloat16 *C, int rsC, int csC);

int test_gemm( int nrepeats, int first, int last, int inc);

float bf16_to_f32(__bfloat16 val);

__bfloat16 f32_to_bf16(float val);

void rand_bf16(int m, int n, __bfloat16 *M, int rs, int cs);

void ref_gemm(int m, int n, int k,
              __bfloat16 *A, int rsA, int csA,
              __bfloat16 *B, int rsB, int csB,
              __bfloat16 *C, int rsC, int csC);

double bf16_maxabsdiff(int m, int n,
                       __bfloat16 *A, int rsA, int csA,
                       __bfloat16 *B, int rsB, int csB);              

#define MC 256
#define NC 3072
#define KC 512
/* Register tile for both BF16 and FP32 microkernels */
#define MR 8
#define NR 6

void microkernel_fp32(int kc, float *A, float *B, float *C, int rsC, int csC);

void gemm_fp32(int m, int n, int k,
               float *A, int rsA, int csA,
               float *B, int rsB, int csB,
               float *C, int rsC, int csC);

void rand_fp32(int m, int n, float *M, int rs, int cs);

void ref_gemm_fp32(int m, int n, int k,
                   float *A, int rsA, int csA,
                   float *B, int rsB, int csB,
                   float *C, int rsC, int csC);

double fp32_maxabsdiff(int m, int n,
                       float *A, int rsA, int csA,
                       float *B, int rsB, int csB);