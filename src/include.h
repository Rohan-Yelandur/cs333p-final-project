#include <immintrin.h>
#include <math.h>
#include "blis.h"

double shpc_maxabsdiff( int m, int n, double *A, int rsA, int csA, double *B, int rsB, int csB );

int get_args( int argc, char **argv, int *nrepeats, int *first, int *last, int *inc );

void microkernel(int kc, __bfloat16 *A, __bfloat16 *B, __bfloat16 *C, int rsC, int csC);

void gemm(int m, int n, int k,
          __bfloat16 *A, int rsA, int csA,
          __bfloat16 *B, int rsB, int csB,
          __bfloat16 *C, int rsC, int csC);

#define MC 48
#define NC 48
#define KC 48
#define MR 4
#define NR 4