#include "assignment1.h"

void gemm_naive( int m, int n, int k,
                    bfloat16 *A, int rsA, int csA,
                    bfloat16 *B, int rsB, int csB,
                    float    *C, int rsC, int csC )
{
    float A_val, B_val;
    for (int j = 0; j < n; j++)
    {
        for (int i = 0; i < m; i++)
        {
            for (int p = 0; p < k; p++)
            {
                A_val = bf16_to_float( *(A + i * rsA + p * csA) );
                B_val = bf16_to_float( *(B + p * rsB + j * csB) );
                *(C + i * rsC + j * csC) += A_val * B_val;
            }
        }
    }
}