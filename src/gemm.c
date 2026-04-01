#include "include.h"

void microkernel(int kc, __bfloat16* A, __bfloat16* B, __bfloat16* C, int rsC, int csC) {
    // Need to load and compute as fp32, then convert back to bfloat16.
    __m256d gamma_01234567_0 = _mm256_loadu_ps(&C[0 * csC]);
    __m256d gamma_01234567_1 = _mm256_loadu_ps(&C[1 * csC]);
    __m256d gamma_01234567_2 = _mm256_loadu_ps(&C[2 * csC]);
    __m256d gamma_01234567_3 = _mm256_loadu_ps(&C[3 * csC]);

    int p = 0;
    while(p < kc) {
        __m256d alpha_0123_p = _mm256_loadu_ps(&A[p * MR]);
        __m256d beta_p_j = _mm256_broadcast_ss(&B[p * NR]);
        gamma_01234567_0 = _mm256_fmadd_ps(alpha_0123_p, beta_p_j, gamma_01234567_0);

        
        p++;
    }

    _mm256_storeu_ps(&C[0 * csC], gamma_01234567_0);
    _mm256_storeu_ps(&C[1 * csC], gamma_01234567_1);
    _mm256_storeu_ps(&C[2 * csC], gamma_01234567_2);
    _mm256_storeu_ps(&C[3 * csC], gamma_01234567_3);
}

void gemm(int m, int n, int k,
          __bfloat16 *A, int rsA, int csA,
          __bfloat16 *B, int rsB, int csB,
          __bfloat16 *C, int rsC, int csC) {
    __bfloat16* Apacked = (__bfloat16*)malloc(MC * KC * sizeof(__bfloat16));
    __bfloat16* Bpacked = (__bfloat16*)malloc(KC * NC * sizeof(__bfloat16));

    for(int jc = 0; jc < n; jc += NC) {
        int nc = n - jc < NC ? n - jc : NC;
        for(int pc = 0; pc < k; pc += KC) {
            int kc = k - pc < KC ? k - pc : KC;

            // Pack B
            int Bindex = 0;
            for(int jr = 0; jr < nc; jr += NR) {
                int nr = nc - jr < NR ? nc - jr : NR;
                for(int p = 0; p < kc; p++) {
                    for(int j = 0; j < NR; j++) {
                        if (j < nr) {
                            Bpacked[Bindex] = B[(pc + p) * rsB + (jc + jr + j) * csB];
                        } else {
                            Bpacked[Bindex] = 0;
                        }
                        Bindex ++;
                    }
                }
            }

            for(int ic = 0; ic < m; ic += MC) {
                int mc = m - ic < MC ? m - ic : MC;

                // Pack A
                int Aindex = 0;
                for(int ir = 0; ir < mc; ir += MR) {
                    int mr = mc - ir < MR ? mc - ir : MR;
                    for(int p = 0; p < kc; p++) {
                        for(int i = 0; i < MR; i++) {
                            if (i < mr) {
                                Apacked[Aindex] = A[(ic + ir + i) * rsA + (pc + p) * csA];
                            } else {
                                Apacked[Aindex] = 0;
                            }
                            Aindex ++;
                        }
                    }
                }

                for(int jr = 0; jr < nc; jr += NR) {
                    for(int ir = 0; ir < mc; ir += MR) {                       
                        __bfloat16* Akernel = &Apacked[ir * KC];
                        __bfloat16* Bkernel = &Bpacked[jr * KC];
                        __bfloat16* Ckernel = &C[(ic + ir) * rsC + (jc + jr) * csC];
                        microkernel(KC, Akernel, Bkernel, Ckernel, rsC, csC);
                    }
                }
            }
        }
    }

    free(Apacked);
    free(Bpacked);
}