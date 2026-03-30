#include "include.h"

void microkernel(int kc, __bfloat16 *A, __bfloat16 *B, __bfloat16 *C, int rsC, int csC) {

}

void gemm(int m, int n, int k,
          __bfloat16 *A, int rsA, int csA,
          __bfloat16 *B, int rsB, int csB,
          __bfloat16 *C, int rsC, int csC) {
    __bfloat16* Apacked = (__bfloat16*)malloc(MC * KC * sizeof(__bfloat16));
    __bfloat16* Bpacked = (__bfloat16*)malloc(KC * NC * sizeof(__bfloat16));

    for(int jc = 0; jc < n; jc += NC) {
        for(int pc = 0; pc < k; pc += KC) {
            // Pack B
            int Bindex = 0;
            for(int jr = 0; jr < NC; jr += NR) {
                for(int p = 0; p < KC; p++) {
                    for(int j = 0; j < NR; j++) {
                        Bpacked[Bindex] = B[(pc + p) * rsB + (jc + jr + j) * csB];
                        Bindex ++;
                    }
                }
            }

            for(int ic = 0; ic < m; ic += MC) {
                // Pack A
                int Aindex = 0;
                for(int ir = 0; ir < MC; ir += MR) {
                    for(int p = 0; p < KC; p++) {
                        for(int i = 0; i < MR; i++) {
                            Apacked[Aindex] = A[(ic + ir + i) * rsA + (pc + p) * csA];
                            Aindex ++;
                        }
                    }
                }

                for(int jr = 0; jr < NC; jr += NR) {
                    for(int ir = 0; ir < MC; ir += MR) {
                        microkernel(KC, Apacked, Bpacked, C, rsC, csC);
                    }
                }
            }
        }
    }
}