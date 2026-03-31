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
                        microkernel(kc, Apacked, Bpacked, C, rsC, csC);
                    }
                }
            }
        }
    }
}