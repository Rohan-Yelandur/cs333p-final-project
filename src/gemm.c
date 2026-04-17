#include "include.h"

void microkernel(int k, __bfloat16* A, __bfloat16* B, __bfloat16* C, int rsC, int csC) {
    __m128i c_raw0 = _mm_loadl_epi64((__m128i*)&C[0 * csC]);
    __m128 gamma_0 = _mm_castsi128_ps(_mm_slli_epi32(_mm_cvtepu16_epi32(c_raw0), 16));
    __m128i c_raw1 = _mm_loadl_epi64((__m128i*)&C[1 * csC]);
    __m128 gamma_1 = _mm_castsi128_ps(_mm_slli_epi32(_mm_cvtepu16_epi32(c_raw1), 16));
    __m128i c_raw2 = _mm_loadl_epi64((__m128i*)&C[2 * csC]);
    __m128 gamma_2 = _mm_castsi128_ps(_mm_slli_epi32(_mm_cvtepu16_epi32(c_raw2), 16));
    __m128i c_raw3 = _mm_loadl_epi64((__m128i*)&C[3 * csC]);
    __m128 gamma_3 = _mm_castsi128_ps(_mm_slli_epi32(_mm_cvtepu16_epi32(c_raw3), 16));

    int p = 0;
    while (p < k) {
        __m128i a_raw = _mm_loadl_epi64((__m128i*)&A[p * MR]);
        __m128 alpha = _mm_castsi128_ps(_mm_slli_epi32(_mm_cvtepu16_epi32(a_raw), 16));

        uint32_t b0 = ((uint32_t)(*(uint16_t*)&B[p * NR + 0])) << 16;
        gamma_0 = _mm_fmadd_ps(alpha, _mm_broadcast_ss((float*)&b0), gamma_0);

        uint32_t b1 = ((uint32_t)(*(uint16_t*)&B[p * NR + 1])) << 16;
        gamma_1 = _mm_fmadd_ps(alpha, _mm_broadcast_ss((float*)&b1), gamma_1);

        uint32_t b2 = ((uint32_t)(*(uint16_t*)&B[p * NR + 2])) << 16;
        gamma_2 = _mm_fmadd_ps(alpha, _mm_broadcast_ss((float*)&b2), gamma_2);

        uint32_t b3 = ((uint32_t)(*(uint16_t*)&B[p * NR + 3])) << 16;
        gamma_3 = _mm_fmadd_ps(alpha, _mm_broadcast_ss((float*)&b3), gamma_3);

        p++;
    }

    float tmp[4];

    _mm_storeu_ps(tmp, gamma_0);
    for (int i = 0; i < MR; i++)
        C[0 * csC + i * rsC] = f32_to_bf16(tmp[i]);

    _mm_storeu_ps(tmp, gamma_1);
    for (int i = 0; i < MR; i++)
        C[1 * csC + i * rsC] = f32_to_bf16(tmp[i]);

    _mm_storeu_ps(tmp, gamma_2);
    for (int i = 0; i < MR; i++)
        C[2 * csC + i * rsC] = f32_to_bf16(tmp[i]);

    _mm_storeu_ps(tmp, gamma_3);
    for (int i = 0; i < MR; i++)
        C[3 * csC + i * rsC] = f32_to_bf16(tmp[i]);
}

void bfloat16_gemm(int m, int n, int k,
          __bfloat16 *A, int rsA, int csA,
          __bfloat16 *B, int rsB, int csB,
          __bfloat16 *C, int rsC, int csC) {

    __bfloat16* Apacked = (__bfloat16*)malloc(m * k * sizeof(__bfloat16));
    __bfloat16* Bpacked = (__bfloat16*)malloc(k * n * sizeof(__bfloat16));

    for (int jr = 0; jr < n; jr += NR) {
        for (int p = 0; p < k; p++) {
            for (int j = 0; j < NR; j++) {
                if (jr + j < n) {
                    Bpacked[(jr / NR) * k * NR + p * NR + j] = B[p * rsB + (jr + j) * csB];
                } else {
                    Bpacked[(jr / NR) * k * NR + p * NR + j] = 0;
                }
            }
        }
    }

    for (int ir = 0; ir < m; ir += MR) {
        for (int p = 0; p < k; p++) {
            for (int i = 0; i < MR; i++) {
                if (ir + i < m) {
                    Apacked[(ir / MR) * k * MR + p * MR + i] = A[(ir + i) * rsA + p * csA];
                } else {
                    Apacked[(ir / MR) * k * MR + p * MR + i] = 0;
                }
            }
        }
    }

    //Saving old version for now just in case
    /*__bfloat16* Apacked = (__bfloat16*)malloc(MC * KC * sizeof(__bfloat16));
    __bfloat16* Bpacked = (__bfloat16*)malloc(KC * NC * sizeof(__bfloat16));

    for(int jc = 0; jc < n; jc += NC) {
        int nc = n - jc < NC ? n - jc : NC;
        for(int pc = 0; pc < k; pc += KC) {
            int kc = k - pc < KC ? k - pc : KC;

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
                        Bindex++;
                    }
                }
            }

            for(int ic = 0; ic < m; ic += MC) {
                int mc = m - ic < MC ? m - ic : MC;

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
                            Aindex++;
                        }
                    }
                }

                The next part stays basically the same
                */
                for(int jr = 0; jr < n; jr += NR) {
                    int nr = n - jr < NR ? n - jr : NR;
                    for(int ir = 0; ir < m; ir += MR) {
                        int mr = m - ir < MR ? m - ir : MR;
                        __bfloat16* Akernel = &Apacked[(ir / MR) * k * MR];
                        __bfloat16* Bkernel = &Bpacked[(jr / NR) * k * NR];
                        __bfloat16* Cpanel = &C[ir * rsC + jr * csC];

                        if (mr == MR && nr == NR) {
                            microkernel(k, Akernel, Bkernel, Cpanel, rsC, csC);
                        } else {
                            for (int j = 0; j < nr; j++) {
                                for (int i = 0; i < mr; i++) {
                                    float sum = bf16_to_f32(Cpanel[i * rsC + j * csC]);
                                    for (int p = 0; p < k; p++) {
                                        float a_val = bf16_to_f32(Akernel[p * MR + i]);
                                        float b_val = bf16_to_f32(Bkernel[p * NR + j]);
                                        sum += a_val * b_val;
                                    }
                                    Cpanel[i * rsC + j * csC] = f32_to_bf16(sum);
                                }
                            }
                        }
                    }
                }
           // }
      //  }
  //  }

    free(Apacked);
    free(Bpacked);
}