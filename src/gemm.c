#include "include.h"

static inline __m128 bf16_x4_to_f32(__m128i raw16)
{
    return _mm_castsi128_ps(_mm_slli_epi32(_mm_cvtepu16_epi32(raw16), 16));
}

void microkernel(int k, __bfloat16* A, __bfloat16* B, __bfloat16* C, int rsC, int csC)
{
    __m256 gamma[NR];

    for (int j = 0; j < NR; j++) {
        __m128i c_raw = _mm_loadu_si128((__m128i*)&C[j * csC]);
        __m128 g_lo = bf16_x4_to_f32(c_raw);
        __m128 g_hi = bf16_x4_to_f32(_mm_srli_si128(c_raw, 8));
        gamma[j] = _mm256_set_m128(g_hi, g_lo);
    }

    for (int p = 0; p < k; p++) {
        __m128i a_raw = _mm_loadu_si128((__m128i*)&A[p * MR]);
        __m128 a_lo = bf16_x4_to_f32(a_raw);
        __m128 a_hi = bf16_x4_to_f32(_mm_srli_si128(a_raw, 8));
        __m256 alpha = _mm256_set_m128(a_hi, a_lo);

        for (int j = 0; j < NR; j++) {
            uint32_t bj = ((uint32_t)(*(uint16_t*)&B[p * NR + j])) << 16;
            __m256 bvec = _mm256_broadcast_ss((float*)&bj);
            gamma[j] = _mm256_fmadd_ps(alpha, bvec, gamma[j]);
        }
    }

    float tmp[MR];
    for (int j = 0; j < NR; j++) {
        _mm256_storeu_ps(tmp, gamma[j]);
        for (int i = 0; i < MR; i++)
            C[j * csC + i * rsC] = f32_to_bf16(tmp[i]);
    }
}

void bfloat16_gemm(int m, int n, int k,
          __bfloat16 *A, int rsA, int csA,
          __bfloat16 *B, int rsB, int csB,
          __bfloat16 *C, int rsC, int csC) {

    /* Packing lays out ceil(m/MR) x k x MR and ceil(n/NR) x k x NR panels. */
    int n_panel_a = (m + MR - 1) / MR;
    int n_panel_b = (n + NR - 1) / NR;
    __bfloat16* Apacked = (__bfloat16*)malloc((size_t)n_panel_a * k * MR * sizeof(__bfloat16));
    __bfloat16* Bpacked = (__bfloat16*)malloc((size_t)n_panel_b * k * NR * sizeof(__bfloat16));

    for (int jr = 0; jr < n; jr += NR) {
        for (int p = 0; p < k; p++) {
            for (int j = 0; j < NR; j++) {
                if (jr + j < n) {
                    Bpacked[(jr / NR) * k * NR + p * NR + j] =
                        B[p * rsB + (jr + j) * csB];
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
                    Apacked[(ir / MR) * k * MR + p * MR + i] =
                        A[(ir + i) * rsA + p * csA];
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
                for (int jr = 0; jr < n; jr += NR) {
                    int nr = n - jr < NR ? n - jr : NR;
                    for (int ir = 0; ir < m; ir += MR) {
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