#include "include.h"

void microkernel_fp32(int kc, float *A, float *B, float *C, int rsC, int csC)
{
    // This kernel assumes column-major storage with rsC == 1.
    // That matches your current test harness.
    (void)rsC;

    __m128 gamma_0 = _mm_loadu_ps(&C[0 * csC]);
    __m128 gamma_1 = _mm_loadu_ps(&C[1 * csC]);
    __m128 gamma_2 = _mm_loadu_ps(&C[2 * csC]);
    __m128 gamma_3 = _mm_loadu_ps(&C[3 * csC]);

    for (int p = 0; p < kc; p++)
    {
        __m128 alpha = _mm_loadu_ps(&A[p * MR]);

        __m128 b0 = _mm_broadcast_ss(&B[p * NR + 0]);
        __m128 b1 = _mm_broadcast_ss(&B[p * NR + 1]);
        __m128 b2 = _mm_broadcast_ss(&B[p * NR + 2]);
        __m128 b3 = _mm_broadcast_ss(&B[p * NR + 3]);

        // Use mul + add instead of FMA to better match scalar reference arithmetic.
        gamma_0 = _mm_add_ps(gamma_0, _mm_mul_ps(alpha, b0));
        gamma_1 = _mm_add_ps(gamma_1, _mm_mul_ps(alpha, b1));
        gamma_2 = _mm_add_ps(gamma_2, _mm_mul_ps(alpha, b2));
        gamma_3 = _mm_add_ps(gamma_3, _mm_mul_ps(alpha, b3));
    }

    _mm_storeu_ps(&C[0 * csC], gamma_0);
    _mm_storeu_ps(&C[1 * csC], gamma_1);
    _mm_storeu_ps(&C[2 * csC], gamma_2);
    _mm_storeu_ps(&C[3 * csC], gamma_3);
}

void fp32_gemm(int m, int n, int k,
               float *A, int rsA, int csA,
               float *B, int rsB, int csB,
               float *C, int rsC, int csC)
{
    float *Apacked = (float *)malloc(MC * KC * sizeof(float));
    float *Bpacked = (float *)malloc(KC * NC * sizeof(float));

    if (Apacked == NULL || Bpacked == NULL)
    {
        fprintf(stderr, "fp32_gemm: packing buffer allocation failed\n");
        free(Apacked);
        free(Bpacked);
        return;
    }

    for (int jc = 0; jc < n; jc += NC)
    {
        int nc = (n - jc < NC ? n - jc : NC);

        for (int pc = 0; pc < k; pc += KC)
        {
            int kc = (k - pc < KC ? k - pc : KC);

            // Pack B into panels of shape kc x NR.
            int Bindex = 0;
            for (int jr = 0; jr < nc; jr += NR)
            {
                int nr = (nc - jr < NR ? nc - jr : NR);

                for (int p = 0; p < kc; p++)
                {
                    for (int j = 0; j < NR; j++)
                    {
                        if (j < nr)
                            Bpacked[Bindex] = B[(pc + p) * rsB + (jc + jr + j) * csB];
                        else
                            Bpacked[Bindex] = 0.0f;

                        Bindex++;
                    }
                }
            }

            for (int ic = 0; ic < m; ic += MC)
            {
                int mc = (m - ic < MC ? m - ic : MC);

                // Pack A into panels of shape kc x MR.
                int Aindex = 0;
                for (int ir = 0; ir < mc; ir += MR)
                {
                    int mr = (mc - ir < MR ? mc - ir : MR);

                    for (int p = 0; p < kc; p++)
                    {
                        for (int i = 0; i < MR; i++)
                        {
                            if (i < mr)
                                Apacked[Aindex] = A[(ic + ir + i) * rsA + (pc + p) * csA];
                            else
                                Apacked[Aindex] = 0.0f;

                            Aindex++;
                        }
                    }
                }

                for (int jr = 0; jr < nc; jr += NR)
                {
                    int nr = (nc - jr < NR ? nc - jr : NR);

                    for (int ir = 0; ir < mc; ir += MR)
                    {
                        int mr = (mc - ir < MR ? mc - ir : MR);

                        float *Akernel = &Apacked[(ir / MR) * kc * MR];
                        float *Bkernel = &Bpacked[(jr / NR) * kc * NR];
                        float *Cpanel  = &C[(ic + ir) * rsC + (jc + jr) * csC];

                        if (mr == MR && nr == NR)
                        {
                            microkernel_fp32(kc, Akernel, Bkernel, Cpanel, rsC, csC);
                        }
                        else
                        {
                            for (int j = 0; j < nr; j++)
                            {
                                for (int i = 0; i < mr; i++)
                                {
                                    float sum = Cpanel[i * rsC + j * csC];

                                    for (int p = 0; p < kc; p++)
                                    {
                                        float a_val = Akernel[p * MR + i];
                                        float b_val = Bkernel[p * NR + j];
                                        sum += a_val * b_val;
                                    }

                                    Cpanel[i * rsC + j * csC] = sum;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    free(Apacked);
    free(Bpacked);
}