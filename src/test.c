#include "include.h"

int main(int argc, char *argv[])
{
    int first, last, inc, nrepeats;

    int err = get_args(argc, argv, &nrepeats, &first, &last, &inc);
    if (err != 0) return 1;

    return test_gemm(nrepeats, first, last, inc);
}

int test_gemm(int nrepeats, int first, int last, int inc)
{
    int size, irep;
    int m, n, k;
    int csA, csB, csC;
    int rsA, rsB, rsC;

    __bfloat16 *A_bf16, *B_bf16, *C_bf16, *Cref_bf16, *Cold_bf16;
    float *A_fp32, *B_fp32, *C_fp32, *Cref_fp32, *Cold_fp32;

    double t_ref_bf16, t_bf16;
    double t_ref_fp32, t_fp32;
    double t_start;

    double gflops_ref_bf16, gflops_bf16;
    double gflops_ref_fp32, gflops_fp32;

    double diff_bf16, diff_fp32;
    double maxdiff_bf16 = 0.0, maxdiff_fp32 = 0.0;

    float alpha_s = 1.0f;
    float beta_s  = 1.0f;

    printf("%% --------- BF16 vs FP32 GEMM ---------\n");
    printf("%% columns: [ m k n bf16_ref bf16_opt fp32_blis fp32_opt bf16_diff fp32_diff ratio_bf16_over_fp32 ]\n");

    printf("data_gemm");
    printf("(%4lu, 1:10) = [ %5lu %5lu %5lu %8.2f %8.2f %8.2f %8.2f %15.4e %15.4e %8.4f ];\n",
           (unsigned long)(last - first) / inc + 1,
           (unsigned long)0,
           (unsigned long)0,
           (unsigned long)0,
           0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);

    for (size = last; size >= first; size -= inc)
    {
        t_ref_bf16 = DBL_MAX;
        t_bf16     = DBL_MAX;
        t_ref_fp32 = DBL_MAX;
        t_fp32     = DBL_MAX;

        m = n = k = size;
        csA = m; csB = k; csC = m;
        rsA = rsB = rsC = 1;

        A_bf16    = (__bfloat16 *)malloc(csA * k * sizeof(__bfloat16));
        B_bf16    = (__bfloat16 *)malloc(csB * n * sizeof(__bfloat16));
        C_bf16    = (__bfloat16 *)malloc(csC * n * sizeof(__bfloat16));
        Cold_bf16 = (__bfloat16 *)malloc(csC * n * sizeof(__bfloat16));
        Cref_bf16 = (__bfloat16 *)malloc(csC * n * sizeof(__bfloat16));

        A_fp32    = (float *)malloc(csA * k * sizeof(float));
        B_fp32    = (float *)malloc(csB * n * sizeof(float));
        C_fp32    = (float *)malloc(csC * n * sizeof(float));
        Cold_fp32 = (float *)malloc(csC * n * sizeof(float));
        Cref_fp32 = (float *)malloc(csC * n * sizeof(float));

        if (!A_bf16 || !B_bf16 || !C_bf16 || !Cold_bf16 || !Cref_bf16 ||
            !A_fp32 || !B_fp32 || !C_fp32 || !Cold_fp32 || !Cref_fp32)
        {
            fprintf(stderr, "Allocation failed for size %d\n", size);

            free(A_bf16);    free(B_bf16);    free(C_bf16);
            free(Cold_bf16); free(Cref_bf16);
            free(A_fp32);    free(B_fp32);    free(C_fp32);
            free(Cold_fp32); free(Cref_fp32);

            return 1;
        }

        /* BF16 inputs */
        srand(42);
        rand_bf16(m, k, A_bf16, rsA, csA);
        rand_bf16(k, n, B_bf16, rsB, csB);
        rand_bf16(m, n, Cold_bf16, rsC, csC);

        /* FP32 inputs */
        srand(42);
        rand_fp32(m, k, A_fp32, rsA, csA);
        rand_fp32(k, n, B_fp32, rsB, csB);
        rand_fp32(m, n, Cold_fp32, rsC, csC);

        /* BF16 reference */
        for (irep = 0; irep < nrepeats; irep++)
        {
            memcpy(Cref_bf16, Cold_bf16, csC * n * sizeof(__bfloat16));

            t_start = bli_clock();

            ref_gemm(m, n, k,
                     A_bf16, rsA, csA,
                     B_bf16, rsB, csB,
                     Cref_bf16, rsC, csC);

            t_ref_bf16 = bli_clock_min_diff(t_ref_bf16, t_start);
        }

        gflops_ref_bf16 = 2.0 * m * n * k / (t_ref_bf16 * 1.0e9);

        /* BF16 optimized */
        for (irep = 0; irep < nrepeats; irep++)
        {
            memcpy(C_bf16, Cold_bf16, csC * n * sizeof(__bfloat16));

            t_start = bli_clock();

            gemm_bf16(m, n, k,
                          A_bf16, rsA, csA,
                          B_bf16, rsB, csB,
                          C_bf16, rsC, csC);

            t_bf16 = bli_clock_min_diff(t_bf16, t_start);
        }

        gflops_bf16 = 2.0 * m * n * k / (t_bf16 * 1.0e9);
        diff_bf16   = bf16_maxabsdiff(m, n, C_bf16, rsC, csC, Cref_bf16, rsC, csC);
        maxdiff_bf16 = max(diff_bf16, maxdiff_bf16);

        /* FP32 BLIS reference */
        for (irep = 0; irep < nrepeats; irep++)
        {
            memcpy(Cref_fp32, Cold_fp32, csC * n * sizeof(float));

            t_start = bli_clock();

            bli_sgemm(BLIS_NO_TRANSPOSE, BLIS_NO_TRANSPOSE,
                      m, n, k,
                      &alpha_s,
                      A_fp32, rsA, csA,
                      B_fp32, rsB, csB,
                      &beta_s,
                      Cref_fp32, rsC, csC);

            t_ref_fp32 = bli_clock_min_diff(t_ref_fp32, t_start);
        }

        gflops_ref_fp32 = 2.0 * m * n * k / (t_ref_fp32 * 1.0e9);

        /* FP32 optimized */
        for (irep = 0; irep < nrepeats; irep++)
        {
            memcpy(C_fp32, Cold_fp32, csC * n * sizeof(float));

            t_start = bli_clock();

            gemm_fp32(m, n, k,
                      A_fp32, rsA, csA,
                      B_fp32, rsB, csB,
                      C_fp32, rsC, csC);

            t_fp32 = bli_clock_min_diff(t_fp32, t_start);
        }

        gflops_fp32 = 2.0 * m * n * k / (t_fp32 * 1.0e9);
        diff_fp32   = fp32_maxabsdiff(m, n, C_fp32, rsC, csC, Cref_fp32, rsC, csC);
        maxdiff_fp32 = max(diff_fp32, maxdiff_fp32);

        printf("data_gemm");
        printf("(%4lu, 1:10) = [ %5lu %5lu %5lu %8.2f %8.2f %8.2f %8.2f %15.4e %15.4e %8.4f ];\n",
               (unsigned long)(size - first) / inc + 1,
               (unsigned long)m,
               (unsigned long)k,
               (unsigned long)n,
               gflops_ref_bf16,
               gflops_bf16,
               gflops_ref_fp32,
               gflops_fp32,
               diff_bf16,
               diff_fp32,
               (gflops_fp32 > 0.0 ? gflops_bf16 / gflops_fp32 : 0.0));

        free(A_bf16);
        free(B_bf16);
        free(C_bf16);
        free(Cold_bf16);
        free(Cref_bf16);

        free(A_fp32);
        free(B_fp32);
        free(C_fp32);
        free(Cold_fp32);
        free(Cref_fp32);
    }

    printf("%% max BF16 diff = %.6e\n", maxdiff_bf16);
    printf("%% max FP32 diff = %.6e\n", maxdiff_fp32);

    return 0;
}