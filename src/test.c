#include "include.h"

int main(int argc, char *argv[])
{
	int first, last, inc, nrepeats;
    
    int err = get_args(argc, argv, &nrepeats, &first, &last, &inc);
    if (err != 0) return 1;
    	
	test_gemm(nrepeats, first, last, inc);
}

int test_gemm(int nrepeats, int first, int last, int inc)
{
	int size, irep;
	int m, n, k;
	int csA, csB, csC;
	int rsA, rsB, rsC;

	__bfloat16 *A, *B, *C, *Cref, *Cold;

	double done = 1.0;


	double t_ref = DBL_MAX;
	double t     = DBL_MAX;
	double t_start; 

	double gflops_ref, gflops;

	double diff, maxdiff = 0.0;

	printf("%% --------- DGEMM --------- \n"); 
	printf("data_dgemm");
	printf("(%4lu, 1:6) = [ %5lu %5lu %5lu %8.2f %8.2f %15.4e ];\n",
						 (unsigned long)(last - first)/inc + 1,
	        			 (unsigned long)0,
	        			 (unsigned long)0,
	        			 (unsigned long)0, 0.0, 0.0, 0.0);
	for(size=last; size>= first; size-=inc)
	{
    	m = n = k = size;
		csA = m; csB = k; csC = m;

		rsA = rsB = rsC = 1;

    	A = (__bfloat16 *)malloc(csA * k * sizeof(__bfloat16));
    	B = (__bfloat16 *)malloc(csB * n * sizeof(__bfloat16));
    	C = (__bfloat16 *)malloc(csC * n * sizeof(__bfloat16));
    	Cold = (__bfloat16 *)malloc(csC * n * sizeof(__bfloat16));
    	Cref = (__bfloat16 *)malloc(csC * n * sizeof(__bfloat16));

		bli_drandm(0, BLIS_DENSE, m, k, A, rsA, csA);
		bli_drandm(0, BLIS_DENSE, k, n, B, rsB, csB);
		bli_drandm(0, BLIS_DENSE, m, n, Cold, rsC, csC);

		for(irep=0; irep<nrepeats; irep++)
		{
			memcpy(Cref, Cold, csC * n * sizeof(__bfloat16));

			t_start = bli_clock();
		
			bli_dgemm(BLIS_NO_TRANSPOSE, BLIS_NO_TRANSPOSE,  
					  m, n, k, &done, 
					  A, rsA, csA, 
					  B, rsB, csB, 
					  &done, Cref, rsC, csC);	
			t_ref = bli_clock_min_diff(t_ref, t_start);
			
		}

		gflops_ref = 2 * m * n * k / (t_ref * 1.0e9);

		for(irep=0; irep<nrepeats; irep++)
		{
			memcpy(C, Cold, csC * n * sizeof(__bfloat16 ));

			t_start = bli_clock();
		
			shpc_dgemm(m, n, k, 
					   A, rsA, csA, 
					   B, rsB, csB, 
					   C, rsC, csC);	
			
			t = bli_clock_min_diff(t , t_start);
			
		}

		gflops = 2 * m * n * k / (t * 1.0e9);
		
		diff    = shpc_maxabsdiff(m, n, C, rsC, csC, Cref, rsC, csC);
        maxdiff = max (diff, maxdiff);

		printf("data_dgemm");
		printf("(%4lu, 1:6) = [ %5lu %5lu %5lu %8.2f %8.2f %15.4e ];\n",
		        (unsigned long)(size - first)/inc + 1,
		        (unsigned long)m,
		        (unsigned long)k,
		        (unsigned long)n, gflops_ref, gflops, diff);

		free(A);
		free(B);
		free(C);
		free(Cold);
		free(Cref);
	}
}