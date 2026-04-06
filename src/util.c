#include "include.h"

float bf16_to_f32(__bfloat16 val) {
    uint32_t bits = ((uint32_t)(*(uint16_t*)&val)) << 16;
    return *(float*)&bits;
}
__bfloat16 f32_to_bf16(float val) {
    uint16_t bits = (*(uint32_t*)&val) >> 16;
    return *(__bfloat16*)&bits;
}

// Fill matrix with random bf16 values
void rand_bf16(int m, int n, __bfloat16 *M, int rs, int cs) {
    for (int j = 0; j < n; j++)
        for (int i = 0; i < m; i++)
            M[i * rs + j * cs] = f32_to_bf16((float)rand() / RAND_MAX);
}

// Naive bf16 gemm for reference: C += A * B
void ref_gemm(int m, int n, int k,
              __bfloat16 *A, int rsA, int csA,
              __bfloat16 *B, int rsB, int csB,
              __bfloat16 *C, int rsC, int csC) {
    for (int j = 0; j < n; j++)
        for (int i = 0; i < m; i++) {
            float sum = bf16_to_f32(C[i * rsC + j * csC]);
            for (int p = 0; p < k; p++) {
                float a_val = bf16_to_f32(A[i * rsA + p * csA]);
                float b_val = bf16_to_f32(B[p * rsB + j * csB]);
                sum += a_val * b_val;
            }
            C[i * rsC + j * csC] = f32_to_bf16(sum);
        }
}

// Max absolute difference between two bf16 matrices
double bf16_maxabsdiff(int m, int n,
                       __bfloat16 *A, int rsA, int csA,
                       __bfloat16 *B, int rsB, int csB) {
    double maxdiff = 0.0;
    for (int j = 0; j < n; j++)
        for (int i = 0; i < m; i++) {
            double a = (double)bf16_to_f32(A[i * rsA + j * csA]);
            double b = (double)bf16_to_f32(B[i * rsB + j * csB]);
            double d = fabs(a - b);
            if (d > maxdiff) maxdiff = d;
        }
    return maxdiff;
}

int get_args( int argc, char **argv, int *nrepeats, int *first, 
                                     int *last, int *inc )
{
     // For optional inputs                                                      
    char* p;                                                                    
    long int arg;                                                               
                                                    

    // Default values                            
    *nrepeats = 3;                                                               
                                                                                
    *first = 100;                                                                
    *last  = 500;                                                                
    *inc   = 50;                        
                                                                                
    if ( argc == 1 )                                                            
    {                                                                           
        printf("%% Using default values\n");                                    
    }                                                                           
                                                                                
    if ( argc >= 2 )                                                            
    {                                                                           
        arg = strtol(argv[1], &p, 10);                                          
        if (*p != '\0' )                                                        
        {                                                                       
            printf("Not a valid input\n");                                      
            return 1;                                                           
        }                                                                       
        if ( arg < INT_MIN  || arg > INT_MAX )                                  
        {                                                                       
            return 1;                                                           
        }                                                                       
        *nrepeats = (int ) arg;                                                  
    }                                                                           
    if ( argc >= 3 )                                                            
    {
        arg = strtol(argv[2], &p, 10);                                      
        if (*p != '\0' )                                                        
        {                                                                       
            printf("Not a valid input\n");                                      
            return 1;                                                           
        }                                                                       
        if ( arg < INT_MIN  || arg > INT_MAX )                                  
        {                                                                       
            return 1;                                                           
        }                                                                       
        *first = ( int ) arg;    
    }
    if ( argc >= 4 )                                                            
    {
        arg = strtol(argv[3], &p, 10);                                      
        if (*p != '\0' )                                                        
        {                                                                       
            printf("Not a valid input\n");                                      
            return 1;                                                           
        }                                                                       
        if ( arg < INT_MIN  || arg > INT_MAX )                                  
        {                                                                       
            return 1;                                                           
        }                                                                       
        *last = ( int ) arg;    
    }
    if ( argc >= 5 )                                                            
    {
        arg = strtol(argv[4], &p, 10);                                      
        if (*p != '\0' )                                                        
        {                                                                       
            printf("Not a valid input\n");                                      
            return 1;                                                           
        }                                                                       
        if ( arg < INT_MIN  || arg > INT_MAX )                                  
        {                                                                       
            return 1;                                                           
        }                                                                       
        *inc = ( int ) arg;    
    }
        
    printf("%% nrepeats = %d\n", *nrepeats);                                 
    printf("%% first = %d\n", *first);                                       
    printf("%% last = %d\n", *last);                                         
    printf("%% inc = %d\n", *inc);    
    return 0;
}
