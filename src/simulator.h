#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define PATHS       1000000 // Number of simulation paths
#define DAYS        252     // Trading days per year
#define T           1.0     // Time to maturity (in years)
#define S0          100.0   // Starting price
#define K           100.0   // Strike price
#define SIGMA       0.2     // Volatility
#define R           0.05    // Risk-free interest rate

#define PI          3.14159265358979323846