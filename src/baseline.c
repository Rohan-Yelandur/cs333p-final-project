#include <stdio.h>
#include <math.h>

#define PATHS       1000000 // Number of simulation paths
#define DAYS        252     // Trading days per year
#define T           1       // Time to maturity (in years)
#define S0          100.0   // Starting price
#define K           100.0   // Strike price
#define SIGMA       0.2     // Volatility
#define R           0.05    // Risk-free interest rate            

int simulate_path () {
    double dt = T / DAYS;
    double drift = R - pow(SIGMA, 2) / 2.0;
    double diffusion = SIGMA * sqrt(T);
    double day_price = S0;
    double sum = 0.0;
    
    for (int i = 0; i < DAYS; i++) {
        // Geometric Brownian Motion
        // price = price * exp(drift * T + diffusion) NEED TO IMPLEMENT RAND FOR NORMAL DISTRIBUTION
        sum += day_price;
    }

    return sum / DAYS;
}

int main () {
    double sum = 0.0;

    for (int i = 0; i < PATHS; i++) {
        double average_price = simulate_path();
        double payoff = average_price > 0 ? average_price : 0;
        sum += payoff;
    }

    double option_price = (sum / PATHS) * exp(-R * T);
    printf("Estimated Option Price: $%.6f", option_price);

    return 0;
}