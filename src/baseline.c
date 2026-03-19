#include "simulator.h"

double rand_normal () {
    double u1 = (rand() + 1.0) / (RAND_MAX + 2.0);
    double u2 = (double)rand() / RAND_MAX;
    return sqrt(-2.0 * log(u1)) * cos(2.0 * PI * u2);
}

double simulate_path() {
    double dt        = T / DAYS;
    double drift     = (R - 0.5 * SIGMA * SIGMA) * dt;
    double diffusion = SIGMA * sqrt(dt);
    double price     = S0;
    double sum       = 0.0;

    for (int i = 0; i < DAYS; i++) {
        price = price * exp(drift + diffusion * rand_normal());
        sum  += price;
    }

    return sum / DAYS;
}

int main () {
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    srand((unsigned int)time(NULL));
    double sum = 0.0;

    for (int i = 0; i < PATHS; i++) {
        double average_price = simulate_path();
        double payoff = average_price - K > 0 ? average_price - K : 0;
        sum += payoff;
    }

    double option_price = (sum / PATHS) * exp(-R * T);

    clock_gettime(CLOCK_MONOTONIC, &end);
    
    double time_taken = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    printf("Elapsed Time: %f\n", time_taken);
    printf("Estimated Option Price: $%f\n", option_price);

    return 0;
}