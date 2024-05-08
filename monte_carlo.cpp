#include "monte_carlo.h"
#include <omp.h>
#include <iostream>
#include <cstdlib>

#define INT_MAX 2147483647

monte_carlo::monte_carlo(int _n, double _radius, double _rib, int _generator_number) {
    n = _n;
    radius = _radius;
    rib = _rib;
    generator_number = _generator_number;
}

double monte_carlo::calc_parallel(int threads_number, int chunk_size) {
    omp_set_num_threads(threads_number);
    int total_hits = 0;
    double semi_rib = rib / 2.0;
    #pragma omp parallel
    {
        unsigned int seed = time(NULL) ^ (time(NULL) - omp_get_thread_num());

        int local_hits = 0;
        #pragma omp for schedule(dynamic, 10000)
        for (int i = 0; i < n; i++) {
            double cur_point[3];
            for (int t = 0; t < 3; t++) {
                if (generator_number == 0) {
                    cur_point[t] = get_random0(-semi_rib, semi_rib, seed);
                } else if (generator_number == 1) {
                    cur_point[t] = get_random1(-semi_rib, semi_rib, seed);
                } else {
                    cur_point[t] = get_random2(-semi_rib, semi_rib, &seed);
                }
            }
            if (check_hit(cur_point[0], cur_point[1], cur_point[2], semi_rib)) {
                local_hits++;
            }
        }
        #pragma omp atomic
        total_hits += local_hits;
    }
    double volume_estimate = pow(2 * radius, 3) * double(total_hits) / double(n);
    return volume_estimate;
}

double monte_carlo::calc() {
    int total_hits = 0;
    double semi_rib = rib / 2.0;


    auto tt = time(NULL);
    unsigned int seed = tt ^ (tt - (omp_get_thread_num() + 177));

    for (int i = 0; i < n; i++) {
        double cur_point[3];
        for (double & t : cur_point) {
            if (generator_number == 2) {
                t = get_random0(-semi_rib, semi_rib, seed);
            } else if (generator_number == 1) {
                t = get_random1(-semi_rib, semi_rib, seed);
            } else {
                t = get_random2(-semi_rib, semi_rib, &seed);
            }
        }
        if (check_hit(cur_point[0], cur_point[1], cur_point[2], semi_rib)) {
            total_hits++;
        }
    }
    double volume_estimate = pow(2 * radius, 3) * static_cast<double>(total_hits) / static_cast<double>(n);
    return volume_estimate;
}

double monte_carlo::get_random0(double min, double max, unsigned int seed) {
    static thread_local std::mt19937 generator(seed);
    uint32_t rnd = generator();
    return min + (max - min) * (double(rnd) / std::mt19937::max());
}

double monte_carlo::get_random1(double min, double max, unsigned int seed) {
    static thread_local std::minstd_rand generator(seed);
    uint32_t rnd = generator();
    return min + (max - min) * (double(rnd) / std::minstd_rand ::max());
}

double monte_carlo::get_random2(double min, double max, unsigned int *seed) {
    int randNum = rand_r(seed);
    double normalized = (double)randNum / (double) INT_MAX;
    return min + (max - min) * normalized;
}

bool monte_carlo::check_hit(const double &x, const double &y, const double &z, const double &semi_rib) {
    return std::abs(x) + std::abs(y) + std::abs(z) <= semi_rib;
}

// https://sourceware.org/git/?p=glibc.git;a=blob;f=stdlib/rand_r.c;h=50bec5deb3e8339168cc70d37b616e2b685f80e1;hb=HEAD
int monte_carlo::rand_r(unsigned int *seed) {
    unsigned int next = *seed;
    int result;

    next *= 1103515245;
    next += 12345;
    result = (unsigned int) (next / 65536) % 2048;

    next *= 1103515245;
    next += 12345;
    result <<= 10;
    result ^= (unsigned int) (next / 65536) % 1024;

    next *= 1103515245;
    next += 12345;
    result <<= 10;
    result ^= (unsigned int) (next / 65536) % 1024;

    *seed = next;

    return result;
}
