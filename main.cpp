#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <omp.h>
#include "monte_carlo.h"
#include "point.h"

using namespace std;

const double SQRT2 = sqrt(2.0);

void read_data(int &n, vector<point> &points, FILE *input_file) {
    fscanf(input_file, "%d\n", &n);
    for (int i = 0; i < 3; i++) {
        double x, y, z;
        fscanf(input_file, "(%lf %lf %lf)\n", &x, &y, &z);
        points.emplace_back(x, y, z);
    }
}

void write_data(const double &res1, const double &res2, FILE *output_file) {
    fprintf(output_file, "%lg %lg\n", res1, res2);
}

double dist(point a, point b) {
    return sqrt(pow(a.x - b.x, 2) + pow(a.y - b.y, 2) + pow(a.z - b.z, 2));
}

double calc_rib(vector<point> &points) {
    return min(dist(points[0], points[1]), dist(points[1], points[2]));
}

double calc_radius(double rib) {
    return (rib * SQRT2) / 2.0;
}

double calc_square(const double &rib) {
    return pow(rib, 3) * sqrt(2) / 3.0;
}

signed main(int argc, char *argv[]) {
    if (argc < 4 || argc > 5) {
        cerr << "Usage: " << argv[0] << " <threads_number> <input_file> <output_file> <generator>\n";
        return 1;
    }

    int threads_number = atoi(argv[1]);

    if (threads_number < -1) {
        cerr << "Error: Invalid number of threads. Please enter an integer >= -1.\n";
        return 1;
    }


    int generator_number = 0;

    if (argc == 5) {
        generator_number = atoi(argv[4]);
        if (generator_number < 0 || generator_number > 2) {
            cerr << "Error: Invalid generator number. Please enter an integer between 0 and 2";
            return 1;
        }
    }


    FILE *input_file = fopen(argv[2], "r");
    if (!input_file) {
        cerr << "Error opening input file\n";
        return 1;
    }

    FILE *output_file = fopen(argv[3], "w");
    if (!output_file) {
        cerr << "Error opening output file\n";
        return 1;
    }

    int n;
    vector<point> input_points;
    read_data(n, input_points, input_file);

    double start_time = omp_get_wtime();

    double rib = calc_rib(input_points);
    double radius = calc_radius(rib);

    double random_result;
    double analytic_result = calc_square(rib);
    monte_carlo random_solver(n, radius, rib, generator_number);
    if (threads_number > -1) {
        if (threads_number == 0) {
            threads_number = omp_get_max_threads();
        }
        random_result = random_solver.calc_parallel(threads_number, 0);
    } else {
        threads_number = 0;
        random_result = random_solver.calc();
    }

    double endTime = omp_get_wtime();
    write_data(analytic_result, random_result, output_file);
    double duration = (endTime - start_time) * 1000.0;
    printf("Time (%i thread(s)): %g ms\n", threads_number, duration);

    fclose(input_file);
    fclose(output_file);
    return 0;
}