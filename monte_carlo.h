#include <random>


class monte_carlo {
public:
    double calc_parallel(int threads_number, int chunk_size);

    double calc();

    monte_carlo(int _n, double _radius, double _rib, int _generator_number);
private:
    static double get_random0(double min, double max, unsigned int seed);
    static double get_random1(double min, double max, unsigned int seed);
    static double get_random2(double min, double max, unsigned int *seed);

    static bool check_hit(const double &x, const double &y, const double &z, const double &semi_rib);
    static int rand_r(unsigned int *seed);

    int n;
    double rib;
    double radius;
    int generator_number;
};