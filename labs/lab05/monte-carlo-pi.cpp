#include <iostream>
#include <random>
#include <mpi.h>

using namespace std;

double monte_carlo_pi(size_t iterations)
{
    // Seed with real random number if available
    random_device r;
    // Create random number generator
    default_random_engine e(r());
    // Create a distribution - we want doubles between 0.0 and 1.0
    uniform_real_distribution<double> distribution(0.0, 1.0);

    // Keep track of number of points in circle
    unsigned int in_circle = 0;
    // Iterate
    for (size_t i = 0; i < iterations; ++i)
    {
        // Generate random point
        auto x = distribution(e);
        auto y = distribution(e);
        // Get length of vector defined - use Pythagarous
        auto length = sqrt((x * x) + (y * y));
        // Check if in circle
        if (length <= 1.0)
            ++in_circle;
    }
    // Calculate pi
    return (4.0 * in_circle) / static_cast<double>(iterations);
}

int main(int argc, char **argv)
{
    // Initialise MPI
    auto result = MPI_Init(nullptr, nullptr);
    if (result != MPI_SUCCESS)
    {
        cout << "ERROR - initialising MPI" << endl;
        MPI_Abort(MPI_COMM_WORLD, result);
        return -1;
    }

    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    double local_sum, global_sum;

    // Calculate local sum - use previously defined function
    local_sum = monte_carlo_pi(static_cast<unsigned int>(pow(2, 24)));
    // Print out local sum
    cout.precision(numeric_limits<double>::digits10);
    cout << my_rank << ":" << local_sum << endl;
    // Reduce
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    // If main process display global reduced sum
    if (my_rank == 0)
    {
        global_sum /= 4.0;
        cout << "Pi=" << global_sum << endl;
    }

    // Shutdown MPI
    MPI_Finalize();

    return 0;
}
