#include <iostream>
#include <memory>
#include <functional>
#include <thread>
#include <cmath>
#include <omp.h>

using namespace std;

void trap(function<double(double)> f, double start, double end, size_t iterations, shared_ptr<double> p)
{
    // Get thread number
    auto my_rank = omp_get_thread_num();
    // Get number of threads
    auto thread_count = omp_get_num_threads();
    // Calculation iteration slice size
    auto slice_size = (end - start) / iterations;
    // Calculate number of iterations per thread
    auto iterations_thread = iterations / thread_count;
    // Calculate this thread's start point
    auto local_start = start + ((my_rank * iterations_thread) * slice_size);
    // Calculate this thread's end point
    auto local_end = local_start + iterations_thread * slice_size;
    // Calculate initial result
    auto my_result = (f(local_start) + f(local_end)) / 2.0;

    // Declare x before the loop - stops it being allocated and destroyed each iteration
    double x;
    // Sum each iteration
    for (size_t i = 0; i <= iterations_thread - 1; ++i)
    {
        // Calculate next slice to calculate
        x = local_start + i * slice_size;
        // Add to current result
        my_result += f(x);
    }
    
    // Multiply the result by the slice size and divide by 2
    my_result *= slice_size / 2;

    // Critical section - add to the shared data
#pragma omp critical
    *p += my_result;
}

int main(int argc, char **argv)
{
    // Declare shared result
    auto result = make_shared<double>(0.0);
    // Define start and end values
    auto start = 0.0;
    auto end = 3.14159265359; // pi
    // Defined number of trapezoids to generation
    auto trapezoids = static_cast<size_t>(pow(2, 24));
    // Get number of threads
    auto thread_count = thread::hardware_concurrency();

    // Create function to calculate integral of.  Use cos
    auto f = [](double x){ return cos(x); };

    // Run trap in parallel
#pragma omp parallel num_threads(thread_count)
    trap(f, start, end, trapezoids, result);

    // Output result
    cout << "Using " << trapezoids << " trapezoids. ";
    cout << "Estimated integral of function " << start << " to " << end << " = " << *result << endl;

    return 0;
}
