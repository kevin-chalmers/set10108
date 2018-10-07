#include <iostream>
#include <vector>
#include <random>
#include <mpi.h>

using namespace std;

constexpr size_t SIZE = 1 << 20;

// Randomly generate vector values
void generate_data(vector<float> &data)
{
    // Create random engine
    random_device r;
    default_random_engine e(r());
    // Fill data
    for (unsigned int i = 0; i < data.size(); ++i)
        data[i] = e();
}

// Normalises 4D vectors
void normalise_vector(vector<float> &data)
{
    // Iterate through each 4-dimensional vector
    for (unsigned int i = 0; i < (data.size() / 4); ++i)
    {
        // Sum the squares of the 4 components
        float sum = 0.0f;
        for (unsigned int j = 0; j < 4; ++j)
            sum += powf(data[(i * 4) + j], 2.0f);
        // Get the square root of the result
        sum = sqrtf(sum);
        // Divide each component by sum
        for (unsigned int j = 0; j < 4; ++j)
            data[(i * 4) + j] /= sum;
    }
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

    int my_rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // Vector containing values to normalise
    vector<float> data;
    // Local storage.  Allocate enough space
    vector<float> my_data(SIZE / num_procs);

    // Check if main process
    if (my_rank == 0)
    {
        // Generate data
        data.resize(SIZE);
        generate_data(data);
        // Scatter the data
        MPI_Scatter(&data[0], SIZE / num_procs, MPI_FLOAT,  // Source
            &my_data[0], SIZE / num_procs, MPI_FLOAT,       // Destination
            0, MPI_COMM_WORLD);
    }
    else
    {
        MPI_Scatter(nullptr, SIZE / num_procs, MPI_FLOAT,  // Source
            &my_data[0], SIZE / num_procs, MPI_FLOAT,      // Destination
            0, MPI_COMM_WORLD);
    }
    
    // Normalise local data
    normalise_vector(my_data);
    
    // Gather the results

    if (my_rank == 0)
    {
        MPI_Gather(&my_data[0], SIZE / num_procs, MPI_FLOAT,// Source
            &data[0], SIZE / num_procs, MPI_FLOAT,          // Dest
            0, MPI_COMM_WORLD);
    }
    else
    {
        MPI_Gather(&my_data[0], SIZE / num_procs, MPI_FLOAT,// Source
            nullptr, SIZE / num_procs, MPI_FLOAT,           // Dest
            0, MPI_COMM_WORLD);
    }

    // Check if main process
    if (my_rank == 0)
    {
        // Display results - first 10
        for (unsigned int i = 0; i < 10; ++i)
        {
            cout << "<";
            for (unsigned int j = 0; j < 3; ++j)
                cout << data[(i * 4) + j] << ", ";
            cout << data[(i * 4) + 3] << ">" << endl;
        }
    }

    // Shutdown MPI
    MPI_Finalize();

    return 0;
}