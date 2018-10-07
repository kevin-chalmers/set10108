#include <iostream>
#include <sstream>
#include <mpi.h>

using namespace std;

constexpr unsigned int MAX_STRING = 100;

int main(int argc, char **argv)
{
    int num_procs, my_rank;

    // Initialise MPI
    auto result = MPI_Init(nullptr, nullptr);
    if (result != MPI_SUCCESS)
    {
        cout << "ERROR - initialising MPI" << endl;
        MPI_Abort(MPI_COMM_WORLD, result);
        return -1;
    }

    // Get MPI Information
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    // Check if we are the main process
    if (my_rank != 0)
    {
        // Not main process - send message
        // Generate message
        stringstream buffer;
        buffer << "Greetings from process " << my_rank << " of " << num_procs << "!\0";
        // Get the character array from the string
        auto data = buffer.str().c_str();
        // Send to the main node
        MPI_Send((void*)data, buffer.str().length() + 1, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
    }
    else
    {
        // Main process - print message
        cout << "Greetings from process " << my_rank << " of " << num_procs << "!" << endl;
        // Read in data from each worker process
        char message[MAX_STRING];
        for (int i = 1; i < num_procs; ++i)
        {
            // Receive message into buffer
            MPI_Recv(message, MAX_STRING, MPI_CHAR, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            // Display message
            cout << message << endl;
        }
    }

    // Shutdown MPI
    MPI_Finalize();

    return 0;
}