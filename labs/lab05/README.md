# Distributed Parallelism with MPI

For the next two practicals it might be useful to work with a partner so you can get work on the distributed work we are undertaking. There is quite a bit of setup to do in this practical, so take your time and ensure everything works correctly.

## Installing MPI

We are going to use [Microsoft MPI](https://www.microsoft.com/en-us/download/details.aspx?id=56727) to support our MPI work as the lab has Windows installed.  You will need **both the runtime and the SDK**.  MPI on Linux is generally easier to manage, and there are a few variants out there.  You will need to add the relevant include and library folders to your project - these can normally be found in the `C:\Program Files (x86)\Microsoft SDKs\MPI\` folder. The library that you need to link against is called `msmpi.lib`.

## First MPI Application

Our first application will initialise MPI, display some local information, and shutdown.

```cpp
#include <iostream>
#include <mpi.h>

using namespace std;

int main(int argc, char **argv)
{
    // Initialise MPI
    auto result = MPI_Init(nullptr, nullptr);
    // Check that we initialised correctly
    if (result != MPI_SUCCESS)
    {
        // Display error and abort
        cout << "ERROR - initialising MPI!" << endl;
        MPI_Abort(MPI_COMM_WORLD, result);
        return -1;
    }

    // Get MPI information
    int num_procs, rank, length;
    char host_name[MPI_MAX_PROCESSOR_NAME];
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Get_processor_name(host_name, &length);

    // Display information
    cout << "Number of processors = " << num_procs << endl;
    cout << "My rank = " << rank << endl;
    cout << "Running on = " << host_name << endl;

    // Shutdown MPI
    MPI_Finalize();

    return 0;
}
```

The methods of interest are `MPI_Init` (initialises MPI), `MPI_Comm_size` (gets the number of processes in the application), `MPI_Comm_rank` (gets the ID of this process) and `MPI_Finalize` (shuts down MPI).

At the moment you should just check the application builds - running an MPI application takes a bit more work.

## Running an MPI Application

You need a command prompt in the directory where you built your application.  Run the following command to execute the application on the local machine in parallel:

```shell
mpiexec -n 4 "exe_name.exe"
```

Make sure to use the name of your application. `-n` denotes the number of processes to create. The output is similar to:

```shell
Number of processors = 4
Number of processors = 4
Number of processors = 4
Number of processors = 4
My rank = 3
My rank = 0
My rank = 2
My rank = 1
Running on = xps-13
Running on = xps-13
Running on = xps-13
Running on = xps-13
```

## Using a Remote Host

We are not doing any distributed parallelism yet.  We want to use one or more remote machines to do our processing.  First you will need to find the IP address of the remote machine using `ipconfig`:

`ipconfig`

This will give you the output similar to:

```shell
Wireless LAN adapter Wi-Fi:

   Connection-specific DNS Suffix  . : napier.ac.uk
   Link-local IPv6 Address . . . . . : fe80::55a3:e0ab:485:ce50%3
   IPv4 Address. . . . . . . . . . . : 146.176.135.164
   Subnet Mask . . . . . . . . . . . : 255.255.252.0
   Default Gateway . . . . . . . . . : 146.176.132.1
```

The value you want is the IPv4 Address - `146.176.135.164` above.  Next you want to run the following command on the remote machine:

`smpd -d`

It will listen on the machine and wait for us to allocate a job. We run `mpiexec` with a few more commands:

`mpiexec -hosts 1 <ip-address> 4 <application-name>`

`1` is the number of hosts we are providing, which is one here.  `4` is the number of processes for that host.  We tell MPI which host to run on and can provide multiple hosts.  

`mpiexec -hosts 2 <ip-address1> 8 <ip-address2> 4 <application-name>`

You will need to copy the application to the other machine.  Running this version will give a similar output to before.

It is worth at this point to look at the different flags for the `mpiexec`. You can find these [here](https://docs.microsoft.com/en-us/powershell/high-performance-computing/mpiexec?view=hpc16-ps).

## Sending and Receiving

The next few examples will examine different methods for communication.  First we will use standard send and receive.

```cpp
const unsigned int MAX_STRING = 100;

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
        buffer << "Greetings from process " << my_rank << " of " << num_procs << "!";
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
```

We use the process rank to determine the behaviour of a process.  The process with rank `0` is the main process.  It will receive messages.  Other processes will send to the main process.

We are using two new commands:

- `MPI_Send` requires the data to be sent, the size of data (we make sure we send an extra byte for a string - *the null terminator*), the type of data, the destination (`0` - the main process), a tag (we will not be using tags), and the communicator.
- `MPI_Recv` requires a buffer to store the message, the maximum size of the buffer, the process to receive from, the tag, the communicator to use, and status conditions.

Running this application will give you an output similar to:

```shell
Greetings from process 0 of 4!
Greetings from process 1 of 4!
Greetings from process 2 of 4!
Greetings from process 3 of 4!
```

There are data types beyond `MPI_CHAR` MPI can use.  The *Introduction to Parallel Programming* book will explain these further.

## Map-Reduce

Another approach to communication is map-reduce. For this you will build a Monte-Carlo &pi; application:

```cpp
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
```

`MPI_Reduce` takes the following values:

- The value to send (`local_sum`).
- The value to reduce into (`global_sum`).
- The number of elements in the send buffer (`1`).
- The type of the send buffer (`MPI_DOUBLE`).
- The reduction operation - we are using `MPI_SUM` to sum.
- The rank of the process that collects the reduction operation - we use rank `0` as the main process.
- The communicator used (`MPI_COMM_WORLD`).

Notice that we only use our main application to calculate &pi;.  Running this application will produce an output similar to:

```shell
2:3.14220428466797
3:3.14110040664673
0:3.14220428466797
1:3.14110040664673
Pi=3.14652345655735
```

## Scatter-Gather

Scatter-gather involves taking an array of data and distributing it evenly amongst the processes. Gathering involves collecting the results back again at the end.

For scatter-gather we will implement a vector normalization application. You need the two helper functions to generate and normalize data.

```cpp
// Randomly generate vector values
void generate_data(vector<float> &data)
{
    // Create random engine
    auto millis = duration_cast<milliseconds>(system_clock::now().time_since_epoch());
    default_random_engine e(static_cast<unsigned int>(millis.count()));
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
```

Our main application calls scatter, normalize, gather.

```cpp
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
```

`MPI_Scatter` has the following parameters:

- The data to scatter - only relevant on the root process (`data`).
- The count of data to send to each process (`SIZE / num_procs`).
- The type of data sent (`MPI_FLOAT`).
- The memory to receive the data into on each process (`my_data`).
- The count of data to receive at each process (`SIZE / num_procs`).
- The type of data received (`MPI_FLOAT`).
- The root process (sender) (`0`).
- The communicator used (`MPI_COMM_WORLD`).

`MPI_Gather` is essentially `MPI_Scatter` in reverse:

- The local data to send (`my_data`).
- The count of data to send from each process (`SIZE / num_procs`).
- The type of data sent (`MPI_FLOAT`).
- The memory to gather results into - only relevant onthe root process (`data`).
- The count of data to receive from each process (`SIZE / num_procs`).
- The type of data received (`MPI_FLOAT`).
- The root process (gatherer) (`0`).
- The communicator used (`MPI_COMM_WORLD`).

Running will provide the following output:

```shell
<0.47411, 0.499097, 0.318984, 0.651438>
<0.328696, 0.597439, 0.312654, 0.661266>
<0.369751, 0.563316, 0.00549236, 0.73887>
<0.513968, 0.515516, 0.55603, 0.401137>
<0.557941, 0.572452, 0.567326, 0.197844>
<0.607652, 0.0981679, 0.781428, 0.102427>
<0.332849, 0.046342, 0.620809, 0.708279>
<0.654357, 0.100316, 0.714417, 0.226631>
<0.791911, 0.187327, 0.226341, 0.535308>
<0.301491, 0.402493, 0.653127, 0.566151>
```

You can check to see if these vectors are normalised.

## Broadcast

The final communication type we will look at is broadcast.  Broadcasting allows us to send a message from one source to all processes on the communicator.

```cpp
// Check if main process
if (my_rank == 0)
{
    // Broadcast message to workers
    string str = "Hello World!";
    MPI_Bcast((void*)&str.c_str()[0], str.length() + 1, MPI_CHAR, 0, MPI_COMM_WORLD);
}
else
{
    // Receive message from main process
    char data[100];
    MPI_Bcast(data, 100, MPI_CHAR, 0, MPI_COMM_WORLD);
    cout << my_rank << ":" << data << endl;
}
```

`MPI_Bcast` takes the following parameters:

- Data to broadcast (sender sends, receiver reads into this data) (`data`).
- Count of data to send / receive) (`100`).
- Data type sent / received (`MPI_CHAR`).
- Root node (`0`).
- Communicator (`MPI_COMM_WORLD`).

## Exercises

1. As always you should be taking timings of your applications.
2. The Mandelbrot is quite an interesting application to distribute.  In particular you will find that our implementation allows some parts to be processed quickly, and other parts slowly.  You should try and divide the work so that you can optimise performance.
3. Try and get an application that works across a number of hosts. Try four machines in the games lab (16 processes in total). Again gather timings. Mandelbrot is another good application here.
