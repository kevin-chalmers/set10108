# More MPI

This will focus on MPI examples from previous labs.  Focus will be on the examples rather than new MPI commands.

## Mandelbrot

The previous implementation ran using rank based execution. We just need to modify the previous one to use MPI ranks.

```cpp
// Broadcast dimension around the workers
unsigned int dim = 0;
if (my_rank == 0)
{
    // Broadcast dimension to all the workers - could read this in from user
    dim = 8192;
    MPI_Bcast(&dim, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
}
else
    // Get dimension
    MPI_Bcast(&dim, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

// Perform Mandelbrot
auto strip_height = dim / num_procs;
auto res = mandelbrot(dim, my_rank * strip_height, (my_rank + 1) * strip_height);

// Gather results back
vector<float> data;
// Check if main process - if so resize data to gather results
if (my_rank == 0)
    data.resize(res.size() * num_procs);
// Gather results
MPI_Gather(&res[0], res.size(), MPI_FLOAT, &data[0], res.size(), MPI_FLOAT, 0, MPI_COMM_WORLD);

// Save image..
```

We use gather to collect the final result.

## Parallel Sort

The parallel sort we used is an [odd-even sort](https://en.wikipedia.org/wiki/Odd%E2%80%93even_sort). This works in phases that require sharing between processes. The algorithm is:

1. Sort local data
2. For number of phases
   1. Exchange data with phase partner
   2. Merge
3. Gather results

First, we define two merge methods - one to merge at the top of the list and one at the bottom.

```cpp
// Merges the largest n values in local_data and temp_b into temp_c
// temp_c is then copied back into local_data
void merge_high(vector<unsigned int> &local_data, vector<unsigned int> &temp_b, vector<unsigned int> &temp_c)
{
    int ai, bi, ci;
    // Get starting size
    ai = bi = ci = local_data.size() - 1;
    // Loop through each value and store relevant largest value in temp_c
    for (; ci >= 0; --ci)
    {
        // Find largest from local data and temp_b
        if (local_data[ai] >= temp_b[bi])
        {
            temp_c[ci] = local_data[ai];
            --ai;
        }
        else
        {
            temp_c[ci] = temp_b[bi];
            --bi;
        }
    }
    // Copy temp_c into local_data
    copy(temp_c.begin(), temp_c.end(), local_data.begin());
}

// Merges the smallest n values in local_data and temp_b into temp_c
// temp_c is then copied back into local_data
void merge_low(vector<unsigned int> &local_data, vector<unsigned int> &temp_b, vector<unsigned int> &temp_c)
{
    int ai, bi, ci;
    // Start at 0
    ai = bi = ci = 0;
    // Loop through each value and store relevant smallest value in temp_c
    for (; ci < local_data.size(); ++ci)
    {
        // Find smallest from local data and temp_b
        if (local_data[ai] <= temp_b[bi])
        {
            temp_c[ci] = local_data[ai];
            ++ai;
        }
        else
        {
            temp_c[ci] = temp_b[bi];
            ++bi;
        }
    }
    // Copy temp_c into local_data
    copy(temp_c.begin(), temp_c.end(), local_data.begin());
}
```

We call `merge_high` and `merge_low` during an odd-even iteration where we exchange data between partners.

```cpp
void odd_even_iter(vector<unsigned int> &local_data, vector<unsigned int> &temp_b, vector<unsigned int> &temp_c, unsigned int phase, int even_partner, int odd_partner, unsigned int my_rank, unsigned int num_procs)
{
    // Operate based on phase
    if (phase % 2 == 0)
    {
        // Check if even partner is valid
        if (even_partner >= 0)
        {
            // Exchange data with even partner
            MPI_Sendrecv(&local_data[0], local_data.size(), MPI_UNSIGNED, even_partner, 0, &temp_b[0], temp_b.size(), MPI_UNSIGNED, even_partner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            // Merge results accordingly
            if (my_rank % 2 == 0)
                merge_low(local_data, temp_b, temp_c);
            else
                merge_high(local_data, temp_b, temp_c);
        }
    }
    else
    {
        // Check if odd partner is valid
        if (odd_partner >= 0)
        {
            // Exchange data with odd partner
            MPI_Sendrecv(&local_data[0], local_data.size(), MPI_UNSIGNED, odd_partner, 0, &temp_b[0], temp_b.size(), MPI_UNSIGNED, odd_partner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            // Merge results accordingly
            if (my_rank % 2 == 0)
                merge_high(local_data, temp_b, temp_c);
            else
                merge_low(local_data, temp_b, temp_c);
        }
    }
}
```

Depending on the phase and if the process has a partner to the left or right, the process exchanges data.  The process merges this data with its current results, resulting in the process having either the sorted upper portion of two process's or the sorted lower portion of two process's data.  Eventually, each process will have its sorted portion which can be sent back to the main process.

The sort method is below. It sorts the local data section before performing the required number of phases (which is equal to the number of processes involved).

```cpp
// Odd-even sort
void odd_even_sort(vector<unsigned int> &local_data, unsigned int my_rank, unsigned int num_procs)
{
    // Temporary storage
    vector<unsigned int> temp_b(local_data);
    vector<unsigned int> temp_c(local_data);
    // Partners.  Even phase look left.  Odd phase looks right
    int even_partner, odd_partner;

    // Find partners
    if (my_rank % 2 == 0)
    {
        even_partner = static_cast<int>(my_rank) + 1;
        odd_partner = static_cast<int>(my_rank) - 1;
        // Check that even_partner is valid
        if (even_partner == num_procs)
            even_partner = MPI_PROC_NULL;
    }
    else
    {
        even_partner = static_cast<int>(my_rank) - 1;
        odd_partner = static_cast<int>(my_rank) + 1;
        // Check that odd_partner is valid
        if (odd_partner == num_procs)
            odd_partner = MPI_PROC_NULL;
    }

    // Sort this processes share of the data
    // std::sort is in the algorithm header
    sort(local_data.begin(), local_data.end());

    // Phased odd-even transposition sort
    for (unsigned int phase = 0; phase < num_procs; ++phase)
        odd_even_iter(local_data, temp_b, temp_c, phase, even_partner, odd_partner, my_rank, num_procs);
}
```

The sort and phases are at the bottom of the method.  We call the sort method after scattering out the data and then gather the data at the end.

```cpp
// Data to sort
vector<unsigned int> data;
// If main process generate the data
if (my_rank == 0)
    data = generate_values(SIZE);

// Allocate enough space for local working data
vector<unsigned int> local_data(SIZE / num_procs);

// Scatter the data
cout << my_rank << ":Scattering" << endl;
MPI_Scatter(&data[0], SIZE / num_procs, MPI_UNSIGNED, &local_data[0], SIZE / num_procs, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

// Sort the data
cout << my_rank << ":Sorting" << endl;
odd_even_sort(local_data, my_rank, num_procs);

// Gather the results
cout << my_rank << ":Gathering" << endl;
MPI_Gather(&local_data[0], SIZE / num_procs, MPI_UNSIGNED, &data[0], SIZE / num_procs, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

// If main process output first 1000 values
if (my_rank == 0)
    for (unsigned int i = 0; i < 1000; ++i)
        cout << data[i] << endl;
```

Running this application will give:

```shell
...
257761
257948
258195
258220
258304
258327
258347
261482
```

## Trapezoidal Rule

For the trapezoidal rule example we will use a barrier to synchronise between processes.

```cpp
// Sync
MPI_Barrier(MPI_COMM_WORLD);

// Function to use
auto f = [](double x) { return cos(x); };

// Perform calculation
unsigned int iterations = static_cast<unsigned int>(pow(2, 24));
auto local_result = trap(f, 0.0, PI, iterations, my_rank, num_procs);

// Reduce result
double global_result = 0.0;
MPI_Reduce(&local_result, &global_result, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

// If main process, display result
if (my_rank == 0)
    cout << "Area under curve: " << global_result << endl;
```

Running this code will give you an output:

```shell
Area under curve: 1.87253e-007
```

## Performance Evaluation of MPI

We will now look at a how we measure latency and bandwidth using MPI.  These values are essential if you undertake any serious distribution of tasks and data communication.  However, you wil probably find that the stated network speed is what we get in a lab environment.

### Measuring Latency

For latency the application below:

```cpp
// Perform 100 timings
for (unsigned int i = 0; i < 100; ++i)
{
    // Sync with other process
    MPI_Barrier(MPI_COMM_WORLD);

    // Get start time
    auto start = system_clock::now();
    // Perform 100000 ping-pongs
    if (my_rank == 0)
    {
        for (unsigned int j = 0; j < 100000; ++j)
        {
            // Ping
            MPI_Send(&send, 1, MPI_CHAR, 1, 0, MPI_COMM_WORLD);
            // Pong
            MPI_Recv(&receive, 1, MPI_CHAR, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
    else
    {
        for (unsigned int j = 0; j < 100000; ++j)
        {
            // Pong
            MPI_Recv(&receive, 1, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            // Ping
            MPI_Send(&send, 1, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
        }
    }
    // Get time
    auto end = system_clock::now();
    auto total = duration_cast<nanoseconds>(end - start).count();
    // Divide total by number of iterations to get time per iteration
    auto res = static_cast<double>(total) / 100000.0;
    // Divide by two to get latency
    res /= 2.0;
    // Output result
    if (my_rank == 0)
        data << res << ",";
}

// Close file if main process
if (my_rank == 0)
{
    data << endl;
    data.close();
}
```

You will need to run this application across two machines to get a latency time.

### Measuring Bandwidth

For bandwidth you change the latency application so that it uses bigger data sizes. Use powers of two as normal, and range from approximately 1K (1024 bytes) to 1MB.  You will have to convert from the time taken to send the message to the actual MBit/s.

You should measure broadcast to see the performance. **REMEMBER** - create the charts and look at the performance. You should be able to predict the performance of an application purely by sequential computation time plus communication time.

## Exercises

1. Time the Mandelbrot.  Take into account the data transmission time.  Try and optimise the application and split across a number of nodes. Also try different data sizes to analyse performance.
2. Now do the same with parallel sort. There are more data exchanges happening so you will need to think about a number of I/O stages taking place.
3. For the trapezoidal rule you will want to greatly increase the number of iterations to get an idea of execution time.
4. Test the latency and the bandwidth of the lab. Does it meet your expectations? Take these values into account and start to estimate the probable performance of Mandelbrot and parallel sort.

## Reading

Read Chapter 3 of *Introduction to Parallel Programming* for more information on MPI.
