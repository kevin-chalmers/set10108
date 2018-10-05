# GPU Programming with OpenCL

We are now going to move onto programming the GPU to perform data parallel processing. Using the GPU in this manner is a relatively new concept. However, if you have done graphics programming, the principals are essentially the same as shader programming.

Before getting started you will need to make sure you have the relevant SDK installed on the machine you are using. This will depend on the hardware you are using. Intel, Nvidia and AMD each provide an SDK (the Intel and AMD ones also support the CPU as an OpenCL device). You will have to work out the setup of your OpenCL projects in Visual Studio. After that, you will be able to run OpenCL applications.

The header we will be using is `CL/cl.hpp`. The library is `OpenCL.lib`.  If you are having problems (e.g. library not found or `cl.hpp` not available) ask in the lab.

## Getting Started with OpenCL

Our first application is purely about setting up OpenCL.

```cpp
#define __CL_ENABLE_EXCEPTIONS

#include <iostream>
#include <vector>
#include <CL/cl.hpp>

using namespace std;
using namespace cl;

int main(int argc, char **argv)
{
    try
    {
        // Get the platforms
        vector<Platform> platforms;
        Platform::get(&platforms);

        // Assume only one platform.  Get GPU devices.
        vector<Device> devices;
        platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);

        // Just to test, print out device 0 name
        cout << devices[0].getInfo<CL_DEVICE_NAME>() << endl;

        // Create a context with these devices
        Context context(devices);

        // Create a command queue for device 0
        CommandQueue queue(context, devices[0]);
    }
    catch (Error error)
    {
        cout << error.what() << "(" << error.err() << ")" << endl;
    }
    return 0;
}
```

The first thing we do is get the platforms supported on the machine. A platform in OpenCL is a different OpenCL runtime - for example a machine could have both Intel and Nvidia platforms.

Next we get the first GPU device supported by `platform[0]` - this assumes that `platform[0]` is your GPU platform so you might have to modify this code. As we also want to only work with the GPU we use the device type `CL_DEVICE_TYPE_GPU`. You can use the following devices types:

- `CL_DEVICE_TYPE_CPU` a CPU device.
- `CL_DEVICE_TYPE_GPU` a GPU device.
- `CL_DEVICE_TYPE_ACCELERATOR` a custom OpenCL device.
- `CL_DEVICE_TYPE_DEFAULT` the default OpenCL device.
- `CL_DEVICE_TYPE_ALL` all devices on the platform.

The final two steps are the creation of a `Context` (allows creation of command queues, kernels, and memory) and a `CommandQueue` (allows sending of commands to the OpenCL device).

We get the device name (`CL_DEVICE_NAME`) using `getInfo` just to test.  If everything works, you should get the name of your GPU output to the screen.  On my old and slow laptop:

```shell
Intel(R) HD Graphics IvyBridge M GT2
```

## Getting OpenCL Info

Our next application will print out the information for our OpenCL devices. To do this, we just need to grab the info using some of the OpenCL functions.

```cpp
int main(int argc, char **argv)
{
    try
    {
        // Get the platforms
        vector<Platform> platforms;
        Platform::get(&platforms);

        // Iterate through each platform
        for (auto &p : platforms)
        {
            // Print out platform name
            cout << "********************" << endl;
            cout << p.getInfo<CL_PLATFORM_NAME>() << endl;

            // Get all devices for the platform
            vector<Device> devices;
            p.getDevices(CL_DEVICE_TYPE_ALL, &devices);

            // Iterate through all the devices
            for (auto &d : devices)
            {
                cout << endl;
                cout << d.getInfo<CL_DEVICE_NAME>() << endl;
                cout << "Vendor: " << d.getInfo<CL_DEVICE_VENDOR>() << endl;
                cout << "Max Compute: " << d.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << endl;
                cout << "Max Memory: " << d.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>() / (1024 * 1024) << "MB" << endl;
                cout << "Clock Freq: " << d.getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>() << "MHz" << endl;
                cout << "Available: " << (d.getInfo<CL_DEVICE_AVAILABLE>() ? "True" : "False") << endl;
            }
            
            cout << endl;
            cout << "********************" << endl;
        }
    }
    catch (Error error)
    {
        cout << error.what() << "(" << error.err() << ")" << endl;
    }
    return 0;
}
```

This is a fairly straightforward method and you should know how to update your main application to use it. Running this will give you the information about your OpenCL devices.

```shell
********************
Intel Gen OCL Driver

Intel(R) HD Graphics IvyBridge M GT2
Vendor: Intel
Max Compute: 16
Max Memory: 1913MB
Clock Freq: 1000MHz
Available: True

********************
```

Modifying this to get all the devices for the platform might change your output. If you are using AMD or Intel hardware, you can also retrieve the CPU values.

## Loading an OpenCL Kernel

We are now going to look at how we have a basic method of setting up OpenCL and displaying the information we require, let us move onto performing some processing. This involves us loading what are called kernels. You can think of this a little bit like loading a shader, but we are doing less specialised programming and more general purpose (hence the name *General Purpose GPU programming*).

The kernel we are using is below.  You should save this in a file called `vec-add.cl`.

```opencl
__kernel void vecadd(__global int *A, __global int *B, __global int *C)
{
    // Get the work item's unique ID
    int idx = get_global_id(0);
    // Add corresponding locations of A and B and store in C
    C[idx] = A[idx] + B[idx];
}
```

We will look at this code in a bit of detail. First, if our function is a kernel we use the keyword `__kernel` at the start of the declaration. Kernels do not return values, so our return value is `void`.

The parameters for our kernel all are declared as `__global`. This means that they are accessible to all the cores when the kernel executes - it is global memory. We declare these as pointers as they will be blocks of memory that we will be accessing and writing to.

Our kernel is adding two vectors - or two arrays of a particular size (we will add two 2048 element vectors together). The `get_global_id` function allows us to get the index of the current executing thread. A thread can have various dimensions for the index - so we can get the `id` for 0, 1, 2, etc. We can also get the local id for work groups. As our kernel adds two 1D vectors, we only need to use the 0 dimension.

The final line of the kernel just stores the value. It is a standard line of code.

The code to run this kernel, adding two 2048-element arrays, is:

```cpp
#define __CL_ENABLE_EXCEPTIONS

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <array>
#include <CL/cl.hpp>

using namespace std;
using namespace cl;

constexpr int ELEMENTS = 2048;
constexpr std::size_t DATA_SIZE = sizeof(int) * ELEMENTS;

int main(int argc, char **argv)
{
    // Initialise memory
    array<int, ELEMENTS> A;
    array<int, ELEMENTS> B;
    array<int, ELEMENTS> C;
    for (std::size_t i = 0; i < ELEMENTS; ++i)
        A[i] = B[i] = i;

    try
    {
        // Get the platforms
        vector<Platform> platforms;
        Platform::get(&platforms);

        // Assume only one platform.  Get GPU devices.
        vector<Device> devices;
        platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);

        // Just to test, print out device 0 name
        cout << devices[0].getInfo<CL_DEVICE_NAME>() << endl;

        // Create a context with these devices
        Context context(devices);

        // Create a command queue for device 0
        CommandQueue queue(context, devices[0]);

        // Create the buffers
        Buffer bufA(context, CL_MEM_READ_ONLY, DATA_SIZE);
        Buffer bufB(context, CL_MEM_READ_ONLY, DATA_SIZE);
        Buffer bufC(context, CL_MEM_WRITE_ONLY, DATA_SIZE);

        // Copy data to the GPU
        queue.enqueueWriteBuffer(bufA, CL_TRUE, 0, DATA_SIZE, &A);
        queue.enqueueWriteBuffer(bufB, CL_TRUE, 0, DATA_SIZE, &B);

        // Read in kernel source
        ifstream file("vec-add.cl");
        string code(istreambuf_iterator<char>(file), (istreambuf_iterator<char>()));
        
        // Create program
        Program::Sources source(1, make_pair(code.c_str(), code.length() + 1));
        Program program(context, source);

        // Build program for devices
        program.build(devices);

        // Create the kernel
        Kernel vecadd_kernel(program, "vecadd");

        // Set kernel arguments
        vecadd_kernel.setArg(0, bufA);
        vecadd_kernel.setArg(1, bufB);
        vecadd_kernel.setArg(2, bufC);

        // Execute kernel
        NDRange global(ELEMENTS);
        NDRange local(256);
        queue.enqueueNDRangeKernel(vecadd_kernel, NullRange, global, local);

        // Copy result back.
        queue.enqueueReadBuffer(bufC, CL_TRUE, 0, DATA_SIZE, &C);

        // Test that the results are correct
        for (int i = 0; i < 2048; ++i)
            if (C[i] != i + i)
                cout << "Error: " << i << endl;

        cout << "Finished" << endl;
    }
    catch (Error error)
    {
        cout << error.what() << "(" << error.err() << ")" << endl;
    }
    return 0;
}
```

**TODO: Add explanation**

At the moment we are only interested in some of these parameters. These are:

- `kernel` the kernel we are executing.
- `NullRange` the number of dimensions for the work.
- `global` the number of elements per dimension.

We will look at some of the other parameters as we work through the rest of the module. Finally we need to copy our data back from the kernel at the end. We do this as follows:

- `bufC` the buffer we are reading from.
- `CL_TRUE` we will wait for the read to complete.
- `0` the offset of the buffer to read from.
- `DATA_SIZE` the amount of data to read from the buffer.
- `C` pointer to the host memory to copy the device buffer to.

Now all we want to do is validate that the data read back is correct. We can do this using the following code:

Running this application will provide:

```shell
Intel(R) HD Graphics IvyBridge M GT2
Finished
```

We can break down our application as follows:

1. Create host memory.
2. Initialise.
3. Create device memory and copy host memory to device.
4. Load kernel.
5. Set kernel arguments.
6. Run kernel - remember to set the work dimensions.
7. Copy device memory back to host to get the results.
8. Clean up resources.

This will be our quite standard approach to working with OpenCL. We might iterate through some of these stages (running kernels and setting / getting results), but typically the process is the same.

## Matrix Multiplication

This is more of an exercise than a straight tutorial. The kernel we are using is given is:

```opencl
__kernel void simple_multiply(__global float *output_C, unsigned int width_A, unsigned int height_A, unsigned int width_B, unsigned int height_B, __global float *input_A, __global float *input_B)
{
    // Get global position in Y direction
    unsigned int row = get_global_id(1);
    // Get global position in X direction
    unsigned int col = get_global_id(0);

    float sum = 0.0f;

    // Calculate result of one element of matrix C
    for (unsigned int i = 0; i < width_A; ++i)
        sum += input_A[row * width_A + i] * input_B[i * width_B + col];

    // Store result in matrix C
    output_C[row * width_B + col] = sum;
}
```

You have enough information to run this kernel. I would recommend using matrices that are square and setting all the values of the matrices to 1. The value of each element of the output matrix will then be the dimension of the square matrix (e.g. a `64 * 64` matrix will mean each element of the output is 64).

If you get really stuck, the Heterogeneous Computing with OpenCL book will help.
