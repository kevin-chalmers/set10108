# GPU Programming with OpenCL

We are now going to move onto programming the GPU to perform data parallel processing. Using the GPU in this manner is a relatively new concept. However, the principals are essentially the same as shader programming.

Before getting started you will need to make sure you have the relevant SDK installed on the machine you are using. This will depend on the hardware you are using. Intel, Nvidia and AMD each provide an SDK (the Intel and AMD ones also support the CPU as an OpenCL device). You will have to work out the setup of your OpenCL projects in Visual Studio. After that, you will be able to run OpenCL applications.

The header we will be using is `CL/cl.h`. There is a C++ header (`cl.hpp`) in some installations but not all - so we will work at a C level rather than C++. The library is `OpenCL.lib`.

## Getting Started with OpenCL

Our first application is purely about setting up OpenCL.

```cpp
// Initialise OpenCL
void initialise_opencl(vector<cl_platform_id> &platforms, vector<cl_device_id> &devices, cl_context &context, cl_command_queue &cmd_queue)
{
    // Status of OpenCL calls
    cl_int status;

    // Get the number of platforms
    cl_uint num_platforms;
    status = clGetPlatformIDs(0, nullptr, &num_platforms);
    // Resize vector to store platforms
    platforms.resize(num_platforms);
    // Fill in platform vector
    status = clGetPlatformIDs(num_platforms, &platforms[0], nullptr);

    // Assume platform 0 is the one we want to use
    // Get devices for platform 0
    cl_uint num_devices;
    status = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 0, nullptr, &num_devices);
    // Resize vector to store devices
    devices.resize(num_devices);
    // Fill in devices vector
    status = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, num_devices, &devices[0], nullptr);

    // Create a context
    context = clCreateContext(nullptr, num_devices, &devices[0], nullptr, nullptr, &status);

    // Create a command queue
    cmd_queue = clCreateCommandQueue(context, devices[0], 0, &status);
}
```

The first thing we do is get the number of platforms (line 9) and use this to resize a vector to store the platform information (lines 11 to 13). A platform in OpenCL is a different OpenCL runtime - for example your machine could have both Intel and Nvidia platforms.

Next we get the devices supported by Platform 0 - this assumes that Platform 0 is your GPU platform so you might have to modify this code. As we also want to only work with the GPU we use the device type `CL_DEVICE_TYPE_GPU`. You can use the following devices types:

- `CL_DEVICE_TYPE_CPU` a CPU device
- `CL_DEVICE_TYPE_GPU` a GPU device
- `CL_DEVICE_TYPE_ACCELERATOR` a custom OpenCL device
- `CL_DEVICE_TYPE_DEFAULT` the default OpenCL device
- `CL_DEVICE_TYPE_ALL` all devices on the platform

The final two steps are the creation of a `cl_context` (allows creation of command queues, kernels, and memory) and a `cl_command_queue` (allows sending of commands to the OpenCL device).

The main application is:

```cpp
int main()
{
    // Status of OpenCL calls
    cl_int status;

    // Initialise OpenCL
    vector<cl_platform_id> platforms;
    vector<cl_device_id> devices;
    cl_context context;
    cl_command_queue cmd_queue;
    initialise_opencl(platforms, devices, context, cmd_queue);

    // Free OpenCL resources
    clReleaseCommandQueue(cmd_queue);
    clReleaseContext(context);

    return 0;
}
```

All we are doing at the moment is calling the initialise method (line 11) and then cleaning up the resources (lines 14 and 15). Running this application will not do anything, but it will allow you to check your OpenCL setup seems to be working.

## Getting OpenCL Info

Our next application will print out the information for our OpenCL devices. To do this, we just need to grab the info using some of the OpenCL functions - again it is much like doing so from OpenGL.

```cpp
// Helper method to print OpenCL device info
void print_opencl_info(vector<cl_device_id> &devices)
{
    // Buffers for device name and vendor
    char device_name[1024], vendor[1024];
    // Declare other necessary variables
    cl_uint num_cores;
    cl_long memory;
    cl_uint clock_freq;
    cl_bool available;

    // Iterate through each device in vector and display information
    for (auto &d : devices)
    {
        // Get info for device
        clGetDeviceInfo(d, CL_DEVICE_NAME, 1024, device_name, nullptr);
        clGetDeviceInfo(d, CL_DEVICE_VENDOR, 1024, vendor, nullptr);
        clGetDeviceInfo(d, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &num_cores, nullptr);
        clGetDeviceInfo(d, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_long), &memory, nullptr);
        clGetDeviceInfo(d, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(cl_uint), &clock_freq, nullptr);
        clGetDeviceInfo(d, CL_DEVICE_AVAILABLE, sizeof(cl_bool), &available, nullptr);

        // Print info
        cout << "Device: " << device_name << endl;
        cout << "Vendor: " << vendor << endl;
        cout << "Cores: " << num_cores << endl;
        cout << "Memory: " << memory / (1024 * 1024) << "MB" << endl;
        cout << "Clock freq: " << clock_freq << "MHz" << endl;
        cout << "Available: " << available << endl;
        cout << "*************************" << endl << endl;
    }
}
```

This is a fairly straightforward method and you should know how to update your main application to use it. Running this will give you the information about your OpenCL devices.

\centering
![GPU Properties from
OpenCL[]{label="fig:opencl-gpu-properties"}](opencl-gpu-properties){#fig:opencl-gpu-properties
width="\textwidth"}

Modifying this to get all the devices for the platform might change your output. If you are using AMD or Intel hardware, you can also retrieve the CPU values. For example, my laptop (running Intel HD4000), provides the following CPU information:

- GPU
  - Device
    - HD Graphics 4000
  - Cores
    - 16
  - Memory
    - 1752 MB
  - Clock freq
    - 350 MHz
- CPU
  - Device
    - i5 CPU
  - Cores
    - 4
  - Memory
    - 2047 MB
  - Clock freq
    - 1900 MHz

So judging from the hardware specs above I have the following potential performance:

- **GPU desktop** `4 * 1800 MHz = 7200MHz = 7.2GHz`.
- **CPU laptop** `4 * 1900MHz = 7600MHz = 7.6GHz`.
- **GPU laptop** `16 * 350MHz = 5600MHz = 5.6GHz`.

My CPU looks more powerful - but these are logical cores not physical.  The CPU really only has 2 physical cores, giving 3.8 GHz. My desktop GPU also boasts 192 CUDA cores, which is theoretically 345.6 GHz. However, the graphics clock according to the specification is only 900 MHz, not 1800 MHz, so we have 172.8 Ghz.

On a standard desktop the difference between the potential processing power of the CPU and the GPU is far more dramatic. The GPU is usually many more cores at 800Mhz or more.

## Loading an OpenCL Kernel

We are now going to look at how we have a basic method of setting up OpenCL and displaying the information we require, let us move onto performing some processing. This involves us loading what are called kernels. You can think of this a little bit like loading a shader, but we are doing less specialised programming and more general purpose (hence the name *General Purpose GPU programming*).

The kernel we are using is below.  You should save this in a file called `kernel.cl`.

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

To load a kernel, we use:.

```cpp
// Loads an OpenCL program
cl_program load_program(const string &filename, cl_context &context, cl_device_id &device, cl_int num_devices)
{
    // Status of OpenCL calls
    cl_int status;

    // Create and compile program
    // Read in kernel file
    ifstream input(filename, ifstream::in);
    stringstream buffer;
    buffer << input.rdbuf();
    // Get the character array of the file contents
    auto file_contents = buffer.str();
    auto char_contents = file_contents.c_str();
    
    // Create program object
    auto program = clCreateProgramWithSource(context, 1, &char_contents, nullptr, &status);
    // Compile / build program
    status = clBuildProgram(program, num_devices, &device, nullptr, nullptr, nullptr);

    // Check if compiled
    if (status != CL_SUCCESS)
    {
        // Error building - get log
        size_t length;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &length);
        char *log = new char[length];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, length, log, &length);
        // Print log
        cout << log << endl;
        delete[] log;
    }

    // Return program object
    return program;
}
```

This method works much how we would load a shader. We read in the file contents (lines 9 to 14), and then create a `cl_program` object (line
17) and build it (line 19). If the build fails, we print out the build
log (lines 22 to 32).

Now that we are loading a program, we need to choose which kernel to
use. We can do this by updating the main function. Our updated version
is in
Listing [\[lst:opencl-main2\]](#lst:opencl-main2){reference-type="ref"
reference="lst:opencl-main2"}.

``` {#lst:opencl-main2 caption="OpenCL Main with Kernel Loading" label="lst:opencl-main2"}
int main()
{
    // Status of OpenCL calls
    cl_int status;

    // Initialise OpenCL
    vector<cl_platform_id> platforms;
    vector<cl_device_id> devices;
    cl_context context;
    cl_command_queue cmd_queue;
    initialise_opencl(platforms, devices, context, cmd_queue);

    // Print info
    print_opencl_info(devices);

    // Load program
    auto program = load_program("kernel.cl", context, devices[0], devices.size());

    // Create the kernel
    auto kernel = clCreateKernel(program, "vecadd", &status);

    // Free OpenCL resources
    clReleaseCommandQueue(cmd_queue);
    clReleaseContext(context);
    clReleaseKernel(kernel);
    clReleaseProgram(program);

    return 0;
}
```

We load the program on line 17, passing in the name of our file. We then
select the kernel we want to use on line 20. Notice as well that we are
now releasing our kernel and program (lines 25 and 26). Running this
application should still print out your OpenCL device properties -
however it should not print out an error log.

Passing Data to OpenCL Kernels
------------------------------

We are now ready to send data to our OpenCL kernel. This involves us
creating memory buffers on the GPU and copying the data to the GPU.

The first thing we need to do is create the "host" memory - that is the
memory that sits in main memory accessible by the CPU. For our
application we will create two arrays of 2048 elements.
Listing [\[lst:opencl-host\]](#lst:opencl-host){reference-type="ref"
reference="lst:opencl-host"} should be added to the main function.

``` {#lst:opencl-host caption="Creating Host Memory for OpenCL Application" label="lst:opencl-host"}
// Number of elements and size of buffer on GPU
const unsigned int elements = 2048;
const unsigned int data_size = sizeof(int) * elements;

// Host data - stored in main memory
array<int, elements> A;
array<int, elements> B;
array<int, elements> C;

// Initialise input data
for (unsigned int i = 0; i < elements; ++i)
    A[i] = B[i] = i;
```

Next we need to create our buffers on the OpenCL device. We do this in
Listing [\[lst:opencl-device\]](#lst:opencl-device){reference-type="ref"
reference="lst:opencl-device"}.

``` {#lst:opencl-device caption="Creating Device Memory for OpenCL Application" label="lst:opencl-device"}
// Create device buffers - stored on GPU
cl_mem buffer_A; // Input array on the device
cl_mem buffer_B; // Input array on the device
cl_mem buffer_C; // Output array on the device
// Allocate buffer size
buffer_A = clCreateBuffer(context, CL_MEM_READ_ONLY, data_size, nullptr, &status);
buffer_B = clCreateBuffer(context, CL_MEM_READ_ONLY, data_size, nullptr, &status);
buffer_C = clCreateBuffer(context, CL_MEM_WRITE_ONLY, data_size, nullptr, &status);
```

Notice that we have to tell OpenCL the type of buffer we are creating
(read, write, etc.) and the size of the buffer. This is just creating a
buffer on the GPU in our instance -- it is allocating memory. All we
need to do is copy our host data to our device buffers. This is done in
Listing [\[lst:opencl-copy\]](#lst:opencl-copy){reference-type="ref"
reference="lst:opencl-copy"}.

``` {#lst:opencl-copy caption="Copying Host Memory to Device Memory" label="lst:opencl-copy"}
// Copy host data to device data
status = clEnqueueWriteBuffer(cmd_queue, buffer_A, CL_FALSE, 0, data_size, A.data(), 0, nullptr, nullptr);
status = clEnqueueWriteBuffer(cmd_queue, buffer_B, CL_FALSE, 0, data_size, B.data(), 0, nullptr, nullptr);
```

We use `clEnqueueWriteBuffer` to write our data to the relevant buffers.
The third parameter is an interesting parameter as it tells OpenCL
whether the application should wait for the write to complete. Typically
we probably do not worry about a write completing - but we normally do
for reading.

The final stage is setting the kernel arguments. Remember our arguments
for the kernel are as follows:

A

:   this is an input vector

B

:   this is an input vector

C

:   this is an output vector

We can set the kernel arguments as shown in
Listing [\[lst:opencl-set\]](#lst:opencl-set){reference-type="ref"
reference="lst:opencl-set"}.

``` {#lst:opencl-set caption="Setting OpenCL Kernel Arguments" label="lst:opencl-set"}
// Set the kernel arguments
status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &buffer_A);
status |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &buffer_B);
status |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &buffer_C);
```

We need the `cl_kernel` object, the parameter number (0 indexed), the
size of the parameter, and a pointer to the parameter value.

We also need to make sure we free our buffers. We do this at the end of
the main application as shown in
Listing [\[lst:opencl-free\]](#lst:opencl-free){reference-type="ref"
reference="lst:opencl-free"}.

``` {#lst:opencl-free caption="Freeing OpenCL Memory Buffers" label="lst:opencl-free"}
// Free OpenCL resources
clReleaseMemObject(buffer_A);
clReleaseMemObject(buffer_B);
clReleaseMemObject(buffer_C);
clReleaseCommandQueue(cmd_queue);
clReleaseContext(context);
clReleaseKernel(kernel);
clReleaseProgram(program);
```

Running this application still will not do anything - we have not
executed the kernel and got our results back.

Running and Getting Results from OpenCL Kernels
-----------------------------------------------

We are finally ready to run our OpenCL kernel. To do this we need to
define a new piece of information - the dimensions of the work. In our
instance we have a single dimension arrays with elements items. We can
therefore set up our work size as shown in
Listing [\[lst:opencl-work-dim\]](#lst:opencl-work-dim){reference-type="ref"
reference="lst:opencl-work-dim"}.

``` {#lst:opencl-work-dim caption="Defining OpenCL Work Dimensions" label="lst:opencl-work-dim"}
// Configure the work dimensions - 1D of elements
array<size_t, 1> global_work_size = { elements };
```

We can now run our kernel. We do this in
Listing [\[lst:opencl-enqueue\]](#lst:opencl-enqueue){reference-type="ref"
reference="lst:opencl-enqueue"}.

``` {#lst:opencl-enqueue caption="Enqueuing a OpenCL Kernel for Execution" label="lst:opencl-enqueue"}
// Enqueue the kernel for execution
status = clEnqueueNDRangeKernel(cmd_queue, kernel, 1, nullptr, global_work_size.data(), nullptr, 0, nullptr, nullptr);
```

At the moment we are only interested in some of these parameters. These
are:

`cmd_queue`

:   the command queue we are using to enqueue the work

`kernel`

:   the kernel we are executing

`1`

:   the number of dimensions for the work

`global_work_size.data()`

:   the number of elements per dimension

We will look at some of the other parameters as we work through the rest
of the module. Finally we need to copy our data back from the kernel at
the end. We do this as follows:

``` {#lst:opencl-read caption="Reading Results Back from GPU Memory in OpenCL" label="lst:opencl-read"}
// Read the output buffer from the GPU to main memory
clEnqueueReadBuffer(cmd_queue, buffer_C, CL_TRUE, 0, data_size, C.data(), 0, nullptr, nullptr);
```

Again we are only interested in a few parameters:

`cmd_queue`

:   the command queue

`buffer_C`

:   the buffer we are reading from

`CL_TRUE`

:   we will wait for the read to complete

`0`

:   the offset of the buffer to read from

`data_size`

:   the amount of data to read from the buffer

`C.data()`

:   pointer to the host memory to copy the device buffer to

Now all we want to do is validate that the data read back is correct. We
can do this using the following code:

``` {#lst:opencl-verify caption="Verifying OpenCL Results" label="lst:opencl-verify"}
// Verify the output
auto result = true;
int i = 0;
// Iterate through each value in result array
for (auto &e : C)
{
    // Check value
    if (e != i + i)
    {
        result = false;
        break;
    }
    ++i;
}

// Check if result is true and display accordingly
if (result)
    cout << "Output is correct" << endl;
else
    cout << "Output is incorrect" << endl;
```

Running this application will provide the output shown in
Figure [1.2](#fig:opencl-output){reference-type="ref"
reference="fig:opencl-output"}.

\centering
![Output from OpenCL
Application[]{label="fig:opencl-output"}](opencl-output){#fig:opencl-output
width="\textwidth"}

We can break down our application as follows:

1.  Initialise

2.  Load kernel

3.  Create host memory

4.  Create device memory and copy host memory to device

5.  Set kernel arguments

6.  Run kernel -- remember to set the work dimensions

7.  Copy device memory back to host to get the results

8.  Clean up resources

This will be our quite standard approach to working with OpenCL. We
might iterate through some of these stages (running kernels and setting
/ getting results), but typically the process is the same.

Matrix Multiplication
---------------------

This is more of an exercise than a straight tutorial. The kernel we are
using is given in
Listing [\[lst:opencl-matrix-mult\]](#lst:opencl-matrix-mult){reference-type="ref"
reference="lst:opencl-matrix-mult"}.

\lstset{language=[OpenCL]C}
``` {#lst:opencl-matrix-mult caption="OpenCL Matrix Multiplication Kernel" label="lst:opencl-matrix-mult"}
__kernel void simply_multiply(__global float *output_C, unsigned int width_A, unsigned int height_A, unsigned int width_B, unsigned int height_B, __global float *input_A, __global float *input_B)
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

You have enough information to run this kernel. I would recommend using
matrices that are square and setting all the values of the matrices to
1. The value of each element of the output matrix will then be the
dimension of the square matrix (e.g. a $64 \times 64$ matrix will mean
each element of the output is 64).

If you get really stuck, the Heterogeneous Computing with OpenCL book
will help.
