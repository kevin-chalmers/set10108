# GPU Programming with CUDA

OpenCL is the equivalent of OpenGL for GPU programming. It runs on most hardware and has tools provided by a number of vendors.  However, generalisation comes at the cost of simplicity and performance.  CUDA can be considered the DirectX of GPU programming.  It is proprietary as it is provided by Nvidia to run on Nvidai hardware.  There is a simplicity and performance benefit because of the proprietary nature of CUDA. You will find the general ideas are the same as OpenCL just fewer lines of code to get going.

## Getting Started with CUDA

If you have the CUDA SDK installed, you should have Nvidia Nsight added to Visual Studio.  This being the case, you will be able to create a new CUDA application by selecting it during the project creation (under NVIDIA in the templates). Visual Studio will provide an initial kernel.  You should delete this code and start from an empty file.

First we we need to initialise CUDA.

```cuda
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

int main(int argc, char **argv)
{
    // Initialise CUDA - select device
    cudaSetDevice(0);
    
    return 0;
}
```

If you are using Linux, then the compile command is `nvcc` which calls `gcc` as needed.  Therefore the command line arguments you can use are similar.

And that is it. Much simpler than OpenCL as we only need to select a device. If your machine only has one Nvidia device, this is just 0. Run the application to test it just to ensure everything is set up OK. Let us now output some information from CUDA.

## Getting CUDA Info

Getting information from CUDA is also fairly trivial.

```cuda
#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

using namespace std;

int main(int argc, char **argv)
{
    // Get number of devices on system
    int deviceCount; 
    cudaGetDeviceCount(&deviceCount); 

    cout << "Number of devices: " << deviceCount << endl;
    for (int i = 0; i < deviceCount; ++i) 
    {
        // Get properties for device
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, i);

        cout << "Device " << i << endl;
        cout << "Name " << deviceProp.name << endl;
        cout << "Revision " << deviceProp.major << "." << deviceProp.minor << endl;
        cout << "Memory " << deviceProp.totalGlobalMem / 1024 / 1024 << "MB" << endl;
        cout << "Warp Size " << deviceProp.warpSize << endl;
        cout << "Clock " << deviceProp.clockRate << endl;
        cout << "Multiprocessors " << deviceProp.multiProcessorCount << endl;
    } 
    return 0;
}
```

An example output is:

```shell
Name:  GeForce GTX 550 Ti
CUDA Capability: 2.1
Cores: 4
Memory: 1024MB
Clock freq: 1800MHz
```

This is the same type of information achieved from interrogating OpenCL. So why do Nvidia state that they have 192 CUDA cores for the graphics card shown? Each multiprocessor (we have 4) has a number of streaming processors (processors that can execute an instruction for us). Each multiprocessor in the graphics card has in fact 48 of these. Depending on the CUDA capability of your graphics card, you will have a different number of streaming processors per multiprocessor. The table illustrates the various capabilities.

| **Microarchitecture** | **CUDA Capability** | **SP per MP** |
|-----------------------|---------------------|---------------|
| Tesla                 |  1.0 - 1.3          |          8    |
| Fermi                 |  2.0                |         32    |
| Fermi                 |  2.1                |         48    |
| Kerpler               |  3.0 - 3.7          |        192    |
| Maxwell               |  5.0 - 5.3          |        128    |
| Pascal                |  6.0 - 6.2          |        128    |
| Volta                 |  7.0 - 7.2          |         64    |
| Turing                |  7.5                |               |

A GTX 780 has Kepler architecture and has 12 multiprocessors. This means that the 780 will have 2304 cores in total, running at approximately 862 MHz. This means that we have close to 2 THz of performance.

## CUDA Kernels

One of the main differences CUDA provides from OpenCL is that we are using a single file solution. That means that we will write our CUDA kernel in the same file as our main method. This keeps us closer to standard C++ development, and we do not need to load and compile external.

```cuda
__global__ void vecadd(const int *A, const int *B, int *C)
{
    // Get block index
    unsigned int block_idx = blockIdx.x;
    // Get thread index
    unsigned int thread_idx = threadIdx.x;
    // Get the number of threads per block
    unsigned int block_dim = blockDim.x;
    // Get the thread's unique ID - (block_idx * block_dim) + thread_idx;
    unsigned int idx = (block_idx * block_dim) + thread_idx;
    // Add corresponding locations of A and B and store in C
    C[idx] = A[idx] + B[idx];
}
```

We will get to how we launch the kernel soon. First let us look at memory management between the CPU and GPU.

## Passing Data to CUDA

As with OpenCL, we have to work between host memory (main memory) and device memory (GPU memory). If you remember with OpenCL we first declared and initialised some host memory. We will do the same this time.

```cuda
// Create host memory
auto data_size = sizeof(int) * ELEMENTS;
vector<int> A(ELEMENTS);    // Input aray
vector<int> B(ELEMENTS);    // Input array
vector<int> C(ELEMENTS);    // Output array

// Initialise input data
for (unsigned int i = 0; i < ELEMENTS; ++i)
    A[i] = B[i] = i;
```

We also need to initialise memory on our device.

```cuda
// Declare buffers
int *buffer_A, *buffer_B, *buffer_C;

// Initialise buffers
cudaMalloc((void**)&buffer_A, data_size);
cudaMalloc((void**)&buffer_B, data_size);
cudaMalloc((void**)&buffer_C, data_size);
```

Notice that we don't need any special types with CUDA. We simply declare an `int` pointer as standard. The only difference is in how we allocate memory. You may be familiar with malloc from standard C. `cudaMalloc` undertakes the same functionality but for allocating memory on the GPU.

All we need to do now is copy the memory from the host to the device. We do this using the `cudaMemcpy` operation:

```cuda
cudaMemcpy(dest, src, size, direction);
```

The `direction` value is used to tell CUDA which way the data is being copied (host to device, device to host, device to device). For our needs, we use:

```cuda
// Write host data to device
cudaMemcpy(buffer_A, &A[0], data_size, cudaMemcpyHostToDevice);
cudaMemcpy(buffer_B, &B[0], data_size, cudaMemcpyHostToDevice);
```

## Running and Getting Results from CUDA

Now we just need to run the kernel.  We do this as:

```cuda
// Run kernel with one thread for each element
// First value is number of blocks, second is threads per block.  Max 1024 threads per block
vecadd<<<ELEMENTS / 1024, 1024>>>(buffer_A, buffer_B, buffer_C);

// Wait for kernel to complete
cudaDeviceSynchronize();
```

Notice that it looks very similar to running the operation normally.  The only difference is that we are defining the number of blocks (`ELEMENTS / 1024`) and the number of threads per block (`1024`). The call to `cudaDeviceSynchronize` means that we wait for the kernel to complete executing before continuing.

To get our results back, we simply call `cudaMemcpy` again.
Listing [\[lst:cuda-copy-host\]](#lst:cuda-copy-host){reference-type="ref"
reference="lst:cuda-copy-host"} provides the necessary call.

```cuda
// Read output buffer back to the host
cudaMemcpy(&C[0], buffer_C, data_size, cudaMemcpyDeviceToHost);
```

You should write the code to check that the output is correct. It is the same as the code in the OpenCL example.

## Freeing Resources

The last thing our application has to do is free any resources used by CUDA. As we have only allocated memory, this requires only a few calls.

```cuda
// Clean up resources
cudaFree(buffer_A);
cudaFree(buffer_B);
cudaFree(buffer_C);
```

Running this application should give you the same output as that in the OpenCL version.

## Matrix Multiplication

As with OpenCL, your task here is to write and test the application required to multiply two matrices using CUDA.

```cuda
__global__ void simple_multiply(float *output_C, unsigned int width_A, unsigned int height_A, unsigned int width_B, unsigned int height_B, const float *input_A, const float *input_B)
{
    // Get global position in Y direction
    unsigned int row = (blockIdx.y * 1024) + threadIdx.y;
    // Get global position in X direction
    unsigned int col = (blockIdx.x * 1024) + threadIdx.x;
    
    float sum = 0.0f;
    
    // Calculate result of one element of matrix C
    for (unsigned int i = 0; i < width_A; ++i)
    sum += input_A[row * width_A + i] * input_B[i * width_B + col];
    
    // Store result in matrix C
    output_C[row * width_B + col] = sum;
}
```
