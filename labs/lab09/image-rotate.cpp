#define __CL_ENABLE_EXCEPTIONS

#include <iostream>
#include <fstream>
#include <CL/cl.hpp>
#include "FreeImage.h"

using namespace std;
using namespace cl;

constexpr float PI = 3.14159265359;

int main(int argc, char **argv)
{
	FreeImage_Initialise();

	// Load an image
	FREE_IMAGE_FORMAT format = FreeImage_GetFileType("pic.png");
	FIBITMAP *image = FreeImage_Load(format, "pic.png", 0);
	// Convert image to 32-bit - how kernel works
	FIBITMAP *temp = image;
	image = FreeImage_ConvertTo32Bits(image);
	FreeImage_Unload(temp);

	// Get the image data
	int *bits = (int*)FreeImage_GetBits(image);
	unsigned int width = FreeImage_GetWidth(image);
	unsigned int height = FreeImage_GetHeight(image);
	// Store in a vector - just makes our life easier
	vector<int> image_data(bits, bits + (width * height));

	// Create rest of host data
	vector<int> result_data(width * height);

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

		// Create device buffers
		Buffer bufImage(context, CL_MEM_READ_ONLY, sizeof(int) * image_data.size());
		Buffer bufResults(context, CL_MEM_WRITE_ONLY, sizeof(int) * result_data.size());

		// Write host data to device
		queue.enqueueWriteBuffer(bufImage, CL_TRUE, 0, sizeof(int) * image_data.size(), &image_data[0]);

		// Read in kernel source
		ifstream file("image-rotate.cl");
		string code(istreambuf_iterator<char>(file), (istreambuf_iterator<char>()));

		// Create program
		Program::Sources source(1, make_pair(code.c_str(), code.length() + 1));
		Program program(context, source);

		// Build program for devices
		program.build(devices);

		// Create the kernel
		Kernel kernel(program, "image_rotate");

		// Set kernel arguments
		float cos_theta = cos(PI / 4.0f);
		float sin_theta = sin(PI / 4.0f);
		kernel.setArg(0, bufResults);
		kernel.setArg(1, bufImage);
		kernel.setArg(2, width);
		kernel.setArg(3, height);
		kernel.setArg(4, cos_theta);
		kernel.setArg(5, sin_theta);

		// Execute the kernel
		NDRange global(width, height);
		NDRange local(1, 1);
		queue.enqueueNDRangeKernel(kernel, NullRange, global, local);

		// Read the output buffer back to the host
		queue.enqueueReadBuffer(bufResults, CL_TRUE, 0, result_data.size() * sizeof(cl_int), &result_data[0]);

		// Write out image data
		FIBITMAP *result = FreeImage_ConvertFromRawBits((BYTE*)&result_data[0], width, height, ((((32 * width) + 31) / 32) * 4), 32, 0xFF000000, 0x00FF0000, 0x0000FF00);
		FreeImage_Save(FIF_BMP, result, "output.bmp", 0);
		FreeImage_Unload(result);

		FreeImage_DeInitialise();

		cout << "Finished" << endl;
	}
	catch (Error error)
	{
		cout << error.what() << "(" << error.err() << ")" << endl;
	}

	getchar();

	return 0;
}