cmake_minimum_required(VERSION 3.10)
project(lab03)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCL)
include_directories(${OpenCL_INCLUDE_DIRS})
link_libraries(${OpenCL_LIBRARIES})

# This assumes FreeImage.lib is in the same folder as the code
link_libraries(FreeImage.lib)

add_executable(monte-carlo-pi-cl monte-carlo-pi.cpp)
add_executable(mandelbrot-cl mandelbrot.cpp)
add_executable(image-rotate-cl image-rotate.cpp)

# This will copy the kernel files to the build output
configure_file(monte-carlo-pi.cl monte-carlo-pi.cl COPYONLY)
configure_file(mandelbrot.cl mandelbrot.cl COPYONLY)
configure_file(image-rotate.cl image-rotate.cl COPYONLY)

# Copy image across
configure_file(pic.png pic.png COPYONLY)

# Copy FreeImage to the build output folder
configure_file(FreeImage.lib FreeImage.lib COPYONLY)
configure_file(FreeImage.dll FreeImage.dll COPYONLY)
