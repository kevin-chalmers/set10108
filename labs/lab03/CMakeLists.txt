cmake_minimum_required(VERSION 3.10)
project(lab03)

set(CMAKE_CXX_STANDARD 11)

find_package(Threads)
link_libraries(${CMAKE_THREAD_LIBS_INIT})

include(FindOpenMP)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")

add_executable(hello-openmp hello-openmp.cpp)
add_executable(pi-openmp pi-openmp.cpp)
add_executable(bubble-sort bubble-sort.cpp)
add_executable(parallel-sort parallel-sort.cpp)
add_executable(trapezoidal trapezoidal.cpp)
add_executable(scheduler-test schedule-test.cpp)