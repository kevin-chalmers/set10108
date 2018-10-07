cmake_minimum_required (VERSION 3.10)
project("lab05")

find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})
link_libraries(${MPI_LIBRARIES})

add_executable(hello-mpi hello-mpi.cpp)
add_executable(send-receive send-receive.cpp)
add_executable(monte-carlo-pi monte-carlo-pi.cpp)
add_executable(normalize normalize.cpp)
add_executable(broadcast broadcast.cpp)