cmake_minimum_required (VERSION 3.10)
project("lab05")

find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})
link_libraries(${MPI_LIBRARIES})