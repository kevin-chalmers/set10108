cmake_minimum_required(VERSION 3.10)
project(lab08 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 11)
 
add_executable(hello hello.cu)
set_property(TARGET hello PROPERTY CUDA_SEPARABLE_COMPILATION ON)