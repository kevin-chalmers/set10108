cmake_minimum_required(VERSION 3.10)
project(block_chain)

set(CMAKE_CXX_STANDARD 11)

add_executable(block_chain main.cpp sha256.cpp block_chain.cpp)