cmake_minimum_required(VERSION 3.10)
project(lab01)

set(CMAKE_CXX_STANDARD 11)

find_package(Threads)
link_libraries(${CMAKE_THREAD_LIBS_INIT})

add_executable(hello_world hello_world.cpp)
add_executable(parameter_thread parameter_thread.cpp)
add_executable(multiple_tasks multiple_tasks.cpp)
add_executable(lambda lambda.cpp)
add_executable(lambda-thread lambda-thread.cpp)
add_executable(timing timing.cpp)
add_executable(monte-carlo-pi monte-carlo-pi.cpp)