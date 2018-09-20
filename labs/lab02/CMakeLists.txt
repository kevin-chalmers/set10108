cmake_minimum_required(VERSION 3.10)
project(block_chain)

set(CMAKE_CXX_STANDARD 11)

find_package(Threads)
link_libraries(${CMAKE_THREAD_LIBS_INIT})

add_executable(increment increment.cpp)
add_executable(increment-mutex increment-mutex.cpp)
add_executable(increment-guarded increment-guarded.cpp)
add_executable(condition-variable condition-variable.cpp)
add_executable(guarded guarded.cpp test-guarded.cpp)
add_executable(threadsafe_stack threadsafe_stack.cpp)
add_executable(increment-atomic increment-atomic.cpp)
add_executable(atomic-flag atomic-flag.cpp)
add_executable(find-max find-max.cpp)
add_executable(monte-carlo-future monte-carlo-future.cpp)