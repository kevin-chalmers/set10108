# Lab 01: Multi-threading

The aim of the first lab is to get you started with threads in C++.  We will start with CMake in different programming environments.  Once setup, we will look at different methods of thread creation: functions and lambda expressions.  We will then examine data gathering to understand parallel performance, using the Monte Carlo Pi benchmark as an example.  Then for those of you with time we will look at user-level threads with Boost.Fiber and how to use R for data analysis.

## Setting Up

### Visual Studio

### CLion

### Command Line

### Our Starting CMake File

## Starting a Thread

Creating a thread in C++ is simple - we need to include the thread header file:

```cpp
#include <thread>
```

We then create a new thread object in our application, passing in the name of the function to execute:

```cpp
thread t(hello_world);
```

This will create a new thread that will execute the function `hello_world`. The thread will start executing the function while the main application continues executing. The main application is itself a thread that is created and launched on application start.

### Waiting for a Thread to Complete

When we launch a thread we want to wait for it to complete. To do this we use the `join` method on the `thread`.

```cpp
t.join();
```

This will pause the current execution until the joined `thread` completes.

### First Multi-threaded Example

We can now create our first multi-threaded application.  This is a *Hello World* example illustrating the basics.

```cpp
#include <thread>
#include <iostream>

using namespace std ;

/*
This is the function called by the thread
*/
void hello_world()
{
    cout << "Hello from thread " << this_thread::get_id() << endl ;
}

int main(int argc, char **argv)
{
    // Create a new thread
    thread t(hello_world);
    // Wait for thread to finish (join it)
    t.join();
    // Return 0 (OK)
    return 0;
}
```

We use the following instruction:

```cpp
this_thread::get_id()
```

which gets the (operating system) assigned ID of the `thread` running.  Each time you run the application, you should get a different value.  The output of your application should be as follows:

```shell
Hello from thread 16196
```

## Multiple Tasks

When multi-threading we really want to execute multiple tasks. Creating multiple threads is easy - we just create multiple thread objects.  We will use a new operation during this next example: `sleep_for`.  It will put a `thread` to sleep for an amount of time. For example, we can put a `thread` to sleep for 10 seconds.

```cpp
sleep_for(seconds(10));
```

The `chrono` header provides access to the duration constructs.

```cpp
#include <chrono>
```

Below is a multiple task test application:

```cpp
#include <thread>
#include <chrono>
#include <iostream>

using namespace std;
using namespace std::chrono;
using namespace std::this_thread;

void task_one()
{
    cout << "Task one starting" << endl;
    cout << "Task one sleeping for 3 seconds" << endl;
    sleep_for(seconds(3));
    cout << "Task one awake again" << endl;
    cout << "Task one sleeping for 5 seconds" << endl;
    sleep_for(milliseconds(5000));
    cout << "Task one awake again" << endl;
    cout << "Task one ending" << endl;
}

void task_two()
{
    cout << "Task two starting" << endl;
    cout << "Task two sleeping for 2 seconds" << endl;
    sleep_for(microseconds(2000000));
    cout << "Task two awake again" << endl;
    cout << "Task two sleeping for 10 seconds" << endl;
    sleep_for(seconds(10));
    cout << "Task two awake again" << endl;
    cout << "Task two ending" << endl;
}

int main(int argc, char **argv)
{
    cout << "Starting task one" << endl;
    thread t1(task_one);
    cout << "Starting task two" << endl;
    thread t2(task_two);
    cout << "Joining task one" << endl;
    t1.join();
    cout << "Task one joined" << endl;
    cout << "Joining task two" << endl;
    t2.join();
    cout << "Task two joined" << endl;
    cout << "Exiting" << endl;
    return 0;
}
```

This application has been designed to show the different time constructs and show multiple tasks interleaving.  The output from this application is:

```shell
Starting task one
Starting task two
Task one starting
Task one sleeping for 3 seconds
Joining task one
Task two starting
Task two sleeping for 2 seconds
Task two awake again
Task two sleeping for 10 seconds
Task one awake again
Task one sleeping for 5 seconds
Task one awake again
Task one ending
Task one joined
Joining task two
Task two awake again
Task two ending
Task two joined
Exiting
```

## Passing Parameters to Threads

We can now create threads in C++ and put them to sleep. Next we are going to pass parameters to a `thread`. It just requires adding the parameters to the `thread` creation call. For example, the following function:

```cpp
void task(size_t n, int val)
```

we can create a `thread` and pass in parameters to `n` and `val` as follows:

```cpp
thread t(task, 1, 20);
```

`n` will be assigned 1 and `val` will be assigned 10. We will use random number generation to set these values.

### Random Numbers in C++

To use random numbers we need to include the `random` header

```cpp
#include <random>
```

We then create a random number generation engine. There are several generation engines, but we will use the default. We create this as follows:

```cpp
default_random_engine e( seed );
```

`seed` is a value used to seed the random number engine.  You should, hopefully, know that we cannot truly create random numbers so a `seed` defines the sequence.  The same `seed` will produce the same sequence of random numbers.  We will available random hardware (if present) to generate our `seed`. To get a random number from the engine we call it:

```cpp
auto num = e();
```

### Ranged `for` Loops

The other new functionality we will use is a ranged `for` loop. You may be familiar with the `foreach` loop in C#. The ranged `for` loop in C has the same functionality:

```cpp
for (auto &t : threads)
```

We use `t` as an object reference in our loop. The `threads` variable is a collection.

### Test Application

Our test application will create 100 threads.  Each `thread` will print out an index and a random number.  The threads will be stored in a `vector` so we can
`join` them all.  Our test application is:

```cpp
#include <thread>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>

using namespace std;
using namespace std::chrono;

constexpr size_t num_threads = 100;

void task(size_t n , int val)
{
    cout << "Thread: " << n << " Random Value: " << val << endl;
}

int main(int argc, char **argv)
{
    // C++ style of creating a random
    // Seed with real random number if available
    std::random_device r;
    // Create random number generator
    default_random_engine e(r());

    // Create 100 threads in a vector
    vector<thread> threads;
    for (size_t i = 0; i < num_threads; ++i)
        threads.push_back(thread(task, i, e()));

    // Use C++ ranged for loop to join the threads
    // Same as foreach in C#
    for (auto &t : threads)
        t.join();

    return 0;
}
```

Running the application gives the following:

```shell
...
Thread: 89 Random Value: 1293822889
Thread: 90 Random Value: 2009369548
Thread: 92 Random Value: 1945950277
Thread: 93 Random Value: 1557845376
Thread: 94 Random Value: 586610208
Thread: 95 Random Value: 60342479
Thread: 96 Random Value: 563763169
Thread: 97 Random Value: 469730819
Thread: 98 Random Value: 615988561
Thread: 99 Random Value: 2048566187
```

## Lambda Expressions

### What is a Lambda Expression

### Lambda Expressions in C++

### Example Application

#### Simple Lambda Expression Example

#### Function Objects

#### Fixed Values

#### Reference Values

#### Complete Application

## Lambda Expressions and Threads

## Gathering Data

### Work Function

### Creating a File

### Capturing Times

### Complete Application

### Getting the Data

## Monte Carlo Pi

### Theory

### Distribution of Random Numbers

### Monte Carlo Pi Algorithm

### Main Application

### Results

## If you have time - User-level Threads with Boost.Fibers

## If you have time - Using R to Analyse Data

## Exercises