# Lab 01: Multi-threading

The aim of the first lab is to get you started with threads in C++.  We will start with CMake in different programming environments.  Once setup, we will look at different methods of thread creation: functions and lambda expressions.  We will then examine data gathering to understand parallel performance, using the Monte Carlo Pi benchmark as an example.  Then for those of you with time we will look at user-level threads with Boost.Fiber and how to use R for data analysis.

## Setting Up

### Visual Studio

### CLion

### Command Line

### Our Starting CMake File

## Starting a Thread

Creating a thread in C++ is simple - we need to include the thread header file:

```C++
#include <thread>
```

We then create a new thread object in our application, passing in the name of the function to execute:

```C++
thread t(hello_world);
```

This will create a new thread that will execute the function `hello world`. The thread will start executing the function while the main application continues executing. The main application is itself a thread that is created and launched on application start.

### Waiting for a Thread to Complete

### First Multi-threaded Example

## Multiple Tasks

## Passing Parameters to Threads

### Random Numbers in C++

### Ranged `for` Loops

### Test Application

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