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