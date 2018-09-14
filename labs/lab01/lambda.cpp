#include <iostream>
#include <functional>

using namespace std;

int main (int argc, char **argv)
{
    // Create lambda expression
    auto add = [](int x, int y) { return x + y; };
    // Call the defined function
    auto x = add(10 , 12);
    // Display answer - should be 22
    cout << "10 + 12 = " << x << endl;

    // Create function object with same lambda expression
    function<int(int, int)> add_function = [](int x, int y) { return x + y; };
    // Call the function object
    x = add_function(20, 12);
    // Display the answer - should be 32
    cout << "20 + 12 = " << x << endl;

    int a = 5;
    int b = 10;
    // Define the values passed to the function
    auto add_fixed = [a, b] { return a + b; };
    // Call the function
    x = add_fixed();
    // Display the answer - should be 15
    cout << "5 + 10 = " << x << endl;

    // Change values of a and b
    a = 20;
    b = 30;
    // Call the fixed function again
    x = add_fixed();
    // Display the answer - will come out as 15
    cout << "20 + 30 = " << x << endl;

    // Define the values passed to the function , but pass as reference
    auto add_reference = [&a, &b] { return a + b; };
    // Call the function
    x = add_reference();
    // Display the answer - should be 50
    cout << "20 + 30 = " << x << endl;

    // Change the values of a and b
    a = 30;
    b = 5;
    // Call the reference based function again
    x = add_reference();
    // Display the answer - should be 35
    cout << "30 + 5 = " << x << endl;

    return 0;
}