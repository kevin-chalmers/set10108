#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <fstream>
#include <cmath>

using namespace std;
using namespace std::chrono;

vector<unsigned int> generate_values(unsigned int size)
{
    // Create random engine
    random_device r;
    default_random_engine e(r());
    // Generate random numbers
    vector<unsigned int> data;
    for (unsigned int i = 0; i < size; ++i)
        data.push_back(e());
    return data;
}

void bubble_sort(vector<unsigned int>& values)
{
	// Bubble values up the vector
	for (unsigned int count = values.size(); count >= 2; --count)
	{
		for (unsigned int i = 0; i < count - 1; ++i)
		{
			// Swap values if they are in wrong order
			if (values[i] > values[i + 1])
			{
				auto tmp = values[i];
				values[i] = values[i + 1];
				values[i + 1] = tmp;
			}
		}
	}
}

int main(int argc, char **argv)
{
    // Create results file
    ofstream results("bubble.csv", ofstream::out);
    // Gather results for 2^8 to 2^16 results
    for (unsigned int size = 8; size <= 16; ++size)
    {
        // Output data size
        results << pow(2, size) << ", ";
        // Gather 100 results
        for (unsigned int i = 0; i < 100; ++i)
        {
            // Generate vector of random values
            cout << "Generating " << i << " for " << pow(2, size) << " values" << endl;
            auto data = generate_values(static_cast<unsigned int>(pow(2, size)));
            // Sort the vector
            cout << "Sorting" << endl;
            auto start = system_clock::now();
            bubble_sort(data);
            auto end = system_clock::now();
            auto total = duration_cast<milliseconds>(end - start).count();
            // Output time
            results << total << ",";
        }
        results << endl;
    }
    results.close();

    return 0;
}