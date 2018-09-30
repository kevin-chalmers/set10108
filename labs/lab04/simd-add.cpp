#include <iostream>
#include <immintrin.h>
#include <cassert>
#include <chrono>

using namespace std;
using namespace std::chrono;

union v4
{
    __m128 v;
    float a[4];
};

union v8
{
    __m256 v;
    float a[8];
};

int main(int argc, char **argv)
{
    // Test all types to ensure add is working
    {
        alignas(16) v4 x;
        alignas(16) v4 y;
        for (size_t i = 0; i < 4; ++i)
        {
            x.a[i] = static_cast<float>(i);
            y.a[i] = static_cast<float>(i);
        }
        x.v = _mm_add_ps(x.v, y.v);
        for (size_t i = 0; i < 4; ++i)
            assert(x.a[i] == static_cast<float>(2 * i));
    }
    {
        alignas(32) v8 x;
        alignas(32) v8 y;
        for (size_t i = 0; i < 8; ++i)
        {
            x.a[i] = static_cast<float>(i);
            y.a[i] = static_cast<float>(i);
        }
        x.v = _mm256_add_ps(x.v, y.v);
        for (size_t i = 0; i < 8; ++i)
            assert(x.a[i] == static_cast<float>(2 * i));
    }

    // Add 1 million float values
    // First using floats
    {
        float *d1 = (float*)_aligned_malloc(sizeof(float) * 1000000, 4);
        float *d2 = (float*)_aligned_malloc(sizeof(float) * 1000000, 4);
        auto start = system_clock::now();
        for (size_t count = 0; count < 100; ++count)
        {
            for (size_t i = 0; i < 1000000; ++i)
                d1[i] = d1[i] + d2[i];
        }
        auto total = duration_cast<nanoseconds>((system_clock::now() - start)).count() / 100;
        cout << "float time: " << total << "ns" << endl;
        _aligned_free(d1);
        _aligned_free(d2);
    }
    // Now using _m128
    {
        v4 *d1 = (v4*)_aligned_malloc(sizeof(v4) * 500000, 16);
        v4 *d2 = (v4*)_aligned_malloc(sizeof(v4) * 500000, 16);
        auto start = system_clock::now();
        for (size_t count = 0; count < 100; ++count)
        {
            for (size_t i = 0; i < 500000; ++i)
                d1[i].v = _mm_add_ps(d1[i].v, d2[i].v);
        }
        auto total = duration_cast<nanoseconds>((system_clock::now() - start)).count() / 100;
        cout << "m128 time: " << total << "ns" << endl;
        _aligned_free(d1);
        _aligned_free(d2);
    }
    // Now using _m256
    {
        v8 *d1 = (v8*)_aligned_malloc(sizeof(v8) * 250000, 32);
        v8 *d2 = (v8*)_aligned_malloc(sizeof(v8) * 250000, 32);
        auto start = system_clock::now();
        for (size_t count = 0; count < 100; ++count)
        {
            for (size_t i = 0; i < 250000; ++i)
                d1[i].v = _mm256_add_ps(d1[i].v, d2[i].v);
        }
        auto total = duration_cast<nanoseconds>((system_clock::now() - start)).count() / 100;
        cout << "m256 time: " << total << "ns" << endl;
        _aligned_free(d1);
        _aligned_free(d2);
    }

    return 0;
}