#if _WIN32
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT
#endif
#include <iostream>
#include <vector>
#include <random>

extern "C" {
DLLEXPORT double* get_Kmeans(int K, double* Xtrain, int exampleCount, int inputsSize, int epochs);
}