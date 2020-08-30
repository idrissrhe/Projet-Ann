#if _WIN32
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT
#endif


#include <chrono>
#include <random>
#include <algorithm>
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/QR>

extern "C" {
DLLEXPORT double* create_linear_model(int nbInputs);

DLLEXPORT void train_linear_model_classification(
        double* W,
        double* allExamplesInputs,
        int exampleCount,
        int nbInputs,
        double* allExamplesExpectedOutputs,
        double alpha,
        int epochs);

DLLEXPORT double predict_linear_model_classification(double* W, double* X, int nbInputs);

DLLEXPORT double* train_linear_model_regression(
        double* allExamplesInputs,
        int exampleCount,
        int nbInputs,
        double* allExamplesExpectedOutputs
);

DLLEXPORT double predict_linear_model_regression(double* W, double* X, int nbInputs);
}