#pragma once
#if _WIN32
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT
#endif
#include <cmath>
#include <Eigen/Dense>
#include <Eigen/QR>

#include <iostream>


extern "C" {
DLLEXPORT int predict_class_RBF_naive(double* W, double* X, double* Xpredict, int inputsSize, double gamma, int N);
DLLEXPORT double predict_reg_RBF_naive(double* W, double* X, double* Xpredict, int inputsSize, double gamma, int N);
DLLEXPORT double* fit_reg_RBF_naive(double* XTrain, double gamma, double* YTrain, int exampleCount, int inputsSize);
}