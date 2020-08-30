#if _WIN32
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT
#endif
#include <random>
#include <chrono>
#include <iostream>


extern "C" {
DLLEXPORT double*** create_mlp_model(int* layers, int layer_count);

DLLEXPORT void train_mlp_classification(double*** W,
                                        double* Xtrain,
                                        int* YTrain,
                                        int* layers,
                                        int layer_count,
                                        int exampleCount,
                                        int inputsSize,
                                        double alpha,
                                        int epochs);

DLLEXPORT void train_mlp_regression(double*** W,
                                    double* Xtrain,
                                    int* YTrain,
                                    int* layers,
                                    int layer_count,
                                    int exampleCount,
                                    int inputsSize,
                                    double alpha,
                                    int epochs);

DLLEXPORT double* predict_mlp_classification(double*** W, int* layers, int layer_count, int inputsSize, double* Xinput);

DLLEXPORT double* predict_mlp_regression(double*** W, int* layers, int layer_count, int inputsSize, double* Xinput);
}
