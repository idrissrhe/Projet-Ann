#include "RBF.h"


extern "C" {

double get_distance(double* Xpredict, double* Xn, int inputsSize) {

    double accum = 0.;

    for (int i = 0; i < inputsSize; ++i) {
        double res = Xpredict[i] - Xn[i];
        accum += res * res;
    }

    double norm = sqrt(accum);
    return norm;
}

double gauss(double* Xpredict, double* Xn, double gamma, int inputsSize) {
    double dist = get_distance(Xpredict, Xn, inputsSize);
    double pow = std::pow(dist, 2);
    double gam_pow = -gamma * pow;
    double exp = std::exp(gam_pow);
    return exp;
}

DLLEXPORT int predict_class_RBF_naive(double* W, double* X, double* Xpredict, int inputsSize, double gamma, int N) {
    if (predict_reg_RBF_naive(W, X, Xpredict, inputsSize, gamma, N) >= 0)
        return 1;
    return -1;
}

DLLEXPORT double predict_reg_RBF_naive(double* W, double* X, double* Xpredict, int inputsSize, double gamma, int N) {
    double* Xn = new double[inputsSize];
    double w_sum = 0;

    for (int n = 0; n < N; n++)
    {
        for (int i = 0; i < inputsSize; i++)
        {
            Xn[i] = X[(inputsSize * n) + i];
        }
        w_sum += W[n] * gauss(Xpredict, Xn, gamma, inputsSize);
    }
    return w_sum;
}

DLLEXPORT double* fit_reg_RBF_naive(double* XTrain, double gamma, double* YTrain, int exampleCount, int inputsSize) {
    Eigen::MatrixXd phi(exampleCount, sampleCount);
    Eigen::MatrixXd Y(exampleCount, 1);

    double* Xn1 = new double[inputsSize];
    double* Xn2 = new double[inputsSize];
    for (int x = 0; x < exampleCount; x++)
    {
        for (int y = 0; y < exampleCount; y++)
        {
            for (int i = 0; i < inputsSize; i++)
            {
                Xn1[i] = XTrain[(inputsSize * x) + i];
            }
            for (int i = 0; i < inputsSize; i++)
            {
                Xn2[i] = XTrain[(inputsSize * y) + i];
            }
            phi(x, y) = gauss(Xn1, Xn2, gamma, inputsSize);

        }
    }

    for (int x = 0; x < exampleCount; x++)
        Y(x, 0) = YTrain[x];

    Eigen::MatrixXd W(inputsSize, 1);
    auto inv = phi.inverse();
    W = inv * Y;


    double* Wmat = new double[exampleCount];



    for (int i = 0; i < exampleCount; i++)
        Wmat[i] = W(i);

    return Wmat;
}
}

