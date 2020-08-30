#include "Linear.h"

extern "C" {
DLLEXPORT double * create_linear_model(int nbInputs) {
    double* W = new double[nbInputs + 1];
    double min = -1.0;
    double max = 1.0;
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed)

    std::uniform_real_distribution<double> distribution(min, max);
    for (int i = 0; i < nbInputs + 1; i++)
    {
        W[i] = distribution(generator);
    }

    return W;
}
double error(double v_true, double v_given)
{
    return pow(v_true - v_given, 2);
}

double Loss(double* v_true, double* v_given, int nb_elem)
{
    double res = 0.0;
    for (int i = 0; i < nb_elem; i++)
    {
        res += error(v_true[i], v_given[i]);
    }
    return res / nb_elem;
}

double Alimenter(double* X, double* W, int nbInputs)
{
    double result = 0;
    for (int i = 0; i < nbInputs; i++)
    {
        result += X[i] * W[i];
    }
    return std::tanh(result);
}

DLLEXPORT double predict_linear_model_regression(double *W, double *X, int nbInputs) {
    double* newX = new double[nbInputs + 1];
    newX[0] = 1;
    int pos = 0;
    for (int i = 1; i < nbInputs + 1; i++)
        newX[i] = X[pos++];

    double result = 0;

    for (int i = 0; i < nbInputs + 1; i++)
        result += newX[i] * W[i];

    return result;
}

DLLEXPORT double predict_linear_model_classification(double* W, double* X, int inputsSize) {
    double* newX = new double[inputsSize + 1];
    newX[0] = 1;
    int pos = 0;
    for (int i = 1; i < inputsSize + 1; i++)
        newX[i] = X[pos++];

    double result = 0;
    for (int i = 0; i < inputsSize + 1; i++)
        result += newX[i] * W[i];

    auto res = std::tanh(result);
    return res >= 0 ? 1.0 : -1.0;
}

DLLEXPORT void train_linear_model_classification(
        double* W,
        double* allExamplesInputs,
        double* allExamplesExpectedOutputs,
        int exampleCount,
        int inputsSize,
        double alpha,
        int epochs) { // alpha : learning rate

    double* CorrPre = new double[inputsSize + 1];

    for (int i = 0; i < inputsSize + 1; i++)
        CorrPre[i] = 0.0;

    double output = 0.0;
    double* Xact = new double[inputsSize + 1];
    Xact[0] = 1;
    for (int e = 1; e < epochs + 1; e++)
    {
        double* Xout = new double[(double)exampleCount];
        int pos = 0;
        for (int img = 0; img < exampleCount; img++)
        {
            for (int input = 1; input < inputsSize + 1; input++)
                Xact[input] = allExamplesInputs[pos++];

            output = Alimenter(Xact, W, inputsSize + 1);
            Xout[img] = output;

            for (int i = 0; i < inputsSize + 1; i++)
            {
                double correction = alpha * (allExamplesExpectedOutputs[img] - output) * Xact[i] + (0.9 * CorrPre[i]);
                W[i] = W[i] + correction;
                CorrPre[i] = correction;
            }

        }
        double loss = Loss(allExamplesExpectedOutputs, Xout, exampleCount);
        if (e % 10 == 0 || e == epochs - 1)
            printf("Epoch: %d loss: %f\n", e, loss);


    }

    DLLEXPORT void train_linear_model_regression(
            double* allExamplesInputs,
            double* allExamplesExpectedOutputs,
            int exampleCount,
            int inputsSize) {

        Eigen::MatrixXd X(exampleCount, inputsSize + 1);
        Eigen::MatrixXd Y(exampleCount, 1);

        int pos = 0;

        for (int x = 0; x < exampleCount; x++)
        {
            for (int y = 0; y < inputsSize + 1; y++)
            {
                if (y == 0)
                    X(x, y) = 1;
                else
                {
                    X(x, y) = allExamplesInputs[pos++];
                    if (x == y)
                        X(x, y) += 0.00001;
                }
            }
        }

        for (int x = 0; x < exampleCount; x++)
            Y(x, 0) = allExamplesExpectedOutputs[x];


        Eigen::MatrixXd WW(inputsSize + 1, 1);
        Eigen::MatrixXd transposeX = X.transpose();
        Eigen::MatrixXd multX = transposeX * X;
        Eigen::MatrixXd pseudo_inverse = multX.completeOrthogonalDecomposition().pseudoInverse();
        Eigen::MatrixXd mult_inv_trans = pseudo_inverse * transposeX;
        WW = mult_inv_trans * Y;


        double* Wmat = new double[inputsSize + 1];

        for (int i = 0; i < inputsSize + 1; i++)
            Wmat[i] = WW(i);

        return Wmat;
    }

    DLLEXPORT void delete_linear_model(double* W) {
        delete[] W;
    }
}