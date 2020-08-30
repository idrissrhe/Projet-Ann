#include "Mlp.h"

extern "C" {

double error_mlp(double v_true, double v_given)
{
    return pow(v_true - v_given, 2);
}

double Loss_mlp(double* v_true, double* v_given, int nb_elem, int index)
{
    double res = 0.0;
    double* v_trueN0 = new double[nb_elem];

    for (int i = 0; i < nb_elem; i++) {
        v_trueN0[i] = -1.0;
        if (index == 0 && i < nb_elem / 3)
            v_trueN0[i] = 1.0;
        if (index == 1 && i >= nb_elem / 3 && i < 2 * nb_elem / 3)
            v_trueN0[i] = 1.0;
        if (index == 2 && i >= 2 * nb_elem / 3)
            v_trueN0[i] = 1.0;
    }

    for (int i = 0; i < nb_elem; i++)
        res += error_mlp(v_trueN0[i], v_given[i]);


    return res / nb_elem;
}

DLLEXPORT double* predict_mlp_classification(double*** W, int* layers, int layer_count, int inputsSize, double* Xinput)
{
    double** X = new double* [layer_count];

    for (int l = 0; l < layer_count; l++)
    {
        if (l == 0)
        {
            X[l] = new double[inputsSize + 1];
            int pos = 0;
            for (int input = 1; input < inputsSize + 1; input++)
                X[l][input] = Xinput[pos++];
        }
        else
            X[l] = new double[layers[l] + 1];
        X[l][0] = 1;
    }


    for (int l = 1; l < layer_count; l++)
    {
        for (int j = 1; j < (layers[l] + 1); j++)
        {
            double res = 0.0;
            for (int i = 0; i < layers[l - 1] + 1; i++)
                res += W[l][j][i] * X[l - 1][i];

            X[l][j] = std::tanh(res);
        }
    }
    return X[layer_count - 1];
}

DLLEXPORT double* predict_mlp_regression(double*** W, int* layers, int layer_count, int inputsSize, double* Xinput)
{
    double** X = new double* [layer_count];

    for (int l = 0; l < layer_count; l++)
    {
        if (l == 0)
        {
            X[l] = new double[inputsSize + 1];
            int pos = 0;
            for (int input = 1; input < inputsSize + 1; input++)
                X[l][input] = Xinput[pos++];
        }
        else
            X[l] = new double[layers[l] + 1];
        X[l][0] = 1;
    }


    for (int l = 1; l < layer_count; l++)
    {
        for (int j = 1; j < (layers[l] + 1); j++)
        {
            double res = 0.0;
            for (int i = 0; i < layers[l - 1] + 1; i++)
                res += W[l][j][i] * X[l - 1][i];

            if (l == layer_count)
                X[l][j] = res;
            else
                X[l][j] = std::tanh(res);
        }
    }
    return X[layer_count - 1];
}

void feed_mlp(double*** W, int* layers, int layer_count, int inputsSize, double** X)
{
    for (int l = 1; l < layer_count; l++)
    {
        for (int j = 1; j < (layers[l] + 1); j++)
        {
            double res = 0.0;
            for (int i = 0; i < layers[l - 1] + 1; i++)
                res += W[l][j][i] * X[l - 1][i];

            X[l][j] = std::tanh(res);
        }
    }
}

void feed_mlp_regression(double*** W, int* layers, int layer_count, int inputsSize, double** X)
{
    for (int l = 1; l < layer_count; l++)
    {
        for (int j = 1; j < (layers[l] + 1); j++)
        {
            double res = 0.0;
            for (int i = 0; i < layers[l - 1] + 1; i++)
                res += W[l][j][i] * X[l - 1][i];

            if (l == layer_count - 1)
                X[l][j] = res;
            else
                X[l][j] = std::tanh(res);
        }
    }
}

void update_delta(double*** W, double** X, int* layers, int layer_count, double** delta, int inputsSize)
{
    for (int l = layer_count - 1; l > 1; l--)
    {

        for (int i = 1; i < layers[l - 1] + 1; i++)
        {
            double res = 0.0;
            for (int j = 1; j < (layers[l] + 1); j++)
                res += W[l][j][i] * delta[l][j];

            delta[l - 1][i] = (1 - std::pow(X[l - 1][i], 2)) * res;
        }
    }
}

void update_W(double*** W, double** X, int* layers, int layer_count, int inputsSize, double** delta, double alpha, double*** prev_corr) {
    for (int l = 1; l < layer_count; l++) {

        for (int j = 1; j < layers[l] + 1; j++)
        {
            for (int i = 0; i < layers[l - 1] + 1; i++) {
                double correction = -alpha * X[l - 1][i] * delta[l][j] + (0.9 * prev_corr[l][j][i]);
                W[l][j][i] = W[l][j][i] + correction;
                prev_corr[l][j][i] = correction;
            }
        }
    }
}

void get_last_delta(double** X, int* layers, int layer_count, int* Y, double** delta) {
    int L = layer_count - 1;
    for (int j = 1; j < layers[L] + 1; j++)
        delta[L][j] = ((1 - std::pow(X[L][j], 2)) * (X[L][j] - Y[j - 1]));
}

double*** init_prev_corr(int* layers, int layer_count) {
    double*** oldW = new double** [layer_count];

    int k = 0;
    for (int l = 1; l < layer_count; l++)
    {
        oldW[l] = new double* [layers[l] + 1];
        for (int j = 1; j < layers[l] + 1; j++)
        {
            oldW[l][j] = new double[layers[l - 1] + 1];
            for (int i = 0; i < (layers[l - 1] + 1); i++)
                oldW[l][j][i] = 0.0;
        }
    }

    return oldW;
}

void get_last_delta_regression(double** X, int* layers, int layer_count, int* Y, double** delta) {
    int L = layer_count - 1;
    for (int j = 1; j < layers[L] + 1; j++)
        delta[L][j] = X[L][j] - Y[j - 1];
}

DLLEXPORT void train_mlp_classification(double*** W,
                                        double* Xtrain,
                                        int* YTrain,
                                        int* layers,
                                        int layer_count,
                                        int exampleCount,
                                        int inputsSize,
                                        double alpha,
                                        int epochs)
{
    double** X = new double* [layer_count];
    double** delta = new double* [layer_count];

    for (int l = 0; l < layer_count; l++)
    {
        X[l] = new double[layers[l] + 1];
        X[l][0] = 1;

        if (l > 0)
            delta[l] = new double[layers[l] + 1];
    }

    std::vector<int> myImageIndex;
    auto rng = std::default_random_engine{};

    for (int i = 0; i < exampleCount; i++) // Créer des vecteurs ordonnées
        myImageIndex.push_back(i);

    int* y = new int[layers[layer_count - 1]];

    double* YT = new double[exampleCount * 3];

    double* Xout0 = new double[exampleCount];
    double* Xout1 = new double[exampleCount];
    double* Xout2 = new double[exampleCount];

    double*** prev_corr = init_prev_corr(layers, layer_count);
    for (int e = 0; e < epochs; e++)
    {

        std::shuffle(std::begin(myImageIndex), std::end(myImageIndex), rng); //shuffle indexes images at each epoch

        for (int img = 0; img < exampleCount; img++)
        {
            // Load Inputs
            for (int n = 1; n < (inputsSize + 1); n++)
                X[0][n] = Xtrain[(inputsSize * myImageIndex[img]) + n - 1];


            feed_mlp(W, layers, layer_count, inputsSize, X);

            for (int subimg = 0; subimg < layers[layer_count - 1]; subimg++) // Télécharger des images aléatoires
                y[subimg] = YTrain[(layers[layer_count - 1] * myImageIndex[img]) + subimg];

            get_last_delta(X, layers, layer_count, y, delta);

            update_delta(W, X, layers, layer_count, delta, inputsSize);

            Xout0[myImageIndex[img]] = X[layer_count - 1][1];
            Xout1[myImageIndex[img]] = X[layer_count - 1][2];
            Xout2[myImageIndex[img]] = X[layer_count - 1][3];
            YT[myImageIndex[img]] = YTrain[myImageIndex[img] + 3];
            update_W(W, X, layers, layer_count, inputsSize, delta, alpha, prev_corr);
        }


        if (e % 100 == 0 || e == epochs - 1) {
            double loss0 = Loss_mlp(YT, Xout0, exampleCount, 0);
            double loss1 = Loss_mlp(YT, Xout1, exampleCount, 1);
            double loss2 = Loss_mlp(YT, Xout2, exampleCount, 2);
            std::cout << loss0 << " \n" << loss1 << " \n" << loss2 << "\n";
            printf("Epoch: %d loss: %f\n", e, (loss0 + loss1 + loss2) / 3);
        }
    }
    //delete[] X, myImageIndex, Xout, YT, delta;
}

DLLEXPORT void train_mlp_regression(double*** W,
                                    double* Xtrain,
                                    int* YTrain,
                                    int* layers,
                                    int layer_count,
                                    int exampleCount,
                                    int inputsSize,
                                    double alpha,
                                    int epochs)
{
    double** X = new double* [layer_count];
    double** delta = new double* [layer_count];

    for (int l = 0; l < layer_count; l++)
    {
        X[l] = new double[layers[l] + 1];
        X[l][0] = 1;

        if (l > 0)
            delta[l] = new double[layers[l] + 1];
    }

    std::vector<int> myImageIndex;
    auto rng = std::default_random_engine{};

    for (int i = 0; i < exampleCount; i++) // Créer des vecteurs ordonnés
        myImageIndex.push_back(i);


    int* y = new int[layers[layer_count - 1]];

    double* YT = new double[exampleCount];

    double* Xout = new double[exampleCount];
    double*** prev_corr = init_prev_corr(layers, layer_count);
    for (int e = 0; e < epochs; e++)
    {
        std::shuffle(std::begin(myImageIndex), std::end(myImageIndex), rng); //shuffle les indexes d'images

        for (int img = 0; img < exampleCount; img++)
        {
            // Load Inputs
            for (int n = 1; n < (inputsSize + 1); n++)
                X[0][n] = Xtrain[(inputsSize * myImageIndex[img]) + n - 1];


            feed_mlp_regression(W, layers, layer_count, inputsSize, X);

            for (int subimg = 0; subimg < layers[layer_count - 1]; subimg++) // Load Random Image
                y[subimg] = YTrain[(layers[layer_count - 1] * myImageIndex[img]) + subimg];

            get_last_delta_regression(X, layers, layer_count, y, delta);

            update_delta(W, X, layers, layer_count, delta, inputsSize);

            Xout[myImageIndex[img]] = X[layer_count - 1][1];

            update_W(W, X, layers, layer_count, inputsSize, delta, alpha, prev_corr);
        }

        for (int k = 0; k < exampleCount; k++)
            YT[k] = (double)YTrain[k];

        double loss = Loss_mlp(YT, Xout, exampleCount, 0);
        if (e % 1000 == 0 || e == epochs - 1)
            printf("Epoch: %d loss: %f\n", e, loss);
    }
    //delete[] X, myImageIndex, Xout, YT, delta;
}


DLLEXPORT double*** create_mlp_model(int* layers, int layer_count)
{
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);
    std::uniform_real_distribution<double> distribution(-1, 1);

    double*** W = new double** [layer_count];

    int k = 0;
    for (int l = 1; l < layer_count; l++)
    {
        W[l] = new double* [layers[l] + 1];
        for (int j = 1; j < layers[l] + 1; j++)
        {
            W[l][j] = new double[layers[l - 1] + 1];
            for (int i = 0; i < (layers[l - 1] + 1); i++)
                W[l][j][i] = distribution(generator);
        }
    }
    return W;
}
}