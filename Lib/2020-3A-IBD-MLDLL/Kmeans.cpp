#include "Kmeans.h"


extern "C" {

double get_distanceK(double* Xpredict, double* Xn, int inputsSize) {

    // l2-norm
    double accum = 0.;

    for (int i = 0; i < inputsSize; ++i) {
        double res = Xpredict[i] - Xn[i];
        accum += res * res;
    }

    double norm = sqrt(accum);
    return norm;
}

double* select_random_k(int K, double* Xtrain, int exampleCount, int inputsSize) {
    double* Kmeans = new double[K * inputsSize];

    std::vector<int> inputIndex;
    auto rng = std::default_random_engine{};

    for (int i = 0; i < exampleCount; i++) // Create ordered vector
        inputIndex.push_back(i);

    std::shuffle(std::begin(inputIndex), std::end(inputIndex), rng); //shuffle indexes inputs


    int pos = 0;
    for (int i = 0; i < K; i++)
    {
        for (int n = 0; n < inputsSize; n++)
        {
            Kmeans[pos] = Xtrain[(inputsSize * inputIndex[i]) + n];
            pos++;
        }
    }

    return Kmeans;
}

void center_to_cluster(double* Kmeans, int K, double* X, int exampleCount, int inputsSize, double* colors) {
    double* Xn = new double[inputsSize]; // Init with big int?
    double* Kn = new double[inputsSize];
    double* distances = new double[exampleCount];
    for (int i = 0; i < exampleCount; i++)
        distances[i] = INT_MAX;

    for (int k = 0; k < K; k++)
    {
        // get one K
        for (int i = 0; i < inputsSize; i++)
            Kn[i] = Kmeans[(inputsSize * k) + i];

        // for Kn get all distances. If distance smaller, replace
        for (int n = 0; n < exampleCount; n++)
        {
            for (int i = 0; i < inputsSize; i++)
                Xn[i] = X[(inputsSize * n) + i];

            double distance = get_distanceK(Kn, Xn, inputsSize);
            if (distance <= distances[n])
            {
                distances[n] = distance;
                colors[n] = k;
            }
        }
    }

}
void cluster_to_center(double* Kmeans, int K, double* Xtrain, int exampleCount, int inputsSize, double* old_Kmeans, double* colors) {
    // Store Kmeans in old_Kmeans
    for	(int i = 0; i < K * inputsSize; i++)
        old_Kmeans[i] = Kmeans[i];

    double* KmeansTmp = new double[K * inputsSize];
    for (int i = 0; i < K * inputsSize; i++)
        KmeansTmp[i] = 0;

    double* color_occur = new double[exampleCount];
    for (int i = 0; i < exampleCount; i++)
        color_occur[i] = 0;


    for (int k = 0; k < K; k++)
    {
        for (int i = 0; i < exampleCount; i++) // accumulate coordinate
        {
            if (colors[i] == k) {
                color_occur[k] += 1;
                for (int n = 0; n < inputsSize; n++)
                {
                    KmeansTmp[(k * inputsSize) + n] += Xtrain[(i * inputsSize) + n];
                }
            }
        }
    }

    for (int i = 0; i < K; i++) // make mean
        for (int n = 0; n < inputsSize; n++)
            KmeansTmp[(i * inputsSize) + n] = KmeansTmp[(i * inputsSize) + n]  / color_occur[i];

    for	(int i = 0; i < K * inputsSize; i++)
        Kmeans[i] = KmeansTmp[i];
}

bool should_stop(double* Kmeans, double* old_Kmeans, int size) {
    for (int i = 0; i < size; i++)
        if (Kmeans[i] != old_Kmeans[i])
            return false;
    return true;
}

DLLEXPORT double* get_Kmeans(int K, double* Xtrain, int exampleCount, int inputsSize, int epochs) {
    double* Kmeans = select_random_k(K, Xtrain, exampleCount, inputsSize);
    double* old_Kmeans = new double[K * inputsSize];
    double* colors = new double[exampleCount * inputsSize];

    for (int i = 0; i < epochs; i++)
    {
        center_to_cluster(Kmeans, K, Xtrain, exampleCount, inputsSize, colors);

        cluster_to_center(Kmeans, K, Xtrain, exampleCount, inputsSize, old_Kmeans, colors);

        if (should_stop(Kmeans, old_Kmeans, K * inputsSize))
            break;
    }

    return Kmeans;
}
}