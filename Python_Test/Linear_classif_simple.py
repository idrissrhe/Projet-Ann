from dll_load import create_linear_model, train_linear_model_classification, get_Kmeans
from pretty_print import predict_2D
import numpy as np
import random

if __name__ == "__main__":
    exampleCount = 8
    inputsSize = 2
    alpha = 0.02
    epochs = 200
    YTrain = []
    XTrain = [0, 0, 0, 0.5, 0.5, 0, 0.5, 0.5, 1, 1, 1, 1.5, 1.5, 1, 1.5, 1.5]
    YTrainKmeans = []
    K = 2

    for val in np.arange(0, exampleCount / 2):
        YTrain.append(-1)
    for val in np.arange(exampleCount / 2, exampleCount):
        YTrain.append(1)

    YTrainKmeans.append(1)
    YTrainKmeans.append(-1)
    W = create_linear_model(inputsSize)
    WKmeans = create_linear_model(inputsSize)

    XTrainKmeans = get_Kmeans(K, XTrain, exampleCount, inputsSize, 2)

    train_linear_model_classification(
        W, XTrain, exampleCount, inputsSize, YTrain, alpha, epochs
    )

    train_linear_model_classification(
        WKmeans, XTrainKmeans, K, inputsSize, YTrainKmeans, alpha, epochs
    )

    predict_2D(W, inputsSize, XTrain, YTrain, 0, 2)
    predict_2D(WKmeans, inputsSize, XTrainKmeans, YTrainKmeans, 0, 2)
