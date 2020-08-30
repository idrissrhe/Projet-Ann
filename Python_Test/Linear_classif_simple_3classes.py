from dll_load import create_linear_model, train_linear_model_classification
from pretty_print import predict_2D_3Class_individual, predict_2D_3Class
import numpy as np
import random

if __name__ == "__main__":
    exampleCount = 50
    inputsSize = 2
    alpha = 0.02
    epochs = 2000
    YTrain1 = []
    YTrain2 = []
    YTrain3 = []
    XTrain = []
    x1 = []
    y1 = []
    x2 = []
    y2 = []
    x3 = []
    y3 = []

    for x in np.arange(0, exampleCount / 3):
        tmp = random.uniform(0, 0.7) * 0.9
        XTrain.append(tmp)
        x1.append(tmp)
        tmp = random.uniform(0, 0.8) * 0.9
        XTrain.append(tmp)
        y1.append(tmp)

    for x in np.arange(0, exampleCount / 3):
        tmp = random.uniform(1.2, 2) * 0.9
        XTrain.append(tmp)
        x2.append(tmp)
        tmp = random.uniform(0, 0.8) * 0.9
        XTrain.append(tmp)
        y2.append(tmp)

    for x in np.arange(0, exampleCount / 3):
        tmp = random.uniform(0.5, 1.5) * 0.9
        XTrain.append(tmp)
        x3.append(tmp)
        tmp = random.uniform(1.5, 2) * 0.9
        XTrain.append(tmp)
        y3.append(tmp)

    for val in np.arange(0, exampleCount / 3):
        YTrain1.append(1)
    for val in np.arange(exampleCount / 3, exampleCount):
        YTrain1.append(-1)

    for val in np.arange(0, exampleCount / 3):
        YTrain2.append(-1)
    for val in np.arange(exampleCount / 3, exampleCount * (2 / 3)):
        YTrain2.append(1)
    for val in np.arange(exampleCount * (2 / 3), exampleCount):
        YTrain2.append(-1)

    for val in np.arange(0, exampleCount * (2 / 3)):
        YTrain3.append(-1)
    for val in np.arange(exampleCount * (2 / 3), exampleCount):
        YTrain3.append(1)

    W1 = create_linear_model(inputsSize)
    W2 = create_linear_model(inputsSize)
    W3 = create_linear_model(inputsSize)

    train_linear_model_classification(
        W1, XTrain, exampleCount, inputsSize, YTrain1, alpha, epochs
    )
    train_linear_model_classification(
        W2, XTrain, exampleCount, inputsSize, YTrain2, alpha, epochs
    )
    train_linear_model_classification(
        W3, XTrain, exampleCount, inputsSize, YTrain3, alpha, epochs
    )

    predict_2D_3Class_individual(W1, inputsSize, x1, y1, x2, y2, x3, y3)
    predict_2D_3Class_individual(W2, inputsSize, x1, y1, x2, y2, x3, y3)
    predict_2D_3Class_individual(W3, inputsSize, x1, y1, x2, y2, x3, y3)
    predict_2D_3Class(W1, W2, W3, inputsSize, x1, y1, x2, y2, x3, y3)
