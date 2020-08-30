from dll_load import create_mlp_model, train_mlp_classification, flatten
from pretty_print import predict_2D_mlp_multi
import numpy as np
import random

if __name__ == "__main__":
    layers = [2, 16, 16, 3]
    layer_count = 4
    exampleCount = 1000
    inputsSize = 2
    alpha = 0.04
    epochs = 5000

    X = np.random.random((1000, 2)) * 2.0 - 1.0
    Y = np.array(
        [
            [1, 0, 0]
            if abs(p[0] % 0.5) <= 0.25 and abs(p[1] % 0.5) > 0.25
            else [0, 1, 0]
            if abs(p[0] % 0.5) > 0.25 and abs(p[1] % 0.5) <= 0.25
            else [0, 0, 1]
            for p in X
        ]
    )
    XTrain = list(flatten(X))
    YTrain = list(flatten(Y))

    W = create_mlp_model(layers, layer_count)

    train_mlp_classification(
        W,
        XTrain,
        YTrain,
        layers,
        layer_count,
        exampleCount,
        inputsSize,
        alpha,
        epochs,
    )

    predict_2D_mlp_multi(W, layers, layer_count, inputsSize, XTrain, Y)
