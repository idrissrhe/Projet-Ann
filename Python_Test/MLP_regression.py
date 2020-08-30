from dll_load import create_mlp_model, train_mlp_regression, flatten
from pretty_print import predict_3D_mlp_reg
import numpy as np

if __name__ == "__main__":
    layers = [2, 2, 1]
    layer_count = 3
    exampleCount = 4
    inputsSize = 2
    alpha = 0.01
    epochs = 5000

    X = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])
    Y = np.array([2, 1, -2, -1])

    XTrain = list(flatten(X))
    YTrain = list(flatten(Y))

    W = create_mlp_model(layers, layer_count, inputsSize)

    train_mlp_regression(
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

    predict_3D_mlp_reg(
        W, layers, layer_count, inputsSize, XTrain, YTrain, -1, 1
    )
