from dll_load import create_linear_model, train_linear_model_classification
from pretty_print import predict_2D

if __name__ == "__main__":
    exampleCount = 4
    inputsSize = 2
    alpha = 0.02
    epochs = 1000
    YTrain = [-1, 1, 1, 1]
    XTrain = [0, 0, 0, 1, 1, 0, 1, 1]

    W = create_linear_model(inputsSize)

    train_linear_model_classification(
        W, XTrain, exampleCount, inputsSize, YTrain, alpha, epochs
    )

    predict_2D(W, inputsSize, XTrain, YTrain, 0, 1)
