from dll_load import create_linear_model, train_linear_model_regression
from pretty_print import predict_2D_reg

if __name__ == "__main__":
    exampleCount = 3
    inputsSize = 1
    XTrain = [1, 2, 3]
    YTrain = [2, 3, 2.5]

    W = train_linear_model_regression(XTrain, exampleCount, inputsSize, YTrain)

    predict_2D_reg(W, inputsSize, XTrain, YTrain)
