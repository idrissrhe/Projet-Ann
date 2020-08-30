from dll_load import fit_reg_RBF_naive

from pretty_print import predict_2D_RBF

gamma = 500
exampleCount = 5
inputsSize = 2
XTrain = [0.0, 0.0, 0.0, 5.0, 5.0, 0.0, 5.0, 5.0, 2.5, 2.5]
YTrain = [-1.0, -1.0, -1.0, -1.0, 1.0]


W = fit_reg_RBF_naive(XTrain, gamma, YTrain, exampleCount, inputsSize)

predict_2D_RBF(W, XTrain, YTrain, inputsSize, gamma, exampleCount, 0, 5)
