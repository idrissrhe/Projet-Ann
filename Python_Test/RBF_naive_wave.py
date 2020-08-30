from dll_load import fit_reg_RBF_naive, flatten

from pretty_print import predict_2D_RBF

from sklearn.datasets import make_moons

# generate 2d classification dataset
X, Y = make_moons(n_samples=200, noise=0.1)
XTrain = list(flatten(X))
YTrain0 = list(flatten(Y))
YTrain = []
for i in YTrain0:
    if i == 0:
        YTrain.append(-1)
    else:
        YTrain.append(1)


gamma = 500
exampleCount = 200
inputsSize = 2

W = fit_reg_RBF_naive(XTrain, gamma, YTrain, exampleCount, inputsSize)

predict_2D_RBF(W, XTrain, YTrain, inputsSize, gamma, exampleCount, -1, 2)
