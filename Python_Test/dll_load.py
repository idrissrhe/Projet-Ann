from ctypes import *
from ctypes.wintypes import *
import ctypes as ct
import os
from collections.abc import Iterable

dll_name = "..\\src\\x64\\Debug\\2020-3A-IBD-MLDLL.dll"
dllabspath = os.path.dirname(os.path.abspath(__file__)) + os.path.sep + dll_name
myDll = CDLL(dllabspath)

# create_linear_model
myDll.create_linear_model.argtypes = [ct.c_int]

# train_linear_model_classification
myDll.create_linear_model.restype = ct.c_void_p
myDll.train_linear_model_classification.argtypes = [
    ct.c_void_p,
    ct.c_void_p,
    ct.c_int,
    ct.c_int,
    ct.c_void_p,
    ct.c_double,
    ct.c_int,
]

# predict_linear_model_classification
myDll.predict_linear_model_classification.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_int]
myDll.predict_linear_model_classification.restype = ct.c_double

# train_linear_model_regression
myDll.train_linear_model_regression.argtypes = [ct.c_void_p, ct.c_int, ct.c_int, ct.c_void_p]
myDll.train_linear_model_regression.restype = ct.c_void_p

# predict_linear_model_regression
myDll.predict_linear_model_regression.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_int]
myDll.predict_linear_model_regression.restype = ct.c_double

# create_mlp_model
myDll.create_mlp_model.argtypes = [ct.c_void_p, ct.c_int]
myDll.create_mlp_model.restype = ct.c_void_p

# train_mlp_classification
myDll.train_mlp_classification.argtypes = [
    ct.c_void_p,
    ct.c_void_p,
    ct.c_void_p,
    ct.c_void_p,
    ct.c_int,
    ct.c_int,
    ct.c_int,
    ct.c_double,
    ct.c_int,
]

# train_mlp_regression
myDll.train_mlp_regression.argtypes = [
    ct.c_void_p,
    ct.c_void_p,
    ct.c_void_p,
    ct.c_void_p,
    ct.c_int,
    ct.c_int,
    ct.c_int,
    ct.c_double,
    ct.c_int,
]


# predict_mlp_classification
myDll.predict_mlp_classification.argtypes = [
    ct.c_void_p,
    ct.c_void_p,
    ct.c_int,
    ct.c_int,
    ct.c_void_p,
]
myDll.predict_mlp_classification.restype = POINTER(ct.c_double)

# predict_mlp_regression
myDll.predict_mlp_regression.argtypes = [
    ct.c_void_p,
    ct.c_void_p,
    ct.c_int,
    ct.c_int,
    ct.c_void_p,
]
myDll.predict_mlp_regression.restype = POINTER(ct.c_double)

# fit_reg_RBF_naive
myDll.fit_reg_RBF_naive.argtypes = [
    ct.c_void_p,
    ct.c_double,
    ct.c_void_p,
    ct.c_int,
    ct.c_int,
]
myDll.fit_reg_RBF_naive.restype = ct.c_void_p

# predict_reg_RBF_naive
myDll.predict_reg_RBF_naive.argtypes = [
    ct.c_void_p,
    ct.c_void_p,
    ct.c_void_p,
    ct.c_int,
    ct.c_double,
    ct.c_int,
]
myDll.predict_reg_RBF_naive.restype = ct.c_double

# get_Kmeans
myDll.get_Kmeans.argtypes = [ct.c_int, ct.c_void_p, ct.c_int, ct.c_int, ct.c_int]
myDll.get_Kmeans.restype = POINTER(ct.c_double)

# saveLinearModel
myDll.saveLinearModel.argtypes = [ct.c_void_p, ct.c_int, ct.c_void_p]

# getinputsSize
myDll.getinputsSize.argtypes = [ct.c_void_p]
myDll.getinputsSize.restype = ct.c_int

# loadLinearModel
myDll.loadLinearModel.argtypes = [ct.c_void_p]
myDll.loadLinearModel.restype = POINTER(ct.c_double)

# saveModel
myDll.saveModel.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_int, ct.c_void_p]

# getLayer_count
myDll.getLayer_count.argtypes = [ct.c_void_p]
myDll.getLayer_count.restype = ct.c_int

# getLayers
myDll.getLayers.argtypes = [ct.c_void_p]
myDll.getLayers.restype = POINTER(ct.c_int)

# loadModel
myDll.loadModel.argtypes = [ct.c_void_p]
myDll.loadModel.restype = ct.c_void_p


def create_linear_model(pyinputsSize):
    inputsSize = ct.c_int(pyinputsSize)
    return myDll.create_linear_model(inputsSize)


def train_linear_model_classification(
    W, pyXTrain, pyexampleCount, pyinputsSize, pyYTrain, pyAlpha, pyEpochs
):
    exampleCount = ct.c_int(pyexampleCount)
    inputsSize = ct.c_int(pyinputsSize)
    alpha = ct.c_double(pyAlpha)
    epochs = ct.c_int(pyEpochs)
    YTrain = (ct.c_double * len(pyYTrain))(*pyYTrain)
    XTrain = (ct.c_double * len(pyXTrain))(*pyXTrain)
    myDll.train_linear_model_classification(
        W, XTrain, exampleCount, inputsSize, YTrain, alpha, epochs
    )


def predict_linear_model_classification(W, pyX, pyinputsSize):
    X = (ct.c_double * len(pyX))(*pyX)
    inputsSize = ct.c_int(pyinputsSize)
    return myDll.predict_linear_model_classification(W, X, inputsSize)


def train_linear_model_regression(pyXTrain, pyexampleCount, pyinputsSize, pyYTrain):
    exampleCount = ct.c_int(pyexampleCount)
    inputsSize = ct.c_int(pyinputsSize)
    YTrain = (ct.c_double * len(pyYTrain))(*pyYTrain)
    XTrain = (ct.c_double * len(pyXTrain))(*pyXTrain)
    return myDll.train_linear_model_regression(XTrain, exampleCount, inputsSize, YTrain)


def predict_linear_model_regression(W, pyX, pyinputsSize):
    inputsSize = ct.c_int(pyinputsSize)
    X = (ct.c_double * len(pyX))(*pyX)
    return myDll.predict_linear_model_regression(W, X, inputsSize)


def create_mlp_model(pyLayers, pyLayer_count):
    layers = (ct.c_int * len(pyLayers))(*pyLayers)
    layer_count = ct.c_int(pyLayer_count)
    return myDll.create_mlp_model(layers, layer_count)


def train_mlp_classification(
    W,
    pyXTrain,
    pyYTrain,
    pyLayers,
    pyLayer_count,
    pyexampleCount,
    pyinputsSize,
    pyAlpha,
    pyEpochs,
):
    layers = (ct.c_int * len(pyLayers))(*pyLayers)
    layer_count = ct.c_int(pyLayer_count)
    exampleCount = ct.c_int(pyexampleCount)
    inputsSize = ct.c_int(pyinputsSize)
    alpha = ct.c_double(pyAlpha)
    epochs = ct.c_int(pyEpochs)
    YTrain = (ct.c_int * len(pyYTrain))(*pyYTrain)
    XTrain = (ct.c_double * len(pyXTrain))(*pyXTrain)
    myDll.train_mlp_classification(
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


def predict_mlp_classification(W, pyLayers, pyLayer_count, pyinputsSize, pyX):
    layers = (ct.c_int * len(pyLayers))(*pyLayers)
    layer_count = ct.c_int(pyLayer_count)
    X = (ct.c_double * len(pyX))(*pyX)
    inputsSize = ct.c_int(pyinputsSize)
    res = myDll.predict_mlp_classification(
        W, layers, layer_count, inputsSize, X
    )
    l = [res[i] for i in range(pyLayers[-1] + 1)]
    return l


def train_mlp_regression(
    W,
    pyXTrain,
    pyYTrain,
    pyLayers,
    pyLayer_count,
    pyexampleCount,
    pyinputsSize,
    pyAlpha,
    pyEpochs,
):
    layers = (ct.c_int * len(pyLayers))(*pyLayers)
    layer_count = ct.c_int(pyLayer_count)
    exampleCount = ct.c_int(pyexampleCount)
    inputsSize = ct.c_int(pyinputsSize)
    alpha = ct.c_double(pyAlpha)
    epochs = ct.c_int(pyEpochs)
    YTrain = (ct.c_int * len(pyYTrain))(*pyYTrain)
    XTrain = (ct.c_double * len(pyXTrain))(*pyXTrain)
    myDll.fit_mlp_regression(
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


def predict_mlp_regression(W, pyLayers, pyLayer_count, pyinputsSize, pyX):
    layers = (ct.c_int * len(pyLayers))(*pyLayers)
    layer_count = ct.c_int(pyLayer_count)
    X = (ct.c_double * len(pyX))(*pyX)
    inputsSize = ct.c_int(pyinputsSize)
    res = myDll.predict_mlp_regression(W, layers, layer_count, inputsSize, X)
    l = [res[i] for i in range(pyLayers[-1] + 1)]
    return l


def flatten(items):
    """Yield items from any nested iterable; see Reference."""
    for x in items:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            for sub_x in flatten(x):
                yield sub_x
        else:
            yield x


def fit_reg_RBF_naive(
    pyXTrain, pyGamma, pyYTrain, pyexampleCount, pyinputsSize
):
    XTrain = (ct.c_double * len(pyXTrain))(*pyXTrain)
    YTrain = (ct.c_double * len(pyYTrain))(*pyYTrain)
    inputsSize = ct.c_int(pyinputsSize)
    exampleCount = ct.c_int(pyexampleCount)
    gamma = ct.c_double(pyGamma)
    return myDll.fit_reg_RBF_naive(
        XTrain, gamma, YTrain, exampleCount, inputsSize
    )


def predict_reg_RBF_naive(
    W, pyXTrain, pyXpredict, pyinputsSize, pyGamma, pyexampleCount
):
    XTrain = (ct.c_double * len(pyXTrain))(*pyXTrain)
    inputsSize = ct.c_int(pyinputsSize)
    exampleCount = ct.c_int(pyexampleCount)
    gamma = ct.c_double(pyGamma)
    Xpredict = (ct.c_double * len(pyXpredict))(*pyXpredict)
    return myDll.predict_reg_RBF_naive(
        W, XTrain, Xpredict, inputsSize, gamma, exampleCount
    )


def get_Kmeans(pyK, pyXTrain, pyexampleCount, pyinputsSize, pyEpochs):
    K = ct.c_int(pyK)
    XTrain = (ct.c_double * len(pyXTrain))(*pyXTrain)
    inputsSize = ct.c_int(pyinputsSize)
    exampleCount = ct.c_int(pyexampleCount)
    epochs = ct.c_int(pyEpochs)
    kmeansC = myDll.get_Kmeans(K, XTrain, exampleCount, inputsSize, epochs)
    kmeans = [kmeansC[i] for i in range(pyK * pyinputsSize)]
    return kmeans


def saveLinearModel(W, pyinputsSize, pyFileName):
    inputsSize = ct.c_int(pyinputsSize)
    fileName = ctypes.c_char_p(pyFileName.encode("utf-8"))
    myDll.saveLinearModel(W, inputsSize, fileName)


def getinputsSize(pyFileName):
    fileName = ctypes.c_char_p(pyFileName.encode("utf-8"))
    return myDll.getInputsSize(fileName)


def loadLinearModel(pyFileName):
    fileName = ctypes.c_char_p(pyFileName.encode("utf-8"))
    inputsSizeC = getinputsSize(pyFileName)
    inputsSize = int(inputsSizeC)
    return inputsSize, myDll.loadLinearModel(fileName)


def saveModel(W, pyLayers, pyLayer_count, pyFileName):
    layers = (ct.c_int * len(pyLayers))(*pyLayers)
    layer_count = ct.c_int(pyLayer_count)
    fileName = ctypes.c_char_p(pyFileName.encode("utf-8"))
    myDll.saveModel(W, layers, layer_count, fileName)


def getLayer_count(pyFileName):
    fileName = ctypes.c_char_p(pyFileName.encode("utf-8"))
    return myDll.getLayer_count(fileName)


def getLayers(pyFileName):
    fileName = ctypes.c_char_p(pyFileName.encode("utf-8"))
    return myDll.getLayers(fileName)


def loadModel(pyFileName):
    fileName = ctypes.c_char_p(pyFileName.encode("utf-8"))
    layer_countC = getLayer_count(pyFileName)
    layer_count = int(layer_countC)
    layersC = getLayers(pyFileName)
    layers = [layersC[i] for i in range(layer_count)]
    return layer_count, layers, myDll.loadModel(fileName)
