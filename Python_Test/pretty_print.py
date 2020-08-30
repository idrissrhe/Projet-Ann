from dll_load import (
    predict_linear_model_regression,
    predict_mlp_classification,
    predict_mlp_regression,
    predict_reg_RBF_naive,
)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def predict_3D_reg(W, inputsSize, x2, z2):
    x1 = []
    y1 = []
    z1 = []

    for x in range(0, 16):
        for y in range(0, 16):
            dot = []
            dot.append(x / 4)
            dot.append(y / 4)
            res = predict_linear_model_regression(W, dot, inputsSize)
            x1.append(x / 4)
            y1.append(y / 4)
            z1.append(res)

    x22 = x2[0::2]
    y22 = x2[1::2]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(x22, y22, z2, "filled", s=100, c="red")
    ax.scatter(x1, y1, z1, c="green", s=1)
    plt.show()


def predict_2D_reg(W, inputsSize, x2, y2):
    x1 = []
    y1 = []
    for x in range(0, 300):
        dot = []
        dot.append(x / 100)
        res = predict_linear_model_regression(W, dot, inputsSize)
        y1.append(res)
        x1.append(x / 100)

    plt.scatter(x1, y1, c="green")
    plt.scatter(x2, y2, c="blue")
    plt.show()


def predict_2D(W, inputsSize, XTrain, YTrain, low, up):
    x1 = []
    x2 = []
    y1 = []
    y2 = []

    # given point true
    x11 = []
    y11 = []

    # given point fasle
    x12 = []
    y12 = []
    pos = 0
    for y in YTrain:
        if y == 1:
            x11.append(XTrain[pos])
            pos = pos + 1
            y11.append(XTrain[pos])
            pos = pos + 1
        else:
            x12.append(XTrain[pos])
            pos = pos + 1
            y12.append(XTrain[pos])
            pos = pos + 1
    for x in range(100 * low, 100 * up):
        for y in range(100 * low, 100 * up):
            dot = []
            dot.append(x / 100)
            dot.append(y / 100)
            res = predict_linear_model_regression(W, dot, inputsSize)
            if res > 0:
                x1.append(x / 100)
                y1.append(y / 100)
            else:
                x2.append(x / 100)
                y2.append(y / 100)

    plt.scatter(x1, y1, c="green")
    plt.scatter(x2, y2, c="red")
    plt.scatter(x11, y11, c="magenta")
    plt.scatter(x12, y12, c="yellow")
    plt.show()


def predict_2D_3Class(W1, W2, W3, inputsSize, x3, y3, x4, y4, x5, y5):
    x11 = []
    y11 = []
    x12 = []
    y12 = []
    x13 = []
    y13 = []

    for x in range(0, 200):
        for y in range(0, 200):
            dot = []
            dot.append(x / 100)
            dot.append(y / 100)
            res1 = predict_linear_model_regression(W1, dot, inputsSize)
            res2 = predict_linear_model_regression(W2, dot, inputsSize)
            res3 = predict_linear_model_regression(W3, dot, inputsSize)
            l = []
            l.append(res1)
            l.append(res2)
            l.append(res3)
            if res1 == max(l):
                x11.append(x / 100)
                y11.append(y / 100)
            if res2 == max(l):
                x12.append(x / 100)
                y12.append(y / 100)
            if res3 == max(l):
                x13.append(x / 100)
                y13.append(y / 100)

    plt.scatter(x11, y11, c="green")
    plt.scatter(x12, y12, c="red")
    plt.scatter(x13, y13, c="cyan")

    plt.scatter(x3, y3, c="yellow")
    plt.scatter(x4, y4, c="magenta")
    plt.scatter(x5, y5, c="blue")
    plt.show()


def predict_2D_3Class_individual(W, inputsSize, x3, y3, x4, y4, x5, y5):
    x1 = []
    y1 = []
    x2 = []
    y2 = []

    for x in range(0, 200):
        for y in range(0, 200):
            dot = []
            dot.append(x / 100)
            dot.append(y / 100)
            res = predict_linear_model_regression(W, dot, inputsSize)
            if res > 0:
                x1.append(x / 100)
                y1.append(y / 100)
            else:
                x2.append(x / 100)
                y2.append(y / 100)

    plt.scatter(x1, y1, c="green")
    plt.scatter(x2, y2, c="red")

    plt.scatter(x3, y3, c="yellow")
    plt.scatter(x4, y4, c="magenta")
    plt.scatter(x5, y5, c="blue")
    plt.show()


def predict_2D_mlp(
    W, layers, layer_count, inputsSize, XTrain, YTrain, low, up
):
    x1 = []
    x2 = []
    y1 = []
    y2 = []

    # given point true
    x11 = []
    y11 = []

    # given point fasle
    x12 = []
    y12 = []
    pos = 0
    for y in YTrain:
        if y == 1:
            x11.append(XTrain[pos])
            pos = pos + 1
            y11.append(XTrain[pos])
            pos = pos + 1
        else:
            x12.append(XTrain[pos])
            pos = pos + 1
            y12.append(XTrain[pos])
            pos = pos + 1

    for x in range(100 * low, 100 * up):
        for y in range(100 * low, 100 * up):
            dot = []
            dot.append(x / 100)
            dot.append(y / 100)
            result = predict_mlp_classification(
                W, layers, layer_count, inputsSize, dot
            )
            res = result[1:]
            maxi = max(res)
            if maxi >= 0:
                x1.append(x / 100)
                y1.append(y / 100)
            else:
                x2.append(x / 100)
                y2.append(y / 100)

    plt.scatter(x1, y1, c="green")
    plt.scatter(x2, y2, c="red")
    plt.scatter(x11, y11, c="magenta")
    plt.scatter(x12, y12, c="yellow")
    plt.show()


def predict_2D_mlp_multi(W, layers, layer_count, inputsSize, XTrain, YTrain):
    x1 = []
    x2 = []
    x3 = []
    y1 = []
    y2 = []
    y3 = []

    # given point A
    x11 = []
    y11 = []

    # given point B
    x12 = []
    y12 = []

    # given point C
    x13 = []
    y13 = []
    pos = 0

    for y in YTrain:
        if y[0] == 1:
            x11.append(XTrain[pos])
            pos = pos + 1
            y11.append(XTrain[pos])
            pos = pos + 1
        if y[1] == 1:
            x12.append(XTrain[pos])
            pos = pos + 1
            y12.append(XTrain[pos])
            pos = pos + 1
        if y[2] == 1:
            x13.append(XTrain[pos])
            pos = pos + 1
            y13.append(XTrain[pos])
            pos = pos + 1

    for x in range(-100, 100):
        for y in range(-100, 100):
            dot = []
            dot.append(x / 100)
            dot.append(y / 100)
            result = predict_mlp_classification(
                W, layers, layer_count, inputsSize, dot
            )
            res = result[1:]
            maxi = res.index(max(res))
            if maxi == 0:
                x1.append(x / 100)
                y1.append(y / 100)
            elif maxi == 1:
                x2.append(x / 100)
                y2.append(y / 100)
            elif maxi == 2:
                x3.append(x / 100)
                y3.append(y / 100)

    plt.scatter(x1, y1, c="green")
    plt.scatter(x2, y2, c="red")
    plt.scatter(x3, y3, c="grey")
    plt.scatter(x11, y11, c="magenta")
    plt.scatter(x12, y12, c="yellow")
    plt.scatter(x13, y13, c="black")
    plt.show()


def predict_3D_mlp_reg(
    W, layers, layer_count, inputsSize, XTrain, YTrain, low, up
):
    x1 = []
    y1 = []
    z1 = []

    # given point true
    x11 = []
    y11 = []
    z11 = []

    posX = 0
    posY = 0
    for y in YTrain:
        x11.append(XTrain[posX])
        posX = posX + 1
        y11.append(XTrain[posX])
        posX = posX + 1
        z11.append(YTrain[posY])
        posY = posY + 1

    for x in range(50 * low, 50 * up):
        for y in range(50 * low, 50 * up):
            dot = []
            dot.append(x / 20)
            dot.append(y / 20)
            result = predict_mlp_regression(
                W, layers, layer_count, inputsSize, dot
            )
            res = result[1:]
            x1.append(x / 20)
            y1.append(y / 20)
            z1.append(res)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(x11, y11, z11, "filled", s=100, c="red")
    ax.scatter(x1, y1, z1, c="green", s=1)
    plt.show()


def predict_2D_RBF(W, XTrain, YTrain, inputsSize, gamma, exampleCount, low, up):

    x1 = []
    x2 = []
    y1 = []
    y2 = []

    # given point true
    x11 = []
    y11 = []

    # given point fasle
    x12 = []
    y12 = []
    pos = 0
    for y in YTrain:
        if y == 1:
            x11.append(XTrain[pos])
            pos = pos + 1
            y11.append(XTrain[pos])
            pos = pos + 1
        else:
            x12.append(XTrain[pos])
            pos = pos + 1
            y12.append(XTrain[pos])
            pos = pos + 1
    for x in range(100 * low, 100 * up):
        for y in range(100 * low, 100 * up):
            dot = []
            dot.append(x / 100)
            dot.append(y / 100)
            res = predict_reg_RBF_naive(
                W, XTrain, dot, inputsSize, gamma, exampleCount
            )
            if res > 0:
                x1.append(x / 100)
                y1.append(y / 100)
            else:
                x2.append(x / 100)
                y2.append(y / 100)

    plt.scatter(x1, y1, c="green")
    plt.scatter(x2, y2, c="red")
    plt.scatter(x11, y11, c="magenta")
    plt.scatter(x12, y12, c="yellow")
    plt.show()
