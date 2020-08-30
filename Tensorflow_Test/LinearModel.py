import numpy as np
from Load_img import *
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import math
import h5py


def load_linear_model(model_path):
    model = load_model(model_path)
    return model


def linear_keras(filename, img_per_folder, height, width, batch_size=1, epochs=100):
    inputsSize = height * width * 3

    # retrieve images
    XTrain, Y = getDataSet("../img", img_per_folder, height, width, False)
    XTrain = np.array(XTrain)
    XTrain = XTrain.reshape(3 * img_per_folder, inputsSize)
    # print(XTrain.shape)

    # create models
    W_Cats = Sequential()
    W_Cats.add(Dense(1, activation="tanh", input_dim=XTrain.shape[1]))

    W_Dogs = Sequential()
    W_Dogs.add(Dense(1, activation="tanh", input_dim=XTrain.shape[1]))

    W_Birds = Sequential()
    W_Birds.add(Dense(1, activation="tanh", input_dim=XTrain.shape[1]))

    # compile models
    W_Cats.compile(optimizer="sgd", loss="mean_squared_error", metrics=["accuracy"])
    W_Dogs.compile(optimizer="sgd", loss="mean_squared_error", metrics=["accuracy"])
    W_Birds.compile(optimizer="sgd", loss="mean_squared_error", metrics=["accuracy"])

    YTrain_Cats = np.array([])
    YTrain_Dogs = np.array([])
    YTrain_Birds = np.array([])

    for i in range(img_per_folder * 3):
        if i < img_per_folder:
            YTrain_Cats = np.append(YTrain_Cats, 1)
        else:
            YTrain_Cats = np.append(YTrain_Cats, -1)
    for i in range(img_per_folder * 3):
        if i >= img_per_folder and i < 2 * img_per_folder:
            YTrain_Dogs = np.append(YTrain_Dogs, 1)
        else:
            YTrain_Dogs = np.append(YTrain_Dogs, -1)
    for i in range(img_per_folder * 3):
        if i >= 2 * img_per_folder and i < 3 * img_per_folder:
            YTrain_Birds = np.append(YTrain_Birds, 1)
        else:
            YTrain_Birds = np.append(YTrain_Birds, -1)

    # fit models
    W_Cats.fit(XTrain, YTrain_Cats, batch_size=batch_size, verbose=1, epochs=epochs)
    W_Dogs.fit(XTrain, YTrain_Dogs, batch_size=batch_size, verbose=1, epochs=epochs)
    W_Birds.fit(XTrain, YTrain_Birds, batch_size=batch_size, verbose=1, epochs=epochs)

    # save models
    W_Cats.save("models/" + filename + "_" + str(inputsSize) + "_Cats.model")
    W_Dogs.save("models/" + filename + "_" + str(inputsSize) + "_Dogs.model")
    W_Birds.save("models/" + filename + "_" + str(inputsSize) + "_Birds.model")


def get_stats_linear_tf(img_per_folder, pathCats, pathDogs, pathBirds, isValidation):
    Cats_model = load_linear_model("models/" + pathCats)
    Dogs_model = load_linear_model("models/" + pathDogs)
    Birds_model = load_linear_model("models/" + pathBirds)

    inputsSize = int(pathCats.split("_")[1])
    size = inputsSize / 3
    size = int(math.sqrt(size))

    files = getImgPath("../img", img_per_folder, size, size, isValidation)
    result = []
    for img in files:
        cats, dogs, birds, index = predict_linear_tf(Cats_model, Dogs_model, Birds_model, img, inputsSize)
        result.append(index)

    stat = []
    for i in range(img_per_folder * 3):
        if result[i] == 0 and i < img_per_folder:
            stat.append(True)
        elif result[i] == 1 and i >= img_per_folder and i < 2 * img_per_folder:
            stat.append(True)
        elif result[i] == 2 and i >= 2 * img_per_folder and i < 3 * img_per_folder:
            stat.append(True)
        else:
            stat.append(False)
    print(sum(stat) / len(stat) * 100)
    return (sum(stat) / len(stat) * 100)


def predict_linear_tf(Cats_model, Dogs_model, Birds_model, imageToPredict, inputsSize):
    # Evaluate models
    Xpredict = []

    size = inputsSize / 3
    size = int(math.sqrt(size))

    im = Image.open(imageToPredict)
    im = im.convert("RGB")
    imResize = im.resize((size, size), Image.ANTIALIAS)
    imgLoad = imResize.load()

    for x in range(size):
        for y in range(size):
            R, G, B = imgLoad[x, y]
            Xpredict.append(R / 255)
            Xpredict.append(G / 255)
            Xpredict.append(B / 255)
    im.close()

    Xpredict = np.array(Xpredict)
    Xpredict = Xpredict.reshape(1, inputsSize)
    cats = Cats_model.predict(Xpredict)[0][0]
    dogs = Dogs_model.predict(Xpredict)[0][0]
    birds = Birds_model.predict(Xpredict)[0][0]
    index = 0
    if cats > dogs and cats > dogs:
        index = 0
    if dogs > cats and dogs > birds:
        index = 1
    if birds > dogs and birds > cats:
        index = 2
    print(cats, dogs, birds, index)
    return (cats,
            dogs,
            birds,
            index
            )
