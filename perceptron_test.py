import numpy as np
from tensorflow.keras.datasets import mnist
import copy
import matplotlib.pyplot as plt

data = mnist
(x_train, y_train),(x_test, y_test) = data.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

input_size = x_train[0].shape[0] * x_train[0].shape[1]
# input_vector = np.array(x_train[0]).flatten()

# Input vector size is 28*28 = 784

# First Layer: 10 Neurons
# b_1 = np.zeros(10)
# w_1 = np.zeros((10, input_size))


def sig(x):
    return 1 / (1 + np.exp(-x))


def df_sig(x):
    return sig(x) * (1 - sig(x))


def layer_0(input_0, w_0, b_0):
    n_0 = sig(b_0 + np.sum(w_0.dot(input_0)))
    return n_0


def layer_1(input_1, w_1, b_1):
    n_1 = sig(b_1 + np.sum(w_1.dot(input_1)))
    return n_1


def output(input_2, w_2, b_2):
    n_2 = sig(b_2 + np.sum(w_2.dot(input_2)))
    return n_2


def network(x, weight, bias):
    w_0, w_1, w_2 = weight[0], weight[1], weight[2]
    b_0, b_1, b_2 = bias[0], bias[1], bias[2]

    input_layer = np.array(x).flatten()

    first_layer = layer_0(input_layer, w_0, b_0)
    second_layer = layer_1(first_layer, w_1, b_1)
    out_layer = output(second_layer, w_2, b_2)

    return out_layer


def training(x_train, y_train, alpha):
    # Initialise random weight and bias
    w_0 = np.random.uniform(-0.1, 0.1, (16, input_size))
    w_1 = np.random.uniform(-0.1, 0.1, (16, 16))
    w_2 = np.random.uniform(-0.1, 0.1, (10, 16))

    b_0 = np.random.uniform(-0.1, 0.1, (16, 1))
    b_1 = np.random.uniform(-0.1, 0.1, (16, 1))
    b_2 = np.random.uniform(-0.1, 0.1, (10, 1))

    alpha = 0.01
    index_i = np.arange(len(x_train))
    e_measure = []

    for iterations in range(1):
        np.random.shuffle(index_i)

        for i in range(len(y_train)-59000):
            y = np.zeros((10, 1))
            y[y_train[i]] = 1
            input_layer = np.expand_dims(np.array(x_train[i]).flatten(), axis=1)

            first_layer = layer_0(input_layer, w_0, b_0)
            second_layer = layer_1(first_layer, w_1, b_1)
            out_layer = output(second_layer, w_2, b_2)

            error = (out_layer - y)
            se = error ** 2
            e_measure.append(np.sum(se) / 10)

            # print(out_layer.T)
            print(iterations, i, np.sum(se) / 10)

            nb_2 = b_2 - alpha * ((2 / 10) * error)
            nw_2 = w_2 - alpha * ((2 / 10) * error.dot(second_layer.T))

            nb_1 = b_1 - alpha * ((2 / 10) * w_2.T.dot(error))
            nw_1 = w_1 - alpha * ((2 / 10) * w_2.T.dot(error.dot(first_layer.T)))

            nb_0 = b_0 - alpha * ((2 / 10) * (w_2.dot(w_1)).T.dot(error))
            nw_0 = w_0 - alpha * ((2 / 10) * (w_2.dot(w_1)).T.dot(error.dot(input_layer.T)))

            b_0, b_1, b_2 = nb_0, nb_1, nb_2
            w_0, w_1, w_2 = nw_0, nw_1, nw_2

    weight = [w_0, w_1, w_2]
    bias = [b_0, b_1, b_2]






# Train function via fitting each function wihin the function according to the other functions output

