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


def layer_0(input_0, w_0, b_0):
    n_0 = np.maximum(0.00, b_0 + w_0.dot(input_0))
    return n_0


def layer_1(input_1, w_1, b_1):
    n_1 = np.maximum(0.00, b_1 + w_1.dot(input_1))
    return n_1


def output(input_2, w_2, b_2):
    n_2 = np.maximum(0.00, b_2 + w_2.dot(input_2))
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
    # He-et-al Initialization
    w_0 = np.random.randn(16, input_size) * np.sqrt(2 / input_size)
    w_1 = np.random.randn(16, 16) * np.sqrt(2 / 16)
    w_2 = np.random.randn(10, 16) * np.sqrt(2 / 16)

    b_0 = np.zeros((16, 1))
    b_1 = np.zeros((16, 1))
    b_2 = np.zeros((10, 1))

    alpha = 0.001
    index_i = np.arange(len(x_train))
    e_measure = []
    test = []
    test_score = []

    for iterations in range(10):
        np.random.shuffle(index_i)
        print('Iterations: ', iterations)

        for i in range(len(y_train)):
            y = np.zeros((10, 1))
            y[y_train[i]] = 1
            input_layer = np.expand_dims(np.array(x_train[i]).flatten(), axis=1)

            first_layer = layer_0(input_layer, w_0, b_0)
            second_layer = layer_1(first_layer, w_1, b_1)
            out_layer = output(second_layer, w_2, b_2)

            error = (out_layer - y)
            se = error ** 2
            e_measure.append(np.sum(se) / 10)
            if np.argmax(y) == np.argmax(out_layer):
                test.append(0)
            else:
                test.append(1)
            test_score.append(sum(test) / len(test))

            # print(out_layer.T)
            # print(iterations, i, np.sum(se) / 10)
            # print(np.argmax(y), np.argmax(out_layer))

            if i % 1000 == 0:
                print(iterations + 1, i)

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

def test(x_test, y_test, weight, bias):
    w_0, w_1, w_2 = weight[0], weight[1], weight[2]
    b_0, b_1, b_2 = bias[0], bias[1], bias[2]

    e_measure = []
    test = []
    test_score = []
    accuracy_matrix = np.zeros((10, 10))

    for i in range(len(y_test)):
            y = np.zeros((10, 1))
            y[y_test[i]] = 1
            input_layer = np.expand_dims(np.array(x_test[i]).flatten(), axis=1)

            first_layer = layer_0(input_layer, w_0, b_0)
            second_layer = layer_1(first_layer, w_1, b_1)
            out_layer = output(second_layer, w_2, b_2)

            error = (out_layer - y)
            se = error ** 2
            e_measure.append(np.sum(se) / 10)
            y_true, y_pred = np.argmax(y), np.argmax(out_layer)

            if y_true == y_pred:
                test.append(0)
            else:
                test.append(1)
            test_score.append(sum(test) / len(test))

            accuracy_matrix[y_true, y_pred] += 1

    am_sum = np.sum(accuracy_matrix, axis=1)
    accuracy = accuracy_matrix / am_sum[:, None]

    return accuracy_matrix / am_sum[:, None], sum(test) / len(test)

np.around(accuracy, decimals=3)

fig, ax = plt.subplots()
im = ax.imshow(accuracy, cmap='Blues')

# Show all ticks
ax.set_xticks(np.arange(10))
ax.set_yticks(np.arange(10))
ax.xaxis.tick_top()

# Loop over data dimensions
for i in range(10):
    for j in range(10):
        if i == j:
            text = ax.text(j, i, np.around(accuracy, decimals=3)[i, j], ha="center", va="center", color="w")
        else:
            text = ax.text(j, i, np.around(accuracy, decimals=3)[i, j], ha="center", va="center", color="k")

plt.show()





# Train function via fitting each function wihin the function according to the other functions output

