import numpy as np
from tensorflow.python.keras.datasets import mnist
import copy
import matplotlib.pyplot as plt

data = mnist
(x_train, y_train),(x_test, y_test) = data.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0


class ReLuNN:
    def __init__(self, learning_rate):
        self.alpha = learning_rate
        self.w_0 = None
        self.w_1 = None
        self.w_2 = None

        self.b_0 = None
        self.b_1 = None
        self.b_2 = None

        self.training_error = None
        self.training_score = None
        self.training_accuracy_score = None

        self.evaluation_error = None
        self.evaluation_score = None
        self.evaluation_accuracy_score = None
        self.classification_matrix = None
        self.confusion_matrix = None

    def relu_layer(self, in_layer, w, b):
        n = np.maximum(0, b + w.dot(in_layer))
        return n

    def softmax_layer(self, in_layer, w, b):
        n = np.exp(b + w.dot(in_layer)) / np.sum(np.exp(b + w.dot(in_layer)), axis=0)
        return n

    def cross_entropy(self, y_t, y_p):
        return -np.sum(y_t * np.log(y_p))

    def training(self, x_train, y_train, repetitions):
        input_size = x_train[0].shape[0] * x_train[0].shape[1]
        y_length = len(y_train)

        # He-et-al Initialization
        w_0 = np.random.randn(16, input_size) * np.sqrt(2 / input_size)
        w_1 = np.random.randn(16, 16) * np.sqrt(2 / 16)
        w_2 = np.random.randn(10, 16) * np.sqrt(2 / 16)

        b_0 = np.zeros((16, 1))
        b_1 = np.zeros((16, 1))
        b_2 = np.zeros((10, 1))

        index_i = np.arange(len(x_train))
        self.training_error = []
        self.training_score = []
        self.training_accuracy_score = []
        count = 0

        for rep in range(repetitions):
            np.random.shuffle(index_i)

            for i in range(y_length):
                y = np.zeros((10, 1))
                y[y_train[i]] = 1
                input_layer = np.expand_dims(np.array(x_train[i]).flatten(), axis=1)

                first_layer = self.relu_layer(input_layer, w_0, b_0)
                second_layer = self.relu_layer(first_layer, w_1, b_1)
                out_layer = self.softmax_layer(second_layer, w_2, b_2)

                error = self.cross_entropy(y, out_layer)
                self.training_error.append(error)

                if np.argmax(y) == np.argmax(out_layer):
                    self.training_score.append(0)
                else:
                    self.training_score.append(1)
                self.training_accuracy_score.append(sum(self.training_score) / len(self.training_score))

                count += 1
                if count / (repetitions * y_length) * 100 % 2 == 0:
                    print('%', int(count / (repetitions * y_length) * 100))

                nb_2 = b_2 - self.alpha * (out_layer - y)
                nw_2 = w_2 - self.alpha * (out_layer - y).dot(second_layer.T)

                nb_1 = b_1 - self.alpha * (out_layer - y).T.dot(w_2).T
                nw_1 = w_1 - self.alpha * ((first_layer.dot((out_layer - y).T)).dot(w_2)).T

                nb_0 = b_0 - self.alpha * (out_layer - y).T.dot(w_2.dot(w_1)).T
                nw_0 = w_0 - self.alpha * ((input_layer.dot((out_layer - y).T)).dot(w_2.dot(w_1))).T

                b_0, b_1, b_2 = nb_0, nb_1, nb_2
                w_0, w_1, w_2 = nw_0, nw_1, nw_2

        self.b_0, self.b_1, self.b_2 = b_0, b_1, b_2
        self.w_0, self.w_1, self.w_2 = w_0, w_1, w_2

    def evaluation(self, x_test, y_test):
        self.evaluation_error = []
        self.evaluation_score = []
        self.evaluation_accuracy_score = []
        self.classification_matrix = np.zeros((10, 10))

        b_0, b_1, b_2 = self.b_0, self.b_1, self.b_2
        w_0, w_1, w_2 = self.w_0, self.w_1, self.w_2

        for i in range(len(y_test)):
                y = np.zeros((10, 1))
                y[y_test[i]] = 1
                input_layer = np.expand_dims(np.array(x_test[i]).flatten(), axis=1)

                first_layer = self.relu_layer(input_layer, w_0, b_0)
                second_layer = self.relu_layer(first_layer, w_1, b_1)
                out_layer = self.softmax_layer(second_layer, w_2, b_2)

                error = self.cross_entropy(y, out_layer)
                self.evaluation_error.append(error)
                y_true, y_pred = np.argmax(y), np.argmax(out_layer)

                if y_true == y_pred:
                    self.evaluation_score.append(0)
                else:
                    self.evaluation_score.append(1)

                self.evaluation_accuracy_score.append(sum(self.evaluation_score) / len(self.evaluation_score))
                self.classification_matrix[y_true, y_pred] += 1

        am_sum = np.sum(self.classification_matrix, axis=1)
        self.confusion_matrix = self.classification_matrix / am_sum[:, None]

    def confusion(self):
        fig, ax = plt.subplots()
        im = ax.imshow(self.confusion_matrix, cmap='Blues')

        # Show all ticks
        ax.set_xticks(np.arange(10))
        ax.set_yticks(np.arange(10))
        ax.xaxis.tick_top()

        # Loop over data dimensions
        for i in range(10):
            for j in range(10):
                if i == j:
                    text = ax.text(j, i, np.around(self.confusion_matrix, decimals=3)[i, j], ha="center", va="center", color="w")
                else:
                    text = ax.text(j, i, np.around(self.confusion_matrix, decimals=3)[i, j], ha="center", va="center", color="k")

        plt.show()

    def classify(self, input_image):
        input_layer = np.expand_dims(np.array(input_image).flatten(), axis=1)
        first_layer = self.relu_layer(input_layer, self.w_0, self.b_0)
        second_layer = self.relu_layer(first_layer, self.w_1, self.b_1)
        out_layer = self.softmax_layer(second_layer, self.w_2, self.b_2)

        print(np.argmax(out_layer))


test = ReLuNN(0.001)
test.training(x_train, y_train, 2)
