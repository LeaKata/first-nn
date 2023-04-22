import numpy as np
from tensorflow.keras.datasets import mnist
import copy
import matplotlib.pyplot as plt

data = mnist
(x_train, y_train),(x_test, y_test) = data.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0


class ReLuNN:
    def __init__(self):
        self.alpha = 0.001
        self.beta_1 = 0.9
        self.beta_2 = 0.999
        self.eps = 10 ** (-8)
        self.w_0 = None
        self.w_1 = None
        self.w_2 = None

        self.b_0 = None
        self.b_1 = None
        self.b_2 = None

        self.training_error = None
        self.average_loss = None
        self.validation_error_score = None
        self.validation_accuracy = None

        self.evaluation_error = None
        self.evaluation_error_score = None
        self.evaluation_accuracy = None
        self.classification_matrix = None
        self.confusion_matrix = None

    def b_relu_layer(self, in_layer, w, b):
        return np.maximum(0, b + np.einsum('ijk, ikl -> ijl', w, in_layer))

    def b_softmax_layer(self, in_layer, w, b):
        w_dot_in = np.einsum('ijk, ikl -> ijl', w, in_layer)
        return np.exp(b + w_dot_in) / np.sum(np.exp(b + w_dot_in), axis=1)[:, :, np.newaxis]

    def relu_layer(self, in_layer, w, b):
        return np.maximum(0, b + w.dot(in_layer))

    def softmax_layer(self, in_layer, w, b):
        return np.exp(b + w.dot(in_layer)) / np.sum(np.exp(b + w.dot(in_layer)), axis=0)

    def cross_entropy(self, y_t, y_p):
        return -np.sum(y_t * np.log(y_p), axis=1)

    def training(self, x_input, y_input, validation_fraction, batch_size, repetitions):
        if validation_fraction == 0:
            validation = False
            x_train, y_train = x_input, y_input
            x_valid, y_valid = [], []
        else:
            validation = True
            input_index = np.arange(len(x_input))
            np.random.shuffle(input_index)
            cut = int(len(x_input) * (1 - validation_fraction))
            index_train, index_valid = input_index[:cut], input_index[cut:]
            x_train, y_train = x_input[index_train], y_input[index_train]
            x_valid, y_valid = x_input[index_valid], y_input[index_valid]

        input_size = x_train[0].shape[0] * x_train[0].shape[1]
        y_length = len(y_train)

        # He-et-al Initialisation
        b_0 = np.zeros((16, 1))
        b_1 = np.zeros((16, 1))
        b_2 = np.zeros((10, 1))

        w_0 = np.random.randn(16, input_size) * np.sqrt(2 / input_size)
        w_1 = np.random.randn(16, 16) * np.sqrt(2 / 16)
        w_2 = np.random.randn(10, 16) * np.sqrt(2 / 16)

        # Initialise Moments
        mb_0, mb_0_p, vb_0, vb_0_p = (np.zeros((16, 1)) for i in range(4))
        mb_1, mb_1_p, vb_1, vb_1_p = (np.zeros((16, 1)) for i in range(4))
        mb_2, mb_2_p, vb_2, vb_2_p = (np.zeros((10, 1)) for i in range(4))

        mw_0, mw_0_p, vw_0, vw_0_p = (np.zeros((16, input_size)) for i in range(4))
        mw_1, mw_1_p, vw_1, vw_1_p = (np.zeros((16, 16)) for i in range(4))
        mw_2, mw_2_p, vw_2, vw_2_p = (np.zeros((10, 16)) for i in range(4))

        index_i = np.arange(len(x_train))
        self.training_error = []
        error_sum = 0
        self.average_loss = []
        progress_index = repetitions * (y_length // batch_size) * batch_size
        progress = 0
        t = 0

        for rep in range(repetitions):
            np.random.shuffle(index_i)

            for batch in range(y_length // batch_size):
                il = batch_size * batch
                iu = il + batch_size
                t += 1 * batch_size

                y = np.zeros((batch_size, 10, 1))
                y[np.arange(batch_size), y_train[il:iu]] = 1
                input_layer = x_train[il:iu].reshape(batch_size, input_size, 1)

                bb_0, bb_1, bb_2 = np.tile(b_0, (batch_size, 1, 1)), np.tile(b_1, (batch_size, 1, 1)), np.tile(b_2, (batch_size, 1, 1))
                bw_0, bw_1, bw_2 = np.tile(w_0, (batch_size, 1, 1)), np.tile(w_1, (batch_size, 1, 1)), np.tile(w_2, (batch_size, 1, 1))

                first_layer = self.b_relu_layer(input_layer, bw_0, bb_0)
                second_layer = self.b_relu_layer(first_layer, bw_1, bb_1)
                out_layer = self.b_softmax_layer(second_layer, bw_2, bb_2)

                error = self.cross_entropy(y, out_layer)
                self.training_error.extend(np.squeeze(error, axis=1))
                #error_sum += error
                #self.average_loss.append(error_sum / progress)

                if t / progress_index * 100 % 5 == 0:
                    print('Training %', int(t / progress_index * 100), ', Loss:', np.round(self.training_error[-1], 5))

                # Gradients
                # Account for ReLu derivative < 0
                i1 = (second_layer != 0).astype(int)
                i0 = (first_layer != 0).astype(int)
                e = out_layer - y

                # Wrong transpose at one point reduces accuracy here:

                # Sum over Batch Gradients
                gb_2 = np.sum(e, axis=0)
                gw_2 = np.einsum('ijk, ikj -> jk', e, second_layer)

                pg_10 = np.einsum('ijk, ijl -> ilk', e, bw_2)
                gb_1 = np.sum(i1 * pg_10, axis=0)
                gw_1 = np.sum(i1 * np.einsum('ijk, ikj -> ikj', first_layer, pg_10), axis=0)

                pg_00 = np.einsum('ijk, ilk -> ilj', bw_2, i1 * bw_1)
                pg_01 = np.einsum('ijk, ilj -> ilk', e, pg_00)
                gb_0 = np.sum(i0 * pg_01, axis=0) 
                gw_0 = np.sum(i0 * np.einsum('ijk, ilk -> ilj', input_layer, pg_01), axis=0)

                # Update first Moments with average Gradients
                mb_0 = self.beta_1 * mb_0_p + (1 - self.beta_1) * (gb_0 / batch_size)
                mb_1 = self.beta_1 * mb_1_p + (1 - self.beta_1) * (gb_1 / batch_size)
                mb_2 = self.beta_1 * mb_2_p + (1 - self.beta_1) * (gb_2 / batch_size)

                mw_0 = self.beta_1 * mw_0_p + (1 - self.beta_1) * (gw_0 / batch_size)
                mw_1 = self.beta_1 * mw_1_p + (1 - self.beta_1) * (gw_1 / batch_size)
                mw_2 = self.beta_1 * mw_2_p + (1 - self.beta_1) * (gw_2 / batch_size)

                # Update second Moments with average Gradients
                vb_0 = self.beta_2 * vb_0_p + (1 - self.beta_2) * ((gb_0 / batch_size) ** 2)
                vb_1 = self.beta_2 * vb_1_p + (1 - self.beta_2) * ((gb_1 / batch_size) ** 2)
                vb_2 = self.beta_2 * vb_2_p + (1 - self.beta_2) * ((gb_2 / batch_size) ** 2)

                vw_0 = self.beta_2 * vw_0_p + (1 - self.beta_2) * ((gw_0 / batch_size) ** 2)
                vw_1 = self.beta_2 * vw_1_p + (1 - self.beta_2) * ((gw_1 / batch_size) ** 2)
                vw_2 = self.beta_2 * vw_2_p + (1 - self.beta_2) * ((gw_2 / batch_size) ** 2)

                # Unbiased first Moments
                mb_0_hat = mb_0 / (1 - self.beta_1 ** t)
                mb_1_hat = mb_1 / (1 - self.beta_1 ** t)
                mb_2_hat = mb_2 / (1 - self.beta_1 ** t)

                mw_0_hat = mw_0 / (1 - self.beta_1 ** t)
                mw_1_hat = mw_1 / (1 - self.beta_1 ** t)
                mw_2_hat = mw_2 / (1 - self.beta_1 ** t)

                # Unbiased second Moments
                vb_0_hat = vb_0 / (1 - self.beta_2 ** t)
                vb_1_hat = vb_1 / (1 - self.beta_2 ** t)
                vb_2_hat = vb_2 / (1 - self.beta_2 ** t)

                vw_0_hat = vw_0 / (1 - self.beta_2 ** t)
                vw_1_hat = vw_1 / (1 - self.beta_2 ** t)
                vw_2_hat = vw_2 / (1 - self.beta_2 ** t)

                # Update Parameter
                nb_0 = b_0 - (self.alpha * mb_0_hat) / (np.sqrt(vb_0_hat) + self.eps)
                nb_1 = b_1 - (self.alpha * mb_1_hat) / (np.sqrt(vb_1_hat) + self.eps)
                nb_2 = b_2 - (self.alpha * mb_2_hat) / (np.sqrt(vb_2_hat) + self.eps)

                nw_0 = w_0 - (self.alpha * mw_0_hat) / (np.sqrt(vw_0_hat) + self.eps)
                nw_1 = w_1 - (self.alpha * mw_1_hat) / (np.sqrt(vw_1_hat) + self.eps)
                nw_2 = w_2 - (self.alpha * mw_2_hat) / (np.sqrt(vw_2_hat) + self.eps)

                b_0, b_1, b_2 = nb_0, nb_1, nb_2
                w_0, w_1, w_2 = nw_0, nw_1, nw_2

                # Update t-1 Moments
                mb_0_p, mb_1_p, mb_2_p = mb_0, mb_1, mb_2
                mw_0_p, mw_1_p, mw_2_p = mw_0, mw_1, mw_2
                vb_0_p, vb_1_p, vb_2_p = vb_0, vb_1, vb_2
                vw_0_p, vw_1_p, vw_2_p = vw_0, vw_1, vw_2

        if validation:
            validation_error_score = 0
            progress_index = len(y_valid)
            progress = 0

            for i in range(len(y_valid)):
                progress += 1
                y = np.zeros((10, 1))
                y[y_valid[i]] = 1
                input_layer = np.expand_dims(np.array(x_valid[i]).flatten(), axis=1)

                first_layer = self.relu_layer(input_layer, w_0, b_0)
                second_layer = self.relu_layer(first_layer, w_1, b_1)
                out_layer = self.softmax_layer(second_layer, w_2, b_2)

                y_true, y_pred = np.argmax(y), np.argmax(out_layer)

                if y_true != y_pred:
                    validation_error_score += 1

                self.validation_accuracy = (1 - validation_error_score / progress)

                if progress / progress_index * 100 % 10 == 0:
                    print('Validation %', int(progress / progress_index * 100), ', Accuracy: ', np.round(self.validation_accuracy, 5))

            print('Validation Accuracy %', np.round(self.validation_accuracy, 5))

        self.b_0, self.b_1, self.b_2 = b_0, b_1, b_2
        self.w_0, self.w_1, self.w_2 = w_0, w_1, w_2

    def evaluation(self, x_test, y_test):
        self.evaluation_error = []
        evaluation_error_score = 0
        self.classification_matrix = np.zeros((10, 10))

        b_0, b_1, b_2 = self.b_0, self.b_1, self.b_2
        w_0, w_1, w_2 = self.w_0, self.w_1, self.w_2

        progress_index = len(y_test)
        progress = 0

        for i in range(len(y_test)):
                progress += 1
                y = np.zeros((10, 1))
                y[y_test[i]] = 1
                input_layer = np.expand_dims(np.array(x_test[i]).flatten(), axis=1)

                first_layer = self.relu_layer(input_layer, w_0, b_0)
                second_layer = self.relu_layer(first_layer, w_1, b_1)
                out_layer = self.softmax_layer(second_layer, w_2, b_2)

                error = self.cross_entropy(y, out_layer)
                self.evaluation_error.append(error)
                y_true, y_pred = np.argmax(y), np.argmax(out_layer)

                if y_true != y_pred:
                    evaluation_error_score += 1

                self.evaluation_accuracy = (1 - evaluation_error_score / progress)
                self.classification_matrix[y_true, y_pred] += 1

                if progress / progress_index * 100 % 10 == 0:
                    print('Evaluation %', int(progress / progress_index * 100), ', Accuracy: ', np.round(self.evaluation_accuracy, 5))

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


test = ReLuNN()
test.training(x_train, y_train, 0.2, 5, 1)
# Batch size of 5 optimal so far (keep testing)

# generate general layer function with inputs 'batch_size, layer_size, prev_layer_size'


