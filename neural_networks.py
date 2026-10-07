import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    return np.exp(x) / np.exp(x).sum()


class NeuralNetwork:

    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.learning_rate = learning_rate
        self.input = np.zeros(input_nodes)

        self.weights_input_hidden = np.random.randn(input_nodes, hidden_nodes) * np.sqrt(1 / input_nodes)
        self.bias_hidden = np.zeros(hidden_nodes)

        self.hidden = np.zeros(hidden_nodes)

        self.weights_hidden_output = np.random.randn(hidden_nodes, output_nodes) * np.sqrt(1 / hidden_nodes)
        self.bias_output = np.zeros(output_nodes)

        self.output = np.zeros(output_nodes)

    def feed_forward(self):
        self.hidden = sigmoid(
            np.dot(self.input, self.weights_input_hidden) + self.bias_hidden
        )

        self.output = softmax(
            np.dot(self.hidden, self.weights_hidden_output) + self.bias_output
        )

        return self.output

    def loss_val(self):
        delta = 1e-7

        return -np.sum(
            self.target * np.log(self.output + delta)
        )

    def train(self, input, target):
        self.input = input
        self.target = target

        self.feed_forward()

        output_error = self.output - self.target

        dW_output = np.outer(
            self.hidden,
            output_error
        )

        hidden_error = np.dot(
            self.weights_hidden_output,
            output_error
        )

        hidden_delta = (
            hidden_error
            * self.hidden
            * (1 - self.hidden)
        )

        dW_hidden = np.outer(
            self.input,
            hidden_delta
        )

        self.weights_hidden_output -= (
            self.learning_rate * dW_output
        )

        self.weights_input_hidden -= (
            self.learning_rate * dW_hidden
        )

        self.bias_output -= (
            self.learning_rate * output_error
        )

        self.bias_hidden -= (
            self.learning_rate * hidden_delta
        )

        return self.loss_val()

    def predict(self, input):
        z2 = np.dot(
            input,
            self.weights_input_hidden
        ) + self.bias_hidden

        a2 = sigmoid(z2)

        z3 = np.dot(
            a2,
            self.weights_hidden_output
        ) + self.bias_output

        a3 = softmax(z3)

        return np.argmax(a3)

    def accuracy(self, test_data):
        matched_list = []
        not_matched_list = []

        for index in range(len(test_data)):
            label = int(test_data.iloc[index, 0])

            data = (
                test_data.iloc[index, 1:] / 255.0 * 0.99
            ) + 0.01

            predicted_num = self.predict(
                np.array(data, ndmin=2)
            )

            if label == predicted_num:
                matched_list.append(index)
            else:
                not_matched_list.append(index)

        accuracy = 100 * (
            len(matched_list) / len(test_data)
        )

        print("Current Accuracy =", accuracy, "%")