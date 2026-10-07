import numpy as np
import pandas as pd
import pickle

from neural_networks import NeuralNetwork


mnist_train = pd.read_csv("data/mnist_train.csv").to_numpy()
mnist_test = pd.read_csv("data/mnist_test.csv").to_numpy()

train_labels = mnist_train[:, 0].astype(int)
train_data = (mnist_train[:, 1:] / 255.0 * 0.99) + 0.01

test_labels = mnist_test[:, 0].astype(int)
test_data = (mnist_test[:, 1:] / 255.0 * 0.99) + 0.01


input_nodes = 784
hidden_nodes = 128
output_nodes = 10
learning_rate = 0.0003
epochs = 12

nn = NeuralNetwork(
    input_nodes,
    hidden_nodes,
    output_nodes,
    learning_rate
)


for epoch in range(epochs):

    for step in range(len(train_data)):

        target_data = np.zeros(output_nodes) + 0.01
        target_data[train_labels[step]] = 0.99

        input_data = train_data[step]

        nn.train(input_data, target_data)

        if step % 500 == 0:
            print(
                f"step = {step}\t"
                f"loss = {nn.loss_val():.3f}\t"
                f"epoch = {epoch + 1}"
            )

    print(f"Epoch {epoch + 1} completed.")


print("\nTraining accuracy:")

train_correct = 0

for i in range(len(train_data)):
    prediction = nn.predict(train_data[i])

    if prediction == train_labels[i]:
        train_correct += 1

print(
    "Current Accuracy =",
    100 * train_correct / len(train_data),
    "%"
)


print("\nTest accuracy:")

test_correct = 0

for i in range(len(test_data)):
    prediction = nn.predict(test_data[i])

    if prediction == test_labels[i]:
        test_correct += 1

print(
    "Current Accuracy =",
    100 * test_correct / len(test_data),
    "%"
)

with open("model.pkl", "wb") as f:
    pickle.dump({
        "weights_input_hidden": nn.weights_input_hidden,
        "bias_hidden": nn.bias_hidden,
        "weights_hidden_output": nn.weights_hidden_output,
        "bias_output": nn.bias_output
    }, f)

print("\nModel saved successfully!")