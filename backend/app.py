from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os
import numpy as np
import pickle

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from neural_networks import NeuralNetwork


app = Flask(__name__)
CORS(app)


input_nodes = 784
hidden_nodes = 128
output_nodes = 10
learning_rate = 0.0003


nn = NeuralNetwork(
    input_nodes,
    hidden_nodes,
    output_nodes,
    learning_rate
)


model_path = os.path.join(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    ),
    "model.pkl"
)


with open(model_path, "rb") as f:
    model = pickle.load(f)


nn.weights_input_hidden = model["weights_input_hidden"]
nn.bias_hidden = model["bias_hidden"]
nn.weights_hidden_output = model["weights_hidden_output"]
nn.bias_output = model["bias_output"]


print("Trained model loaded successfully!")


@app.route("/")
def home():
    return "Neural Network API is running!"


@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    pixels = np.array(
        data["pixels"],
        dtype=float
    )

    pixels = (
        pixels / 255.0 * 0.99
    ) + 0.01


    # Give the current drawing to the neural network

    nn.input = pixels

    output = nn.feed_forward()


    prediction = int(
        np.argmax(output)
    )


    probabilities = output.tolist()


    return jsonify({
        "prediction": prediction,
        "probabilities": probabilities
    })


if __name__ == "__main__":
    app.run(debug=True)