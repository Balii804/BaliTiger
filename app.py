from flask import Flask, request, jsonify
import numpy as np

app = Flask(__name__)

# Load the trained model weights and normalization parameters
theta = np.load('model_weights.npy')
X_mean = np.load('model_mean.npy')
X_std = np.load('model_std.npy')

# Prediction function
def predict_price(area, basement, garage):
    X = np.array([area, basement, garage])
    X_normalized = (X - X_mean) / X_std
    X_normalized = np.hstack(([1], X_normalized))
    price = X_normalized.dot(theta)
    return price

# Define a prediction API endpoint
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    try:
        area = data['area']
        basement = data['basement']
        garage = data['garage']
    except KeyError:
        return jsonify({'error': 'Invalid input format'}), 400

    price = predict_price(area, basement, garage)
    return jsonify({'predicted_price': price})

@app.route('/')
def home():
    return "Welcome to the Home Page"

if __name__ == '__main__':
    app.run(debug=True)