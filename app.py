from flask import Flask, request, jsonify, render_template
import numpy as np

app = Flask(__name__)

# Load the trained model weights and normalization parameters
theta = np.load('model_weights.npy')
X_mean = np.load('model_mean.npy')
X_std = np.load('model_std.npy')

def predict_price(area, basement, garage):
    """Predict house price based on input features."""
    temp = np.array([area, basement, garage])
    temp_normalized = (temp - X_mean) / X_std
    temp_normalized = np.hstack(([1], temp_normalized))
    price = temp_normalized.dot(theta)
    return price

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for predicting house price."""
    data = request.json
    try:
        area = float(data['area'])
        basement = float(data['basement'])
        garage = float(data['garage'])
    except (KeyError, ValueError):
        return jsonify({'error': 'Invalid input format'}), 400

    price = predict_price(area, basement, garage)
    return jsonify({'predicted_price': price})

@app.route('/')
def home():
    """Serve the homepage."""
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
