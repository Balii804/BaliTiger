import numpy as np

# Create a synthetic dataset
def create_dataset():
    # Features: [Size (sq ft), Number of Bedrooms, Age (years)]
    data = np.array([
        [1500, 3, 10],
        [1800, 4, 15],
        [2000, 3, 8],
        [2500, 5, 20],
        [3000, 4, 12]
    ])
    # Target: Prices in thousands
    target = np.array([400, 500, 450, 600, 650])
    
    return data, target
# Dummy dataset preprocessing with normalization
def preprocess(data, temp_mean, temp_std):
    # Normalize the data
    data_normalized = (data - temp_mean) / temp_std
    # Add a bias column with ones
    ones = np.ones((data.shape[0], 1))
    return np.hstack([ones, data_normalized])

# Dummy model training using linear regression
def train_model(temp, y):
    # Use the Normal Equation: weights = (temp^T * temp)^-1 * temp^T * y
    temp_transpose = temp.T
    weights = np.linalg.pinv(temp_transpose @ temp) @ temp_transpose @ y
    return weights

# Make predictions using the trained model
def predict(model, temp):
    return temp @ model

# Save the model weights to a file
def save_model_weights(model, filename):
    np.save(filename, model)

# Load the model weights from a file
def load_model_weights(filename):
    return np.load(filename)

# Save the mean and std normalization parameters to files
def save_normalization_params(mean, std, mean_filename, std_filename):
    np.save(mean_filename, mean)
    np.save(std_filename, std)

# Load the mean and std normalization parameters from files
def load_normalization_params(mean_filename, std_filename):
    mean = np.load(mean_filename)
    std = np.load(std_filename)
    return mean, std

if __name__ == "__main__":
    # Create the dataset
    data, target = create_dataset()
    # Compute normalization parameters
    temp_mean = np.mean(data, axis=0)
    temp_std = np.std(data, axis=0)
    # Preprocess the data
    temp = preprocess(data, temp_mean, temp_std)
    
    # Train the model
    model = train_model(temp, target)
    print(f"Model weights: {model}")
    # Save the model weights and normalization parameters to files
    save_model_weights(model, 'model_weights.npy')
    save_normalization_params(temp_mean, temp_std, 'model_mean.npy', 'model_std.npy')
    # Load the model weights and normalization parameters (if needed later)
    model_loaded = load_model_weights('model_weights.npy')
    temp_mean_loaded, temp_std_loaded = load_normalization_params('model_mean.npy', 'model_std.npy')
    print(f"Loaded model weights: {model_loaded}")
    print(f"Loaded normalization parameters - Mean: {temp_mean_loaded}, Std: {temp_std_loaded}")
    # Predict on new data
    new_data = np.array([
        [1600, 3, 12],
        [2200, 4, 10]
    ])
    temp_new = preprocess(new_data, temp_mean_loaded, temp_std_loaded)
    predictions = predict(model_loaded, temp_new)
    print(f"Predictions: {predictions}")
