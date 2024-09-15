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
def preprocess(data, X_mean, X_std):
    # Normalize the data
    data_normalized = (data - X_mean) / X_std
    # Add a bias column with ones
    ones = np.ones((data.shape[0], 1))
    return np.hstack([ones, data_normalized])

# Dummy model training using linear regression
def train_model(X, y):
    # Use the Normal Equation: weights = (X^T * X)^-1 * X^T * y
    X_transpose = X.T
    weights = np.linalg.pinv(X_transpose @ X) @ X_transpose @ y
    return weights

# Make predictions using the trained model
def predict(model, X):
    return X @ model

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
    X_mean = np.mean(data, axis=0)
    X_std = np.std(data, axis=0)
    
    # Preprocess the data
    X = preprocess(data, X_mean, X_std)
    
    # Train the model
    model = train_model(X, target)
    print(f"Model weights: {model}")
    
    # Save the model weights and normalization parameters to files
    save_model_weights(model, 'model_weights.npy')
    save_normalization_params(X_mean, X_std, 'model_mean.npy', 'model_std.npy')
    
    # Load the model weights and normalization parameters (if needed later)
    model_loaded = load_model_weights('model_weights.npy')
    X_mean_loaded, X_std_loaded = load_normalization_params('model_mean.npy', 'model_std.npy')
    print(f"Loaded model weights: {model_loaded}")
    print(f"Loaded normalization parameters - Mean: {X_mean_loaded}, Std: {X_std_loaded}")
    
    # Predict on new data
    new_data = np.array([
        [1600, 3, 12],
        [2200, 4, 10]
    ])
    X_new = preprocess(new_data, X_mean_loaded, X_std_loaded)
    predictions = predict(model_loaded, X_new)
    
    print(f"Predictions: {predictions}")