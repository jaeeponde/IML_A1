import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Extract features and target
df=pd.read_csv("/Users/jaeeponde/IML_A1/IML_A1/trail_training_data.csv")
# Shuffle the data and reset the index
df_shuffled = df.sample(frac=1).reset_index(drop=True)

train_data = df_shuffled

# Extract features and target
X_train = train_data.drop(columns='FUEL CONSUMPTION').values
y_train = train_data['FUEL CONSUMPTION'].values


# Step 1: Add Bias Term
def add_bias_term(X):
    return np.column_stack([np.ones(X.shape[0]), X])

# Step 2: Polynomial Feature Transformation
def polynomial_features(X, degree):
    poly_X = X.copy()
    for deg in range(2, degree + 1):
        poly_X = np.column_stack([poly_X, X ** deg])
    return poly_X

# Step 3: Initialize Weights
def initialize_weights(n_features):
    return np.full(n_features, 1)

# Step 4: Hypothesis Function (for predictions)
def hypothesis(X, weights):
    return np.dot(X, weights)

# Step 5: Mean Squared Error (MSE) with Regularization (L2)
def mse_loss(y_true, y_pred, weights):
    return np.mean((y_true - y_pred) ** 2) 

# Step 6: Root Mean Squared Error (RMSE)
def rmse_loss(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

# Step 7: R-squared (R²)
def r_squared(y_true, y_pred):
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    return 1 - (ss_residual / ss_total)

# Step 8: Gradient Descent with Regularization and Automatic Convergence
def gradient_descent(X, y, weights, learning_rate, tolerance=1e-6):
    m = X.shape[0]
    prev_loss = float('inf')
    epoch = 0
    while True:
        y_pred = hypothesis(X, weights)
        gradients = (1/m) * np.dot(X.T, (y_pred - y))   # Apply L2 regularization (skip bias term)
        weights = weights - learning_rate * gradients

        # Compute the current loss with regularization
        loss = mse_loss(y, y_pred, weights)
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: MSE = {loss:.4f}")
        
        # Check for convergence
        if abs(prev_loss - loss) < tolerance:
            print(f"Converged after {epoch} epochs with a tolerance of {tolerance}")
            break
        
        prev_loss = loss
        epoch += 1

    return weights, epoch

# Step 9: Train the Model
degree = 3  # Degree of the polynomial
X_train_poly = polynomial_features(X_train, degree)
X_train_bias = add_bias_term(X_train_poly)

n_features = X_train_bias.shape[1]
weights = initialize_weights(n_features)

learning_rate = 0.39
tolerance = 1e-6  # Convergence threshold

# Train the model using gradient descent with regularization and convergence check
trained_weights, n_epochs = gradient_descent(X_train_bias, y_train, weights, learning_rate, tolerance)

# Step 10: Predictions on Training and Cross-Validation Data
y_train_pred = hypothesis(X_train_bias, trained_weights)

# Step 11: Calculate Metrics for Training Data
train_mse = mse_loss(y_train, y_train_pred, trained_weights)
train_rmse = rmse_loss(y_train, y_train_pred)
train_r2 = r_squared(y_train, y_train_pred)


# Output training metrics to file
with open('/Users/jaeeponde/IML_A1/IML_A1/Regression_Task/results/train_metrics.txt', 'w') as f:
    f.write(f"Training MSE: {train_mse:.4f}\n")
    f.write(f"Training RMSE: {train_rmse:.4f}\n")
    f.write(f"Training R²: {train_r2:.4f}\n")


def create_and_overwrite_predictions_csv(actual, predicted, filename):
    # Create a new DataFrame with 'Actual' and 'Predicted' columns
    new_df = pd.DataFrame({
        'Actual': actual,
        'Predicted': predicted
    })
    
    # Overwrite the existing CSV file with the new DataFrame
    new_df.to_csv(filename, index=False)
    print(f"New predictions saved and file overwritten: {filename}")



# Call the function to overwrite the CSV file
create_and_overwrite_predictions_csv(y_train, y_train_pred, "/Users/jaeeponde/IML_A1/IML_A1/Regression_Task/results/train_predictions.csv")