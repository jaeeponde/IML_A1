import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

# Load and shuffle the dataset
df = pd.read_csv("/Users/jaeeponde/IML_A1/IML_A1/trail_training_data.csv")
df_shuffled = df.sample(frac=1).reset_index(drop=True)

# Extract features and target
X = df_shuffled.drop(columns='FUEL CONSUMPTION').values
y = df_shuffled['FUEL CONSUMPTION'].values

# Split data into training and cross-validation sets (80% train, 20% CV)
X_train, X_cv, y_train, y_cv = train_test_split(X, y, test_size=0.2, random_state=42)

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
    return np.full(n_features, 3)

# Step 4: Hypothesis Function (for predictions)
def hypothesis(X, weights):
    return np.dot(X, weights)

# Step 5: Mean Squared Error (MSE) with Regularization (L2)
def mse_loss(y_true, y_pred, weights, lambda_):
    regularization_term = lambda_ * np.sum(weights[1:] ** 2)  # Regularize all weights except the bias term
    return np.mean((y_true - y_pred) ** 2) + regularization_term

# Step 6: Root Mean Squared Error (RMSE)
def rmse_loss(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

# Step 7: R-squared (R²)
def r_squared(y_true, y_pred):
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    return 1 - (ss_residual / ss_total)

# Step 8: Gradient Descent with Regularization and Automatic Convergence
def gradient_descent(X, y, weights, learning_rate, lambda_, tolerance=1e-6):
    m = X.shape[0]
    prev_loss = float('inf')
    epoch = 0
    while True:
        y_pred = hypothesis(X, weights)
        gradients = (1/m) * np.dot(X.T, (y_pred - y)) + lambda_ * np.r_[0, weights[1:]]  # Apply L2 regularization (skip bias term)
        weights = weights - learning_rate * gradients

        # Compute the current loss with regularization
        loss = mse_loss(y, y_pred, weights, lambda_)
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

X_cv_poly = polynomial_features(X_cv, degree)
X_cv_bias = add_bias_term(X_cv_poly)

n_features = X_train_bias.shape[1]
weights = initialize_weights(n_features)

learning_rate = 0.3
lambda_ = 0.1  # Regularization strength
tolerance = 1e-6  # Convergence threshold

# Train the model using gradient descent with regularization and convergence check
trained_weights, n_epochs = gradient_descent(X_train_bias, y_train, weights, learning_rate, lambda_, tolerance)

# Step 10: Predictions on Training and Cross-Validation Data
y_train_pred = hypothesis(X_train_bias, trained_weights)
y_cv_pred = hypothesis(X_cv_bias, trained_weights)

# Step 11: Calculate Metrics for Training Data
train_mse = mse_loss(y_train, y_train_pred, trained_weights, lambda_)
train_rmse = rmse_loss(y_train, y_train_pred)
train_r2 = r_squared(y_train, y_train_pred)

# Step 12: Calculate Metrics for Cross-Validation Data
cv_mse = mse_loss(y_cv, y_cv_pred, trained_weights, lambda_)
cv_rmse = rmse_loss(y_cv, y_cv_pred)
cv_r2 = r_squared(y_cv, y_cv_pred)

# Output training and cross-validation metrics
print(f"Training MSE: {train_mse:.4f}")
print(f"Training RMSE: {train_rmse:.4f}")
print(f"Training R²: {train_r2:.4f}")

print(f"CV MSE: {cv_mse:.4f}")
print(f"CV RMSE: {cv_rmse:.4f}")
print(f"CV R²: {cv_r2:.4f}")

# Step 13: Plot Actual vs Predicted for Training and CV sets
def plot_actual_vs_predicted(y_true, y_pred, dataset_type):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, color='blue', edgecolors='k', alpha=0.7, s=100)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color='red', linestyle='--', linewidth=2)
    plt.title(f'Actual vs Predicted ({dataset_type})')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.grid(True)
    plt.show()

# Plot for training data
plot_actual_vs_predicted(y_train, y_train_pred, 'Training Set')

# Plot for cross-validation data
plot_actual_vs_predicted(y_cv, y_cv_pred, 'Cross-Validation Set')

# Save predictions and metrics to files
def create_and_overwrite_predictions_csv(actual, predicted, filename):
    new_df = pd.DataFrame({
        'Actual': actual,
        'Predicted': predicted
    })
    new_df.to_csv(filename, index=False)
    print(f"New predictions saved and file overwritten: {filename}")

# Save predictions for training and cross-validation sets
create_and_overwrite_predictions_csv(y_train, y_train_pred, "/Users/jaeeponde/IML_A1/IML_A1/Regression_Task/results/train_predictions.csv")
create_and_overwrite_predictions_csv(y_cv, y_cv_pred, "/Users/jaeeponde/IML_A1/IML_A1/Regression_Task/results/cv_predictions.csv")
