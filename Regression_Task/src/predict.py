import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import mean_squared_error

# Function to apply polynomial transformation
def apply_polynomial_features(X, degree):
    X_poly = np.ones((X.shape[0], 1))  # Start with bias (intercept term)
    for d in range(1, degree + 1):
        X_poly = np.hstack((X_poly, np.power(X, d)))  # Add polynomial terms
    return X_poly

# Load your dataset (assuming it's a pandas DataFrame)
df_normalized = pd.read_csv('/Users/jaeeponde/Jaee_Ponde_A1/new_data_train.csv')  # Update with your actual file path

# Separate the features and target (assuming 'FUEL CONSUMPTION' is the target variable)
X = df_normalized.drop(columns=['FUEL CONSUMPTION']).values  # Convert features to NumPy array
y = df_normalized['FUEL CONSUMPTION'].values  # Convert target to NumPy array

# Load the saved model from the pickle file
with open('/Users/jaeeponde/Jaee_Ponde_A1/Regression Task/Regression_Task/models/regression_model_final.pkl', 'rb') as model_file:
    model_info = pickle.load(model_file)

# Extract the weights and degree from the loaded model
weights = model_info['weights']
degree = model_info['degree']

# Apply polynomial transformation to the features
X_poly = apply_polynomial_features(X, degree)

# Make predictions using the learned weights
y_pred = np.dot(X_poly, weights)

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y, y_pred)
print(f"Mean Squared Error: {mse}")

# Output predicted vs actual values
results = pd.DataFrame({'Actual': y, 'Predicted': y_pred})
print(results)

# Save the results to a CSV file
results.to_csv('predicted_vs_actual.csv', index=False)

