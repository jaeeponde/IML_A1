import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

df=pd.read_csv("training_data.csv")

#using shuffling to make sure the samples are well distributes
df_shuffled = df.sample(frac=1).reset_index(drop=True)

train_data = df_shuffled

#dividing into train and test 
X_train = train_data.drop(columns='FUEL CONSUMPTION').values
y_train = train_data['FUEL CONSUMPTION'].values


#adding a bias term 
def add_bias_term(X):
    return np.column_stack([np.ones(X.shape[0]), X])

#curve firring 
def polynomial_features(X, degree):
    poly_X = X.copy()
    for deg in range(2, degree + 1):
        poly_X = np.column_stack([poly_X, X ** deg])
    return poly_X

# initialising all weights to 1
def initialize_weights(n_features):
    return np.full(n_features, 1)

# function for predictions
def hypothesis(X, weights):
    return np.dot(X, weights)

# calculating MSE
def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# calculating RMSE
def rmse_loss(y_true, y_pred):
    return np.sqrt(mse_loss(y_true, y_pred))

# calculating R-squared (R²)
def r_squared(y_true, y_pred):
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    return 1 - (ss_residual / ss_total)

# gradient descent 
def gradient_descent(X, y, weights, learning_rate, n_iterations):
    m = X.shape[0]
    for i in range(n_iterations):
        y_pred = hypothesis(X, weights)
        gradients = (1/m) * np.dot(X.T, (y_pred - y))
        weights = weights - learning_rate * gradients
        
        if i % 1000 == 0:
            loss = mse_loss(y, y_pred)
            print(f"Iteration {i}: MSE = {loss:.4f}")
    return weights

# training the model
degree = 3
X_train_poly = polynomial_features(X_train, degree)
X_train_bias = add_bias_term(X_train_poly)

n_features = X_train_bias.shape[1]
weights = initialize_weights(n_features)

learning_rate = 0.39
n_iterations = 20000


trained_weights = gradient_descent(X_train_bias, y_train, weights, learning_rate, n_iterations)

y_train_pred = hypothesis(X_train_bias, trained_weights)


#y_test_pred = hypothesis(X_test_bias, trained_weights)

train_mse = mse_loss(y_train, y_train_pred)
train_rmse = rmse_loss(y_train, y_train_pred)
train_r2 = r_squared(y_train, y_train_pred)

# output training metrics to file
with open('/Users/jaeeponde/Jaee_Ponde_A1/Regression Task/Regression_Task/results/train_metrics.txt', 'w') as f:
    f.write(f"Regression Metrics\n")
    f.write(f"Mean Squared Error (MSE): {train_mse:.4f}\n")
    f.write(f"Root Mean Squared Error (RMSE): {train_rmse:.4f}\n")
    f.write(f"R-squared (R²) Score {train_r2:.4f}\n")


def create_and_overwrite_predictions_csv(actual, predicted, filename):
    # create a new DataFrame with 'Actual' and 'Predicted' columns
    new_df = pd.DataFrame({
        'Actual': actual,
        'Predicted': predicted
    })
    
    # overwrite the existing CSV file with the new DataFrame
    new_df.to_csv(filename, index=False)
    print(f"New predictions saved and file overwritten: {filename}")



# call the function to overwrite the CSV file
create_and_overwrite_predictions_csv(y_train, y_train_pred, "/Users/jaeeponde/Jaee_Ponde_A1/Regression Task/Regression_Task/results/train_predictions.csv")

import pickle

#create information dictionary for pickled module 
model_info = {
    'weights': trained_weights,
    'degree': degree  
}

model_path = '/Users/jaeeponde/Jaee_Ponde_A1/Regression Task/Regression_Task/models/regression_model2.pkl'

# save the model to a pickle file
with open(model_path, 'wb') as file:
    pickle.dump(model_info, file)

print(f"Model saved to {model_path}")
