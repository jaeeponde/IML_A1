import numpy as np
import pandas as pd
import pickle
import argparse
from data_preprocessing import preprocess





# now we do almost the same things as the prediction model
def add_bias_term(X):
    return np.column_stack([np.ones(X.shape[0]), X])


def load_model(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def polynomial_features(X, degree):
    poly_X = X.copy()
    for deg in range(2, degree + 1):
        poly_X = np.column_stack([poly_X, X ** deg])
    return poly_X


def predict(X, weights):
    return np.dot(X, weights)


def compute_metrics(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    r_squared = 1 - (ss_residual / ss_total)
    return mse, rmse, r_squared


#in this function we load and process data 
def main(model_path, data_path, metrics_output_path, predictions_output_path):
    
    data=preprocess(data_path)
    
    
    X = data.iloc[:, :-1].values
    y_true = data.iloc[:, -1].values

    
    model = load_model(model_path)
    weights = model['weights']
    degree = model['degree']

    
    X_poly = polynomial_features(X, degree)
    X_bias = add_bias_term(X_poly)

    
    y_pred = predict(X_bias, weights)

    
    mse, rmse, r_squared = compute_metrics(y_true, y_pred)


    np.savetxt(predictions_output_path, y_pred, delimiter=',', fmt='%f')

   
    with open(metrics_output_path, 'w') as f:
        f.write(f"Regression Metrics:\n")
        f.write(f"Mean Squared Error (MSE): {mse:.4f}\n")
        f.write(f"Root Mean Squared Error (RMSE): {rmse:.4f}\n")
        f.write(f"R-squared (R²) Score: {r_squared:.4f}\n")


#this takes in the arguments from the bash command 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict using a trained regression model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the saved model file')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the data CSV file that includes features and true labels')
    parser.add_argument('--metrics_output_path', type=str, required=True, help='Path where the evaluation metrics will be saved')
    parser.add_argument('--predictions_output_path', type=str, required=True, help='Path where the predictions will be saved')
    
    args = parser.parse_args()

    main(args.model_path, args.data_path, args.metrics_output_path, args.predictions_output_path)


