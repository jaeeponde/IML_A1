import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import mean_squared_error

df = pd.read_csv('/Users/jaeeponde/Jaee_Ponde_A1/synthetic_vehicle_data.csv')

# Drop the 'Year' column (if it exists)
if 'Year' in df.columns:
    df = df.drop(columns=['Year'])

# One-hot encode the 'FUEL' column if it exists

    one_hot_encoded_df = pd.get_dummies(df['FUEL'], prefix='FUEL')
    df = pd.concat([df, one_hot_encoded_df], axis=1)
    df = df.drop(columns=[col for col in ['FUEL', 'FUEL_D', 'FUEL_E', 'FUEL_N'] if col in df.columns])

# One-hot encode the 'TRANSMISSION' column if it exists
if 'TRANSMISSION' in df.columns:
    one_hot_encoded_df = pd.get_dummies(df['TRANSMISSION'], prefix='TRANSMISSION')
    df = pd.concat([df, one_hot_encoded_df], axis=1)
    df = df.drop(columns=[col for col in ['TRANSMISSION', 'TRANSMISSION_M6', 'TRANSMISSION_AS4', 'TRANSMISSION_A3', 'TRANSMISSION_AS5'] if col in df.columns])

required_columns = ['FUEL_X', 'FUEL_Z', 'TRANSMISSION_A4','TRANSMISSION_A5','TRANSMISSION_M5']
for col in required_columns:
    if col not in df.columns:
        df[col] = 0

# Drop the 'MODEL' column (if it exists)
if 'MODEL' in df.columns:
    df = df.drop(columns=['MODEL'])

# Map the 'VEHICLE CLASS' column if it exists
vehicle_class_mapping = {
    'COMPACT': 3.08, 'FULL-SIZE': 3.61, 'MID-SIZE': 3.42, 'MINICOMPACT': 3.67,
    'MINIVAN': 3.97, 'PICKUP TRUCK - SMALL': 3.53, 'PICKUP TRUCK - STANDARD': 4.34,
    'STATION WAGON - MID-SIZE': 3.31, 'STATION WAGON - SMALL': 2.99, 'SUBCOMPACT': 3.02,
    'SUV': 4.00, 'TWO-SEATER': 4.23, 'VAN - CARGO': 4.50, 'VAN - PASSENGER': 4.83
}

def map_vehicle_class(vehicle_class):
    return vehicle_class_mapping.get(vehicle_class, None)

if 'VEHICLE CLASS' in df.columns:
    df['VEHICLE CLASS'] = df['VEHICLE CLASS'].apply(map_vehicle_class)

# Map the 'MAKE' column if it exists
vehicle_brand_mapping = {
    'SUZUKI': 2.75, 'SATURN': 2.75, 'HONDA': 2.75, 'VOLKSWAGEN': 3.00, 'DAEWOO': 3.00,
    'HYUNDAI': 3.00, 'SUBARU': 3.00, 'KIA': 3.00, 'PONTIAC': 3.25, 'ACURA': 3.25,
    'TOYOTA': 3.25, 'OLDSMOBILE': 3.25, 'INFINITI': 3.50, 'MAZDA': 3.50, 'VOLVO': 3.50,
    'SAAB': 3.50, 'CHRYSLER': 3.50, 'BUICK': 3.50, 'MERCEDES-BENZ': 3.50, 'AUDI': 3.50,
    'BMW': 3.75, 'LEXUS': 3.75, 'NISSAN': 3.75, 'CADILLAC': 3.75, 'PLYMOUTH': 3.75,
    'CHEVROLET': 3.75, 'PORSCHE': 4.00, 'ISUZU': 4.00, 'JAGUAR': 4.00, 'FORD': 4.00,
    'LINCOLN': 4.00, 'JEEP': 4.00, 'GMC': 4.50, 'DODGE': 4.75, 'LAND ROVER': 5.00,
    'FERRARI': 6.25
}

def map_vehicle_brand(brand):
    return vehicle_brand_mapping.get(brand, None)

if 'MAKE' in df.columns:
    df['MAKE'] = df['MAKE'].apply(map_vehicle_brand)

# Convert certain columns to integers only if they exist
for col in ['FUEL_X', 'FUEL_Z', 'TRANSMISSION_A4', 'TRANSMISSION_A5', 'TRANSMISSION_M5']:
    if col in df.columns:
        df[col] = df[col].astype(int)


df_normalized=df.drop(columns=['FUEL CONSUMPTION'])

custom_min_max = {
    'FUEL_X': {'min': 0, 'max': 1},  # Example custom values for column1
    'FUEL_X': {'min': 0, 'max': 1},  # Example custom values for column2
    'TRANSMISSION_A4': {'min': 0, 'max': 1}, 
    'TRANSMISSION_A5': {'min': 0, 'max': 1},
    'TRANSMISSION_M5': {'min': 0, 'max': 1},
    'MAKE': {'min': 2.75, 'max': 6.25},
    'VEHICLE CLASS': {'min': 2.99, 'max': 4.83},
    'ENGINE SIZE': {'min': 1.0, 'max': 8.0},
    'CYLINDERS': {'min': 3, 'max': 12},
    'COEMISSIONS': {'min': 104, 'max': 495} # Example custom values for column3
    # Add more columns and their custom min/max values
}


# Normalize each column individually using its custom min and max values
for col in df_normalized.columns:
    if col in custom_min_max:
        col_min = custom_min_max[col]['min']
        col_max = custom_min_max[col]['max']
        df_normalized[col] = (df_normalized[col] - col_min) / (col_max - col_min)


df_normalized['FUEL CONSUMPTION'] = df['FUEL CONSUMPTION']

# Function to apply polynomial transformation
def apply_polynomial_features(X, degree):
    X_poly = np.ones((X.shape[0], 1))  # Start with bias (intercept term)
    for d in range(1, degree + 1):
        X_poly = np.hstack((X_poly, np.power(X, d)))  # Add polynomial terms
    return X_poly

# Load your dataset (assuming it's a pandas DataFrame)
# Update with your actual file path

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

