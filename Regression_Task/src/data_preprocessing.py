import pandas as pd

# Load the dataset
df = pd.read_csv('/Users/jaeeponde/Jaee_Ponde_A1/synthetic_vehicle_data.csv')

# Drop the 'Year' column (if it exists)
if 'Year' in df.columns:
    df = df.drop(columns=['Year'])

# One-hot encode the 'FUEL' column if it exists
if 'FUEL' in df.columns:
    one_hot_encoded_df = pd.get_dummies(df['FUEL'], prefix='FUEL')
    df = pd.concat([df, one_hot_encoded_df], axis=1)
    df = df.drop(columns=[col for col in ['FUEL', 'FUEL_D', 'FUEL_E', 'FUEL_N'] if col in df.columns])

# One-hot encode the 'TRANSMISSION' column if it exists
if 'TRANSMISSION' in df.columns:
    one_hot_encoded_df = pd.get_dummies(df['TRANSMISSION'], prefix='TRANSMISSION')
    df = pd.concat([df, one_hot_encoded_df], axis=1)
    df = df.drop(columns=[col for col in ['TRANSMISSION', 'TRANSMISSION_M6', 'TRANSMISSION_AS4', 'TRANSMISSION_A3', 'TRANSMISSION_AS5'] if col in df.columns])

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

# Normalize the dataframe (excluding 'FUEL CONSUMPTION' if it exists)
df_normalized = (df - df.min()) / (df.max() - df.min())
if 'FUEL CONSUMPTION' in df.columns:
    df_normalized = df.drop(columns=['FUEL CONSUMPTION'])

# Specify the file path where you want to save the CSV
output_path = "/Users/jaeeponde/Jaee_Ponde_A1/synthetic_data_new.csv"  # Replace with your desired path

# Output the result to the specified CSV file
df_normalized.to_csv(output_path, index=False)

print(f"CSV file '{output_path}' has been created successfully.")
