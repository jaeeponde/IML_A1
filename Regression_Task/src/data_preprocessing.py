import pandas as pd

# Load the dataset
df = pd.read_csv('/Users/jaeeponde/IML_A1/IML_A1/Regression_Task/data/fuel_train - fuel_train.csv')

# Drop the 'Year' column
df = df.drop(columns=['Year'])


# One-hot encode the 'FUEL' column
one_hot_encoded_df = pd.get_dummies(df['FUEL'], prefix='FUEL')
df = pd.concat([df, one_hot_encoded_df], axis=1)
df = df.drop(columns=['FUEL', 'FUEL_D', 'FUEL_E', 'FUEL_N'])

# One-hot encode the 'TRANSMISSION' column
one_hot_encoded_df = pd.get_dummies(df['TRANSMISSION'], prefix='TRANSMISSION')
df = pd.concat([df, one_hot_encoded_df], axis=1)
df = df.drop(columns=['TRANSMISSION', 'TRANSMISSION_M6', 'TRANSMISSION_AS4', 'TRANSMISSION_A3', 'TRANSMISSION_AS5'])

# Drop the 'MODEL' column
df = df.drop(columns=['MODEL'])

# Calculate mean fuel consumption per vehicle class
mean_fuel_consumption = df.groupby('VEHICLE CLASS')['FUEL CONSUMPTION'].mean().reset_index()
mean_fuel_consumption.columns = ['VEHICLE_TYPE', 'MEAN_FUEL_CONSUMPTION']
mean_fuel_consumption['MEAN_FUEL_CONSUMPTION'] = (mean_fuel_consumption['MEAN_FUEL_CONSUMPTION'] / 4).round(2)

# Create a map of fuel consumption
fuel_consumption_map = dict(zip(mean_fuel_consumption['VEHICLE_TYPE'], mean_fuel_consumption['MEAN_FUEL_CONSUMPTION']))
df['VEHICLE CLASS'] = df['VEHICLE CLASS'].map(fuel_consumption_map)

# Calculate mean fuel consumption per make
mean_make = df.groupby('MAKE')['FUEL CONSUMPTION'].mean().reset_index()
mean_make.columns = ['make', 'MEAN_FUEL_CONSUMPTION']
mean_make = mean_make.sort_values(by='MEAN_FUEL_CONSUMPTION')

# Define bins and labels for 'MEAN_FUEL_CONSUMPTION'
bins = [10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5, 20.5, 30]
labels = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 25]
mean_make['Binned_MEAN_FUEL_CONSUMPTION'] = pd.cut(mean_make['MEAN_FUEL_CONSUMPTION'], bins=bins, labels=labels, right=False)

# Drop unnecessary columns and finalize the 'Binned_MEAN_FUEL_CONSUMPTION'
mean_make = mean_make.drop(columns=['MEAN_FUEL_CONSUMPTION'])
mean_make['Binned_MEAN_FUEL_CONSUMPTION'] = (mean_make['Binned_MEAN_FUEL_CONSUMPTION'].astype(int) / 4).round(2)

# Map the binned fuel consumption to the dataframe
consumption_map = dict(zip(mean_make['make'], mean_make['Binned_MEAN_FUEL_CONSUMPTION']))
df['MAKE'] = df['MAKE'].map(consumption_map)

# Convert certain columns to integers
df['FUEL_X'] = df['FUEL_X'].astype(int)
df['FUEL_Z'] = df['FUEL_Z'].astype(int)
df['TRANSMISSION_A4'] = df['TRANSMISSION_A4'].astype(int)
df['TRANSMISSION_A5'] = df['TRANSMISSION_A5'].astype(int)
df['TRANSMISSION_M5'] = df['TRANSMISSION_M5'].astype(int)

# Reorder the columns to have 'FUEL CONSUMPTION' as the last column
df = df[[col for col in df.columns if col != 'FUEL CONSUMPTION'] + ['FUEL CONSUMPTION']]

# Normalize the dataframe
df_normalized = (df - df.min()) / (df.max() - df.min())
df_normalized['FUEL CONSUMPTION'] = df['FUEL CONSUMPTION'].values

# Output the result to a CSV file
df_normalized.to_csv('trail_training_data.csv', index=False)

print("CSV file 'trail_training_data.csv' has been created successfully.")





