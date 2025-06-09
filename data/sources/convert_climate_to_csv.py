import json
import pandas as pd

# Load the climate data JSON
with open('openmeteo_climate.json', 'r') as f:
    climate_data = json.load(f)

# Convert to DataFrame
df = pd.DataFrame(climate_data)

# Show the structure
print('Climate data structure:')
print(f'Shape: {df.shape}')
print(f'Columns: {list(df.columns)}')
print(f'Cities: {df["city"].nunique()}')
print(f'Months: {sorted(df["travel_month"].unique())}')
print()
print('Sample data:')
print(df.head())

# Save as CSV
output_path = '../cities_weather.csv'
df.to_csv(output_path, index=False)
print(f'\nSaved climate data to {output_path}')
print(f'File size: {len(df)} records')

# Show a sample of the saved CSV
print('\nFirst few rows of saved CSV:')
saved_df = pd.read_csv(output_path)
print(saved_df.head()) 