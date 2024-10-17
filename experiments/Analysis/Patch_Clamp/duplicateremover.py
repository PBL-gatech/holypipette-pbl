import os
import pandas as pd

# Define the folder containing the CSV files
folder_path = r"C:\Users\sa-forest\OneDrive - Georgia Institute of Technology\Documents\Grad-school\Gatech\Fall2024\ForestLab\AD\9-30_pres_data\CurrentProtocol"

# Iterate through all files in the folder
for file_name in os.listdir(folder_path):
    # Check if the file starts with "CurrentProtocol" and has a .csv extension
    if file_name.startswith("CurrentProtocol") and file_name.endswith(".csv"):
        file_path = os.path.join(folder_path, file_name)

        # Load the CSV file
        data = pd.read_csv(file_path, delimiter=r"\s+")

        # Rename columns for clarity
        data.columns = ['Time', 'CommandVoltage', 'Current']

        # Remove duplicate rows
        data_cleaned = data.drop_duplicates()
        # make sure to sort the data by time
        data_cleaned = data_cleaned.sort_values(by='Time')

        # Adjust the first timepoint to be 0.0 (optional, can be commented out if not needed)
        # data_cleaned['Time'] = data_cleaned['Time'] - data_cleaned['Time'].iloc[0]

        # Filter data to keep only rows where Time is <= 2.2 seconds (optional, can be commented out if not needed)
        # data_cleaned = data_cleaned[data_cleaned['Time'] <= 2.2]

        
        

        # Save the cleaned data in the same format it was read
        data_cleaned.to_csv(file_path, sep=' ', index=False, header=False)

        # Output the cleaned file path
        print(f"Cleaned file saved to: {file_path}")