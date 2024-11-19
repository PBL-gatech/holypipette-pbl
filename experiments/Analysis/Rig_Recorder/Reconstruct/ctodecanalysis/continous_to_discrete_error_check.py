import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

class PipetteDisplacementAnalyzer:
    def __init__(self, file_path1, file_path2):
        """
        Initialize the analyzer with two file paths.

        Parameters:
        - file_path1: str, path to the first CSV file.
        - file_path2: str, path to the second CSV file.
        """
        self.file_path1 = file_path1
        self.file_path2 = file_path2
        self.columns = ['timestamp', 'st_x', 'st_y', 'st_z', 'pi_x', 'pi_y', 'pi_z']
        self.data1 = None
        self.data2 = None

    def load_data(self):
        """
        Load the CSV files into pandas DataFrames and assign column names.
        """
        try:
            # Read data from both files with whitespace separator and no header
            self.data1 = pd.read_csv(self.file_path1, sep='\s+', header=None)
            self.data2 = pd.read_csv(self.file_path2, sep='\s+', header=None)
            
            # Assign column names
            self.data1.columns = self.columns
            self.data2.columns = self.columns
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
    def calculate_error(self):
        """
        Calculate the mean squared error between the two datasets.
        """
        try:
            # Ensure the data has been loaded
            if self.data1 is None or self.data2 is None:
                raise ValueError("Data not loaded. Please run load_data() first.")
            
            # data set two not the same length as data set one, so must calculate error based on timepoints, not length of data
            
            
            # Calculate the mean squared error between the two datasets
            mse = mean_squared_error(self.data1['pipette_resultant'], self.data2['pipette_resultant'])
            print(f"Mean Squared Error: {mse}")
        except Exception as e:
            print(f"Error calculating error: {e}")

    def process_data(self):
        """
        Process the loaded data by extracting numeric values, zeroing timestamps and positions,
        and calculating the resultant pipette displacement.
        """
        try:
            for col in self.columns:
                # Extract numeric values after the colon and convert to float
                self.data1[col] = self.data1[col].str.split(':').str[1].astype(float)
                self.data2[col] = self.data2[col].str.split(':').str[1].astype(float)
            
            # Zero the timestamp for both datasets
            self.data1['timestamp'] -= self.data1['timestamp'].iloc[0]
            self.data2['timestamp'] -= self.data2['timestamp'].iloc[0]
            
            # Zero the pipette movement to the initial position for both datasets
            for data in [self.data1, self.data2]:
                initial_coords = data[['pi_x', 'pi_y', 'pi_z']].iloc[0].values
                data['pi_x'] -= initial_coords[0]
                data['pi_y'] -= initial_coords[1]
                data['pi_z'] -= initial_coords[2]
                # Calculate resultant displacement
                data['pipette_resultant'] = np.sqrt(data['pi_x']**2 + data['pi_y']**2 + data['pi_z']**2)
            
        except Exception as e:
            print(f"Error processing data: {e}")
            raise

    def plot_data(self):
        """
        Plot the resultant pipette displacement over time for both datasets.
        """
        try:
            plt.figure(figsize=(12, 6))
            plt.plot(self.data1['timestamp'], self.data1['pipette_resultant'], label='Dataset 1 - Pipette Resultant')
            plt.plot(self.data2['timestamp'], self.data2['pipette_resultant'], label='Dataset 2 - Pipette Resultant')
            plt.xlabel('Time (s)')
            plt.ylabel('Pipette Resultant Displacement')
            plt.title('Pipette Resultant Displacement Comparison')
            plt.legend()
            plt.grid(True)
            plt.show()
        except Exception as e:
            print(f"Error plotting data: {e}")
            raise

    def run_analysis(self):
        """
        Execute the full analysis pipeline: load, process, and plot data.
        """
        self.load_data()
        self.process_data()
        self.plot_data()

# Example Usage
if __name__ == "__main__":
    # Define file paths
    sinusoid_path = r"C:\Users\sa-forest\Documents\GitHub\holypipette-pbl\testing\movement\sinusoid_signal.csv"
    chirp_path = r"C:\Users\sa-forest\Documents\GitHub\holypipette-pbl\testing\movement\chirp_signal.csv"
    exponential_path  = r"C:\Users\sa-forest\Documents\GitHub\holypipette-pbl\testing\movement\exponential_position_signal.csv"
    
    # Choose the first dataset (uncomment as needed)
    # file_path_1 = sinusoid_path
    # file_path_1 = chirp_path
    file_path_1 = exponential_path
    
    file_path_2 = r"C:\Users\sa-forest\Documents\GitHub\holypipette-pbl\experiments\Data\TEST_rig_recorder_data\2024_11_19-16_51\movement_recording_tr.csv"
    
    # Initialize the analyzer
    analyzer = PipetteDisplacementAnalyzer(file_path1=file_path_1, file_path2=file_path_2)
    
    # Run the analysis
    analyzer.run_analysis()
