import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from scipy.signal import butter, filtfilt, savgol_filter, correlate
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

class PipetteDisplacementAnalyzer:
    def __init__(self, file_path1, file_path2,fit):
        """
        Initialize the analyzer with two file paths.

        Parameters:
        - file_path1: str, path to the first CSV file.
        - file_path2: str, path to the second CSV file.
        - fit: str, the type of fit to be used
        """
        self.file_path1 = file_path1
        self.file_path2 = file_path2
        self.fit = fit
        self.columns = ['timestamp', 'st_x', 'st_y', 'st_z', 'pi_x', 'pi_y', 'pi_z']
        self.data1 = None
        self.data2 = None
        self.filtered_displacement1 = None
        self.filtered_displacement2 = None
        self.aligned_time = None
        self.aligned_displacement1 = None
        self.aligned_displacement2 = None
        self.velocity1 = None
        self.velocity2 = None
        self.fit_coefficients1 = None
        self.fit_coefficients2 = None

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

    def apply_low_pass_filter(self, cutoff_frequency=1.0, order=4):
        """
        Apply a Butterworth low-pass filter to the pipette resultant displacement data.

        Parameters:
        - cutoff_frequency: float, the cutoff frequency of the filter in Hz.
        - order: int, the order of the Butterworth filter.
        """
        try:
            # Calculate sampling rate from timestamp
            sampling_rate1 = 1.0 / np.median(np.diff(self.data1['timestamp']))
            sampling_rate2 = 1.0 / np.median(np.diff(self.data2['timestamp']))
            sampling_rate = (sampling_rate1 + sampling_rate2) / 2.0  # Average sampling rate

            nyquist_frequency = 0.5 * sampling_rate
            normalized_cutoff = cutoff_frequency / nyquist_frequency

            # Get Butterworth filter coefficients
            b, a = butter(order, normalized_cutoff, btype='low', analog=False)

            # Apply filter using filtfilt to avoid phase shift
            self.filtered_displacement1 = filtfilt(b, a, self.data1['pipette_resultant'].values)
            self.filtered_displacement2 = filtfilt(b, a, self.data2['pipette_resultant'].values)
        except Exception as e:
            print(f"Error applying Butterworth filter: {e}")
            raise

    def apply_savgol_filter(self, window_length=11, polyorder=3):
        """
        Apply a Savitzky-Golay filter to the pipette resultant displacement data.

        Parameters:
        - window_length: int, the length of the filter window (must be odd).
        - polyorder: int, the order of the polynomial used to fit the samples.
        """
        try:
            # Ensure window_length is odd and greater than polyorder
            if window_length % 2 == 0:
                window_length += 1
            if window_length <= polyorder:
                window_length = polyorder + 2 if (polyorder + 2) % 2 != 0 else polyorder + 3

            # Apply Savitzky-Golay filter
            self.filtered_displacement1 = savgol_filter(self.data1['pipette_resultant'].values, 
                                                       window_length=window_length, 
                                                       polyorder=polyorder)
            self.filtered_displacement2 = savgol_filter(self.data2['pipette_resultant'].values, 
                                                       window_length=window_length, 
                                                       polyorder=polyorder)
        except Exception as e:
            print(f"Error applying Savitzky-Golay filter: {e}")
            raise

    def apply_filter(self):
        """
        Apply a smoothing filter to the pipette resultant displacement data.
        Uncomment the desired filter to apply.
        """
        try:
            # Apply Butterworth low-pass filter
            # Uncomment the following lines to apply Butterworth filter
            cutoff_frequency = 1 # Example value; adjust as needed
            order = 4               # Example value; adjust as needed
            self.apply_low_pass_filter(cutoff_frequency=cutoff_frequency, order=order)

            # Apply Savitzky-Golay filter
            # Uncomment the following lines to apply Savitzky-Golay filter
            # window_length = 100  # Must be odd and > polyorder
            # polyorder = 3       # Typically 2 or 3
            # self.apply_savgol_filter(window_length=window_length, polyorder=polyorder)

            self.data1['filtered_pipette_resultant'] = self.filtered_displacement1
            self.data2['filtered_pipette_resultant'] = self.filtered_displacement2
        except Exception as e:
            print(f"Error applying filter: {e}")
            raise

    def align_datasets(self, method='linear', num_points=1000):
        """
        Align the displacement data of both datasets based on time by interpolating them onto a common time base.

        Parameters:
        - method: str, interpolation method (default is 'linear').
        - num_points: int, number of points in the common time base (default is 1000).
        """
        try:
            # Ensure data is processed
            if self.data1 is None or self.data2 is None:
                raise ValueError("Data not processed. Please run process_data() first.")
            
            # Use filtered displacement if available, else use raw
            displacement1 = self.filtered_displacement1 if self.filtered_displacement1 is not None else self.data1['pipette_resultant'].values
            displacement2 = self.filtered_displacement2 if self.filtered_displacement2 is not None else self.data2['pipette_resultant'].values

            # Define common time range based on overlapping time
            start_time = max(self.data1['timestamp'].min(), self.data2['timestamp'].min())
            end_time = min(self.data1['timestamp'].max(), self.data2['timestamp'].max())
            
            if start_time >= end_time:
                raise ValueError("No overlapping time range between the two datasets.")
            
            # Define a common time base
            self.aligned_time = np.linspace(start_time, end_time, num=num_points)
            
            # Create interpolation functions for displacement
            interp1 = interp1d(self.data1['timestamp'], displacement1, kind=method, fill_value="extrapolate")
            interp2 = interp1d(self.data2['timestamp'], displacement2, kind=method, fill_value="extrapolate")
            
            # Interpolate displacements onto the common time base
            self.aligned_displacement1 = interp1(self.aligned_time)
            self.aligned_displacement2 = interp2(self.aligned_time)
        except Exception as e:
            print(f"Error aligning datasets: {e}")
            raise
   

    def detect_variable_region(self, data, gradient_threshold=1e-3):
        """
        Detect the variable (non-flatline) region of the data using gradient thresholds.

        Parameters:
        - data: pandas DataFrame, the dataset containing 'timestamp' and 'pipette_resultant'.
        - gradient_threshold: float, the minimum gradient value to consider as non-flatline.

        Returns:
        - start_idx: int, the starting index of the variable region.
        - end_idx: int, the ending index of the variable region.
        """
        gradient = np.gradient(data['pipette_resultant'], data['timestamp'])
        variable_indices = np.where(abs(gradient) > gradient_threshold)[0]
        if len(variable_indices) > 0:
            start_idx = variable_indices[0]
            end_idx = variable_indices[-1]
        else:
            start_idx, end_idx = 0, len(data) - 1  # Default to full range if no variability detected
        return start_idx, end_idx

    def compute_fit(self):
        """
        computes fits for the data to be used for velocity calculations
        """

        if self.fit == 'quadratic':
            try:
                # Detect and truncate variable region for Dataset 1
                start1, end1 = self.detect_variable_region(self.data1)
                data1_variable = self.data1.iloc[start1:end1 + 1]

                # Perform constrained quadratic fitting on Dataset 1
                x1 = data1_variable['timestamp']
                y1 = data1_variable['pipette_resultant']
                self.fit_coefficients1, _ = curve_fit(self.constrained_quadratic, x1, y1)
                self.y_fit1 = self.constrained_quadratic(x1, *self.fit_coefficients1)

                # Detect and truncate variable region for Dataset 2
                start2, end2 = self.detect_variable_region(self.data2)
                data2_variable = self.data2.iloc[start2:end2 + 1]

                # Perform constrained quadratic fitting on Dataset 2
                x2 = data2_variable['timestamp']
                y2 = data2_variable['pipette_resultant']
                self.fit_coefficients2, _ = curve_fit(self.constrained_quadratic, x2, y2)
                self.y_fit2 = self.constrained_quadratic(x2, *self.fit_coefficients2)

                # print equations
                print(f"Fit Equation Dataset 1: y(t) = {self.fit_coefficients1[0]:.4f} * t^2 + {self.fit_coefficients1[1]:.4f} * t")
                print(f"Fit Equation Dataset 2: y(t) = {self.fit_coefficients2[0]:.4f} * t^2 + {self.fit_coefficients2[1]:.4f} * t")

                # Add fitted data to the DataFrame for visualization
                self.data1['fit'] = np.nan
                self.data2['fit'] = np.nan
                self.data1.loc[start1:end1, 'fit'] = self.y_fit1
                self.data2.loc[start2:end2, 'fit'] = self.y_fit2

            except Exception as e:
                print(f"Error computing fit: {e}")
                raise
        elif self.fit == 'linear':
            # this ramp signal is linear, so we can use a linear fit
            try:
                # Detect and truncate variable region for Dataset 1
                start1, end1 = self.detect_variable_region(self.data1)
                data1_variable = self.data1.iloc[start1:end1 + 1]

                # Perform constrained linear fitting on Dataset 1
                x1 = data1_variable['timestamp']
                y1 = data1_variable['pipette_resultant']
                self.fit_coefficients1 = np.polyfit(x1, y1 - y1.iloc[0], 1)
                self.y_fit1 = np.polyval(self.fit_coefficients1, x1) + y1.iloc[0]

                # Detect and truncate variable region for Dataset 2
                start2, end2 = self.detect_variable_region(self.data2)
                data2_variable = self.data2.iloc[start2:end2 + 1]

                # Perform constrained linear fitting on Dataset 2
                x2 = data2_variable['timestamp']
                y2 = data2_variable['pipette_resultant']
                self.fit_coefficients2 = np.polyfit(x2, y2 - y2.iloc[0], 1)
                self.y_fit2 = np.polyval(self.fit_coefficients2, x2) + y2.iloc[0]

                # print equations
                print(f"Fit Equation Dataset 1: y(t) = {self.fit_coefficients1[0]:.4f} * t + {self.fit_coefficients1[1]:.4f}")
                print(f"Fit Equation Dataset 2: y(t) = {self.fit_coefficients2[0]:.4f} * t + {self.fit_coefficients2[1]:.4f}")

                # Add fitted data to the DataFrame for visualization
                self.data1['fit'] = np.nan
                self.data2['fit'] = np.nan
                self.data1.loc[start1:end1, 'fit'] = self.y_fit1
                self.data2.loc[start2:end2, 'fit'] = self.y_fit2
            except Exception as e:
                print(f"Error computing fit: {e}")
                raise



    def compute_velocity(self):
        """
        Compute the velocity (derivative of displacement) for both datasets based on the fits.
        """
        if self.fit == 'quadratic':
            try:
                # Ensure the fit coefficients exist
                if self.fit_coefficients1 is None or self.fit_coefficients2 is None:
                    raise ValueError("Fit coefficients are not computed. Please run compute_fit() first.")

                # Derive velocity equations from the fit coefficients
                a1, b1 = self.fit_coefficients1
                a2, b2 = self.fit_coefficients2

                # Store the velocity equations as strings for reference
                self.velocity_equation1 = f"v_1(t) = {2 * a1:.4f} * t + {b1:.4f}"
                self.velocity_equation2 = f"v_2(t) = {2 * a2:.4f} * t + {b2:.4f}"

                # Compute numerical velocity arrays for plotting
                self.velocity1 = 2 * a1 * self.aligned_time + b1
                self.velocity2 = 2 * a2 * self.aligned_time + b2

                print(f"Velocity Equation Dataset 1: {self.velocity_equation1}")
                print(f"Velocity Equation Dataset 2: {self.velocity_equation2}")

            except Exception as e:
                print(f"Error computing velocity: {e}")
                raise
        elif self.fit == 'linear':
            try: 
                # Ensure the fit coefficients exist
                if self.fit_coefficients1 is None or self.fit_coefficients2 is None:
                    raise ValueError("Fit coefficients are not computed. Please run compute_fit() first.")

                # Derive velocity equations from the fit coefficients
                a1, b1 = self.fit_coefficients1
                a2, b2 = self.fit_coefficients2

                # Store the velocity equations as strings for reference
                self.velocity_equation1 = f"v_1(t) = {a1:.4f}"
                self.velocity_equation2 = f"v_2(t) = {a2:.4f}"

                # Compute numerical velocity arrays for plotting
                self.velocity1 = np.full_like(self.aligned_time, a1)
                self.velocity2 = np.full_like(self.aligned_time, a2)

                print(f"Velocity Equation Dataset 1: {self.velocity_equation1}")
                print(f"Velocity Equation Dataset 2: {self.velocity_equation2}")
            except Exception as e:
                print(f"Error computing velocity: {e}")
                raise

    def constrained_quadratic(self,t, a, b):
        return a * t**2 + b * t
    
    def compute_drift_error(self):
        """
        Compute the drift error (lag) between the two datasets as a function of velocity.
        """
        try:
            # take in aligned position data
            command = self.aligned_displacement1
            response = self.aligned_displacement2
            # compute cross-correlation to get the lag
            cross_corr = correlate(response, command, mode='full')
            lags = np.arange(-len(command) + 1, len(response))
            lag = lags[np.argmax(cross_corr)]
            lag_time = lag * np.mean(np.diff(self.aligned_time))
            print(f"Drift error (lag) between datasets: {lag_time:.4f} seconds")
            
            # Ensure velocity data is computed
            if self.velocity1 is None or self.velocity2 is None:
                raise ValueError("Velocity data not computed. Please run compute_velocity() first.")
            # get fit coefficients
            a1, b1 = self.fit_coefficients1
            a2, b2 = self.fit_coefficients2
            # subtract the fit coefficients
            a_diff = a1 - a2
            b_diff = b1 - b2
            # calculate position vs time and velocty vs time with difference fit coefficients
            position = (a_diff/2) * self.aligned_time**2 + b_diff * self.aligned_time
            velocity =  a1 * self.aligned_time + b1
            self.lag_vs_velocity = (velocity, position)

        except Exception as e:
            print(f"Error computing drift error: {e}")
            raise


    def plot_displacement(self):
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
            print(f"Error plotting displacement data: {e}")
            raise

    def plot_velocity(self):
        """
        Plot the velocity over time for both aligned datasets.
        """
        try:
            plt.figure(figsize=(12, 6))
            plt.plot(self.aligned_time, self.velocity1, label='Dataset 1 - Velocity')
            plt.plot(self.aligned_time, self.velocity2, label='Dataset 2 - Velocity')
            plt.xlabel('Time (s)')
            plt.ylabel('Velocity (units/s)')
            plt.title('Pipette Velocity Comparison')
            plt.legend()
            plt.grid(True)
            plt.show()
        except Exception as e:
            print(f"Error plotting velocity data: {e}")
            raise

    def plot_drift_error(self):
        """
        Plot the drift error (lag) as a function of data1's velocity.
        """
        try:
            if self.lag_vs_velocity is None:
                raise ValueError("Drift error not calculated. Please run calculate_drift_error() first.")
            
            # Debugging statement
            # print(f"self.lag_vs_velocity: {self.lag_vs_velocity}")
            # print(f"type(self.lag_vs_velocity): {type(self.lag_vs_velocity)}")
            
            position_error, velocity = self.lag_vs_velocity
            
            plt.figure(figsize=(12, 6))
            plt.scatter(velocity, position_error, c='blue', alpha=0.6, edgecolors='w', label='Lag vs Velocity')
            plt.xlabel('Data1 Velocity (um/s)')
            plt.ylabel('Position Error (um)')
            plt.title('Lag Between Datasets as a Function of Data1\'s Velocity')
            plt.legend()
            plt.grid(True)
            plt.show()
        except Exception as e:
            print(f"Error plotting drift error: {e}")
            raise

    def plot_fits(self):
        """
        Plot the fitted data for both datasets alongside the original data.
        """
        try:
            plt.figure(figsize=(12, 6))
            # Plot Dataset 1
            plt.plot(self.data1['timestamp'], self.data1['pipette_resultant'], label='Dataset 1 - Pipette Resultant', alpha=0.5)
            plt.plot(self.data1['timestamp'], self.data1['fit'], label='Dataset 1 - Quadratic Fit', linestyle='--')
            # Plot Dataset 2
            plt.plot(self.data2['timestamp'], self.data2['pipette_resultant'], label='Dataset 2 - Pipette Resultant', alpha=0.5)
            plt.plot(self.data2['timestamp'], self.data2['fit'], label='Dataset 2 - Quadratic Fit', linestyle='--')

            plt.xlabel('Time (s)')
            plt.ylabel('Pipette Resultant Displacement')
            plt.title('Quadratic Fits for Pipette Resultant Displacement')
            plt.legend()
            plt.grid(True)
            plt.show()

        except Exception as e:
            print(f"Error plotting fits: {e}")
            raise


    def plot_filtered_data(self):
        """
        Plot the filtered pipette resultant displacement over time for both datasets.
        """
        try:
            if self.filtered_displacement1 is None or self.filtered_displacement2 is None:
                raise ValueError("Filtered data not available. Please run apply_filter() first.")
            
            plt.figure(figsize=(12, 6))
            plt.plot(self.data1['timestamp'], self.data1['pipette_resultant'], 
                     label='Dataset 1 - Raw Displacement', alpha=0.5)
            plt.plot(self.data1['timestamp'], self.filtered_displacement1, 
                     label='Dataset 1 - Filtered Displacement', linewidth=2)
            plt.plot(self.data2['timestamp'], self.data2['pipette_resultant'], 
                     label='Dataset 2 - Raw Displacement', alpha=0.5)
            plt.plot(self.data2['timestamp'], self.filtered_displacement2, 
                     label='Dataset 2 - Filtered Displacement', linewidth=2)
            plt.xlabel('Time (s)')
            plt.ylabel('Pipette Resultant Displacement')
            plt.title('Filtered Pipette Resultant Displacement Comparison')
            plt.legend()
            plt.grid(True)
            plt.show()
        except Exception as e:
            print(f"Error plotting filtered data: {e}")
            raise


    def plot_all(self):
        """
        Plot displacement, velocity, and drift error in separate subplots.
        """
        try:
            fig, axs = plt.subplots(2, 1, figsize=(12, 24))
            
            # Displacement Plot
            axs[0].plot(self.data1['timestamp'], self.data1['pipette_resultant'], label='Dataset 1 - Pipette Resultant')
            axs[0].plot(self.data2['timestamp'], self.data2['pipette_resultant'], label='Dataset 2 - Pipette Resultant')
            axs[0].set_xlabel('Time (s)')
            axs[0].set_ylabel('Pipette Resultant Displacement')
            axs[0].set_title('Pipette Resultant Displacement Comparison')
            axs[0].legend()
            axs[0].grid(True)
            
            # Velocity Plot
            axs[1].plot(self.aligned_time, self.velocity1, label='Dataset 1 - Velocity')
            axs[1].plot(self.aligned_time, self.velocity2, label='Dataset 2 - Velocity')
            axs[1].set_xlabel('Time (s)')
            axs[1].set_ylabel('Velocity (units/s)')
            axs[1].set_title('Pipette Velocity Comparison')
            axs[1].legend()
            axs[1].grid(True)
            

            
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Error plotting all data: {e}")
            raise

        
    def run_analysis(self):
        """
        Execute the full analysis pipeline:
        load, process, apply filter, align displacements, compute velocity,
        calculate drift error, compute MSE, and plot data.
        """
        self.load_data()
        self.process_data()
        self.compute_fit()
        # self.apply_filter()
        self.align_datasets()
        self.compute_velocity()
        # self.compute_drift_error()
        self.plot_all()
        # self.plot_displacement()
        # self.plot_velocity()
        self.plot_fits()
        # self.plot_drift_error()
        # self.plot_filtered_data()

# Example Usage
if __name__ == "__main__":
    # Define file paths
    sinusoid_path = r"C:\Users\sa-forest\Documents\GitHub\holypipette-pbl\testing\movement\sinusoid_signal.csv"
    chirp_path = r"C:\Users\sa-forest\Documents\GitHub\holypipette-pbl\testing\movement\chirp_signal.csv"
    exponential_path  = r"C:\Users\sa-forest\Documents\GitHub\holypipette-pbl\testing\movement\exponential_position_signal.csv"
    ramp_path = r"C:\Users\sa-forest\Documents\GitHub\holypipette-pbl\testing\movement\ramp_signal.csv"
    
    # Choose the first dataset (uncomment as needed)
    # file_path1 = sinusoid_path
    # file_path1 = chirp_path
    file_path1 = exponential_path
    # file_path1 = ramp_path
    
    file_path2 = r"C:\Users\sa-forest\Documents\GitHub\holypipette-pbl\experiments\Data\TEST_rig_recorder_data\2024_11_20-16_57\movement_recording_truncated.csv" # exponential signal
    # file_path2 = r"C:\Users\sa-forest\Documents\GitHub\holypipette-pbl\experiments\Data\TEST_rig_recorder_data\2024_11_21-12_47\movement_recording_truncated.csv" # linear signal at 10 um/s
    # file_path2 = r"C:\Users\sa-forest\Documents\GitHub\holypipette-pbl\experiments\Data\TEST_rig_recorder_data\2024_11_21-13_05\movement_recording_truncated.csv" # linear signal at 5 um/s
    # file_path2 = r"C:\Users\sa-forest\Documents\GitHub\holypipette-pbl\experiments\Data\TEST_rig_recorder_data\2024_11_21-13_11\movement_recording_truncated.csv" # linear signal at 2.5 um/s
    
    # Initialize the analyzer
    analyzer = PipetteDisplacementAnalyzer(file_path1=file_path1, file_path2=file_path2,fit = 'quadratic')
    
    # Run the full analysis
    analyzer.run_analysis()
