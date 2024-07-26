import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize

class EPhysCalc:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.positive_peak_index = None
        self.negative_peak_index = None
        self.peak_current_index = None
        self.peak_current_time = None
        self.zero_gradient_time = None
        self.mean_highlighted_1 = None
        self.peak_value_response_current = None
        self.post_peak_current = None
        self.exp_fit_params = None

    def read_and_convert_data(self):
        # Load the CSV file into a DataFrame
        try:
            self.data = pd.read_csv(self.file_path, delim_whitespace=True, header=None)
            self.data.columns = ['Time', 'Command Voltage', 'Response Current']
            #shift the time axis to start at 0
            start = self.data['Time'].iloc[0].copy()
            # print(f"Start time: {start}")
            self.data['Time'] = self.data['Time'] - start
            #print first time index
            # print(f"First time index: {self.data['Time'].iloc[0]}")
            self.data['Time (ms)'] = self.data['Time'] * 1000  # converting seconds to milliseconds
            # print(f"First time index in ms: {self.data['Time (ms)'].iloc[0]}")
            self.data['Command Voltage (mV)'] = self.data['Command Voltage'] * 1000  # converting volts to millivolts
            self.data['Response Current (pA)'] = self.data['Response Current'] * 1e12  # converting amps to picoamps
        except Exception as e:
            print(f"Error reading the file: {e}")
            print("Please check the file format and delimiter.")

    def calculate_peaks_and_averages(self):
        # Calculate the gradient of Command Voltage using np.gradient
        X_mV = self.data['Command Voltage (mV)'].to_numpy()
        T_ms = self.data['Time (ms)'].to_numpy()
        X_dT = np.gradient(X_mV, T_ms)
        # Update the DataFrame with the new gradient values
        self.data['X_dT'] = X_dT
        # Find the positive and negative peaks of the gradient
        self.positive_peak_index = np.argmax(X_dT)
        self.negative_peak_index = np.argmin(X_dT)
        # Find the peak of the response current
        self.peak_current_index = np.argmax(self.data['Response Current (pA)'])
        self.peak_current_time = self.data.loc[self.peak_current_index, 'Time (ms)']
        # print(f"peak current time: {self.peak_current_time}")
        # Highlight the portions of the response current curve
        highlighted_response_current_1 = self.data.loc[:self.positive_peak_index, 'Response Current (pA)']
        highlighted_response_current_2 = self.data.loc[self.peak_current_index:self.negative_peak_index, 'Response Current (pA)']
        highlighted_time_2 = self.data.loc[self.peak_current_index:self.negative_peak_index, 'Time (ms)']
        # Calculate means
        self.mean_highlighted_1 = highlighted_response_current_1.mean()
        # Get the peak value on the response current plot
        self.peak_value_response_current = self.data.loc[self.peak_current_index, 'Response Current (pA)']
        # Calculate the gradient of the highlighted portion between the positive peak of the response current and the negative peak of the derivative plot
        highlighted_gradient = np.gradient(highlighted_response_current_2, highlighted_time_2)
        # Find the index where the gradient becomes close to zero
        close_to_zero_index = np.where(np.isclose(highlighted_gradient, 0, atol=1e-2))[0]
        if close_to_zero_index.size > 0:
            zero_gradient_index = close_to_zero_index[0]
            self.zero_gradient_time = highlighted_time_2.iloc[zero_gradient_index]
        else:
            self.zero_gradient_time = None
        # Calculate the mean of the data between the zero gradient time and the min time
        if self.zero_gradient_time:
            post_peak_current_data = self.data[(self.data['Time (ms)'] >= self.zero_gradient_time) & (self.data['Time (ms)'] <= self.data.loc[self.negative_peak_index, 'Time (ms)'])]
            self.post_peak_current = post_peak_current_data['Response Current (pA)'].mean()
        else:
            self.post_peak_current = None

    def monoExp(self, x, m, t, b):
        return m * np.exp(-t * x) + b

    def optimizer(self, filtered_data):
        p0 = (self.peak_value_response_current, 0.01 , self.post_peak_current)
        print(f"Initial guess: {p0}")
        try:
            cure_params, _ = scipy.optimize.curve_fit(self.monoExp, filtered_data["Time (ms)"], filtered_data['Response Current (pA)'], maxfev=1000000, p0=p0)
            m, t, b = cure_params
            return m, t, b
        except Exception as e:
            print("Error:", e)
            return None, None, None

    def fit_exponential(self):
        # Extract the data for fitting
        fit_data = self.data[(self.data['Time (ms)'] >= self.peak_current_time) & (self.data['Time (ms)'] <= self.zero_gradient_time)]
        # shift the data to start at 0 again
        start = fit_data['Time (ms)'].iloc[0]
        fit_data['Time (ms)'] = fit_data['Time (ms)'] - start
        # Fit the exponential function to the data
        m, t, b = self.optimizer(fit_data)
        self.exp_fit_params = (m, t, b)
        return self.exp_fit_params

    def plot_graphs(self):
        # Plotting the graphs with the highlighted portions on the response current plot
        plt.figure(figsize=(10, 12))
        # Top graph: Command Voltage vs Time
        plt.subplot(3, 1, 1)
        plt.plot(self.data['Time (ms)'], self.data['Command Voltage (mV)'], color='blue')
        plt.axvline(x=self.peak_current_time, color='green', linestyle='--', label='Peak Current Time')
        plt.axvline(x=self.data.loc[self.negative_peak_index, 'Time (ms)'], color='purple', linestyle='--', label='Min Time')
        if self.zero_gradient_time:
            plt.axvline(x=self.zero_gradient_time, color='cyan', linestyle='--', label='Zero Gradient Time')
        plt.xlabel('Time (ms)')
        plt.ylabel('Command Voltage (mV)')
        plt.title('Command Voltage vs Time')
        plt.legend()
        # Middle graph: Current vs Time with corrected highlighted portions and zero gradient line
        plt.subplot(3, 1, 2)
        plt.plot(self.data['Time (ms)'], self.data['Response Current (pA)'], color='red')
        highlighted_time_1 = self.data.loc[:self.positive_peak_index, 'Time (ms)']
        highlighted_response_current_1 = self.data.loc[:self.positive_peak_index, 'Response Current (pA)']
        highlighted_time_2 = self.data.loc[self.peak_current_index:self.negative_peak_index, 'Time (ms)']
        highlighted_response_current_2 = self.data.loc[self.peak_current_index:self.negative_peak_index, 'Response Current (pA)']
        plt.plot(highlighted_time_1, highlighted_response_current_1, color='blue', label='Highlighted before Positive Derivative Peak')
        plt.plot(highlighted_time_2, highlighted_response_current_2, color='yellow', label='Highlighted between Peaks')
        plt.axvline(x=self.peak_current_time, color='green', linestyle='--', label='Peak Current Time')
        plt.axvline(x=self.data.loc[self.negative_peak_index, 'Time (ms)'], color='purple', linestyle='--', label='Min Time')
        if self.zero_gradient_time:
            plt.axvline(x=self.zero_gradient_time, color='cyan', linestyle='--', label='Zero Gradient Time')
        # Plot the exponential fit
        if self.exp_fit_params is not None:
            x_fit = np.linspace(self.peak_current_time, self.zero_gradient_time, 100)
            y_fit = self.monoExp(x_fit, *self.exp_fit_params)
            plt.plot(x_fit, y_fit, 'k--', label='Exponential Fit')
        plt.xlabel('Time (ms)')
        plt.ylabel('Response Current (pA)')
        plt.title('Response Current vs Time')
        plt.legend()
        # Bottom graph: Derivative of Command Voltage vs Time
        plt.subplot(3, 1, 3)
        plt.plot(self.data['Time (ms)'], self.data['X_dT'], color='orange')
        plt.axvline(x=self.peak_current_time, color='green', linestyle='--', label='Peak Current Time')
        plt.axvline(x=self.data.loc[self.negative_peak_index, 'Time (ms)'], color='purple', linestyle='--', label='Min Time')
        if self.zero_gradient_time:
            plt.axvline(x=self.zero_gradient_time, color='cyan', linestyle='--', label='Zero Gradient Time')
        plt.xlabel('Time (ms)')
        plt.ylabel('d(Command Voltage)/dT (mV/ms)')
        plt.title('Derivative of Command Voltage vs Time')
        plt.legend()
        # Adjust layout
        plt.tight_layout()
        # Show the plots
        plt.show()

    def fit_plotter(self):
        if self.exp_fit_params is None:
            print("Exponential fit parameters not found. Please run fit_exponential() first.")
            return

        # Extract the data for fitting
        fit_data = self.data[(self.data['Time (ms)'] >= self.peak_current_time) & (self.data['Time (ms)'] <= self.zero_gradient_time)]
        # shift the data to start at 0 again
        start = fit_data['Time (ms)'].iloc[0]
        fit_data['Time (ms)'] = fit_data['Time (ms)'] - start

        # Plot the extracted data
        plt.figure(figsize=(10, 6))
        plt.plot(fit_data['Time (ms)'], fit_data['Response Current (pA)'], 'o', label='Extracted Data')

        # Plot the exponential fit
        x_fit = fit_data['Time (ms)']
        y_fit = self.monoExp(x_fit, *self.exp_fit_params)
        plt.plot(x_fit, y_fit, 'r-', label='Exponential Fit')

        plt.xlabel('Time (ms)')
        plt.ylabel('Response Current (pA)')
        plt.title('Exponential Fit of Extracted Data')
        plt.legend()
        plt.show()

    def calc_param(self):
        tau = self.exp_fit_params[1]
        I_peak = self.peak_value_response_current
        I_prev = self.mean_highlighted_1
        I_ss = self.post_peak_current
        I_d = I_peak - I_prev
        I_dss = I_ss - I_prev
        
        dmV = self.data.loc[self.positive_peak_index:self.negative_peak_index, 'Command Voltage (mV)'].mean()
        print(f"tau (ms): {tau}, dmV: {dmV}, I_peak (pA): {I_peak}, I_prev (pA): {I_prev}, I_ss (pA): {I_ss}")
        print(f"I_d (pA): {I_d}, I_dss (pA): {I_dss}")
        #calculate access resistance:
        R_a = ((dmV*1e-3) / (I_d*1e-12))*1e-6 # 10 mV / 800 pA = 12.5 MOhms --> supposed to be 10 MOhms
        #calculate membrane resistance:
        R_m = (((dmV*1e-3)- R_a*1e6*I_dss*1e-12)/(I_dss*1e-12))*1e-6 #530 Mohms --> supposed to be 500 MOhms   
        #calculate membrane capacitance:
        C_m = ((tau*1e-3)/(1/(1/(R_a*1e6) + 1/(R_m*1e6))))*1e12 # 250 pF --> supposed to be 33 pF
        print(f"Access Resistance (MOhms): {R_a}, Membrane Resistance (MOhms): {R_m}, Membrane Capacitance (pF): {C_m}")


# Example usage:
file_path = r"C:\Users\sa-forest\Documents\GitHub\holypipette-pbl\experiments\Data\patch_clamp_data\2024_07_25-17_02\VoltageProtocol_1721941384.317241_1_k.csv"
ephys_calc = EPhysCalc(file_path)
ephys_calc.read_and_convert_data()
ephys_calc.calculate_peaks_and_averages()
ephys_calc.plot_graphs()
ephys_calc.fit_exponential()
ephys_calc.fit_plotter()
ephys_calc.calc_param()
# Print the calculated values
print("Mean of the first highlighted portion (before positive derivative peak):", ephys_calc.mean_highlighted_1)
print("Peak value of the response current:", ephys_calc.peak_value_response_current)
print("Post peak current mean (between zero gradient time and min time):", ephys_calc.post_peak_current)
print("Exponential fit parameters (m, t, b):", ephys_calc.exp_fit_params)

