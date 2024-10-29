# vprotAnalyzer.py
import pandas as pd
import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt

class VoltageProtocolAnalyzer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.holding_current = None
        self.latestAccessResistance = None
        self.latestMembraneResistance = None
        self.latestMembraneCapacitance = None
        self.totalResistance = None

    def read_and_convert_data(self):
        """
        Load the CSV file into a DataFrame and convert the necessary units.
        """
        try:
            self.data = pd.read_csv(self.file_path, delim_whitespace=True, header=None)
            self.data.columns = ['Time', 'Command Voltage', 'Response Current']
            # Shift the time axis to start at 0
            start = self.data['Time'].iloc[0]
            self.data['Time'] = self.data['Time'] - start
            # Convert units
            self.data['Time (ms)'] = self.data['Time'] * 1000  # converting seconds to milliseconds
            self.data['Command Voltage (mV)'] = self.data['Command Voltage'] * 1000  # converting volts to millivolts
            self.data['Response Current (pA)'] = self.data['Response Current'] * 1e12  # converting amps to picoamps
        except Exception as e:
            print(f"Error reading the file: {e}")
            print("Please check the file format and delimiter.")

    def filter_data(self):
        """
        Filter the data to extract key parameters from the response current.
        """
        if self.data is None:
            raise ValueError("Data has not been loaded. Please run read_and_convert_data() first.")

        X_mV = self.data['Command Voltage (mV)'].to_numpy()
        T_ms = self.data['Time (ms)'].to_numpy()
        Y_pA = self.data['Response Current (pA)'].to_numpy()
        X_dT = np.gradient(X_mV, T_ms)
        self.data["X_dT"] = X_dT

        # Find the index of the maximum and minimum values
        positive_peak_index = np.argmax(X_dT)
        negative_peak_index = np.argmin(X_dT)
        peak_current_index = np.argmax(Y_pA)
        peak_time = self.data.loc[peak_current_index, 'Time (ms)']
        negative_peak_time = self.data.loc[negative_peak_index, 'Time (ms)']

        # Extract the data between peaks
        pre_peak_current = self.data.loc[:positive_peak_index, "Response Current (pA)"]
        sub_data = self.data.loc[peak_current_index:negative_peak_index, "Response Current (pA)"]
        sub_time = self.data.loc[peak_current_index:negative_peak_index, "Time (ms)"]
        sub_command = self.data.loc[peak_current_index:negative_peak_index, "Command Voltage (mV)"]

        # Calculate the mean current prior to the voltage pulse (I_prev)
        mean_pre_peak = pre_peak_current.mean()

        # Calculate the mean current post voltage pulse (I_ss)
        gradient = np.gradient(sub_data, sub_time)
        close_to_zero_index = np.where(np.isclose(gradient, 0, atol=1e-2))[0]
        zero_gradient_time = None
        if close_to_zero_index.size > 0:
            zero_gradient_index = close_to_zero_index[0]
            zero_gradient_time = sub_time.iloc[zero_gradient_index]

        if zero_gradient_time:
            post_peak_current_data = self.data[(self.data['Time (ms)'] >= zero_gradient_time) & (self.data['Time (ms)'] <= negative_peak_time)]
            mean_post_peak = post_peak_current_data['Response Current (pA)'].mean()
        else:
            mean_post_peak = None

        return sub_data, sub_time, sub_command, [peak_time, peak_current_index, negative_peak_time, negative_peak_index], mean_pre_peak, mean_post_peak

    def monoExp(self, x, m, t, b):
        return m * np.exp(-t * x) + b

    def optimizer(self, fit_data, I_peak_pA, I_peak_time, I_ss):
        start = fit_data['Time (ms)'].iloc[0]
        fit_data['Time (ms)'] = fit_data['Time (ms)'] - start
        p0 = (I_peak_pA, I_peak_time, I_ss)
        try:
            params, _ = scipy.optimize.curve_fit(self.monoExp, fit_data['Time (ms)'], fit_data['Response Current (pA)'], maxfev=1000000, p0=p0)
            m, t, b = params
            return m, t, b
        except Exception as e:
            print(f"Error in the optimizer: {e}")
            return None, None, None

    def calc_param(self, tau, mean_voltage, I_peak, I_prev, I_ss):
        I_d = I_peak - I_prev  # in pA
        I_dss = I_ss - I_prev  # in pA

        R_a_Mohms = ((mean_voltage * 1e-3) / (I_d * 1e-12)) * 1e-6
        R_m_Mohms = (((mean_voltage * 1e-3) - R_a_Mohms * 1e6 * I_dss * 1e-12) / (I_dss * 1e-12)) * 1e-6
        C_m_pF = (tau * 1e-3) / (1 / (1 / (R_a_Mohms * 1e6) + 1 / (R_m_Mohms * 1e6))) * 1e12

        return R_a_Mohms, R_m_Mohms, C_m_pF

    def analyze_voltage_protocol(self):
        self.read_and_convert_data()
        filtered_data, filtered_time, filtered_command, plot_params, I_prev_pA, I_post_pA = self.filter_data()
        I_peak_pA = self.data.loc[plot_params[1] + 1, 'Response Current (pA)']
        I_peak_time = self.data.loc[plot_params[1] + 1, 'Time (ms)']
        mean_voltage = filtered_command.mean()

        m, t, b = self.optimizer(pd.DataFrame({'Time (ms)': filtered_time, 'Command Voltage (mV)': filtered_command, 'Response Current (pA)': filtered_data}), I_peak_pA, I_peak_time, I_post_pA)
        if m is not None and t is not None and b is not None:
            tau = 1 / t
            R_a_MOhms, R_m_MOhms, C_m_pF = self.calc_param(tau, mean_voltage, I_peak_pA, I_prev_pA, I_post_pA)
            self.latestAccessResistance = R_a_MOhms
            print(f"Access Resistance: {R_a_MOhms} MOhms")
            self.latestMembraneResistance = R_m_MOhms
            print(f"Membrane Resistance: {R_m_MOhms} MOhms")
            self.latestMembraneCapacitance = C_m_pF
            print(f"Membrane Capacitance: {C_m_pF} pF")
            self.totalResistance = R_a_MOhms + R_m_MOhms
            print(f"Total Resistance: {self.totalResistance} MOhms")
            return R_a_MOhms, R_m_MOhms, C_m_pF, self.totalResistance
        else:
            return None, None, None, None

    def plot_data(self):
        """
        Plot the command data and response data.
        """
        if self.data is None:
            raise ValueError("Data has not been loaded. Please run read_and_convert_data() first.")

        time = self.data['Time (ms)']
        command = self.data['Command Voltage (mV)']
        response = self.data['Response Current (pA)']

        plt.figure(figsize=(10, 6))

        # Plot Command Voltage
        plt.subplot(2, 1, 1)
        plt.plot(time, command, label='Command Voltage (mV)', color='b')
        plt.xlabel('Time (ms)')
        plt.ylabel('Command Voltage (mV)')
        plt.title('Command Voltage vs Time')
        plt.grid(True)
        plt.legend()

        # Plot Response Current
        plt.subplot(2, 1, 2)
        plt.plot(time, response, label='Response Current (pA)', color='r')
        plt.xlabel('Time (ms)')
        plt.ylabel('Response Current (pA)')
        plt.title('Response Current vs Time')
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.show()

# Usage
example_file_path = r"C:\Users\sa-forest\OneDrive - Georgia Institute of Technology\Documents\Grad-school\Gatech\Fall2024\ForestLab\AD\9-30_pres_data\VoltageProtocol\VoltageProtocol_1_k.csv"
if __name__ == "__main__":
    analyzer = VoltageProtocolAnalyzer(example_file_path)
    results = analyzer.analyze_voltage_protocol()
    print(results)
    analyzer.plot_data()
