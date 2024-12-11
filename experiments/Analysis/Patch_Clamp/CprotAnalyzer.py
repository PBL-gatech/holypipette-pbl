#CprotAnalyzer.py
import pandas as pd
import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt

class CurrentProtocolAnalyzer:
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
            self.data = pd.read_csv(self.file_path, sep='\s+', header=None)
            self.data.columns = ['Time', 'Command Current', 'Response Voltage']
            # Shift the time axis to start at 0
            start = self.data['Time'].iloc[0]
            self.data['Time'] = self.data['Time'] - start
            # Convert units
            self.data['Time (ms)'] = self.data['Time'] * 1000  # converting seconds to milliseconds
            self.data['Command Current (pA)'] = self.data['Command Current'] * 400 # converting volts to pico amps with conversion factor of 400 (1V = 400 pA)
            self.data['Response Voltage (mV)'] = self.data['Response Voltage'] * 1000  # converting volts to millivolts with conversion factor of 1000
        except Exception as e:
            print(f"Error reading the file: {e}")
            print("Please check the file format and delimiter.") 

    def filter_data(self):
        """
        Filter the data to extract key parameters from the response current.
        """
        if self.data is None:
            raise ValueError("Data has not been loaded. Please run read_and_convert_data() first.")

        X_pA = self.data['Command Current (pA)'].to_numpy()
        T_ms = self.data['Time (ms)'].to_numpy()
        Y_mV = self.data['Response Voltage (mV)'].to_numpy()
        X_dT = np.gradient(X_pA, T_ms)
        self.data["X_dT"] = X_dT

        cutoff_time = 700 # ms
        cutoff_index = np.where(T_ms > cutoff_time)[0][0]
        print(f"Cut off index: {cutoff_index}")
        print(f"Cut off time: {T_ms[cutoff_index]}")


                # Split the data into two parts
        X_dT_before_cutoff = X_dT[:cutoff_index]
        X_dT_after_cutoff = X_dT[cutoff_index:]


        memtest_negative_peak_index = np.argmin(X_dT_before_cutoff)
        memtest_positive_peak_index = np.argmax(X_dT_before_cutoff)
        memtest_negative_peak_current_dT = X_dT[memtest_negative_peak_index]
        memtest_positive_peak_current_dT = X_dT[memtest_positive_peak_index]

        memtest_negative_peak_current = X_pA[memtest_negative_peak_index +1]
        memtest_positive_peak_current = X_pA[memtest_positive_peak_index + 1]
        # print(f"Memtest negative peak current: {memtest_negative_peak_current} pA")
        # print(f"Memtest positive peak current: {memtest_positive_peak_current} pA")
        # print(f"Memtest negative peak current derivative: {memtest_negative_peak_current_dT} pA")
        # print(f"Memtest positive peak current derivative: {memtest_positive_peak_current_dT} pA")
        stim_positive_peak_index = np.argmax(X_dT_after_cutoff)
        stim_negative_peak_index = np.argmin(X_dT_after_cutoff)
        stim_positive_peak_time = T_ms[stim_positive_peak_index + cutoff_index]
        stim_negative_peak_time = T_ms[stim_negative_peak_index + cutoff_index]
        stim_positive_peak_current_dT = X_dT_after_cutoff[stim_positive_peak_index + cutoff_index]
        stim_negative_peak_current_dT = X_dT_after_cutoff[stim_negative_peak_index  + cutoff_index]
        stim_positive_peak_current = X_pA[stim_positive_peak_index+ cutoff_index + 1]
        stim_negative_peak_current = X_pA[stim_negative_peak_index + cutoff_index + 1]
        # print(f"Stim negative peak current: {stim_negative_peak_current} pA")
        # print(f"Stim positive peak current: {stim_positive_peak_current} pA")
        # print(f"Stim negative peak time: {stim_negative_peak_time} ms")
        # print(f"Stim positive peak time: {stim_positive_peak_time} ms")
        # print(f"Stim negative peak current derivative: {stim_negative_peak_current_dT} pA")
        # print(f"Stim positive peak current derivative: {stim_positive_peak_current_dT} pA")

        # get data between membrane test peaks
        memtest_data = self.data.loc[:cutoff_index, :]
        # get data between stimulus peaks
        stim_data = self.data.loc[cutoff_index:, :]
        return 

    def firing_frequency_analysis(self):
        """
        assess action poential frequency
        """
        if self.data is None:
            raise ValueError("Data has not been loaded. Please run read_and_convert_data() first.")
        return None
    

    def monoExp(self, x, m, t, b):
            return m * np.exp(-t * x) + b
    def  analyze_current_protocol(self):
        """
        Analyze the current protocol data to extract key parameters.
        """
        self.read_and_convert_data()


        return None
    
    def plot_data(self):
        """
        Plot the command data and response data.
        """
        if self.data is None:
            raise ValueError("Data has not been loaded. Please run read_and_convert_data() first.")

        time = self.data['Time (ms)']
        command = self.data['Command Current (pA)']
        command_dT = self.data['X_dT']
        response = self.data['Response Voltage (mV)']

        plt.figure(figsize=(10, 6))

        # Plot Command Voltage
        plt.subplot(2, 1, 1)
        plt.plot(time, command, label='Command Current (pA)', color='b')
        plt.xlabel('Time (ms)')
        plt.ylabel('Command Current (pA)')
        plt.title('Command Current vs Time')
        plt.grid(True)
        plt.legend()

        # Plot Response Current
        plt.subplot(2, 1, 2)
        plt.plot(time, response, label='Response Voltage (mV)', color='r')
        plt.xlabel('Time (ms)')
        plt.ylabel('Response Voltage (mV)')
        plt.title('Response Voltage vs Time')
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.show()   

example_file_path = r"C:\Users\sa-forest\OneDrive - Georgia Institute of Technology\Documents\Grad-school\Gatech\Fall2024\ForestLab\AD\9-30_pres_data\CurrentProtocol\CurrentProtocol_3_#dde4e8_60.0.csv"
if __name__ == "__main__":
    analyzer = CurrentProtocolAnalyzer(example_file_path)
    # results = analyzer.analyze_voltage_protocol()
    # print(results)
    analyzer.read_and_convert_data()
    analyzer.filter_data()
    analyzer.plot_data()
