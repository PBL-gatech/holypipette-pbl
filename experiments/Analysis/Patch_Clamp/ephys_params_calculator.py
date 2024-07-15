# capacitance calculation code


import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import find_peaks


def ephys_params(data):

    print('Rename columns for easier access')
    data.columns = ['color','time', 'current']

    print('Convert time and current to numeric values')
    data['time'] = pd.to_numeric(data['time'], errors='coerce')
    data['current'] = pd.to_numeric(data['current'], errors='coerce')

    print('Calculate the first and second derivatives of the current data')
    data['first_derivative'] = data['current'].diff()
    print(data['first_derivative'])
    data['second_derivative'] = data['first_derivative'].diff()
    print(data['second_derivative'])


    # Drop any rows with NaN values
    data = data.dropna()

    print('Find the largest peak in the current data')
    peaks, _ = find_peaks(data['current'])
    largest_peak_index = peaks[data['current'][peaks].argmax()]
    print('Largest peak index:', largest_peak_index)
    

    # Verify the location of the largest peak
    largest_peak_time = data['time'].iloc[largest_peak_index]
    print('Largest peak value:', data['current'].iloc[largest_peak_index])

    print('Plot the current data with the largest peak')
    plt.figure(figsize=(10, 6))
    plt.plot(data['time'], data['current'], color='green', label='Current')
    plt.axvline(x=largest_peak_time, color='black', linestyle='--', label='Largest Peak')

    # Labeling the plot
    plt.xlabel('Time (ms)')
    plt.ylabel('Current (picoamps)')
    plt.title('Current vs Time with Largest Peak')
    plt.legend()

    # Show the plot
    plt.show()

path_to_data = r'C:\Users\sa-forest\Downloads\VoltageProtocol_0_1719874436.622976.csv'
print('Load the data')
data = pd.read_csv(path_to_data, header=None, delim_whitespace=True)
# print(data)
# Calculate the electrophysiology parameters
ephys_params(data)