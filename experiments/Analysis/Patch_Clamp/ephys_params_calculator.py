import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize

# Step 1: Read in the data
def read_data(file_path):
    data = pd.read_csv(file_path, sep='\t', header=None, names=['Color', 'X', 'Y'])
    # Cleaning the data by splitting and removing unnecessary spaces
    data[['Color', 'X', 'Y']] = data['Color'].str.split(expand=True)
    data['X'] = pd.to_numeric(data['X'], errors='coerce')
    data['Y'] = pd.to_numeric(data['Y'], errors='coerce')
    # Dropping rows with NaN values if any
    data.dropna(inplace=True)
    # Convert X to ms and Y to pA
    data['X_ms'] = data['X'] * 1000  # converting seconds to milliseconds
    data['Y_pA'] = data['Y'] * 1e12  # converting amps to picoamps
    return data

def decay_filter(data):
    peak_index = data['Y_pA'].idxmax()
    peak_time = data.loc[peak_index, 'X_ms']
    peak_value = data.loc[peak_index, 'Y_pA']
    min_index = data['Y_pA'].idxmin()
    min_time = data.loc[min_index, 'X_ms']
    min_value = data.loc[min_index, 'Y_pA']
    print(f'Peak at {peak_time:.2f} ms with value {peak_value:.2f} pA')
    print(f'Minimum at {min_time:.2f} ms with value {min_value:.2f} pA')
    # Step 3: Extract the data between peaks
    sub_data = data[(data['X_ms'] >= peak_time) & (data['X_ms'] <= min_time)]
    # Step 4: Calculate the first numerical derivative
    sub_data['Y_derivative'] = sub_data['Y_pA'].diff() / sub_data['X_ms'].diff()
    # Step 5: Remove the section with sudden changes
    change_threshold = sub_data['Y_derivative'].quantile(0.01)  # Taking the 1st percentile as the threshold
    drop_index = sub_data[sub_data['Y_derivative'] < change_threshold].index[0]
    filtered_sub_data = sub_data.loc[:drop_index - 1]
    # Step 6: Plot the data with the extracted portion highlighted
    fig, axs = plt.subplots(2, 1, figsize=(12, 12))
    # Original plot with highlighted extracted portion
    for color in data['Color'].unique():
        subset = data[data['Color'] == color]
        axs[0].plot(subset['X_ms'], subset['Y_pA'], color=color, label=f'Data with color {color}')
    axs[0].axvline(x=peak_time, color='blue', linestyle='--', label=f'Positive Peak at {peak_time:.2f} ms')
    axs[0].axvline(x=min_time, color='green', linestyle='--', label=f'Negative Peak at {min_time:.2f} ms')
    axs[0].plot(filtered_sub_data['X_ms'], filtered_sub_data['Y_pA'], color='blue', label='Extracted Portion', linewidth=2)
    axs[0].set_xlabel('Time (ms)')
    axs[0].set_ylabel('Current (pA)')
    axs[0].set_title('Original Voltage Protocol Data Plot with Highlighted Extracted Portion')
    axs[0].legend()
    axs[0].grid(True)
    # Extracted plot
    for color in filtered_sub_data['Color'].unique():
        subset = filtered_sub_data[filtered_sub_data['Color'] == color]
        axs[1].plot(subset['X_ms'], subset['Y_pA'], color=color, label=f'Data with color {color}')
    axs[1].set_xlabel('Time (ms)')
    axs[1].set_ylabel('Current (pA)')
    axs[1].set_title('Extracted Data Between Peaks')
    axs[1].legend()
    axs[1].grid(True)
    # Adjusting x-axis to show only whole milliseconds for both plots
    for ax in axs:
        ax.set_xticks(range(int(data['X_ms'].min()), int(data['X_ms'].max()) + 1, 1))
    plt.tight_layout()
    plt.show()
    return filtered_sub_data

def monoExp(x, m, t, b):
    return m * np.exp(-t * x) + b

def optimizer(p0, data):
    print("Optimizing parameters...")
    try:
        params, _ = scipy.optimize.curve_fit(monoExp, data['X_ms'], data['Y_pA'], p0, maxfev=100000000)
        m, t, b = params
        print("Optimization successful")
        print("m: ", m)
        print("t: ", t)
        print("b: ", b)
        return m, t, b
    except Exception as e:
        print("Error:", e)
        print("Failed to optimize parameters")
        return None, None, None

def plotter(m, t, b, data):
    # plot the results
    plt.plot(data['X_ms'], data['Y_pA'], '.', label="data")
    plt.plot(data['X_ms'], monoExp(data['X_ms'], m, t, b), '--', label="fitted")
    plt.title("Fitted Exponential Curve")
    plt.xlabel('Time (ms)')
    plt.ylabel('Current (pA)')
    plt.legend()
    plt.show()

# Assuming file_path and initial parameters are provided
file_path = r'C:\Users\sa-forest\Downloads\VoltageProtocol_0_1719874436.622976.csv'

p0 = (670, 54.097, -48.3)

# Read the data
data = read_data(file_path)

# Filter the data
filtered_data = decay_filter(data)

# Optimize parameters
m, t, b = optimizer(p0, filtered_data)

# Plot the fitted curve
if m is not None and t is not None and b is not None:
    plotter(m, t, b, filtered_data)
    print(f"Y = {m} * e^(-{t} * x) + {b}")
