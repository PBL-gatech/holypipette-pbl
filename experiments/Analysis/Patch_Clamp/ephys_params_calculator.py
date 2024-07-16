import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
# inspired by the following blog post
# https://swharden.com/blog/2020-10-11-model-neuron-ltspice/

# Step 1: Read in the data
def read_data(file_path):
    data = pd.read_csv(file_path, sep='\t', header=None, names=['Color', 'X', 'Y'])
    #sort the data by X
    # Cleaning the data by splitting and removing unnecessary spaces
    data[['Color', 'X', 'Y']] = data['Color'].str.split(expand=True)
    data.dropna(inplace=True)
    data.sort_values(by='X', inplace=True)
    data['X'] = pd.to_numeric(data['X'], errors='coerce')
    # shift all the X values by the vlaue of the first X value
    print(data['X'].iloc[0])
    data['X'] = data['X'] - data['X'].iloc[0]
    print(data['X'].iloc[0])
    data['Y'] = pd.to_numeric(data['Y'], errors='coerce')
    # Convert X to ms and Y to pA
    data['X_ms'] = data['X'] * 1000  # converting seconds to milliseconds
    data['Y_pA'] = data['Y'] * 1e12 # converting picoamps / amps

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
    sub_data = data[(data['X_ms'] >= peak_time) & (data['X_ms'] <= min_time)].copy()  # Use copy to avoid SettingWithCopyWarning
    # Step 4: Calculate the first numerical derivative
    sub_data['Y_derivative'] = sub_data['Y_pA'].diff() / sub_data['X_ms'].diff()
    # Step 5: Remove the section with sudden changes
    if not sub_data.empty:
        change_threshold = sub_data['Y_derivative'].quantile(0.01)  # Taking the 1st percentile as the threshold
        drop_indices = sub_data[sub_data['Y_derivative'] < change_threshold].index
        if not drop_indices.empty:
            drop_index = drop_indices[0]
            filtered_sub_data = sub_data.loc[:drop_index - 1]
            if not filtered_sub_data.empty:
                peak_index = filtered_sub_data['Y_pA'].idxmax()
                peak_time = filtered_sub_data.loc[peak_index, 'X_ms']
                min_index = filtered_sub_data['Y_pA'].idxmin()
                min_time = filtered_sub_data.loc[min_index, 'X_ms']

                # sort the data by X_ms again
                # filtered_sub_data.sort_values(by='X_ms', inplace=True)

    plot_params = [peak_time, peak_index, min_time, min_index]

    return filtered_sub_data, plot_params


def pre_peak_filter(data, peak_index, peak_time):
    peak_value = data.loc[peak_index, 'Y_pA']
    print(f'Peak at {peak_time:.2f} ms with value {peak_value:.2f} pA')

    # Extract the data before the peak
    pre_peak_data = data[data['X_ms'] < peak_time].copy()

    # get the std of the data
    std = pre_peak_data['Y_pA'].std()
    print(f'Standard deviation of current before peak: {std:.2f} pA')

    # get the data under that standard deviation
    pre_peak_data = pre_peak_data[pre_peak_data['Y_pA'] < peak_value - 3 * std]
    # plot it
    plt.plot(pre_peak_data['X_ms'], pre_peak_data['Y_pA'], '.')
    plt.title("Data Before Peak")
    plt.xlabel('Time (ms)')
    plt.ylabel('Current (pA)')
    plt.show()

    # Recompute the mean with filtered derivative values
    mean_filtered_pre_peak = pre_peak_data['Y_pA'].mean()
    print(f'Filtered Average derivative before peak: {mean_filtered_pre_peak:.2f} pA/ms')

    # prepend a colu,mn with the values r
    pre_peak_data['Color'] = 'r'

    return pre_peak_data, mean_filtered_pre_peak

def post_peak_filter(input_data):
    # get the std of the data
    post_peak_data = input_data.copy()
    # deriviate the data
    post_peak_data['Y_derivative'] = post_peak_data['Y_pA'].diff() / post_peak_data['X_ms'].diff()
    std = post_peak_data['Y_pA'].std()
    print(f'Standard deviation of current AFTER peak: {std:.2f} pA')

    # get the data under that standard deviation
    # get the peak value
    peak_value = post_peak_data.loc[peak_index, 'Y_pA']
    post_peak_data = post_peak_data[post_peak_data['Y_pA'] < peak_value - 3 * std]
    # plot it
    plt.plot(post_peak_data['X_ms'], post_peak_data['Y_pA'], '.')
    plt.title("Data AFTER Peak")
    plt.xlabel('Time (ms)')
    plt.ylabel('Current (pA)')
    plt.show()

    # Recompute the mean with filtered derivative values
    mean_filtered_pre_peak = post_peak_data['Y_pA'].mean()
    print(f'Filtered Average derivative AFTER peak: {mean_filtered_pre_peak:.2f} pA/ms')

    # prepend a colu,mn with the values r
    post_peak_data['Color'] = 'r'

    return post_peak_data, mean_filtered_pre_peak


def manual_exponential_fit(data):
    # calculate the log of the data
    log_data = np.log(data['Y_pA'])
    #plot the log data
    plt.plot(data['X_ms'], log_data, '.')
    plt.title("Logarithm of the Data")
    plt.xlabel('Time (ms)')
    plt.ylabel('Log(Current) (pA)')
    plt.show()
    # calculate the linear fit
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(data['X_ms'], log_data)
    # calculate the exponential fit
    m = np.exp(intercept)
    t = -slope
    b = 0
    # print what the linear fit equation is in log form
    print(f"Y = {slope} * x + {intercept}")
    #plot the log data wiht the linear fit
    plt.plot(data['X_ms'], log_data, '.')
    plt.plot(data['X_ms'], slope * data['X_ms'] + intercept, '--')
    plt.title("Logarithm of the Data")
    plt.xlabel('Time (ms)')
    plt.ylabel('Log(Current) (pA)')
    plt.show()
    
    print(f"Y = {m} * e^(-{t} * x) + {b}")
    #plot the results together in log scale
    plt.plot(data['X_ms'], data['Y_pA'], '.', label="data")
    plt.plot(data['X_ms'], m * np.exp(-t * data['X_ms']) + b, '--', label="fitted")
    plt.title("Fitted Exponential Curve")

    return m, t, b


def filtered_plot(data, peak_time, min_time, filtered_sub_data, pre_filtered_data):
    #print data between peak
    # Step 6: Plot the data with the extracted portion highlighted
    fig, axs = plt.subplots(2, 1, figsize=(12, 12))
    # Original plot with highlighted extracted portion
    for color in data['Color'].unique():
        subset = data[data['Color'] == color]
        axs[0].plot(subset['X_ms'], subset['Y_pA'], color=color, label=f'Data with color {color}')

    axs[0].axvline(x=peak_time, color='blue', linestyle='--', label=f'Positive Peak at {peak_time:.2f} ms')
    axs[0].axvline(x=min_time, color='green', linestyle='--', label=f'Negative Peak at {min_time:.2f} ms')
    axs[0].plot(filtered_sub_data['X_ms'], filtered_sub_data['Y_pA'], color='blue', label='Extracted Portion', linewidth=2)
    axs[0].plot(pre_filtered_data['X_ms'], pre_filtered_data['Y_pA'], color='red', label='Extracted Portion', linewidth=2)
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


def frequency_plotter(input_data):
    #plot the fft of the data
    fft = np.fft.fft(input_data['Y_pA'])
    # what does the line below do?
    
    freq = np.fft.fftfreq(len(input_data['X_ms']), input_data['X_ms'].iloc[1] - input_data['X_ms'].iloc[0])
    plt.plot(freq, np.abs(fft))
    plt.title("Frequency of the Data")
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.show()


def monoExp(x, m, t, b):
    return m * np.exp(-t * x) + b


def optimizer(filtered_data):
    start = filtered_data['X_ms'].iloc[0]
    # shift the data to start at 0
    filtered_data['X_ms'] = filtered_data['X_ms'] - start
    print("Optimizing parameters...")
    p0 = (664, 0.24, 15)
    try:
        params, _ = scipy.optimize.curve_fit(monoExp, filtered_data['X_ms'], filtered_data['Y_pA'], maxfev=1000000,p0=p0)
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
# # Read the data
data = read_data(file_path)
# # Filter the data
filtered_data, plot_params = decay_filter(data)
# Plot the filtered data

peak_time = plot_params[0]
peak_index = plot_params[1]
min_time = plot_params[2]

print(plot_params)

# print(filtered_data)

pre_filtered_data, I_prev = pre_peak_filter(data, peak_time = peak_time, peak_index = peak_index)
print(f'Average current before peak: {I_prev:.2f} pA')

post_filtered_data, I_post = post_peak_filter(filtered_data)
print(f'Average current AFTER peak: {I_post:.2f} pA')

# print(pre_filtered_data)

# combine the pre_filtered_data and filtered_data

filtered_plot(filtered_sub_data = filtered_data, peak_time = plot_params[0], min_time = plot_params[2], data = data, pre_filtered_data=pre_filtered_data)
# # # Plot the frequency of the data
# frequency_plotter(filtered_data)

# Optimize parameters
m, t, b = optimizer(filtered_data)
# m, t, b = manual_exponential_fit(filtered_data)
# Plot the fitted curve
if m is not None and t is not None and b is not None:
    plotter(m, t, b, filtered_data)
    print(f"Y = {m} * e^(-{t} * x) + {b}")

#Calculate Cm, Rm, Tau, and Ra
tau = 1 / t
# get peak current using peak_index
I_peak = data.loc[peak_index, 'Y_pA']
# get the steady state current after the peak
I_ss = data.loc[peak_index + 1, 'Y_pA']
I_prev = pre_filtered_data['Y_pA'].mean()
I_d = I_peak 













##### example to check with exponential data
# example exponential data:
# xs = xs = np.arange(12)
# ys = np.array([304.08994, 229.13878, 173.71886, 135.75499,
#                111.096794, 94.25109, 81.55578, 71.30187, 
#                62.146603, 54.212032, 49.20715, 46.765743])
# data = pd.DataFrame({'X_ms': xs, 'Y_pA': ys})
# # plot the data to see what it looks like
# plt.plot(data['X_ms'], data['Y_pA'], '.')
# plt.title("Exponential Data")
# plt.xlabel('Time (ms)')
# plt.ylabel('Current (pA)')
# plt.show()
# # # now we can try to fit the data
# # m, t, b = manual_exponential_fit(data)
# m, t, b = optimizer(data)
# plotter(m, t, b, data)
# print(f"Y = {m} * e^(-{t} * x) + {b}")

