import pandas as pd
import numpy as np
import scipy.optimize

# Step 1: Read in the data
def read_data(file_path):
    data = pd.read_csv(file_path, sep='\t', header=None, names=['Color', 'X', 'Y'])
    # Cleaning the data by splitting and removing unnecessary spaces
    data[['Color', 'X', 'Y']] = data['Color'].str.split(expand=True)
    data.dropna(inplace=True)
    data.sort_values(by='X', inplace=True)
    data['X'] = pd.to_numeric(data['X'], errors='coerce')
    # Shift all the X values by the value of the first X value
    data['X'] = data['X'] - data['X'].iloc[0]
    data['Y'] = pd.to_numeric(data['Y'], errors='coerce')
    # Convert X to ms and Y to pA
    data['X_ms'] = data['X'] * 1000  # converting seconds to milliseconds
    data['Y_pA'] = data['Y'] * 1e12 # converting picoamps / amps

    return data

def filter_data(data):
    # Decay filter part
    peak_index = data['Y_pA'].idxmax()
    peak_time = data.loc[peak_index, 'X_ms']

    min_index = data['Y_pA'].idxmin()
    min_time = data.loc[min_index, 'X_ms']

    # Extract the data between peaks
    sub_data = data[(data['X_ms'] >= peak_time) & (data['X_ms'] <= min_time)].copy()
    # Calculate the first numerical derivative
    sub_data['Y_derivative'] = sub_data['Y_pA'].diff() / sub_data['X_ms'].diff()
    # Remove the section with sudden changes
    change_threshold = sub_data['Y_derivative'].quantile(0.01)
    drop_indices = sub_data[sub_data['Y_derivative'] < change_threshold].index
    if not drop_indices.empty:
        drop_index = drop_indices[0]
        filtered_sub_data = sub_data.loc[:drop_index - 1]

    # Pre-peak filter part
    peak_value = data.loc[peak_index, 'Y_pA']
    pre_peak_data = data[data['X_ms'] < peak_time].copy()
    std = pre_peak_data['Y_pA'].std()
    pre_peak_data = pre_peak_data[pre_peak_data['Y_pA'] < peak_value - 3 * std]
    mean_filtered_pre_peak = pre_peak_data['Y_pA'].mean()

    # Post-peak filter part
    post_peak_data = filtered_sub_data.copy()
    post_peak_data['Y_derivative'] = post_peak_data['Y_pA'].diff() / post_peak_data['X_ms'].diff()
    std = post_peak_data['Y_pA'].std()
    post_peak_data = post_peak_data[post_peak_data['Y_pA'] < peak_value - 3 * std]
    mean_filtered_post_peak = post_peak_data['Y_pA'].mean()

    return filtered_sub_data, pre_peak_data, post_peak_data, [peak_time, peak_index, min_time, min_index], mean_filtered_pre_peak, mean_filtered_post_peak

def monoExp(x, m, t, b):
    return m * np.exp(-t * x) + b

def optimizer(filtered_data):
    start = filtered_data['X_ms'].iloc[0]
    # Shift the data to start at 0
    filtered_data['X_ms'] = filtered_data['X_ms'] - start
    p0 = (664, 0.24, 15)
    try:
        params, _ = scipy.optimize.curve_fit(monoExp, filtered_data['X_ms'], filtered_data['Y_pA'], maxfev=1000000, p0=p0)
        m, t, b = params
        return m, t, b
    except Exception as e:
        print("Error:", e)
        return None, None, None

def calc_param(tau, dV, I_peak, I_prev, I_ss):
    tau_s = tau / 1000  # Convert ms to seconds
    dV_V = dV * 1e-3  # Convert mV to V
    I_d = I_peak - I_prev  # in pA
    I_dss = I_ss - I_prev  # in pA
    I_d_A = I_d * 1e-12
    I_dss_A = I_dss * 1e-12

    # Calculate Access Resistance (R_a) in ohms
    R_a_Ohms = dV_V / I_d_A  # Ohms
    R_a_MOhms = R_a_Ohms * 1e-6  # Convert to MOhms

    # Calculate Membrane Resistance (R_m) in ohms
    R_m_Ohms = (dV_V - (R_a_Ohms * I_dss_A)) / I_dss_A  # Ohms
    R_m_MOhms = R_m_Ohms * 1e-6  # Convert to MOhms

    # Calculate Membrane Capacitance (C_m) in farads
    C_m_F = tau_s / (1/(1 / R_a_Ohms) + (1 / R_m_Ohms))  # Farads
    C_m_pF = C_m_F * 1e12  # Convert to pF

    return R_a_MOhms, R_m_MOhms, C_m_pF

# Assuming file_path and initial parameters are provided
file_path = r"C:\Users\sa-forest\Downloads\VoltageProtocol_0_1719874436.622976.csv"

# Read the data
data = read_data(file_path)

# Filter the data
filtered_data, pre_filtered_data, post_filtered_data, plot_params, I_prev, I_post = filter_data(data)
peak_time, peak_index, min_time, min_index = plot_params

# Optimize parameters
m, t, b = optimizer(filtered_data)
if m is not None and t is not None and b is not None:
    tau = 1 / t

    # Get peak current using peak_index
    I_peak = data.loc[peak_index, 'Y_pA']

    # Calculate parameters
    R_a_MOhms, R_m_MOhms, C_m_pF = calc_param(tau, 5, I_peak, I_prev, I_post)

    # Print the results
    # print(f"Access Resistance (R_a): {R_a_MOhms:.2f} MOhms")
    # print(f"Membrane Resistance (R_m): {R_m_MOhms:.2f} MOhms")
    # print(f"Membrane Capacitance (C_m): {C_m_pF:.2f} pF")
else:
    # print("Failed to optimize parameters.")
