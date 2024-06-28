
import pandas as pd
import matplotlib.pyplot as plt

def load_and_prepare_data(filepath):
    # Load the CSV file with updated argument for handling bad lines
    data = pd.read_csv(filepath, on_bad_lines='skip')
    
    # Ensure columns are treated as strings
    data['Time(HH:MM:SS)'] = data['Time(HH:MM:SS)'].astype(str)
    data['Time(ms)'] = data['Time(ms)'].astype(str)

    # Extract relevant parts of messages and convert times
    data['FPS Value'] = data['Message'].str.extract(r'FPS .*: (\d+\.\d+)').astype(float)
    data['FPS Type'] = data['Message'].str.extract(r'FPS in ([\w\s]+):')
    
    # Concatenate time and milliseconds, then convert to datetime
    data['Time'] = pd.to_datetime(data['Time(HH:MM:SS)'] + '.' + data['Time(ms)'], format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')
    
    return data


def plot_fps_data_with_markers(data, title, program_start_time, recording_starts, recording_stops):
    # Ensure data is sorted by time
    data = data.sort_values('Time')

    # Setup figure
    plt.figure(figsize=(14, 7))
    colors = {'Camera': 'blue', 'Acquisition Thread': 'orange', 'LIVEFEED': 'green'}
    markers = {'Camera': 'o', 'Acquisition Thread': 'x', 'LIVEFEED': '^'}

    # Filter data from the first "FPS in LIVEFEED" and process each type separately
    livefeed_start_time = data[data['FPS Type'] == 'LIVEFEED']['Time'].min()
    data = data[data['Time'] >= livefeed_start_time]

    for fps_type, color in colors.items():
        subset = data[data['FPS Type'] == fps_type]
        if not subset.empty:
            # Calculate per-type statistics
            overall_average = subset['FPS Value'].mean()
            running_avg = subset['FPS Value'].rolling(window=20).mean()
            cumulative_avg = subset['FPS Value'].expanding().mean()
            print(f'{fps_type} FPS: {overall_average:.2f} fps')
            print(f'{fps_type} Running Avg: {running_avg.mean():.2f} fps')
            print(f'{fps_type} Cumulative Avg: {cumulative_avg.mean():.2f} fps')
            




            # Plotting
            plt.plot(subset['Time'], subset['FPS Value'], color=color, marker=markers[fps_type], linestyle='-', label=f'{fps_type} FPS')
            plt.plot(subset['Time'], running_avg, color=color, linestyle='--', label=f'{fps_type} Running Avg')
            plt.plot(subset['Time'], cumulative_avg, color=color, linestyle=':', label=f'{fps_type} Cumulative Avg')
            plt.axhline(y=overall_average, color=color, linestyle='-.', label=f'{fps_type} Avg: {overall_average:.2f} fps')

    # Add recording markers
    for start in recording_starts['Time']:
        if livefeed_start_time <= start <= data['Time'].max():
            plt.axvline(x=start, color='g', linestyle='--', label='Recording Started' if 'Recording Started' not in plt.gca().get_legend_handles_labels()[1] else "")
    for stop in recording_stops['Time']:
        if livefeed_start_time <= stop <= data['Time'].max():
            plt.axvline(x=stop, color='r', linestyle='--', label='Recording Stopped' if 'Recording Stopped' not in plt.gca().get_legend_handles_labels()[1] else "")

    # Finalize plot
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('FPS')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    directory = r'C:\Users\sa-forest\Documents\GitHub\holypipette-pbl\experiments\Analysis\Rig_Recorder'
    filepath = directory + '\logs.csv'

    data = load_and_prepare_data(filepath)
    program_starts = data[data['Message'].str.contains("Program Started", case=False, na=False)]
    recording_starts = data[data['Message'].str.contains("recording started", case=False, na=False)]
    recording_stops = data[data['Message'].str.contains("recording stopped", case=False, na=False)]

    # Process each program start segment
    if not program_starts.empty:
        for i in range(len(program_starts)):
            start_time = program_starts.iloc[i]['Time']
            if i < len(program_starts) - 1:
                end_time = program_starts.iloc[i + 1]['Time']
            else:
                end_time = data['Time'].max()
            segment_data = data[(data['Time'] >= start_time) & (data['Time'] <= end_time)]
            plot_fps_data_with_markers(segment_data, f'FPS Over Time After Program Start {i + 1}', start_time, recording_starts, recording_stops)

if __name__ == "__main__":
    main()
