import os
import csv
import matplotlib.pyplot as plt
from PIL import Image
import time
import tkinter as tk
from tkinter import filedialog, messagebox
from collections import deque
import numpy as np

class DataVisualizer:
    def __init__(self):
        self.graphs = {}
        self.images = {}
        self.current_timestamp = None
        self.setup_gui()

    def setup_gui(self):
        self.root = tk.Tk()
        self.root.title("Data Visualizer")
        self.load_button = tk.Button(self.root, text="Load Data", command=self.load_data)
        self.load_button.pack()
        self.prev_button = tk.Button(self.root, text="Previous", command=self.show_previous)
        self.prev_button.pack(side=tk.LEFT)
        self.next_button = tk.Button(self.root, text="Next", command=self.show_next)
        self.next_button.pack(side=tk.RIGHT)
        self.timestamp_label = tk.Label(self.root, text="")
        self.timestamp_label.pack()
        self.filename_label = tk.Label(self.root, text="")
        self.filename_label.pack()
        
        # Bind arrow keys for navigation
        self.root.bind('<Left>', lambda event: self.show_previous())
        self.root.bind('<Right>', lambda event: self.show_next())

    def load_data(self):
        graph_dir = filedialog.askdirectory(title="Select Graph Directory")
        image_dir = filedialog.askdirectory(title="Select Image Directory")
        self.load_graphs(graph_dir)
        self.load_images(image_dir)
        if self.graphs and self.images:
            graph_timestamps = list(self.graphs.keys())
            image_timestamps = list(self.images.keys())
            
            valid_timestamps = graph_timestamps + image_timestamps
            
            if valid_timestamps:
                self.current_timestamp = min(valid_timestamps)
            else:
                messagebox.showerror("Error", "No valid timestamps found in the data")
                self.current_timestamp = None
            self.update_display()
        else:
            messagebox.showerror("Error", "No data loaded")

    def load_graphs(self, directory):
        pressure_deque = deque(maxlen=100)
        resistance_deque = deque(maxlen=100)
        for filename in os.listdir(directory):
            if filename == "graph_recording.csv":
                filepath = os.path.join(directory, filename)
                with open(filepath, 'r') as file:
                    reader = csv.reader(file)
                    for row in reader:
                        joined_row = ''.join(row)
                        timestamp_str = joined_row.split("pressure:")[0].replace("timestamp:", "").strip()
                        timestamp = float(timestamp_str)
                        
                        if timestamp not in self.graphs:
                            self.graphs[timestamp] = {}
                        
                        # Extract pressure
                        pressure_val = float(joined_row.split('pressure:')[1].split('resistance', 1)[0].strip())
                        pressure_deque.append(pressure_val)
                        self.graphs[timestamp]['pressure_deque'] = pressure_deque.copy()
                        
                        # Extract resistance
                        resistance_vals = joined_row.split('resistance:')[1].split('current:')[0].replace("[","").replace("]","")
                        resistance_val = float(resistance_vals.strip().split()[0])  # Take the first value
                        resistance_deque.append(resistance_val)
                        self.graphs[timestamp]['resistance_deque'] = resistance_deque.copy()
                        
                        # Extract time and current
                        current_vals = joined_row.split('current:')[1].split('voltage:')[0]
                        current_vals_list = [float(val.strip(']')) for val in current_vals.strip('[').split()]
                        self.graphs[timestamp]['current'] = current_vals_list
                        self.graphs[timestamp]['time'] = np.linspace(0, len(current_vals_list) - 1, len(current_vals_list))

                        # Extract voltage
                        voltage_vals = joined_row.split('voltage:')[1]
                        voltage_vals_list = [float(val.strip(']')) for val in voltage_vals.strip('[').split()]
                        self.graphs[timestamp]['voltage'] = voltage_vals_list

    def load_images(self, directory):
        for filename in os.listdir(directory):
            if filename.endswith(('.png', '.jpg', '.jpeg', '.webp')):
                frame_number, timestamp_with_ext = filename.split('_')
                timestamp = float(timestamp_with_ext.rsplit('.', 1)[0])
                self.images[timestamp] = os.path.join(directory, filename)

    def find_closest_timestamp(self, target, data):
        return min(data.keys(), key=lambda x: abs(x - target))

    def update_display(self):
        plt.clf()
        # Display graphs
        graph_timestamp = self.find_closest_timestamp(self.current_timestamp, self.graphs)
        graph_data = self.graphs[graph_timestamp]
        
        # Pressure plot
        plt.subplot(221)
        indices = list(range(len(graph_data['pressure_deque'])))
        plt.plot(indices, list(graph_data['pressure_deque']))
        plt.xlabel('Index')
        plt.ylabel('Pressure')
        plt.title('Pressure Plot')
        
        # Resistance plot
        plt.subplot(222)
        resistance_indices = list(range(len(graph_data['resistance_deque'])))
        plt.plot(resistance_indices, list(graph_data['resistance_deque']))
        plt.title('Resistance')
        plt.ylabel('Resistance')
        
        # Current plot
        plt.subplot(223)
        plt.plot(graph_data['time'], graph_data['current'])
        plt.title('Current vs Time')
        plt.xlabel('Time')
        plt.ylabel('Current')
        
        # Display image
        image_timestamp = self.find_closest_timestamp(self.current_timestamp, self.images)
        img = Image.open(self.images[image_timestamp])
        plt.subplot(224)
        plt.imshow(img)
        plt.title(f"Image at {image_timestamp}")
        
        plt.tight_layout()
        plt.draw()
        
        # Update labels with timestamp and filename
        self.timestamp_label.config(text=f"Current Timestamp: {time.ctime(self.current_timestamp)} (UNIX: {self.current_timestamp})")
        self.filename_label.config(text=f"File: {os.path.basename(self.images[image_timestamp])}")

    def show_previous(self):
        if self.current_timestamp:
            all_timestamps = sorted(set(self.graphs.keys()) | set(self.images.keys()))
            current_index = all_timestamps.index(self.current_timestamp)
            if current_index > 0:
                self.current_timestamp = all_timestamps[current_index - 1]
                self.update_display()

    def show_next(self):
        if self.current_timestamp:
            all_timestamps = sorted(set(self.graphs.keys()) | set(self.images.keys()))
            current_index = all_timestamps.index(self.current_timestamp)
            if current_index < len(all_timestamps) - 1:
                self.current_timestamp = all_timestamps[current_index + 1]
                self.update_display()

    def run(self):
        plt.ion()
        plt.show()
        self.root.mainloop()

if __name__ == "__main__":
    visualizer = DataVisualizer()
    visualizer.run()