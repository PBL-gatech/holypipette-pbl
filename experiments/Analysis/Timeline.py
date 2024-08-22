import os
import csv
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog, QMessageBox, QShortcut
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QKeySequence
from PIL import Image
import time
from collections import deque
import numpy as np

class DataVisualizer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.graphs = {}
        self.images = {}
        self.current_timestamp = None
        self.setup_ui()
        self.setup_shortcuts()

    def setup_ui(self):
        self.setWindowTitle("Data Visualizer")
        self.setGeometry(100, 100, 1200, 800)

        main_widget = QWidget()
        main_layout = QHBoxLayout()
        
        # Sidebar
        sidebar = QWidget()
        sidebar_layout = QVBoxLayout()
        
        self.load_button = QPushButton("Load Data")
        self.load_button.clicked.connect(self.load_data)
        sidebar_layout.addWidget(self.load_button)
        
        self.prev_button = QPushButton("Previous")
        self.prev_button.clicked.connect(self.show_previous)
        sidebar_layout.addWidget(self.prev_button)
        
        self.next_button = QPushButton("Next")
        self.next_button.clicked.connect(self.show_next)
        sidebar_layout.addWidget(self.next_button)
        
        self.timestamp_label = QLabel("")
        sidebar_layout.addWidget(self.timestamp_label)
        
        self.filename_label = QLabel("")
        sidebar_layout.addWidget(self.filename_label)
        
        sidebar_layout.addStretch()
        sidebar.setLayout(sidebar_layout)
        sidebar.setFixedWidth(200)
        
        main_layout.addWidget(sidebar)
        
        # Matplotlib figure
        self.figure, self.ax = plt.subplots(2, 2, figsize=(10, 8))
        self.canvas = FigureCanvas(self.figure)
        main_layout.addWidget(self.canvas)
        
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

    def setup_shortcuts(self):
        # Create shortcuts for next and previous
        self.shortcut_next = QShortcut(QKeySequence(Qt.Key_Right), self)
        self.shortcut_next.activated.connect(self.show_next)

        self.shortcut_prev = QShortcut(QKeySequence(Qt.Key_Left), self)
        self.shortcut_prev.activated.connect(self.show_previous)

    def load_data(self):
        graph_dir = QFileDialog.getExistingDirectory(self, "Select Graph Directory")
        image_dir = QFileDialog.getExistingDirectory(self, "Select Image Directory")
        self.load_graphs(graph_dir)
        self.load_images(image_dir)
        if self.graphs and self.images:
            graph_timestamps = list(self.graphs.keys())
            image_timestamps = list(self.images.keys())
            
            valid_timestamps = graph_timestamps + image_timestamps
            
            if valid_timestamps:
                self.current_timestamp = min(valid_timestamps)
            else:
                QMessageBox.critical(self, "Error", "No valid timestamps found in the data")
                self.current_timestamp = None
            self.update_display()
        else:
            QMessageBox.critical(self, "Error", "No data loaded")

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
        for ax in self.ax.flat:
            ax.clear()

        # Display graphs
        graph_timestamp = self.find_closest_timestamp(self.current_timestamp, self.graphs)
        graph_data = self.graphs[graph_timestamp]
        
        # Pressure plot
        indices = list(range(len(graph_data['pressure_deque'])))
        self.ax[0, 0].plot(indices, list(graph_data['pressure_deque']))
        self.ax[0, 0].set_xlabel('Index')
        self.ax[0, 0].set_ylabel('Pressure')
        self.ax[0, 0].set_title('Pressure Plot')
        
        # Resistance plot
        resistance_indices = list(range(len(graph_data['resistance_deque'])))
        self.ax[0, 1].plot(resistance_indices, list(graph_data['resistance_deque']))
        self.ax[0, 1].set_title('Resistance')
        self.ax[0, 1].set_ylabel('Resistance')
        
        # Current plot
        self.ax[1, 0].plot(graph_data['time'], graph_data['current'])
        self.ax[1, 0].set_title('Current vs Time')
        self.ax[1, 0].set_xlabel('Time')
        self.ax[1, 0].set_ylabel('Current')
        
        # Display image
        image_timestamp = self.find_closest_timestamp(self.current_timestamp, self.images)
        img = Image.open(self.images[image_timestamp])
        self.ax[1, 1].imshow(img)
        self.ax[1, 1].set_title(f"Image at {image_timestamp}")
        self.ax[1, 1].axis('off')
        
        self.figure.tight_layout()
        self.canvas.draw()
        
        # Update labels with timestamp and filename
        self.timestamp_label.setText(f"Current Timestamp: {time.ctime(self.current_timestamp)}\n(UNIX: {self.current_timestamp})")
        self.filename_label.setText(f"File: {os.path.basename(self.images[image_timestamp])}")

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

if __name__ == "__main__":
    app = QApplication([])
    visualizer = DataVisualizer()
    visualizer.show()
    app.exec_()