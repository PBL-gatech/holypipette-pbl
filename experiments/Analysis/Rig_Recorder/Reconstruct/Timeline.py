# import os
# import csv
# from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog, QMessageBox, QShortcut
# from PyQt5.QtCore import Qt, QRect, QSize, QLineF
# from PyQt5.QtGui import QKeySequence, QPixmap, QPainter, QPen, QColor
# from PIL import Image
# import time
# from collections import deque
# import numpy as np

# class DataVisualizer(QMainWindow):
#     def __init__(self):
#         super().__init__()
#         self.graphs = {}
#         self.images = {}
#         self.current_timestamp = None
#         self.setup_ui()
#         self.setup_shortcuts()

#     def setup_ui(self):
#         self.setWindowTitle("Data Visualizer")
#         self.setGeometry(100, 100, 1200, 800)

#         main_widget = QWidget()
#         main_layout = QHBoxLayout()
        
#         # Sidebar
#         sidebar = QWidget()
#         sidebar_layout = QVBoxLayout()
        
#         self.load_button = QPushButton("Load Data")
#         self.load_button.clicked.connect(self.load_data)
#         sidebar_layout.addWidget(self.load_button)
        
#         self.prev_button = QPushButton("Previous")
#         self.prev_button.clicked.connect(self.show_previous)
#         sidebar_layout.addWidget(self.prev_button)
        
#         self.next_button = QPushButton("Next")
#         self.next_button.clicked.connect(self.show_next)
#         sidebar_layout.addWidget(self.next_button)
        
#         self.timestamp_label = QLabel("")
#         sidebar_layout.addWidget(self.timestamp_label)
        
#         self.filename_label = QLabel("")
#         sidebar_layout.addWidget(self.filename_label)
        
#         sidebar_layout.addStretch()
#         sidebar.setLayout(sidebar_layout)
#         sidebar.setFixedWidth(200)
        
#         main_layout.addWidget(sidebar)
        
#         # Graph display area
#         self.graph_widget = QWidget()
#         self.graph_widget.setFixedSize(800, 800)
#         main_layout.addWidget(self.graph_widget)
        
#         main_widget.setLayout(main_layout)
#         self.setCentralWidget(main_widget)

#     def setup_shortcuts(self):
#         # Create shortcuts for next and previous
#         self.shortcut_next = QShortcut(QKeySequence(Qt.Key_Right), self)
#         self.shortcut_next.activated.connect(self.show_next)

#         self.shortcut_prev = QShortcut(QKeySequence(Qt.Key_Left), self)
#         self.shortcut_prev.activated.connect(self.show_previous)

#     def load_data(self):
#         graph_dir = QFileDialog.getExistingDirectory(self, "Select Data Directory")
#         image_dir = os.path.join(graph_dir, 'camera_frames')
#         self.load_graphs(graph_dir)
#         self.load_images(image_dir)
#         if self.graphs and self.images:
#             graph_timestamps = list(self.graphs.keys())
#             image_timestamps = list(self.images.keys())
            
#             valid_timestamps = graph_timestamps + image_timestamps
            
#             if valid_timestamps:
#                 self.current_timestamp = min(valid_timestamps)
#             else:
#                 QMessageBox.critical(self, "Error", "No valid timestamps found in the data")
#                 self.current_timestamp = None
#             self.update_display()
#         else:
#             QMessageBox.critical(self, "Error", "No data loaded")

#     def load_graphs(self, directory):
#         print("loading graphs...")
#         pressure_deque = deque(maxlen=100)
#         resistance_deque = deque(maxlen=100)
#         for filename in os.listdir(directory):
#             if filename == "graph_recording.csv":
#                 filepath = os.path.join(directory, filename)
#                 with open(filepath, 'r') as file:
#                     reader = csv.reader(file)
#                     for row in reader:
#                         joined_row = ''.join(row)
#                         timestamp_str = joined_row.split("pressure:")[0].replace("timestamp:", "").strip()
#                         timestamp = float(timestamp_str)
                        
#                         if timestamp not in self.graphs:
#                             self.graphs[timestamp] = {}
                        
#                         # Extract pressure
#                         pressure_val = float(joined_row.split('pressure:')[1].split('resistance', 1)[0].strip())
#                         pressure_deque.append(pressure_val)
#                         self.graphs[timestamp]['pressure_deque'] = pressure_deque.copy()
                        
#                         # Extract resistance
#                         resistance_vals = joined_row.split('resistance:')[1].split('current:')[0].replace("[","").replace("]","")
#                         resistance_val = float(resistance_vals.strip().split()[0])  # Take the first value
#                         resistance_deque.append(resistance_val)
#                         self.graphs[timestamp]['resistance_deque'] = resistance_deque.copy()
                        
#                         # Extract time and current
#                         current_vals = joined_row.split('current:')[1].split('voltage:')[0]
#                         current_vals_list = [float(val.strip(']')) for val in current_vals.strip('[').split()]
#                         self.graphs[timestamp]['current'] = current_vals_list
#                         self.graphs[timestamp]['time'] = np.linspace(0, len(current_vals_list) - 1, len(current_vals_list))

#                         # Extract voltage
#                         voltage_vals = joined_row.split('voltage:')[1]
#                         voltage_vals_list = [float(val.strip(']')) for val in voltage_vals.strip('[').split()]
#                         self.graphs[timestamp]['voltage'] = voltage_vals_list
#         print("graphs loaded")

#     def load_images(self, directory):
#         print("loading images...")
#         for filename in os.listdir(directory):
#             if filename.endswith(('.png', '.jpg', '.jpeg', '.webp')):
#                 frame_number, timestamp_with_ext = filename.split('_')
#                 timestamp = float(timestamp_with_ext.rsplit('.', 1)[0])
#                 self.images[timestamp] = os.path.join(directory, filename)
#         print("images loaded")

#     def find_closest_timestamp(self, target, data):
#         return min(data.keys(), key=lambda x: abs(x - target))

#     def update_display(self):
#         pixmap = QPixmap(self.graph_widget.size())
#         pixmap.fill(Qt.white)
        
#         painter = QPainter(pixmap)
        
#         # Display graphs
#         graph_timestamp = self.find_closest_timestamp(self.current_timestamp, self.graphs)
#         graph_data = self.graphs[graph_timestamp]
        
#         # Pressure plot
#         self.draw_graph(painter, graph_data['pressure_deque'], QRect(50, 50, 350, 350), "Pressure", "Index", "Pressure")
        
#         # Resistance plot
#         self.draw_graph(painter, graph_data['resistance_deque'], QRect(450, 50, 350, 350), "Resistance", "Index", "Resistance")
        
#         # Current plot
#         self.draw_graph(painter, graph_data['current'], QRect(50, 450, 350, 350), "Current vs Time", "Time", "Current", x_data=graph_data['time'])

#         # Voltage plot
#         self.draw_graph(painter, graph_data['voltage'], QRect(450, 450, 350, 350), "Voltage vs Time", "Time", "Voltage", x_data=graph_data['time'])
        
#         # Display image
#         image_timestamp = self.find_closest_timestamp(self.current_timestamp, self.images)
#         img = Image.open(self.images[image_timestamp])
#         img = img.resize((350, 350))
#         img_qt = QPixmap.fromImage(img.toqimage())
#         painter.drawPixmap(QRect(450, 450, 350, 350), img_qt)
#         painter.drawText(QRect(450, 450, 350, 30), Qt.AlignCenter, f"Image at {image_timestamp}")
        
#         painter.end()
        
#         self.graph_widget.setPixmap(pixmap)
        
#         # Update labels with timestamp and filename
#         self.timestamp_label.setText(f"Current Timestamp: {time.ctime(self.current_timestamp)}\n(UNIX: {self.current_timestamp})")
#         self.filename_label.setText(f"File: {os.path.basename(self.images[image_timestamp])}")

#     def draw_graph(self, painter, y_data, rect, title, x_label, y_label, x_data=None):
#         if x_data is None:
#             x_data = range(len(y_data))
        
#         painter.setPen(QPen(Qt.black, 2))
#         painter.drawRect(rect)
        
#         # Draw title
#         painter.drawText(QRect(rect.left(), rect.top() - 20, rect.width(), 20), Qt.AlignCenter, title)
        
#         # Draw axes labels
#         painter.drawText(QRect(rect.left(), rect.bottom(), rect.width(), 20), Qt.AlignCenter, x_label)
#         painter.drawText(QRect(rect.left() - 40, rect.top(), 40, rect.height()), Qt.AlignCenter | Qt.AlignVCenter, y_label)
        
#         # Scale data
#         x_min, x_max = min(x_data), max(x_data)
#         y_min, y_max = min(y_data), max(y_data)
        
#         if x_max - x_min != 0:
#             x_scaled = [(x - x_min) / (x_max - x_min) * rect.width() for x in x_data]
#         else:
#             x_scaled = [rect.width() / 2] * len(x_data)
        
#         if y_max - y_min != 0:
#             y_scaled = [(y - y_min) / (y_max - y_min) * rect.height() for y in y_data]
#         else:
#             y_scaled = [rect.height() / 2] * len(y_data)
        
#         # Draw graph using QLineF
#         painter.setPen(QPen(Qt.red, 2))
#         for i in range(len(x_scaled) - 1):
#             x1 = rect.left() + x_scaled[i]
#             y1 = rect.bottom() - y_scaled[i]
#             x2 = rect.left() + x_scaled[i + 1]
#             y2 = rect.bottom() - y_scaled[i + 1]
#             painter.drawLine(QLineF(x1, y1, x2, y2))


#     def show_previous(self):
#         if self.current_timestamp:
#             all_timestamps = sorted(set(self.graphs.keys()) | set(self.images.keys()))
#             current_index = all_timestamps.index(self.current_timestamp)
#             if current_index > 0:
#                 self.current_timestamp = all_timestamps[current_index - 1]
#                 self.update_display()

#     def show_next(self):
#         if self.current_timestamp:
#             all_timestamps = sorted(set(self.graphs.keys()) | set(self.images.keys()))
#             current_index = all_timestamps.index(self.current_timestamp)
#             if current_index < len(all_timestamps) - 1:
#                 self.current_timestamp = all_timestamps[current_index + 1]
#                 self.update_display()

# if __name__ == "__main__":
#     app = QApplication([])
#     visualizer = DataVisualizer()
#     visualizer.show()
#     app.exec_()


# import sys
# import os
# from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QPushButton, QFileDialog, QSlider, QShortcut
# from PyQt5.QtGui import QPixmap
# from PyQt5.QtCore import Qt

# class ImageViewer(QMainWindow):
#     def __init__(self):
#         super().__init__()
#         self.setWindowTitle("Image Viewer")
#         self.setGeometry(100, 100, 1000, 700)
        
#         # Main layout
#         self.main_layout = QVBoxLayout()
        
#         # Label to display the current image file name
#         self.filename_label = QLabel("No image loaded")
#         self.filename_label.setAlignment(Qt.AlignCenter)
#         self.main_layout.addWidget(self.filename_label)
        
#         # Image display area
#         self.image_label = QLabel()
#         self.image_label.setAlignment(Qt.AlignCenter)
#         self.main_layout.addWidget(self.image_label)
        
#         # Timeline slider
#         self.timeline_slider = QSlider(Qt.Horizontal)
#         self.timeline_slider.setMinimum(0)
#         self.timeline_slider.setTickPosition(QSlider.TicksBelow)
#         self.timeline_slider.setTickInterval(1)
#         self.timeline_slider.valueChanged.connect(self.slider_changed)
#         self.main_layout.addWidget(self.timeline_slider)
        
#         # Label to display the current time on the scrubber
#         self.time_label = QLabel("Time: 0")
#         self.time_label.setAlignment(Qt.AlignCenter)
#         self.main_layout.addWidget(self.time_label)
        
#         # Button to select directory
#         self.open_dir_button = QPushButton("Open Directory")
#         self.open_dir_button.clicked.connect(self.open_directory)
#         self.main_layout.addWidget(self.open_dir_button)
        
#         # Set the main widget
#         self.main_widget = QWidget()
#         self.main_widget.setLayout(self.main_layout)
#         self.setCentralWidget(self.main_widget)
        
#         # Initialize image list
#         self.image_paths = []
#         self.current_image_index = -1
        
#         # Keyboard shortcuts
#         self.shortcut_left = QShortcut(Qt.Key_Left, self)
#         self.shortcut_left.activated.connect(self.show_previous_image)
#         self.shortcut_right = QShortcut(Qt.Key_Right, self)
#         self.shortcut_right.activated.connect(self.show_next_image)
    
#     def open_directory(self):
#         directory = QFileDialog.getExistingDirectory(self, "Open Directory", "")
#         if directory:
#             camera_frames_dir = os.path.join(directory, 'camera_frames')
#             if os.path.exists(camera_frames_dir):
#                 self.load_images_from_directory(camera_frames_dir)
#             else:
#                 self.filename_label.setText("No camera_frames directory found in the selected folder.")
    
#     def load_images_from_directory(self, directory):
#         self.image_paths = [os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp'))]
#         self.image_paths.sort(key=self.extract_timestamp)  # Sort images by timestamp

#         if self.image_paths:
#             first_timestamp = self.extract_timestamp(self.image_paths[0])
#             last_timestamp = self.extract_timestamp(self.image_paths[-1])
#             self.duration = last_timestamp - first_timestamp
#             self.timeline_slider.setMaximum(self.duration)
#             self.timeline_slider.setTickInterval(max(1, self.duration // 10))
#             self.current_image_index = 0
#             self.display_image(0)
    
#     def extract_timestamp(self, filepath):
#         try:
#             filename = os.path.basename(filepath)
#             timestamp_str = filename.split('_')[1].split('.')[0]
#             return int(timestamp_str)
#         except (IndexError, ValueError):
#             return 0
    
#     def display_image(self, index):
#         if index >= 0 and index < len(self.image_paths):
#             pixmap = QPixmap(self.image_paths[index])
#             # Adjust the image display to fit within the window while keeping the aspect ratio
#             self.image_label.setPixmap(pixmap.scaled(1280, 1280, Qt.KeepAspectRatio))
#             self.setWindowTitle(f"Image Viewer - {os.path.basename(self.image_paths[index])}")
#             self.filename_label.setText(f"Current Image: {os.path.basename(self.image_paths[index])}")
#             self.timeline_slider.setValue(self.calculate_slider_value(index))
#             self.update_time_label(self.calculate_slider_value(index))
        
#     def calculate_slider_value(self, index):
#         first_timestamp = self.extract_timestamp(self.image_paths[0])
#         current_timestamp = self.extract_timestamp(self.image_paths[index])
#         return current_timestamp - first_timestamp
    
#     def show_previous_image(self):
#         if self.current_image_index > 0:
#             self.current_image_index -= 1
#             self.display_image(self.current_image_index)
    
#     def show_next_image(self):
#         if self.current_image_index < len(self.image_paths) - 1:
#             self.current_image_index += 1
#             self.display_image(self.current_image_index)
    
#     def slider_changed(self, value):
#         closest_index = self.find_closest_image_index(value)
#         self.display_image(closest_index)
    
#     def find_closest_image_index(self, slider_value):
#         first_timestamp = self.extract_timestamp(self.image_paths[0])
#         target_timestamp = first_timestamp + slider_value
#         closest_index = min(range(len(self.image_paths)), key=lambda i: abs(self.extract_timestamp(self.image_paths[i]) - target_timestamp))
#         self.current_image_index = closest_index
#         return closest_index
    
#     def update_time_label(self, slider_value):
#         self.time_label.setText(f"Time: {slider_value}s")

# if __name__ == "__main__":
#     app = QApplication(sys.argv)
#     viewer = ImageViewer()
#     viewer.show()
#     sys.exit(app.exec_())


import sys
import os
import csv
import numpy as np
from collections import deque
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QHBoxLayout, QWidget, QPushButton, QFileDialog, QSlider, QShortcut
from PyQt5.QtGui import QPixmap, QPainter, QPen, QColor
from PyQt5.QtCore import Qt, QRectF

class ImageViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Viewer")
        self.setGeometry(100, 100, 1800, 900)
        
        # Main layout
        self.main_layout = QHBoxLayout()
        
        # Left layout for the image
        self.image_layout = QVBoxLayout()
        
        self.filename_label = QLabel("No image loaded")
        self.filename_label.setAlignment(Qt.AlignCenter)
        self.image_layout.addWidget(self.filename_label)
        
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_layout.addWidget(self.image_label)
        
        self.timeline_slider = QSlider(Qt.Horizontal)
        self.timeline_slider.setMinimum(0)
        self.timeline_slider.setTickPosition(QSlider.TicksBelow)
        self.timeline_slider.setTickInterval(1)
        self.timeline_slider.valueChanged.connect(self.slider_changed)
        self.image_layout.addWidget(self.timeline_slider)
        
        self.time_label = QLabel("Time: 0")
        self.time_label.setAlignment(Qt.AlignCenter)
        self.image_layout.addWidget(self.time_label)
        
        self.open_dir_button = QPushButton("Open Directory")
        self.open_dir_button.clicked.connect(self.open_directory)
        self.image_layout.addWidget(self.open_dir_button)
        
        self.main_layout.addLayout(self.image_layout, 2)
        
        # Right layout for the plots
        self.plot_layout = QVBoxLayout()
        
        self.voltage_plot = QLabel()
        self.current_plot = QLabel()
        self.pressure_plot = QLabel()
        self.resistance_plot = QLabel()

        self.plot_layout.addWidget(self.voltage_plot)
        self.plot_layout.addWidget(self.current_plot)
        self.plot_layout.addWidget(self.pressure_plot)
        self.plot_layout.addWidget(self.resistance_plot)

        self.main_layout.addLayout(self.plot_layout, 1)
        
        self.main_widget = QWidget()
        self.main_widget.setLayout(self.main_layout)
        self.setCentralWidget(self.main_widget)
        
        self.image_paths = []
        self.current_image_index = -1
        self.graphs = {}
        
        self.shortcut_left = QShortcut(Qt.Key_Left, self)
        self.shortcut_left.activated.connect(self.show_previous_image)
        self.shortcut_right = QShortcut(Qt.Key_Right, self)
        self.shortcut_right.activated.connect(self.show_next_image)

    def open_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Open Directory", "")
        if directory:
            camera_frames_dir = os.path.join(directory, 'camera_frames')
            if os.path.exists(camera_frames_dir):
                self.load_images_from_directory(camera_frames_dir)
                self.load_graphs(directory)
                print(f"Loaded {len(self.image_paths)} images and {len(self.graphs)} graph data points")
            else:
                self.filename_label.setText("No camera_frames directory found in the selected folder.")
    
    def load_images_from_directory(self, directory):
        self.image_paths = [os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp'))]
        self.image_paths.sort(key=self.extract_timestamp)

        if self.image_paths:
            first_timestamp = self.extract_timestamp(self.image_paths[0])
            last_timestamp = self.extract_timestamp(self.image_paths[-1])
            self.duration = last_timestamp - first_timestamp
            self.timeline_slider.setMaximum(self.duration)
            self.timeline_slider.setTickInterval(max(1, self.duration // 10))
            self.current_image_index = 0
            self.display_image(0)
    
    def load_graphs(self, directory):
        print("Loading graphs...")
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
                        
                        pressure_val = float(joined_row.split('pressure:')[1].split('resistance', 1)[0].strip())
                        pressure_deque.append(pressure_val)
                        self.graphs[timestamp]['pressure_deque'] = list(pressure_deque)
                        
                        resistance_vals = joined_row.split('resistance:')[1].split('current:')[0].replace("[","").replace("]","")
                        resistance_val = float(resistance_vals.strip().split()[0])
                        resistance_deque.append(resistance_val)
                        self.graphs[timestamp]['resistance_deque'] = list(resistance_deque)
                        
                        current_vals = joined_row.split('current:')[1].split('voltage:')[0]
                        current_vals_list = [float(val.strip(']')) for val in current_vals.strip('[').split()]
                        self.graphs[timestamp]['current'] = current_vals_list
                        self.graphs[timestamp]['time'] = list(range(len(current_vals_list)))

                        voltage_vals = joined_row.split('voltage:')[1]
                        voltage_vals_list = [float(val.strip(']')) for val in voltage_vals.strip('[').split()]
                        self.graphs[timestamp]['voltage'] = voltage_vals_list
        print(f"Graphs loaded. Total timestamps: {len(self.graphs)}")
        # testing for loaded data
        print("Sample of loaded graph data:")
        for timestamp in list(self.graphs.keys())[:3]:  # Print first 3 timestamps
            print(f"Timestamp {timestamp}:")
            for key, value in self.graphs[timestamp].items():
                print(f"  {key}: {value[:5]}...") 
            
    def extract_timestamp(self, filepath):
        try:
            filename = os.path.basename(filepath)
            timestamp_str = filename.split('_')[1].split('.')[0]
            return int(timestamp_str)
        except (IndexError, ValueError):
            return 0
    
    def display_image(self, index):
        if index >= 0 and index < len(self.image_paths):
            pixmap = QPixmap(self.image_paths[index])
            self.image_label.setPixmap(pixmap.scaled(800, 600, Qt.KeepAspectRatio))
            self.setWindowTitle(f"Image Viewer - {os.path.basename(self.image_paths[index])}")
            self.filename_label.setText(f"Current Image: {os.path.basename(self.image_paths[index])}")
            self.timeline_slider.setValue(self.calculate_slider_value(index))
            self.update_time_label(self.calculate_slider_value(index))
            self.update_plots(index)
    
    def calculate_slider_value(self, index):
        first_timestamp = self.extract_timestamp(self.image_paths[0])
        current_timestamp = self.extract_timestamp(self.image_paths[index])
        return current_timestamp - first_timestamp
    
    def show_previous_image(self):
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.display_image(self.current_image_index)
    
    def show_next_image(self):
        if self.current_image_index < len(self.image_paths) - 1:
            self.current_image_index += 1
            self.display_image(self.current_image_index)
    
    def slider_changed(self, value):
        closest_index = self.find_closest_image_index(value)
        self.display_image(closest_index)
    
    def find_closest_image_index(self, slider_value):
        first_timestamp = self.extract_timestamp(self.image_paths[0])
        target_timestamp = first_timestamp + slider_value
        closest_index = min(range(len(self.image_paths)), key=lambda i: abs(self.extract_timestamp(self.image_paths[i]) - target_timestamp))
        self.current_image_index = closest_index
        return closest_index
    
    def update_time_label(self, slider_value):
        self.time_label.setText(f"Time: {slider_value}s")

    def update_plots(self, index):
        timestamp = self.extract_timestamp(self.image_paths[index])
        if timestamp in self.graphs:
            graph_data = self.graphs[timestamp]
            
            self.plot_graph(self.voltage_plot, graph_data['time'], graph_data['voltage'], "Voltage", Qt.blue)
            self.plot_graph(self.current_plot, graph_data['time'], graph_data['current'], "Current", Qt.red)
            self.plot_graph(self.pressure_plot, range(len(graph_data['pressure_deque'])), graph_data['pressure_deque'], "Pressure", Qt.green)
            self.plot_graph(self.resistance_plot, range(len(graph_data['resistance_deque'])), graph_data['resistance_deque'], "Resistance", Qt.magenta)
        else:
            print(f"No graph data for timestamp {timestamp}")

    def plot_graph(self, plot_widget, x_data, y_data, title, color):
        if not x_data or not y_data:
            print(f"Empty data for {title} graph")
            return

        pixmap = QPixmap(400, 200)
        pixmap.fill(Qt.white)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)

        plot_rect = QRectF(40, 10, 350, 180)

        # Draw axes
        pen = QPen(Qt.black, 1)
        painter.setPen(pen)
        painter.drawLine(plot_rect.left(), plot_rect.bottom(), plot_rect.right(), plot_rect.bottom())
        painter.drawLine(plot_rect.left(), plot_rect.top(), plot_rect.left(), plot_rect.bottom())

        # Draw the graph
        pen = QPen(color, 2)
        painter.setPen(pen)
        
        x_min, x_max = min(x_data), max(x_data)
        y_min, y_max = min(y_data), max(y_data)
        x_range = x_max - x_min
        y_range = y_max - y_min if y_max > y_min else 1  # Prevent division by zero

        for i in range(len(x_data) - 1):
            x1 = plot_rect.left() + (x_data[i] - x_min) * plot_rect.width() / x_range
            y1 = plot_rect.bottom() - (y_data[i] - y_min) * plot_rect.height() / y_range
            x2 = plot_rect.left() + (x_data[i+1] - x_min) * plot_rect.width() / x_range
            y2 = plot_rect.bottom() - (y_data[i+1] - y_min) * plot_rect.height() / y_range
            painter.drawLine(int(x1), int(y1), int(x2), int(y2))

        # Draw the title
        painter.setPen(Qt.black)
        painter.drawText(QRectF(0, 0, pixmap.width(), 30), Qt.AlignCenter, title)

        painter.end()
        plot_widget.setPixmap(pixmap)
        plot_widget.setMinimumSize(400, 200)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = ImageViewer()
    viewer.show()
    sys.exit(app.exec_())