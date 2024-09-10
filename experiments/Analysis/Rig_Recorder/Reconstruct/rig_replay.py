import sys
import os
import csv
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QPushButton, QFileDialog, QShortcut, QHBoxLayout, QToolButton, QFrame, QSlider, QMessageBox
from PyQt5.QtGui import QPixmap, QColor
from PyQt5.QtCore import Qt
import pyqtgraph as pg
import ast
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QHBoxLayout, QFrame
from PyQt5.QtCore import Qt
from collections import deque

class Timeline(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Timeline")
        self.setGeometry(50, 50, 1600, 800)  # Adjusted width for both image and graph

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.main_layout = QVBoxLayout(self.central_widget)

        # Horizontal layout for image and graph
        self.top_layout = QHBoxLayout()

        # Set up the image label and frame
        self.image = QLabel("No Image loaded yet!")
        self.image.setAlignment(Qt.AlignCenter)
        self.image.setMinimumSize(800, 800)
        self.image.setMaximumSize(800, 800)

        self.image_frame = QFrame()
        frame_layout = QVBoxLayout(self.image_frame)
        frame_layout.addWidget(self.image)
        self.image_frame.setFrameShape(QFrame.StyledPanel)
        self.image_frame.setFrameShadow(QFrame.Sunken)


        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')

        # Add graph frame next to the image with 2x2 grid of graphs
        self.graph_frame = QFrame()
        self.graph_frame.setMinimumSize(900, 900)
        self.graph_frame.setMaximumSize(900, 900)
        self.graph_frame.setFrameShape(QFrame.StyledPanel)
        self.graph_frame.setFrameShadow(QFrame.Sunken)
        
        # Create a layout to hold the 2x2 graphs
        self.graph_layout = QVBoxLayout(self.graph_frame)

        # Create the GraphicsLayoutWidget for plotting
        self.graphics_layout = pg.GraphicsLayoutWidget()
        self.graph_layout.addWidget(self.graphics_layout)

        # Create 2x2 grid of plots
        self.plots = [
            self.graphics_layout.addPlot(row=0, col=0),
            self.graphics_layout.addPlot(row=0, col=1),
            self.graphics_layout.addPlot(row=1, col=0),
            self.graphics_layout.addPlot(row=1, col=1)
        ]
        self.plots[0].setTitle("voltage 1")
        self.plots[0].setLabels(left='Voltage (V)', bottom='Time (s)')
        self.plots[1].setTitle("current 2")
        self.plots[1].setLabels(left='Current (A)', bottom='Time (s)')
        self.plots[2].setTitle("pressure 3")
        self.plots[2].setLabels(left='Pressure (mBAR)', bottom='Time (s)')
        self.plots[3].setTitle("resistance 4")
        self.plots[3].setLabels(left='Resistance (Ohm)', bottom='Time (s)')

        # Add the image and graph frames to the top layout
        self.top_layout.addWidget(self.image_frame)
        self.top_layout.addWidget(self.graph_frame)

        # Add the top layout to the main layout
        self.main_layout.addLayout(self.top_layout)

        # Set up Timeline Scrubber
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMaximumHeight(20)  # Match the height of the label box
        self.slider.setTickPosition(QSlider.NoTicks)  # Remove tick marks
        self.slider.valueChanged.connect(self.slider_changed)

        self.slider_frame = QFrame()
        slider_frame_layout = QHBoxLayout(self.slider_frame)
        slider_frame_layout.addWidget(self.slider)
        self.slider_frame.setFrameShape(QFrame.StyledPanel)
        self.slider_frame.setFrameShadow(QFrame.Sunken)
        self.main_layout.addWidget(self.slider_frame)

        # Create the QLabel to show info text
        self.info = QLabel("No info yet!")
        self.info.setAlignment(Qt.AlignLeft)
        self.info.setAlignment(Qt.AlignTop)
        self.info.setFixedHeight(20)

        self.middle_layout = QVBoxLayout()

        # Create a frame to hold the middle layout
        self.info_frame = QFrame()
        info_frame_layout = QHBoxLayout(self.info_frame)
        info_frame_layout.addWidget(self.info)
        self.info_frame.setFrameShape(QFrame.StyledPanel)
        self.info_frame.setFrameShadow(QFrame.Sunken)

        # Add the info_frame to the middle layout
        self.middle_layout.addWidget(self.info_frame)

        self.buttons_layout = QHBoxLayout()
        # Buttons in desired order
        self.prev_image_button = QPushButton("Previous timepoint")
        self.prev_image_button.clicked.connect(self.show_previous_timepoint)
        self.buttons_layout.addWidget(self.prev_image_button)

        self.select_image_dir_button = QPushButton("Select Data Directory")
        self.select_image_dir_button.clicked.connect(self.open_directory)
        self.buttons_layout.addWidget(self.select_image_dir_button)

        self.next_image_button = QPushButton("Next timepoint")
        self.next_image_button.clicked.connect(self.show_next_timepoint)
        self.buttons_layout.addWidget(self.next_image_button)

        # Add the buttons layout to the middle layout
        self.middle_layout.addLayout(self.buttons_layout)

        # Add the middle layout to the main layout
        self.main_layout.addLayout(self.middle_layout)

        self.image_paths = []
        self.image_index = []
        self.timestamps = []
        self.first_index = 0
        self.last_index = 0
        self.current_index = 0
        self.first_timestamp = 0
        self.last_timestamp = 0
        self.current_timestamp = 0
        self.duration = 0
        self.movement_data = []
        self.graph_data = []
        self.directory = None
        self.pressure_deque = deque(maxlen=100) # add 100 value to the deque
        self.resistance_deque = deque(maxlen=100) # add 100 value to the deque
        self.direction = 1

        # Keyboard shortcuts
        self.shortcut_left = QShortcut(Qt.Key_Left, self)
        self.shortcut_left.activated.connect(self.show_previous_timepoint)
        self.shortcut_right = QShortcut(Qt.Key_Right, self)
        self.shortcut_right.activated.connect(self.show_next_timepoint)

    def open_directory(self):
        self.directory = QFileDialog.getExistingDirectory(self, "Open Directory", "")
        if self.directory:
            camera_frames_dir = os.path.join(self.directory, 'camera_frames')
            movement_file_path = os.path.join(self.directory, 'movement_recording.csv')
            graph_file_path = os.path.join(self.directory, 'graph_recording.csv')
            if os.path.exists(camera_frames_dir):
                self.load_images_from_directory(camera_frames_dir)
            else:
                self.image.setText("No camera_frames directory found in the selected folder.")
            if os.path.exists(movement_file_path):
                self.load_movement_data(movement_file_path)
                if self.movement_data:
                    self.info.setText(f"Loaded {len(self.movement_data)} movement data entries.")
            else:
                QMessageBox.warning(self, "No Movement Data Found", "The selected directory does not contain a valid movement_recording.csv file.")
            if os.path.exists(graph_file_path):
                self.load_graph_data(graph_file_path)
                if self.graph_data:
                    self.info.setText(f"Loaded {len(self.graph_data)} graph data entries,and {len(self.movement_data)} movement data entries.")
            else:
                QMessageBox.warning(self, "No Graph Data Found", "The selected directory does not contain a valid graph_recording.csv file.")

    def check_data_loaded(self):
        """Check if image paths are loaded before allowing navigation"""
        if not self.image_paths:
            QMessageBox.warning(self, "No Data Loaded", "Please load a directory with images first.")
            return False
        return True  # Return True if data is loaded
    
    def show_previous_timepoint(self):
        """Show the previous timepoint"""
        if not self.check_data_loaded():
            return  # Only proceed if data is loaded

        if self.current_index > 0:
            self.current_index -= 1  # Move one timepoint back
            self.slider.setValue(self.current_index)  # Update slider to reflect the current index
            self.update_view()

    def show_next_timepoint(self):
        """Show the next timepoint"""
        if not self.check_data_loaded():
            return  # Only proceed if data is loaded

        if self.current_index < len(self.image_paths) - 1:
            self.current_index += 1  # Move one timepoint forward
            self.slider.setValue(self.current_index)  # Update slider to reflect the current index
            self.update_view()

    def slider_changed(self):
        """Update timepoint based on the slider's position"""
        if not self.check_data_loaded():
            return  # Only proceed if data is loaded

        new_index = self.slider.value()  # Get the new index from the slider
        
        # Update the current index
        self.current_index = new_index

        # Update the view based on the slider's new value
        self.update_view()

    def update_view(self):
        """Update the displayed image, graphs, and info based on the current timestamp"""
        if 0 <= self.current_index < len(self.image_paths):
            # Display the selected image
            self.display_image(self.image_paths[self.current_index])
            
            # Update info and graphs
            self.update_info()
            self.update_graphs()

        else:
            print(f"Invalid index: {self.current_index}. The image paths list has {len(self.image_paths)} items.")


    def update_info(self):
        """Update the info label with current timepoint details"""
        if not self.check_data_loaded():
            return  # Only proceed if data is loaded
        timepoint = self.timestamps[self.current_index] - self.first_timestamp
        # print(f"image timepoint: {self.timestamps[self.current_index]}")

        # Find the closest movement data timepoint from dictionary to the current timepoint
        movement_index = next((i for i, data in enumerate(self.movement_data) if data['time'] >= self.timestamps[self.current_index]), None)

        if movement_index is not None:
            movement_data = self.movement_data[movement_index]
            stage_x, stage_y, stage_z = movement_data['stage']
            pipette_x, pipette_y, pipette_z = movement_data['pipette']
            # print(f"movement timepoint: {movement_data['time']}")
            self.info.setText(f"Time: {timepoint} seconds, Stage: ({stage_x}, {stage_y}, {stage_z}), Pipette: ({pipette_x}, {pipette_y}, {pipette_z})")
    
        else:
            self.info.setText(f"Time: {timepoint} seconds, No movement data available.")

    def update_graphs(self):
        """Update the graphs with the current timepoint data and manage deques based on the closest timestamp"""
        if not self.check_data_loaded():
            return  # Only proceed if data is loaded

        # Get the current timestamp for the current frame
        current_timestamp = self.timestamps[self.current_index]

        # Clear deques before repopulating
        self.pressure_deque.clear()
        self.resistance_deque.clear()

        # Find the closest graph_data entries for the current timestamp
        graph_index = next((i for i, data in enumerate(self.graph_data) if data['time'] >= current_timestamp), None)

        # If a matching entry is found, backtrack up to 99 previous entries
        if graph_index is not None:
            start_index = max(0, graph_index - 99)  # Make sure we don't go below 0
            
            # Loop through the 99 previous entries (or as many as available)
            for i in range(start_index, graph_index + 1):
                graph_data = self.graph_data[i]
                self.pressure_deque.append(graph_data['pressure'])
                self.resistance_deque.append(graph_data['resistance'])

        # Now that the deques have been updated, plot the values
        self.plot_graphs()

    def plot_graphs(self):
        """Plot the values from the deques and the main graph data"""
        # Clear the plots
        self.plots[0].clear()
        self.plots[1].clear()
        self.plots[2].clear()
        self.plots[3].clear()

        # Plot the new data (voltage and current are continuous lists)
        current_timestamp = self.timestamps[self.current_index]
        graph_index = next((i for i, data in enumerate(self.graph_data) if data['time'] >= current_timestamp), None)

        if graph_index is not None:
            graph_data = self.graph_data[graph_index]
            current = graph_data['current']
            voltage = graph_data['voltage']

            self.plots[0].plot(voltage, pen=pg.mkPen(color='b', width=2))  # Plot voltage
            self.plots[1].plot(current, pen=pg.mkPen(color='r', width=2))  # Plot current

        # Plot the deque values for pressure and resistance
        self.plots[2].plot(list(self.pressure_deque), pen=pg.mkPen(color='g', width=2))  # Pressure
        self.plots[3].plot(list(self.resistance_deque), pen=pg.mkPen(color='m', width=2))  # Resistance

    def load_images_from_directory(self, directory):
        "will load images from the directory"
        print(f"Loading images from {directory}")
        if not os.path.exists(directory):
            print(f"Directory {directory} does not exist")
            return
        else:
            self.image_paths = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.webp')]
        if not self.image_paths:
            print(f"No images found in {directory}")
            return
        else:
            print(f"Found {len(self.image_paths)} images")
            print(f"First image path: {self.image_paths[0]}")
            self.first_index,self.first_timestamp = self.extract_image_data(self.image_paths[0])
            print(f"First index: {self.first_index} and timestamp: {self.first_timestamp}")
            self.last_index, self.last_timestamp = self.extract_image_data(self.image_paths[-1])
            print(f"Last index: {self.last_index} and timestamp: {self.last_timestamp}")
            self.current_index = 0
            self.display_image(self.image_paths[self.current_index])
            self.duration = self.last_timestamp - self.first_timestamp
            print(f"Duration: {self.duration} seconds")
            slider_max = len(self.image_paths) - 1
            self.slider.setMaximum(slider_max)
            print(f"Slider max: {slider_max}, Image count: {len(self.image_paths)}")
            self.slider.repaint()


            self.slider.setSingleStep(1)
            self.slider.setMinimum(0)
            self.timestamps = [self.extract_image_data(f)[1] for f in self.image_paths]
            self.image_index = [self.extract_image_data(f)[0] for f in self.image_paths]
        return

    def load_movement_data(self, file_path):
        "will load movement data from the file"
        print(f"Loading movement data from {file_path}")
        self.movement_data.clear()
        try:
            with open(file_path, mode='r') as file:
                for line in file:
                    # Split by spaces and filter out empty strings
                    parts = [part for part in line.strip().split(' ') if part]
                    
                    # Check if we have the expected number of parts (7 values)
                    if len(parts) < 7:
                        QMessageBox.warning(self, "Data Error", f"Skipping invalid data line: {line}.")
                        continue
                    
                    try:
                        # Extract values using the correct format
                        time_value = float(parts[0].split(':')[1])
                        st_x = float(parts[1].split(':')[1])
                        st_y = float(parts[2].split(':')[1])
                        st_z = float(parts[3].split(':')[1])
                        pi_x = float(parts[4].split(':')[1])
                        pi_y = float(parts[5].split(':')[1])
                        pi_z = float(parts[6].split(':')[1])
                        
                        self.movement_data.append({
                            'time': time_value,
                            'stage': (st_x, st_y, st_z),
                            'pipette': (pi_x, pi_y, pi_z)
                        })

                    except (IndexError, ValueError) as e:
                        QMessageBox.warning(self, "Data Error", f"Skipping invalid data line: {line}. Error: {e}")
        except Exception as e:
            QMessageBox.critical(self, "Error Loading Movement Data", f"An error occurred while loading movement data: {e}")

    def load_graph_data(self, file_path):
        "will load graph data from the file"
        print(f"Loading graph data from {file_path}")
        self.graph_data.clear()  # This will hold the list of dictionaries for each row

        try:
            with open(file_path, mode='r') as file:
                for line in file:
                    # Split by spaces and filter out empty strings
                    parts = [part for part in line.strip().split(':') if part]
                    
                    # Print first 3 parts for debugging
                    # print(parts[:4])
                    
                    # Parse each value
                    time_value = float(parts[1].split(' ')[0])
                    # print(f"Time value: {time_value} s")
                    pressure_value = float(parts[2].split(' ')[0])
                    # print(f"Pressure value: {pressure_value} mBAR")
                    resistance_value = float(parts[3].split(' ')[0])
                    # print(f"Resistance value: {resistance_value} MOhm")
                    
                    # Extract the current and voltage values as lists
                    current_data = parts[4].split('[')[-1].split(']')[0]
                    current_value = [float(x) for x in current_data.split(',')]
                    # print(f"Current values: {current_value[:5]} ...")  # Print only the first 5 for brevity
                    
                    voltage_data = parts[5].split('[')[-1].split(']')[0]
                    voltage_value = [float(x) for x in voltage_data.split(',')]
                    # print(f"Voltage values: {voltage_value[:5]} ...")  # Print only the first 5 for brevity

                    # Create a dictionary for this row
                    row_data = {
                        'time': time_value,
                        'pressure': pressure_value,
                        'resistance': resistance_value,
                        'current': current_value,
                        'voltage': voltage_value
                    }

                    # Append the dictionary to the list
                    self.graph_data.append(row_data)

        except Exception as e:
            QMessageBox.critical(self, "Error Loading Graph Data", f"An error occurred while loading graph data: {e}")

        # print(f"Loaded {len(self.graph_data)} graph data entries.")

    
    def extract_image_data(self, image_path):
        "will extract the image data based on file path"
        try:
            filename = os.path.basename(image_path)
            index_str = int(filename.split('_')[0])
            timestamp_str = float(filename.split('_')[1].rsplit('.', 1)[0])
            # print(f"Extracted index: {index_str} and timestamp: {timestamp_str}")
        except:
            print(f"Error extracting index and timestamp from {image_path}")
        return index_str, timestamp_str
    

    def display_image(self, image_path):
        """Display the image on the QLabel"""
        if not os.path.exists(image_path):
            QMessageBox.warning(self, "Error", f"Image file {image_path} does not exist.")
            return  # Exit if image path is invalid
        self.slider.blockSignals(True)
        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            QMessageBox.warning(self, "Error", "Unable to load image.")
        else:
            self.image.setPixmap(pixmap.scaled(self.image.size(), Qt.KeepAspectRatio))
            self.image.show()
        self.slider.blockSignals(False)

    

if __name__ == '__main__':
    app = QApplication(sys.argv)
    timeline = Timeline()
    timeline.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    app = QApplication(sys.argv)
    timeline = Timeline()
    timeline.show()
    sys.exit(app.exec_())


