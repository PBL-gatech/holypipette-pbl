import sys
import os
import csv
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QPushButton,
    QFileDialog, QShortcut, QHBoxLayout, QFrame, QSlider, QMessageBox, QSizePolicy, QSpacerItem
)
from PyQt5.QtGui import QPixmap, QColor
from PyQt5.QtCore import Qt, QTimer
import pyqtgraph as pg
from collections import deque

class Timeline(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Timeline")
        self.setGeometry(50, 50, 1600, 800)  # Initial size; will be resizable

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.main_layout = QVBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(10, 10, 10, 10)  # Add some padding
        self.main_layout.setSpacing(10)  # Add spacing between elements

        # Horizontal layout for image and graph
        self.top_layout = QHBoxLayout()
        self.top_layout.setSpacing(10)  # Spacing between image and graph

        # Set up the image label and frame
        self.image = QLabel("No Image loaded yet!")
        self.image.setAlignment(Qt.AlignCenter)
        self.image.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)  # Allow expanding

        self.image_frame = QFrame()
        frame_layout = QVBoxLayout(self.image_frame)
        frame_layout.addWidget(self.image)
        self.image_frame.setFrameShape(QFrame.StyledPanel)
        self.image_frame.setFrameShadow(QFrame.Sunken)
        self.image_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)  # Allow expanding

        # Configure pyqtgraph
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')

        # Add graph frame next to the image with 2x2 grid of graphs
        self.graph_frame = QFrame()
        self.graph_frame.setFrameShape(QFrame.StyledPanel)
        self.graph_frame.setFrameShadow(QFrame.Sunken)
        self.graph_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)  # Allow expanding
        self.graph_frame.setMinimumSize(600, 400)  # Set minimum size for graph_frame

        # Create a layout to hold the 2x2 graphs
        self.graph_layout = QVBoxLayout(self.graph_frame)
        self.graph_layout.setContentsMargins(5, 5, 5, 5)
        self.graph_layout.setSpacing(5)

        # Create the GraphicsLayoutWidget for plotting
        self.graphics_layout = pg.GraphicsLayoutWidget()
        self.graphics_layout.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.graph_layout.addWidget(self.graphics_layout)

        # Create 2x2 grid of plots
        self.plots = [
            self.graphics_layout.addPlot(row=0, col=0),
            self.graphics_layout.addPlot(row=0, col=1),
            self.graphics_layout.addPlot(row=1, col=0),
            self.graphics_layout.addPlot(row=1, col=1)
        ]
        self.plots[0].setTitle("Voltage 1")
        self.plots[0].setLabels(left='Voltage (V)', bottom='Time (s)')
        self.plots[1].setTitle("Current 2")
        self.plots[1].setLabels(left='Current (A)', bottom='Time (s)')
        self.plots[2].setTitle("Pressure 3")
        self.plots[2].setLabels(left='Pressure (mBAR)', bottom='Time (s)')
        self.plots[3].setTitle("Resistance 4")
        self.plots[3].setLabels(left='Resistance (Ohm)', bottom='Time (s)')

        # Add image and graph frames to the top layout with stretch factors
        self.top_layout.addWidget(self.image_frame, stretch=1)  # Image takes 1 part
        self.top_layout.addWidget(self.graph_frame, stretch=1)  # Graphs take 1 part

        # Add the top layout to the main layout
        self.main_layout.addLayout(self.top_layout, stretch=8)  # Allocate most space to top_layout

        # Set up Timeline Scrubber
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMaximumHeight(30)  # Slightly increased height for better visibility
        self.slider.setTickPosition(QSlider.NoTicks)  # Remove tick marks
        self.slider.valueChanged.connect(self.slider_changed)
        self.slider.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)  # Expand horizontally

        self.slider_frame = QFrame()
        slider_frame_layout = QHBoxLayout(self.slider_frame)
        slider_frame_layout.setContentsMargins(0, 0, 0, 0)
        slider_frame_layout.addWidget(self.slider)
        self.slider_frame.setFrameShape(QFrame.StyledPanel)
        self.slider_frame.setFrameShadow(QFrame.Sunken)
        self.slider_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.main_layout.addWidget(self.slider_frame, stretch=1)  # Allocate less space

        # Create the QLabel to show info text
        self.info = QLabel("No info yet!")
        self.info.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.info.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)  # Expand horizontally

        self.middle_layout = QVBoxLayout()
        self.middle_layout.setSpacing(5)

        # Create a frame to hold the info label and buttons
        self.info_frame = QFrame()
        info_frame_layout = QHBoxLayout(self.info_frame)
        info_frame_layout.setContentsMargins(5, 5, 5, 5)
        info_frame_layout.setSpacing(10)

        # Add the info label to the left
        info_frame_layout.addWidget(self.info, stretch=1)

        # Create a horizontal layout for buttons aligned to the right
        self.buttons_layout = QHBoxLayout()
        self.buttons_layout.setSpacing(10)

        # Buttons in desired order
        self.prev_image_button = QPushButton("Previous timepoint")
        self.prev_image_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.prev_image_button.clicked.connect(self.show_previous_timepoint)
        self.buttons_layout.addWidget(self.prev_image_button)

        self.select_image_dir_button = QPushButton("Select Data Directory")
        self.select_image_dir_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.select_image_dir_button.clicked.connect(self.open_directory)
        self.buttons_layout.addWidget(self.select_image_dir_button)

        self.next_image_button = QPushButton("Next timepoint")
        self.next_image_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.next_image_button.clicked.connect(self.show_next_timepoint)
        self.buttons_layout.addWidget(self.next_image_button)

        self.toggle_video_button = QPushButton("Play")
        self.toggle_video_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.toggle_video_button.clicked.connect(self.toggle_video)
        self.buttons_layout.addWidget(self.toggle_video_button)

        # Add a spacer to push buttons to the right
        self.buttons_layout.addStretch()

        # Add the buttons layout to the info_frame layout
        info_frame_layout.addLayout(self.buttons_layout)

        # Add the info_frame to the middle layout
        self.middle_layout.addWidget(self.info_frame)

        # Add the middle layout to the main layout
        self.main_layout.addLayout(self.middle_layout, stretch=1)  # Allocate less space

        # Initialize required variables
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
        self.pressure_deque = deque(maxlen=100)  # Add 100 values to the deque
        self.resistance_deque = deque(maxlen=100)  # Add 100 values to the deque
        self.direction = 1

        self.timer = QTimer()
        self.interval = 33 / 1000  # 33ms for ~30 frames per second
        self.timer.setInterval(33)  # 33ms for ~30 frames per second
        self.timer.timeout.connect(self.advance_timepoint)

        # Keyboard shortcuts
        self.shortcut_left = QShortcut(Qt.Key_Left, self)
        self.shortcut_left.activated.connect(self.show_previous_timepoint)
        self.shortcut_right = QShortcut(Qt.Key_Right, self)
        self.shortcut_right.activated.connect(self.show_next_timepoint)

    def toggle_video(self):
        """Toggle play/pause functionality"""
        if not self.check_data_loaded():
            return  # Only proceed if data is loaded
        if self.timer.isActive():
            self.toggle_video_button.setText("Play")  # Update button to show "Play"
            self.timer.stop()
        else:
            self.toggle_video_button.setText("Pause")  # Update button to show "Pause"
            self.timer.start()

    def advance_timepoint(self):
        """Advance to the next time point"""
        if not self.check_data_loaded():
            return

        timepoint = self.timestamps[self.current_index]
        new_timepoint = timepoint + self.interval

        # Find the next closest timestamp
        new_index = next((i for i, t in enumerate(self.timestamps) if t >= new_timepoint), None)

        if new_index is not None:
            self.current_index = new_index
            self.slider.setValue(self.current_index)
            self.update_view()

    def open_directory(self):
        self.directory = QFileDialog.getExistingDirectory(self, "Open Directory", "")
        if self.directory:
            camera_frames_dir = os.path.join(self.directory, 'camera_frames')
            movement_file_path = os.path.join(self.directory, 'movement_recording.csv')
            graph_file_path = os.path.join(self.directory, 'graph_recording.csv')
            self.info.setText(f"Loading data from {self.directory}")
            if os.path.exists(camera_frames_dir):
                self.load_images_from_directory(camera_frames_dir)
            else:
                self.image.setText("No camera_frames directory found in the selected folder.")
            if os.path.exists(movement_file_path):
                self.load_movement_data(movement_file_path)
            else:
                QMessageBox.warning(self, "No Movement Data Found", "The selected directory does not contain a valid movement_recording.csv file.")
            if os.path.exists(graph_file_path):
                self.load_graph_data(graph_file_path)
            else:
                QMessageBox.warning(self, "No Graph Data Found", "The selected directory does not contain a valid graph_recording.csv file.")

            self.info.setText(f"Loaded images, movement data, and graph data entries!")

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

        # Find the closest movement data timepoint from dictionary to the current timepoint
        movement_index = next((i for i, data in enumerate(self.movement_data) if data['time'] >= self.timestamps[self.current_index]), None)

        if movement_index is not None:
            movement_data = self.movement_data[movement_index]
            stage_x, stage_y, stage_z = movement_data['stage']
            pipette_x, pipette_y, pipette_z = movement_data['pipette']
            self.info.setText(f"Time: {int(timepoint)} seconds, Stage: ({stage_x}, {stage_y}, {stage_z}), Pipette: ({pipette_x}, {pipette_y}, {pipette_z})")
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
            start_index = max(0, graph_index - 99)  # Ensure we don't go below 0

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
        for plot in self.plots:
            plot.clear()

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
        """Load images from the directory"""
        print(f"Loading images from {directory}")
        if not os.path.exists(directory):
            print(f"Directory {directory} does not exist")
            return
        else:
            self.image_paths = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.webp')]
        if not self.image_paths:
            print(f"No images found in {directory}")
            self.image.setText("No images found.")
            return
        else:
            print(f"Found {len(self.image_paths)} images")
            print(f"First image path: {self.image_paths[0]}")
            self.first_index, self.first_timestamp = self.extract_image_data(self.image_paths[0])
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
        """Load movement data from the file"""
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
        """Load graph data from the file"""
        print(f"Loading graph data from {file_path}")
        self.graph_data.clear()  # This will hold the list of dictionaries for each row

        try:
            with open(file_path, mode='r') as file:
                for line in file:
                    # Split by spaces and filter out empty strings
                    parts = [part for part in line.strip().split(':') if part]

                    # Parse each value
                    if len(parts) < 6:
                        # Not enough parts, skip this line
                        continue

                    try:
                        time_value = float(parts[1].split(' ')[0])
                        pressure_value = float(parts[2].split(' ')[0])
                        resistance_value = float(parts[3].split(' ')[0])

                        # Extract the current and voltage values as lists
                        current_data = parts[4].split('[')[-1].split(']')[0]
                        current_value = [float(x) for x in current_data.split(',') if x]

                        voltage_data = parts[5].split('[')[-1].split(']')[0]
                        voltage_value = [float(x) for x in voltage_data.split(',') if x]

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

                    except (IndexError, ValueError) as e:
                        # Skip lines with parsing errors
                        continue

        except Exception as e:
            QMessageBox.critical(self, "Error Loading Graph Data", f"An error occurred while loading graph data: {e}")

    def extract_image_data(self, image_path):
        """Extract the image data based on file path"""
        try:
            filename = os.path.basename(image_path)
            index_str = int(filename.split('_')[0])
            timestamp_str = float(filename.split('_')[1].rsplit('.', 1)[0])
        except Exception as e:
            print(f"Error extracting index and timestamp from {image_path}: {e}")
            index_str, timestamp_str = 0, 0.0  # Default values in case of error
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
            # Scale the pixmap to fit the label while maintaining aspect ratio
            scaled_pixmap = pixmap.scaled(
                self.image.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.image.setPixmap(scaled_pixmap)
            self.image.show()
        self.slider.blockSignals(False)

    def resizeEvent(self, event):
        """Handle window resize events to scale the image appropriately"""
        if self.image_paths and 0 <= self.current_index < len(self.image_paths):
            self.display_image(self.image_paths[self.current_index])
        super().resizeEvent(event)  # Ensure the base class resizeEvent is also called

if __name__ == '__main__':
    app = QApplication(sys.argv)
    timeline = Timeline()
    timeline.show()
    sys.exit(app.exec_())
