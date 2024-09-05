import sys
import os
import csv
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QPushButton, QFileDialog, QShortcut, QHBoxLayout, QToolButton, QFrame, QSlider, QMessageBox
from PyQt5.QtGui import QPixmap, QColor
from PyQt5.QtCore import Qt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class ImageViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Rig Replay")
        self.setGeometry(0, 0, 800, 800)
        self.setMaximumSize(800, 800)
        
        # Main layout
        self.main_layout = QVBoxLayout()

        # Top section (Image display area)
        self.image_label = QLabel("No Image Loaded")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(500, 500)
        self.image_label.setMaximumSize(800, 800)
        
        # Add dividing line above the bottom widget
        self.image_frame = QFrame()
        self.image_frame.setLayout(QVBoxLayout())
        self.image_frame.layout().addWidget(self.image_label)
        self.image_frame.setFrameShape(QFrame.StyledPanel)
        self.image_frame.setFrameShadow(QFrame.Sunken)
        
        self.main_layout.addWidget(self.image_frame)
        self.set_white_box()  # Set a white box as the placeholder
        
        # Scrubbing feature (Slider)
        self.timeline_slider = QSlider(Qt.Horizontal)
        self.timeline_slider.setMaximumHeight(40)  # Match the height of the label box
        self.timeline_slider.setMinimum(0)
        self.timeline_slider.setTickPosition(QSlider.NoTicks)  # Remove tick marks
        self.timeline_slider.valueChanged.connect(self.slider_changed)
        
        # Scrubbing frame
        self.scrubbing_frame = QFrame()
        self.scrubbing_frame.setLayout(QVBoxLayout())
        self.scrubbing_frame.setMaximumHeight(40)
        self.scrubbing_frame.layout().addWidget(self.timeline_slider)
        self.scrubbing_frame.setFrameShape(QFrame.StyledPanel)
        self.scrubbing_frame.setFrameShadow(QFrame.Sunken)
        
        self.main_layout.addWidget(self.scrubbing_frame)

        # Bottom section
        self.bottom_layout = QVBoxLayout()
        
        # Combine time, stage, and pipette information in one widget
        self.info_label = QLabel("Time: 0.0s\nStage (X, Y, Z): N/A, N/A, N/A\nPipette (X, Y, Z): N/A, N/A, N/A")
        self.info_label.setAlignment(Qt.AlignLeft)
        
        # Add dividing line above the info display
        self.info_frame = QFrame()
        self.info_frame.setLayout(QHBoxLayout())
        self.info_frame.layout().addWidget(self.info_label)
        self.info_frame.setFrameShape(QFrame.StyledPanel)
        self.info_frame.setFrameShadow(QFrame.Sunken)
        
        self.bottom_layout.addWidget(self.info_frame)
        
        # Add bottom layout to the main layout
        self.main_layout.addLayout(self.bottom_layout)
        
        # Sidebar layout
        self.sidebar_widget = QFrame()
        self.sidebar_widget.setLayout(QVBoxLayout())
        self.sidebar_widget.setFrameShape(QFrame.StyledPanel)
        self.sidebar_widget.setFrameShadow(QFrame.Sunken)

        # Buttons
        self.select_image_dir_button = QPushButton("Select Data Directory")
        self.select_image_dir_button.clicked.connect(self.open_directory)
        self.sidebar_widget.layout().addWidget(self.select_image_dir_button)

        self.prev_image_button = QPushButton("Previous timepoint")
        self.prev_image_button.clicked.connect(self.show_previous_image)
        self.sidebar_widget.layout().addWidget(self.prev_image_button)
        
        self.next_image_button = QPushButton("Next timepoint")
        self.next_image_button.clicked.connect(self.show_next_image)
        self.sidebar_widget.layout().addWidget(self.next_image_button)
        
        # Add spacing and stretch to align buttons to the top
        self.sidebar_widget.layout().addStretch()

        # Create a container layout to include the sidebar and main layout
        self.container_layout = QHBoxLayout()
        self.container_layout.addLayout(self.main_layout)
        self.container_layout.addWidget(self.sidebar_widget)

        # Set the container layout as the central widget
        self.main_widget = QWidget()
        self.main_widget.setLayout(self.container_layout)
        self.setCentralWidget(self.main_widget)
        
        # Initialize image list
        self.image_paths = []
        self.current_image_index = -1
        self.first_timestamp = 0

        # Initialize movement data
        self.movement_data = {}
        
        # Keyboard shortcuts
        self.shortcut_left = QShortcut(Qt.Key_Left, self)
        self.shortcut_left.activated.connect(self.show_previous_image)
        self.shortcut_right = QShortcut(Qt.Key_Right, self)
        self.shortcut_right.activated.connect(self.show_next_image)
    
        # 3D Plot
        self.figure = plt.figure()
        self.ax = self.figure.add_subplot(111, projection='3d')
        self.ax.set_title('Movement Data')
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        plt.ion()  # Interactive mode on
        
    def set_white_box(self):
        """Display a white box as a placeholder in the image label."""
        pixmap = QPixmap(self.image_label.size())
        pixmap.fill(QColor("white"))
        self.image_label.setPixmap(pixmap)

    def toggle_sidebar(self):
        """Toggle the sidebar visibility."""
        if self.sidebar_widget.isVisible():
            self.sidebar_widget.hide()
        else:
            self.sidebar_widget.show()

    def open_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Open Directory", "")
        if directory:
            camera_frames_dir = os.path.join(directory, 'camera_frames')
            movement_file_path = os.path.join(directory, 'movement_recording.csv')
            if os.path.exists(camera_frames_dir):
                self.load_images_from_directory(camera_frames_dir)
            else:
                self.image_label.setText("No camera_frames directory found in the selected folder.")
            if os.path.exists(movement_file_path):
                self.load_movement_data(movement_file_path)
            else:
                QMessageBox.warning(self, "No Movement Data Found", "The selected directory does not contain a valid movement_recording.csv file.")
    
    def load_images_from_directory(self, directory):
        self.image_paths = [os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp'))]
        self.image_paths.sort(key=self.extract_timestamp)  # Sort images by timestamp

        if self.image_paths:
            self.first_timestamp = self.extract_timestamp(self.image_paths[0])
            last_timestamp = self.extract_timestamp(self.image_paths[-1])
            self.duration = (last_timestamp - self.first_timestamp) / 1000  # Convert to seconds
            self.timeline_slider.setMaximum(int(self.duration * 1000))  # Set max to milliseconds
            self.current_image_index = 0
            self.display_image(0)
        else:
            QMessageBox.warning(self, "No Images Found", "The selected directory does not contain any valid images.")

    def find_closest_time(self, target_time):
        """Find the closest time in movement_data to the target time."""
        if not self.movement_data:
            return None
        return min(self.movement_data.keys(), key=lambda x: abs(x - target_time))

    def load_movement_data(self, file_path):
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
                        
                        # Store the cleaned data
                        self.movement_data[time_value] = {
                            'stage': (st_x, st_y, st_z),
                            'pipette': (pi_x, pi_y, pi_z)
                        }
                    except (IndexError, ValueError) as e:
                        QMessageBox.warning(self, "Data Error", f"Skipping invalid data line: {line}. Error: {e}")
        except Exception as e:
            QMessageBox.critical(self, "Error Loading Movement Data", f"An error occurred while loading movement data: {e}")

    def extract_timestamp(self, filepath):
        try:
            filename = os.path.basename(filepath)
            timestamp_str = filename.split('_')[1].split('.')[0]
            return int(timestamp_str)
        except (IndexError, ValueError):
            return 0
    
    def check_images_loaded(self):
        """Check if images are loaded. If not, prompt the user to open a directory."""
        if not self.image_paths:
            QMessageBox.warning(self, "No Images Loaded", "Please select a directory containing images first.")
            return False
        return True
    
    def display_image(self, index):
        """Display the image at the specified index and update movement data."""
        if index >= 0 and index < len(self.image_paths): 
            pixmap = QPixmap(self.image_paths[index]) 
            self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio)) 
            self.setWindowTitle(f"Rig Replay - {os.path.basename(self.image_paths[index])}") 
            time_elapsed = self.calculate_time_elapsed(index)
            self.update_info_label(time_elapsed)
            self.update_movement_plot(time_elapsed)
            
            # Set the slider position to match the current image time
            self.timeline_slider.blockSignals(True)  # Temporarily block signals to avoid recursion
            self.timeline_slider.setValue(int(time_elapsed * 1000))  # Convert to milliseconds
            self.timeline_slider.blockSignals(False)

    def update_info_label(self, time_elapsed):
        """Update the info label with time, stage, and pipette positions."""
        closest_time = self.find_closest_time(time_elapsed)
        if closest_time is not None:
            stage_pos = self.movement_data[closest_time]['stage']
            print(stage_pos)
            pipette_pos = self.movement_data[closest_time]['pipette']
            print(pipette_pos)
            self.info_label.setText(f"Time: {time_elapsed:.1f}s (Data: {closest_time:.1f}s)\n"
                                    f"Stage (X, Y, Z): {stage_pos[0]:.2f}, {stage_pos[1]:.2f}, {stage_pos[2]:.2f}\n"
                                    f"Pipette (X, Y, Z): {pipette_pos[0]:.2f}, {pipette_pos[1]:.2f}, {pipette_pos[2]:.2f}")
        else:
            self.info_label.setText(f"Time: {time_elapsed:.1f}s\n"
                                    "Stage (X, Y, Z): N/A, N/A, N/A\n"
                                    "Pipette (X, Y, Z): N/A, N/A, N/A")

    def slider_changed(self, value):
        """Handle slider movement to display the closest image and update movement data."""
        if not self.check_images_loaded():
            return
        time_elapsed = value / 1000  # Convert milliseconds to seconds from the start
        self.current_image_index = self.find_closest_image_index(time_elapsed)
        self.display_image(self.current_image_index)
        
        # Update the info label even when sliding between images
        time_elapsed = self.first_timestamp + value
        self.update_info_label(time_elapsed)

    def calculate_time_elapsed(self, index):
        """Calculate time elapsed in seconds from the start of the video."""
        return (self.extract_timestamp(self.image_paths[index]) - self.first_timestamp) / 1000  # Convert to seconds

    def find_closest_image_index(self, target_time):
        """Find the index of the image closest to the target time."""
        return min(range(len(self.image_paths)), 
                   key=lambda i: abs(self.calculate_time_elapsed(i) - target_time))

    def show_previous_image(self):
        """Navigate to the previous image and update the slider."""
        if not self.check_images_loaded():
            return
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.display_image(self.current_image_index)
            # Update slider to reflect the correct time for this image
            time_elapsed = self.calculate_time_elapsed(self.current_image_index)
            self.timeline_slider.setValue(int(time_elapsed * 1000))  # Convert to milliseconds

    def show_next_image(self):
        """Navigate to the next image and update the slider."""
        if not self.check_images_loaded():
            return
        if self.current_image_index < len(self.image_paths) - 1:
            self.current_image_index += 1
            self.display_image(self.current_image_index)
            # Update slider to reflect the correct time for this image
            time_elapsed = self.calculate_time_elapsed(self.current_image_index)
            self.timeline_slider.setValue(int(time_elapsed * 1000))  # Convert to milliseconds

    def update_movement_plot(self, time_elapsed):
        closest_time = self.find_closest_time(time_elapsed)
        if closest_time is not None:
            stage_pos = self.movement_data[closest_time]['stage']
            pipette_pos = self.movement_data[closest_time]['pipette']
            
            self.ax.clear()
            self.ax.scatter(*stage_pos, color='r', label='Stage')
            self.ax.scatter(*pipette_pos, color='b', label='Pipette')
            
            self.ax.set_title(f'Movement Data at {time_elapsed:.1f}s (Data: {closest_time:.1f}s)')
            self.ax.legend()
            plt.draw()
        else:
            self.ax.clear()
            self.ax.set_title('No movement data for this time')
            plt.draw()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = ImageViewer()
    viewer.show()
    sys.exit(app.exec_())
