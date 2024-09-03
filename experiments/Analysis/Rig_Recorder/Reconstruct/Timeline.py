import sys
import os
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QPushButton, QFileDialog, QShortcut, QHBoxLayout, QToolButton, QFrame, QSlider, QMessageBox
from PyQt5.QtGui import QPixmap, QColor
from PyQt5.QtCore import Qt

class ImageViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Rig Replay")
        self.setGeometry(100, 100, 1200, 800)
        
        # Main layout
        self.main_layout = QVBoxLayout()

        # Top section (Image display area)
        self.image_label = QLabel("No Image Loaded")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(800, 600)
        
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
        self.bottom_layout = QHBoxLayout()
        
        # Time display
        self.time_label = QLabel("0.0s")
        self.time_label.setAlignment(Qt.AlignCenter)
        
        # Add dividing line above the time display
        self.time_frame = QFrame()
        self.time_frame.setLayout(QHBoxLayout())
        self.time_frame.layout().addWidget(self.time_label)
        self.time_frame.setFrameShape(QFrame.StyledPanel)
        self.time_frame.setFrameShadow(QFrame.Sunken)
        
        self.bottom_layout.addWidget(self.time_frame)
        
        # Hamburger menu button
        self.toggle_button = QToolButton()
        self.toggle_button.setText("☰")  # Hamburger menu icon
        self.toggle_button.setCheckable(True)
        self.toggle_button.clicked.connect(self.toggle_sidebar)
        self.bottom_layout.addWidget(self.toggle_button)
        
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
        
        # Keyboard shortcuts
        self.shortcut_left = QShortcut(Qt.Key_Left, self)
        self.shortcut_left.activated.connect(self.show_previous_image)
        self.shortcut_right = QShortcut(Qt.Key_Right, self)
        self.shortcut_right.activated.connect(self.show_next_image)
    
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
            if os.path.exists(camera_frames_dir):
                self.load_images_from_directory(camera_frames_dir)
            else:
                self.image_label.setText("No camera_frames directory found in the selected folder.")
    
    def load_images_from_directory(self, directory):
        self.image_paths = [os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp'))]
        self.image_paths.sort(key=self.extract_timestamp)  # Sort images by timestamp

        if self.image_paths:
            self.first_timestamp = self.extract_timestamp(self.image_paths[0])
            last_timestamp = self.extract_timestamp(self.image_paths[-1])
            self.duration = last_timestamp - self.first_timestamp
            self.timeline_slider.setMaximum(self.duration)
            self.current_image_index = 0
            self.display_image(0)
        else:
            QMessageBox.warning(self, "No Images Found", "The selected directory does not contain any valid images.")

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
        if index >= 0 and index < len(self.image_paths):
            pixmap = QPixmap(self.image_paths[index])
            self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio))
            self.setWindowTitle(f"Rig Replay - {os.path.basename(self.image_paths[index])}")
            time_elapsed = self.extract_timestamp(self.image_paths[index]) - self.first_timestamp  # Correct time calculation in seconds
            self.time_label.setText(f"{time_elapsed:.0f}s")  # Show time as seconds
            self.timeline_slider.setValue(self.calculate_slider_value(index))
    
    def calculate_slider_value(self, index):
        current_timestamp = self.extract_timestamp(self.image_paths[index])
        return current_timestamp - self.first_timestamp
    
    def slider_changed(self, value):
        if not self.check_images_loaded():
            return
        closest_index = self.find_closest_image_index(value)
        self.display_image(closest_index)
    
    def find_closest_image_index(self, slider_value):
        target_timestamp = self.first_timestamp + slider_value
        closest_index = min(range(len(self.image_paths)), key=lambda i: abs(self.extract_timestamp(self.image_paths[i]) - target_timestamp))
        self.current_image_index = closest_index
        return closest_index
    
    def show_previous_image(self):
        if not self.check_images_loaded():
            return
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.display_image(self.current_image_index)
    
    def show_next_image(self):
        if not self.check_images_loaded():
            return
        if self.current_image_index < len(self.image_paths) - 1:
            self.current_image_index += 1
            self.display_image(self.current_image_index)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = ImageViewer()
    viewer.show()
    sys.exit(app.exec_())
