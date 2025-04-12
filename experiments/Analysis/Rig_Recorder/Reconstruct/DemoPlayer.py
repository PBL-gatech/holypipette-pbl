import sys
import h5py
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QHBoxLayout, QVBoxLayout, QLabel,
    QGraphicsScene, QGraphicsView, QPushButton, QMessageBox
)
from PyQt5.QtGui import QImage, QPixmap, QPen
from PyQt5.QtCore import Qt, QTimer

class DemoPlayer(QWidget):
    def __init__(self, hdf5_path):
        super().__init__()
        self.hdf5_path = hdf5_path
        self.playing = True  # video plays by default
        
        # Try opening the HDF5 file.
        try:
            self.hdf5_file = h5py.File(self.hdf5_path, 'r')
        except OSError as e:
            QMessageBox.critical(self, "File Error", f"Could not open file {hdf5_path}:\n{e}")
            sys.exit(1)
        
        if 'data' not in self.hdf5_file:
            QMessageBox.critical(self, "File Error", f"The file {hdf5_path} does not contain a 'data' group.")
            sys.exit(1)
        
        self.demo_keys = sorted(self.hdf5_file['data'].keys())
        if not self.demo_keys:
            QMessageBox.critical(self, "Data Error", "No demos found in the HDF5 file!")
            sys.exit(1)
        
        self.current_demo_idx = 0
        self.current_frame = 0
        
        self.init_ui()
        self.load_demo(self.current_demo_idx)

    def init_ui(self):
        # Set fixed window size.
        self.setWindowTitle("Demo Viewer")
        self.resize(800, 500)

        main_layout = QHBoxLayout(self)

        # Left side: video display and navigation buttons.
        left_layout = QVBoxLayout()

        # Video display area (fixed to 400x400).
        self.video_label = QLabel(self)
        self.video_label.setFixedSize(500, 500)
        left_layout.addWidget(self.video_label)
        
        # Button layout for navigation.
        button_layout = QHBoxLayout()
        
        self.prev_button = QPushButton("Previous Demo", self)
        self.prev_button.clicked.connect(self.prev_demo)
        button_layout.addWidget(self.prev_button)
        
        self.play_pause_button = QPushButton("Pause", self)
        self.play_pause_button.clicked.connect(self.toggle_play_pause)
        button_layout.addWidget(self.play_pause_button)
        
        self.next_button = QPushButton("Next Demo", self)
        self.next_button.clicked.connect(self.next_demo)
        button_layout.addWidget(self.next_button)
        
        left_layout.addLayout(button_layout)
        main_layout.addLayout(left_layout)
        
        # Right side: resistance plot area (fixed to 400x400).
        self.scene = QGraphicsScene(self)
        self.plot_view = QGraphicsView(self.scene, self)
        self.plot_view.setFixedSize(500, 500)
        main_layout.addWidget(self.plot_view)

        self.setLayout(main_layout)

        # Timer for video frame updates (~33 fps).
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def load_demo(self, idx):
        """Load images (video) and resistance data for the given demo."""
        self.current_frame = 0
        demo_key = self.demo_keys[idx]
        demo_path = f'data/{demo_key}/obs'
        print(f"Loading demo: {demo_key}")  # Debug message
        
        try:
            self.images = self.hdf5_file[f'{demo_path}/camera_image'][:]
            self.resistance = self.hdf5_file[f'{demo_path}/resistance'][:]
        except KeyError as e:
            QMessageBox.critical(self, "Data Error", f"Missing dataset in {demo_path}:\n{e}")
            self.images = np.array([])
            self.resistance = np.array([])
            return

        # Validate the resistance data.
        if self.resistance.ndim != 1:
            QMessageBox.critical(
                self, 
                "Data Error", 
                f"Expected 'resistance' to be 1D, got shape {self.resistance.shape}"
            )
            self.resistance = np.array([])

        self.plot_resistance()

    def plot_resistance(self):
        """Draw the resistance data as a simple line plot with numeric labels."""
        self.scene.clear()
        if self.resistance.size == 0:
            self.scene.addText("No resistance data available")
            return

        # Define the drawing area.
        plot_w, plot_h = 400, 350  # Width and height for the plot area.
        max_res = float(np.max(self.resistance))
        min_res = float(np.min(self.resistance))
        points = len(self.resistance)

        if points < 2:
            self.scene.addText("Not enough data points for plot")
            return

        pen = QPen(Qt.blue)
        step = plot_w / (points - 1)
        diff = max_res - min_res
        if diff < 1e-6:
            diff = 1e-6  # Avoid division-by-zero

        # Draw the line plot.
        for i in range(points - 1):
            x1 = i * step
            y1 = plot_h - ((self.resistance[i] - min_res) / diff) * plot_h
            x2 = (i + 1) * step
            y2 = plot_h - ((self.resistance[i + 1] - min_res) / diff) * plot_h
            self.scene.addLine(x1, y1, x2, y2, pen)

        # Draw y-axis numeric labels (min, max, and intermediate values).
        num_y_labels = 6
        for i in range(num_y_labels):
            # Compute the position from the plot's coordinate system.
            y = plot_h - (i / (num_y_labels - 1)) * plot_h
            value = min_res + (i / (num_y_labels - 1)) * diff
            label = f"{value:.2f}"
            text_item = self.scene.addText(label)
            # Position the label to the left of the plot.
            text_item.setPos(-50, y - text_item.boundingRect().height() / 2)

        # Draw x-axis labels: first and last point.
        x_label_0 = self.scene.addText("0")
        x_label_0.setPos(0, plot_h + 5)
        x_label_last = self.scene.addText(f"{points-1}")
        x_label_last.setPos(plot_w - x_label_last.boundingRect().width(), plot_h + 5)

    def update_frame(self):
        """Advance the current frame and draw it in the video label."""
        if self.images.size == 0:
            return

        # Loop back to the first frame if we reach the end.
        if self.current_frame >= len(self.images):
            self.current_frame = 0

        try:
            img_array = self.images[self.current_frame]
        except IndexError:
            self.current_frame = 0
            return

        # Handle grayscale vs. color frames.
        if img_array.ndim == 2:
            # Grayscale.
            h_img, w_img = img_array.shape
            bytes_per_line = w_img
            q_img = QImage(img_array.data, w_img, h_img, bytes_per_line, QImage.Format_Grayscale8)
        elif img_array.ndim == 3 and img_array.shape[-1] in (3, 4):
            # Color.
            h_img, w_img, channels = img_array.shape
            bytes_per_line = w_img * channels
            q_format = QImage.Format_RGB888 if channels == 3 else QImage.Format_RGBA8888
            q_img = QImage(img_array.data, w_img, h_img, bytes_per_line, q_format)
        else:
            # Unexpected shape; do nothing.
            return

        pixmap = QPixmap.fromImage(q_img).scaled(
            self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.video_label.setPixmap(pixmap)
        self.current_frame += 1

    def toggle_play_pause(self):
        if self.playing:
            self.timer.stop()
            self.play_pause_button.setText("Play")
        else:
            self.timer.start(30)
            self.play_pause_button.setText("Pause")
        self.playing = not self.playing

    def next_demo(self):
        """Load the next demo and a new video."""
        self.current_demo_idx = (self.current_demo_idx + 1) % len(self.demo_keys)
        self.load_demo(self.current_demo_idx)

    def prev_demo(self):
        """Load the previous demo and a new video."""
        self.current_demo_idx = (self.current_demo_idx - 1) % len(self.demo_keys)
        self.load_demo(self.current_demo_idx)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Right:
            self.next_demo()
        elif event.key() == Qt.Key_Left:
            self.prev_demo()
        elif event.key() == Qt.Key_Space:
            self.toggle_play_pause()

    def closeEvent(self, event):
        if hasattr(self, 'hdf5_file'):
            self.hdf5_file.close()
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    data_path = r"C:\Users\sa-forest\Documents\GitHub\holypipette-pbl\experiments\Datasets\HEK_dataset_filter_inaction.hdf5"  # Replace with your actual file path.
    viewer = DemoPlayer(data_path)  # Replace with your actual file path.
    viewer.show()
    sys.exit(app.exec_())
