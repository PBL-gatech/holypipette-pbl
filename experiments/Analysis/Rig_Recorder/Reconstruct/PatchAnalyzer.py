import os
import sys
import csv
import pyqtgraph as pg
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QPushButton,
    QFileDialog, QShortcut, QHBoxLayout, QFrame, QSlider, QMessageBox, QSizePolicy, QSpacerItem,QComboBox
)
from PyQt5.QtGui import QPixmap, QColor
from PyQt5.QtCore import Qt, QTimer
from collections import deque

class PatchAnalyzer(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle('PatchAnalyzer')
        self.setGeometry(100, 100, 800, 800)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.main_layout = QVBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(10, 10, 10, 10)
        self.main_layout.setSpacing(10)

        # Configure pyqtgraph
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')

        # Add graph frame with 2x1 grid of graphs
        self.graph_frame = QFrame()
        self.graph_frame.setFrameShape(QFrame.StyledPanel)
        self.graph_frame.setFrameShadow(QFrame.Sunken)
        self.graph_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.graph_frame.setMinimumSize(600, 400)

        # Create a layout to hold the 2x1 graphs
        self.graph_layout = QVBoxLayout(self.graph_frame)
        self.graph_layout.setContentsMargins(5, 5, 5, 5)
        self.graph_layout.setSpacing(5)

        # Create the GraphicsLayoutWidget for plotting
        self.graphics_layout = pg.GraphicsLayoutWidget()
        self.graphics_layout.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.graph_layout.addWidget(self.graphics_layout)

        # Create 2x1 grid of plots
        self.plots = [
            self.graphics_layout.addPlot(row=0, col=0),
            self.graphics_layout.addPlot(row=1, col=0)
        ]

        # Set up vertical slider
        self.vertical_slider = QSlider(Qt.Vertical)
        self.vertical_slider.setTickPosition(QSlider.NoTicks)
        self.vertical_slider.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)

        # Create a horizontal layout to hold the graph frame and the vertical slider
        self.horizontal_layout = QHBoxLayout()
        self.horizontal_layout.addWidget(self.graph_frame)
        self.horizontal_layout.addWidget(self.vertical_slider)

        # Add the horizontal layout to the main layout
        self.main_layout.addLayout(self.horizontal_layout)

        # Create a single row of buttons
        self.buttons_layout = QHBoxLayout()
        self.buttons_layout.setSpacing(10)

        # Navigation buttons
        self.left_button = QPushButton("←")
        self.left_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.left_button.clicked.connect(self.show_previous_graphs)
        self.buttons_layout.addWidget(self.left_button)

        self.right_button = QPushButton("→")
        self.right_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.right_button.clicked.connect(self.show_next_graphs)
        self.buttons_layout.addWidget(self.right_button)

        # Add a small spacer between navigation and control buttons
        self.buttons_layout.addSpacing(20)

        # Control buttons
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
        # add a text label for the protocol number
        self.protocol_label = QLabel("Protocol: ")
        # Add a drop-down menu for protocol number
        self.protocol_dropdown = QComboBox()
        self.protocol_dropdown.addItems(["1", "2", "Protocol 3", "Protocol 4", "Protocol 5"])
        self.protocol_dropdown.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.buttons_layout.addWidget(self.protocol_dropdown)

        # Add a spacer to push buttons to the left
        self.buttons_layout.addStretch()

        # Add the buttons layout to the main layout
        self.main_layout.addLayout(self.buttons_layout)

        # Keyboard shortcuts
        self.shortcut_left = QShortcut(Qt.Key_Left, self)
        self.shortcut_left.activated.connect(self.show_previous_timepoint)
        self.shortcut_right = QShortcut(Qt.Key_Right, self)
        self.shortcut_right.activated.connect(self.show_next_timepoint)

    def show_previous_graphs(self):
        # Implement logic to show previous set of graphs
        pass

    def show_next_graphs(self):
        # Implement logic to show next set of graphs
        pass

    def show_previous_timepoint(self):
        # Implement logic to show previous timepoint
        pass

    def show_next_timepoint(self):
        # Implement logic to show next timepoint
        pass

    def open_directory(self):
        # Implement logic to open directory
        pass

    def toggle_video(self):
        # Implement logic to toggle video playback
        pass

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = PatchAnalyzer()
    window.show()
    sys.exit(app.exec_())