import sys
import os
import csv
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import (
    QApplication, QLabel, QVBoxLayout, QWidget, QPushButton,
    QFileDialog, QShortcut, QHBoxLayout, QFrame, QSlider, QMessageBox, QSizePolicy
)
from PyQt5.QtGui import QPixmap, QColor, QKeySequence
from PyQt5.QtCore import Qt, QTimer
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import numpy as np
from collections import deque


def create_cylinder_mesh(radius=0.5, height=5.0, sectors=32):
    """
    Create vertices and faces for a cylinder.

    :param radius: Radius of the cylinder in cm.
    :param height: Height of the cylinder in cm.
    :param sectors: Number of sectors to approximate the circle.
    :return: vertices, faces
    """
    vertices = []
    faces = []

    # Bottom circle
    for i in range(sectors):
        angle = 2 * np.pi * i / sectors
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        z = 0
        vertices.append([x, y, z])

    # Top circle
    for i in range(sectors):
        angle = 2 * np.pi * i / sectors
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        z = height
        vertices.append([x, y, z])

    # Center points
    vertices.append([0, 0, 0])      # Bottom center
    vertices.append([0, 0, height]) # Top center

    bottom_center = len(vertices) - 2
    top_center = len(vertices) - 1

    # Side faces
    for i in range(sectors):
        next_i = (i + 1) % sectors
        # Bottom to top
        faces.append([i, next_i, sectors + next_i])
        faces.append([i, sectors + next_i, sectors + i])

    # Bottom faces
    for i in range(sectors):
        next_i = (i + 1) % sectors
        faces.append([bottom_center, next_i, i])

    # Top faces
    for i in range(sectors):
        next_i = (i + 1) % sectors
        faces.append([top_center, sectors + i, sectors + next_i])

    vertices = np.array(vertices)
    faces = np.array(faces)

    return vertices, faces


def create_box_mesh(width=5.0, depth=10.0, height=0.5):
    """
    Create vertices and faces for a rectangular prism.

    :param width: Width along the X-axis in cm.
    :param depth: Depth along the Y-axis in cm.
    :param height: Height along the Z-axis in cm.
    :return: vertices, faces
    """
    w = width / 2
    d = depth / 2
    h = height

    vertices = np.array([
        [-w, -d, 0],  # 0: Bottom Front Left
        [w, -d, 0],   # 1: Bottom Front Right
        [w, d, 0],    # 2: Bottom Back Right
        [-w, d, 0],   # 3: Bottom Back Left
        [-w, -d, h],  # 4: Top Front Left
        [w, -d, h],   # 5: Top Front Right
        [w, d, h],    # 6: Top Back Right
        [-w, d, h],   # 7: Top Back Left
    ])

    faces = np.array([
        [0, 1, 2], [0, 2, 3],  # Bottom
        [4, 5, 6], [4, 6, 7],  # Top
        [0, 1, 5], [0, 5, 4],  # Front
        [1, 2, 6], [1, 6, 5],  # Right
        [2, 3, 7], [2, 7, 6],  # Back
        [3, 0, 4], [3, 4, 7],  # Left
    ])

    return vertices, faces


def create_sphere_mesh(radius=0.5, sectors=32, stacks=16):
    """
    Create vertices and faces for a sphere.

    :param radius: Radius of the sphere in cm.
    :param sectors: Number of sectors (longitude divisions).
    :param stacks: Number of stacks (latitude divisions).
    :return: vertices, faces
    """
    vertices = []
    faces = []

    for stack in range(stacks + 1):
        phi = np.pi / 2 - stack * np.pi / stacks  # from pi/2 to -pi/2
        y = radius * np.sin(phi)
        r = radius * np.cos(phi)

        for sector in range(sectors + 1):
            theta = 2 * np.pi * sector / sectors
            x = r * np.cos(theta)
            z = r * np.sin(theta)
            vertices.append([x, y, z])

    for stack in range(stacks):
        for sector in range(sectors):
            first = stack * (sectors + 1) + sector
            second = first + sectors + 1

            faces.append([first, second, first + 1])
            faces.append([second, second + 1, first + 1])

    vertices = np.array(vertices)
    faces = np.array(faces)

    return vertices, faces


class GLViewWidgetWithGrid(gl.GLViewWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setBackgroundColor('k')  # Set background to black

        # Set initial isometric view
        self.restore_isometric_view()

        # Add grid to XY plane
        self.add_grid()

        # Add XYZ axes
        self.add_axes()

        # Initialize objects
        self.init_objects()

    def add_grid(self):
        # Create grid lines on the XY plane with millimeter squares
        grid_size = 100  # Define the extent of the grid (100 mm in each direction)
        step = 1  # 1 mm squares

        # Create lines parallel to X-axis
        for y in np.arange(-grid_size, grid_size + step, step):
            pts = np.array([[-grid_size, y, 0], [grid_size, y, 0]])
            plt = gl.GLLinePlotItem(pos=pts, color=(1, 1, 1, 1), width=1, antialias=True)
            self.addItem(plt)

        # Create lines parallel to Y-axis
        for x in np.arange(-grid_size, grid_size + step, step):
            pts = np.array([[x, -grid_size, 0], [x, grid_size, 0]])
            plt = gl.GLLinePlotItem(pos=pts, color=(1, 1, 1, 1), width=1, antialias=True)
            self.addItem(plt)

    def add_axes(self):
        # Define axis length
        axis_length = 50  # 50 mm

        # X-axis (Red)
        x_axis = gl.GLLinePlotItem(pos=np.array([[0, 0, 0], [axis_length, 0, 0]]),
                                   color=(1, 0, 0, 1), width=2, antialias=True)
        self.addItem(x_axis)

        # Y-axis (Green)
        y_axis = gl.GLLinePlotItem(pos=np.array([[0, 0, 0], [0, axis_length, 0]]),
                                   color=(0, 1, 0, 1), width=2, antialias=True)
        self.addItem(y_axis)

        # Z-axis (Blue)
        z_axis = gl.GLLinePlotItem(pos=np.array([[0, 0, 0], [0, 0, axis_length]]),
                                   color=(0, 0, 1, 1), width=2, antialias=True)
        self.addItem(z_axis)

    def restore_isometric_view(self):
        # Set camera elevation and azimuth for isometric view
        self.opts['azimuth'] = 45  # Rotate 45 degrees around Z-axis
        self.opts['elevation'] = 30  # Tilt 30 degrees from horizontal
        self.update()

    def init_objects(self):
        # Create Microscope as a cylinder
        cyl_vertices, cyl_faces = create_cylinder_mesh(radius=0.5, height=5.0, sectors=32)
        self.microscope_mesh = gl.GLMeshItem(vertexes=cyl_vertices, faces=cyl_faces,
                                            smooth=True, color=(1, 0, 0, 1), shader='shaded', drawEdges=False)
        self.addItem(self.microscope_mesh)

        # Create Stage as a box (rectangular prism)
        box_vertices, box_faces = create_box_mesh(width=5.0, depth=10.0, height=0.5)
        self.stage_mesh = gl.GLMeshItem(vertexes=box_vertices, faces=box_faces,
                                       smooth=False, color=(0, 1, 0, 1), shader='shaded', drawEdges=False)
        self.addItem(self.stage_mesh)

        # Create Pipette as a sphere
        sphere_vertices, sphere_faces = create_sphere_mesh(radius=0.5, sectors=32, stacks=16)
        self.pipette_mesh = gl.GLMeshItem(vertexes=sphere_vertices, faces=sphere_faces,
                                         smooth=True, color=(0, 0, 1, 1), shader='shaded', drawEdges=False)
        self.addItem(self.pipette_mesh)

    def update_objects(self, microscope_z, stage_pos, pipette_pos):
        """
        Update the positions of Stage and Pipette.

        :param microscope_z: Z-coordinate for Microscope (fixed X and Y at 0).
        :param stage_pos: (x, y) position for Stage in the XY-plane.
        :param pipette_pos: (x, y, z) position for Pipette.
        """
        # Update Microscope position (fixed X and Y, variable Z)
        self.microscope_mesh.resetTransform()
        self.microscope_mesh.translate(0, 0, microscope_z)

        # Update Stage position (only in XY-plane, Z is fixed at 0)
        self.stage_mesh.resetTransform()
        self.stage_mesh.translate(stage_pos[0], stage_pos[1], 0)

        # Update Pipette position (in XYZ)
        self.pipette_mesh.resetTransform()
        self.pipette_mesh.translate(pipette_pos[0], pipette_pos[1], pipette_pos[2])


class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('3D Movement Data Visualization')
        self.resize(1200, 900)  # Adjusted size for better visibility

        # Define scaling factor (micrometers to millimeters)
        self.scaling_factor = 1 / 1000  # 1 mm = 1,000 µm

        # Create main layout
        main_layout = QtWidgets.QVBoxLayout()
        self.setLayout(main_layout)

        # Create OpenGL view widget
        self.view = GLViewWidgetWithGrid()
        main_layout.addWidget(self.view, stretch=8)  # Allocate most space to 3D view

        # Create control panel layout
        control_panel = QFrame()
        control_panel.setFrameShape(QFrame.StyledPanel)
        control_panel.setFrameShadow(QFrame.Sunken)
        control_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        control_layout = QVBoxLayout()
        control_layout.setContentsMargins(10, 10, 10, 10)
        control_layout.setSpacing(10)
        control_panel.setLayout(control_layout)
        main_layout.addWidget(control_panel, stretch=2)  # Allocate less space

        # Information Pane
        self.info = QLabel("No data loaded.")
        self.info.setStyleSheet("color: white;")
        self.info.setWordWrap(True)
        self.info.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        control_layout.addWidget(self.info)

        # Timeline Slider
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMaximumHeight(30)
        self.slider.setTickPosition(QSlider.NoTicks)
        self.slider.valueChanged.connect(self.slider_changed)
        self.slider.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        control_layout.addWidget(self.slider)

        # Buttons Layout
        buttons_layout = QHBoxLayout()
        control_layout.addLayout(buttons_layout)

        # Load Data Button
        self.load_data_button = QPushButton("Load Movement Data")
        self.load_data_button.clicked.connect(self.load_movement_data)
        buttons_layout.addWidget(self.load_data_button)

        # Previous Button
        self.prev_button = QPushButton("Previous")
        self.prev_button.clicked.connect(self.show_previous_timepoint)
        buttons_layout.addWidget(self.prev_button)

        # Play/Pause Button
        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self.toggle_playback)
        buttons_layout.addWidget(self.play_button)

        # Next Button
        self.next_button = QPushButton("Next")
        self.next_button.clicked.connect(self.show_next_timepoint)
        buttons_layout.addWidget(self.next_button)

        # Restore View Button
        self.restore_view_button = QPushButton("Restore Isometric View")
        self.restore_view_button.clicked.connect(self.view.restore_isometric_view)
        buttons_layout.addWidget(self.restore_view_button)

        # Spacer to push buttons to the left
        buttons_layout.addStretch()

        # Set background color for control panel
        control_panel.setStyleSheet("""
            QFrame {
                background-color: #2e2e2e;
            }
            QPushButton {
                background-color: #4a4a4a;
                color: white;
                border: 1px solid #5a5a5a;
                padding: 5px 10px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #5a5a5a;
            }
            QSlider::groove:horizontal {
                height: 8px;
                background: #4a4a4a;
                border: 1px solid #5a5a5a;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #ffffff;
                border: 1px solid #5a5a5a;
                width: 14px;
                margin: -3px 0;
                border-radius: 7px;
            }
        """)

        # Set overall background to dark gray
        self.setStyleSheet("""
            QWidget {
                background-color: #2e2e2e;
                color: white;
            }
        """)

        # Initialize variables
        self.movement_data = []
        self.current_index = 0
        self.timer = QTimer()
        self.timer.setInterval(33)  # ~30 FPS
        self.timer.timeout.connect(self.advance_timepoint)
        self.is_playing = False

        # Keyboard Shortcuts
        self.shortcut_prev = QShortcut(QKeySequence(Qt.Key_Left), self)
        self.shortcut_prev.activated.connect(self.show_previous_timepoint)

        self.shortcut_next = QShortcut(QKeySequence(Qt.Key_Right), self)
        self.shortcut_next.activated.connect(self.show_next_timepoint)

    def load_movement_data(self):
        # Open file dialog to select CSV file
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Movement Data CSV", "",
                                                   "CSV Files (*.csv);;All Files (*)", options=options)
        if file_path:
            try:
                with open(file_path, mode='r') as file:
                    reader = csv.reader(file, delimiter=' ')
                    self.movement_data = []
                    for line in reader:
                        # Split by spaces and filter out empty strings
                        parts = [part for part in line if part]

                        if len(parts) < 7:
                            continue  # Skip invalid lines

                        try:
                            time_value = float(parts[0].split(':')[1])
                            st_x = float(parts[1].split(':')[1]) * self.scaling_factor
                            st_y = float(parts[2].split(':')[1]) * self.scaling_factor
                            st_z = float(parts[3].split(':')[1]) * self.scaling_factor  # Microscope Z
                            pi_x = float(parts[4].split(':')[1]) * self.scaling_factor
                            pi_y = float(parts[5].split(':')[1]) * self.scaling_factor
                            pi_z = float(parts[6].split(':')[1]) * self.scaling_factor

                            self.movement_data.append({
                                'time': time_value,
                                'microscope_z': -(st_z),       # Correctly map to microscope_z
                                'stage': (-st_x, -st_y),      # Only X and Y for stage
                                'pipette': (-pi_x, -pi_y, -(pi_z - 1.365))
                            })
                        except (IndexError, ValueError):
                            continue  # Skip lines with parsing errors

                if not self.movement_data:
                    QMessageBox.warning(self, "No Data", "No valid movement data found in the selected file.")
                    return

                # Zero the time by subtracting the first timestamp
                initial_time = self.movement_data[0]['time']
                for entry in self.movement_data:
                    entry['time'] -= initial_time

                self.current_index = 0
                self.slider.setMinimum(0)
                self.slider.setMaximum(len(self.movement_data) - 1)
                self.slider.setValue(self.current_index)

                self.update_view()

                self.info.setText(f"Loaded {len(self.movement_data)} movement entries.")

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load movement data:\n{e}")

    def show_previous_timepoint(self):
        if not self.movement_data:
            QMessageBox.warning(self, "No Data", "Please load movement data first.")
            return

        if self.current_index > 0:
            self.current_index -= 1
            self.slider.setValue(self.current_index)
            self.update_view()

    def show_next_timepoint(self):
        if not self.movement_data:
            QMessageBox.warning(self, "No Data", "Please load movement data first.")
            return

        if self.current_index < len(self.movement_data) - 1:
            self.current_index += 1
            self.slider.setValue(self.current_index)
            self.update_view()

    def toggle_playback(self):
        if not self.movement_data:
            QMessageBox.warning(self, "No Data", "Please load movement data first.")
            return

        if self.is_playing:
            self.timer.stop()
            self.play_button.setText("Play")
            self.is_playing = False
        else:
            self.timer.start()
            self.play_button.setText("Pause")
            self.is_playing = True

    def advance_timepoint(self):
        if not self.movement_data:
            return

        current_time = self.movement_data[self.current_index]['time']
        new_time = current_time + self.timer.interval() / 1000.0  # Convert ms to seconds

        # Find the next index with time >= new_time
        new_index = self.current_index
        while new_index < len(self.movement_data) and self.movement_data[new_index]['time'] < new_time:
            new_index += 1

        if new_index < len(self.movement_data):
            self.current_index = new_index
            self.slider.setValue(self.current_index)
            self.update_view()
        else:
            self.timer.stop()
            self.play_button.setText("Play")
            self.is_playing = False

    def slider_changed(self, value):
        if not self.movement_data:
            return

        self.current_index = value
        self.update_view()

    def update_view(self):
        if not self.movement_data:
            return

        data = self.movement_data[self.current_index]
        time = data['time']
        microscope_z = data['microscope_z']
        stage_x, stage_y = data['stage']
        pi_x, pi_y, pi_z = data['pipette']

        # Stage moves only in XY plane; Z is fixed at 0
        stage_pos = (stage_x, stage_y)

        # Pipette moves freely in XYZ
        pipette_pos = (pi_x, pi_y, pi_z)

        # Update objects in the 3D viewer
        self.view.update_objects(microscope_z, stage_pos, pipette_pos)

        # Update information pane with both millimeters and micrometers
        time_text = f"Time: {time:.2f} s"
        stage_mm = stage_pos
        pipette_mm = pipette_pos
        stage_um = tuple([coord * 1000 for coord in stage_mm])  # mm to µm
        pipette_um = tuple([coord * 1000 for coord in pipette_mm])  # mm to µm

        stage_text = f"Stage Position: (X: {stage_mm[0]:.3f} mm / {stage_um[0]:.2f} µm, " \
                     f"Y: {stage_mm[1]:.3f} mm / {stage_um[1]:.2f} µm, " \
                     f"Z: 0.000 mm / 0.00 µm)"
        pipette_text = f"Pipette Position: (X: {pipette_mm[0]:.3f} mm / {pipette_um[0]:.2f} µm, " \
                       f"Y: {pipette_mm[1]:.3f} mm / {pipette_um[1]:.2f} µm, " \
                       f"Z: {pipette_mm[2]:.3f} mm / {pipette_um[2]:.2f} µm)"
        self.info.setText(f"{time_text}\n{stage_text}\n{pipette_text}")


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
