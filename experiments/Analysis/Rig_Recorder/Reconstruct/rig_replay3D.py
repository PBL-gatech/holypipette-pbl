import sys
import os
import csv
import bisect
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QPushButton,
    QFileDialog, QShortcut, QHBoxLayout, QFrame, QSlider, QMessageBox, QSizePolicy, QStackedWidget
)
from PyQt5.QtGui import QPixmap, QColor, QKeySequence
from PyQt5.QtCore import Qt, QTimer
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import numpy as np
from collections import deque

# ------------------- 3D Mesh Creation Functions -------------------

def create_cylinder_mesh(radius=0.5, height=5.0, sectors=32):
    """
    Create vertices and faces for a cylinder.
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
    Origin is set to the top center of the box.
    """
    w = width / 2
    d = depth / 2
    h = height

    vertices = np.array([
        [-w, -d, -h],  # 0: Bottom Front Left
        [w, -d, -h],   # 1: Bottom Front Right
        [w, d, -h],    # 2: Bottom Back Right
        [-w, d, -h],   # 3: Bottom Back Left
        [-w, -d, 0],    # 4: Top Front Left
        [w, -d, 0],     # 5: Top Front Right
        [w, d, 0],      # 6: Top Back Right
        [-w, d, 0],     # 7: Top Back Left
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

def rotate_vertices(vertices, angle_degrees, axis='x'):
    """
    Rotate vertices by a specified angle around a given axis.

    :param vertices: Nx3 numpy array of vertex coordinates.
    :param angle_degrees: Angle in degrees to rotate.
    :param axis: Axis to rotate around ('x', 'y', or 'z').
    :return: Rotated vertices as Nx3 numpy array.
    """
    angle_rad = np.deg2rad(angle_degrees)
    if axis == 'x':
        rotation_matrix = np.array([
            [1, 0, 0],
            [0, np.cos(angle_rad), -np.sin(angle_rad)],
            [0, np.sin(angle_rad), np.cos(angle_rad)]
        ])
    elif axis == 'y':
        rotation_matrix = np.array([
            [np.cos(angle_rad), 0, np.sin(angle_rad)],
            [0, 1, 0],
            [-np.sin(angle_rad), 0, np.cos(angle_rad)]
        ])
    elif axis == 'z':
        rotation_matrix = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad), 0],
            [np.sin(angle_rad), np.cos(angle_rad), 0],
            [0, 0, 1]
        ])
    else:
        raise ValueError("Invalid axis. Choose from 'x', 'y', or 'z'.")

    rotated_vertices = vertices.dot(rotation_matrix.T)
    return rotated_vertices

# ------------------- 3D Visualization Widget -------------------

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
        
        # X-axis label
        x_label = gl.GLTextItem(text='X', color=(1, 0, 0, 1))
        x_label.translate(axis_length + 2, 0, 0)  # Slightly beyond the axis tip
        self.addItem(x_label)

        # Y-axis (Green)
        y_axis = gl.GLLinePlotItem(pos=np.array([[0, 0, 0], [0, axis_length, 0]]),
                                color=(0, 1, 0, 1), width=2, antialias=True)
        self.addItem(y_axis)
        
        # Y-axis label
        y_label = gl.GLTextItem(text='Y', color=(0, 1, 0, 1))
        y_label.translate(0, axis_length + 2, 0)  # Slightly beyond the axis tip
        self.addItem(y_label)

        # Z-axis (Blue)
        z_axis = gl.GLLinePlotItem(pos=np.array([[0, 0, 0], [0, 0, axis_length]]),
                                color=(0, 0, 1, 1), width=2, antialias=True)
        self.addItem(z_axis)
        
        # Z-axis label
        z_label = gl.GLTextItem(text='Z', color=(0, 0, 1, 1))
        z_label.translate(0, 0, axis_length + 2)  # Slightly beyond the axis tip
        self.addItem(z_label)


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

        # Create Stage as a box (rectangular prism) with origin at top center
        box_vertices, box_faces = create_box_mesh(width=5.0, depth=10.0, height=0.5)
        self.stage_mesh = gl.GLMeshItem(vertexes=box_vertices, faces=box_faces,
                                       smooth=False, color=(0, 1, 0, 1), shader='shaded', drawEdges=False)
        self.addItem(self.stage_mesh)

        # Create Pipette as a rotated cylinder
        pipette_vertices, pipette_faces = create_cylinder_mesh(radius=0.2, height=3.0, sectors=32)
        # Rotate 25 degrees with respect to the XY plane (i.e., 65 degrees away from the Z-axis)
        pipette_vertices = rotate_vertices(pipette_vertices, 65, 'y')
        self.pipette_mesh = gl.GLMeshItem(vertexes=pipette_vertices, faces=pipette_faces,
                                         smooth=True, color=(0, 0, 1, 1), shader='shaded', drawEdges=False)
        # Adjust pipette position so that its bottom is at the origin
        pipette_change = np.array([3.0 * np.cos(np.radians(65)), 0, 0])
        self.pipette_mesh.translate(*pipette_change)
        pipette_change = np.array([3.0 * (np.cos(np.radians(65))-np.sin(np.radians(65))), 0, 0])
        self.pipette_mesh.translate(*pipette_change)
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

# ------------------- Data Manager -------------------

class DataManager:
    def __init__(self, scaling_factor=1/1000):
        self.image_paths = []
        self.timestamps = []
        self.image_index = []
        self.movement_data = []
        self.graph_data = []
        self.directory = None
        self.scaling_factor = scaling_factor  # micrometers to millimeters
        self.init_image_time = 0.0

    def load_directory(self, directory):
        self.directory = directory
        camera_frames_dir = os.path.join(directory, 'camera_frames')
        movement_file_path = os.path.join(directory, 'movement_recording.csv')
        graph_file_path = os.path.join(directory, 'graph_recording.csv')

        # Load images
        self.load_images(camera_frames_dir)

        # Load movement data
        if os.path.exists(movement_file_path):
            self.load_movement_data(movement_file_path)
        else:
            raise FileNotFoundError("movement_recording.csv not found in the selected directory.")

        # Load graph data
        if os.path.exists(graph_file_path):
            self.load_graph_data(graph_file_path)
        else:
            raise FileNotFoundError("graph_recording.csv not found in the selected directory.")

    def load_images(self, directory):
        if not os.path.exists(directory):
            raise FileNotFoundError(f"{directory} does not exist.")

        self.image_paths = sorted([os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.webp')])
        if not self.image_paths:
            raise FileNotFoundError("No .webp images found in the camera_frames directory.")

        # Extract indices and timestamps
        self.image_index = []
        self.timestamps = []
        for img_path in self.image_paths:
            idx, ts = self.extract_image_data(img_path)
            self.image_index.append(idx)
            self.timestamps.append(ts)
        self.init_image_time = self.timestamps[0]

    def load_movement_data(self, file_path):
        self.movement_data.clear()
        try:
            with open(file_path, mode='r') as file:
                first_line = file.readline().strip()
                # New format detection: if semicolons are present and header contains "timestamp"
                if ';' in first_line and 'timestamp' in first_line.lower():
                    file.seek(0)
                    reader = csv.DictReader(file, delimiter=';')
                    for row in reader:
                        try:
                            time_value = float(row['timestamp'])
                            st_x = float(row['st_x']) * self.scaling_factor
                            st_y = float(row['st_y']) * self.scaling_factor
                            st_z = float(row['st_z']) * self.scaling_factor
                            pi_x = float(row['pi_x']) * self.scaling_factor
                            pi_y = float(row['pi_y']) * self.scaling_factor
                            pi_z = float(row['pi_z']) * self.scaling_factor

                            self.movement_data.append({
                                'time': time_value,
                                'microscope_z': -(st_z),
                                'stage': (-st_x, -st_y),
                                'pipette': (-pi_x, -pi_y, -(pi_z - 1.365))
                            })
                        except Exception:
                            continue
                else:
                    # Assume old format with space-delimited values and colon-split key-value pairs
                    file.seek(0)
                    reader = csv.reader(file, delimiter=' ')
                    for line in reader:
                        parts = [part for part in line if part]
                        if len(parts) < 7:
                            continue  # Skip invalid lines
                        try:
                            time_value = float(parts[0].split(':')[1])
                            st_x = float(parts[1].split(':')[1]) * self.scaling_factor
                            st_y = float(parts[2].split(':')[1]) * self.scaling_factor
                            st_z = float(parts[3].split(':')[1]) * self.scaling_factor
                            pi_x = float(parts[4].split(':')[1]) * self.scaling_factor
                            pi_y = float(parts[5].split(':')[1]) * self.scaling_factor
                            pi_z = float(parts[6].split(':')[1]) * self.scaling_factor

                            self.movement_data.append({
                                'time': time_value,
                                'microscope_z': -(st_z),
                                'stage': (-st_x, -st_y),
                                'pipette': (-pi_x, -pi_y, -(pi_z - 1.365))
                            })
                        except Exception:
                            continue

            if not self.movement_data:
                raise ValueError("No valid movement data found in the selected file.")

            # Sort movement data by time
            self.movement_data.sort(key=lambda x: x['time'])
            # Align movement data's time with image data's time
            if self.timestamps:
                initial_image_time = self.timestamps[0]
                # Shift movement_data's time to align with image_data's initial timestamp
                # Assuming that movement_data starts after image_data's start
                self.movement_data = [
                    {k: v if k != 'time' else v - 0 for k, v in entry.items()}
                    for entry in self.movement_data
                ]

        except Exception as e:
            raise IOError(f"Error loading movement data: {e}")

    def get_closest_movement_entry(self, timestamp):
        """
        Retrieve the movement data entry closest to the given timestamp.
        """
        if not self.movement_data:
            return None

        times = [entry['time'] for entry in self.movement_data]
        index = bisect.bisect_left(times, timestamp)

        if index == 0:
            return self.movement_data[0]
        if index == len(times):
            return self.movement_data[-1]

        before = self.movement_data[index - 1]
        after = self.movement_data[index]

        if abs(after['time'] - timestamp) < abs(before['time'] - timestamp):
            return after
        else:
            return before

    def load_graph_data(self, file_path):
        self.graph_data.clear()
        try:
            with open(file_path, mode='r') as file:
                first_line = file.readline().strip()
                # New format detection: if semicolons are present and header contains "timestamp"
                if ';' in first_line and 'timestamp' in first_line.lower():
                    file.seek(0)
                    reader = csv.DictReader(file, delimiter=';')
                    for row in reader:
                        try:
                            time_value = float(row['timestamp'])
                            pressure_value = float(row['pressure'])
                            resistance_value = float(row['resistance'])
                            # Parse list values for current and voltage
                            current_str = row['current'].strip()
                            if current_str.startswith('[') and current_str.endswith(']'):
                                current_str = current_str[1:-1]
                            current_value = [float(x) for x in current_str.split(',') if x.strip() != '']

                            voltage_str = row['voltage'].strip()
                            if voltage_str.startswith('[') and voltage_str.endswith(']'):
                                voltage_str = voltage_str[1:-1]
                            voltage_value = [float(x) for x in voltage_str.split(',') if x.strip() != '']

                            self.graph_data.append({
                                'time': time_value,
                                'pressure': pressure_value,
                                'resistance': resistance_value,
                                'current': current_value,
                                'voltage': voltage_value
                            })
                        except Exception:
                            continue
                else:
                    # Assume old format with colon-separated values
                    file.seek(0)
                    for line in file:
                        parts = [part for part in line.strip().split(':') if part]
                        if len(parts) < 6:
                            continue  # Skip invalid lines
                        try:
                            time_value = float(parts[1].split(' ')[0])
                            pressure_value = float(parts[2].split(' ')[0])
                            resistance_value = float(parts[3].split(' ')[0])
                            current_data = parts[4].split('[')[-1].split(']')[0]
                            current_value = [float(x) for x in current_data.split(',') if x]
                            voltage_data = parts[5].split('[')[-1].split(']')[0]
                            voltage_value = [float(x) for x in voltage_data.split(',') if x]
                            self.graph_data.append({
                                'time': time_value,
                                'pressure': pressure_value,
                                'resistance': resistance_value,
                                'current': current_value,
                                'voltage': voltage_value
                            })
                        except Exception:
                            continue

            self.graph_data.sort(key=lambda x: x['time'])

        except Exception as e:
            raise IOError(f"Error loading graph data: {e}")

    def extract_image_data(self, image_path):
        """
        Extract the image index and timestamp from the image filename.
        Assumes filenames are in the format 'index_timestamp.webp'.
        """
        try:
            filename = os.path.basename(image_path)
            index_str = int(filename.split('_')[0])
            timestamp_str = float(filename.split('_')[1].rsplit('.', 1)[0])
        except Exception as e:
            print(f"Error extracting index and timestamp from {image_path}: {e}")
            index_str, timestamp_str = 0, 0.0  # Default values in case of error
        return index_str, timestamp_str

# ------------------- Main Integrated Window -------------------

class IntegratedTimeline(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Integrated Timeline and 3D Visualization")
        self.setGeometry(50, 50, 1600, 900)  # Adjusted size for better visibility

        # Central Widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # Main Layout
        self.main_layout = QVBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(10, 10, 10, 10)
        self.main_layout.setSpacing(10)

        # Top Layout: Image and Graphs/3D Plot
        self.top_layout = QHBoxLayout()
        self.top_layout.setSpacing(10)
        self.main_layout.addLayout(self.top_layout, stretch=8)  # Allocate most space to top_layout

        # Image Display
        self.image = QLabel("No Image loaded yet!")
        self.image.setAlignment(Qt.AlignCenter)
        self.image.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.image_frame = QFrame()
        image_layout = QVBoxLayout(self.image_frame)
        image_layout.addWidget(self.image)
        self.image_frame.setFrameShape(QFrame.StyledPanel)
        self.image_frame.setFrameShadow(QFrame.Sunken)
        self.image_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.top_layout.addWidget(self.image_frame, stretch=1)  # Image takes 1 part

        # Graphs and 3D Plot
        self.graph_stack = QStackedWidget()
        self.graph_stack.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # 2D Graphs Widget
        self.graph_frame = QFrame()
        self.graph_frame.setFrameShape(QFrame.StyledPanel)
        self.graph_frame.setFrameShadow(QFrame.Sunken)
        self.graph_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.graph_frame.setMinimumSize(600, 400)  # Set minimum size for graph_frame

        self.graph_layout = QVBoxLayout(self.graph_frame)
        self.graph_layout.setContentsMargins(5, 5, 5, 5)
        self.graph_layout.setSpacing(5)

        # GraphicsLayoutWidget for 2x2 plots
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
        self.plots[0].setTitle("Voltage")
        self.plots[0].setLabels(left='Voltage (V)', bottom='Time (s)')
        self.plots[1].setTitle("Current")
        self.plots[1].setLabels(left='Current (A)', bottom='Time (s)')
        self.plots[2].setTitle("Pressure")
        self.plots[2].setLabels(left='Pressure (mBAR)', bottom='Time (s)')
        self.plots[3].setTitle("Resistance")
        self.plots[3].setLabels(left='Resistance (Ohm)', bottom='Time (s)')

        # Initialize plot data objects
        self.plot_voltage = self.plots[0].plot([], pen=pg.mkPen(color='b', width=2))
        self.plot_current = self.plots[1].plot([], pen=pg.mkPen(color='r', width=2))
        self.plot_pressure = self.plots[2].plot([], pen=pg.mkPen(color='g', width=2))
        self.plot_resistance = self.plots[3].plot([], pen=pg.mkPen(color='m', width=2))

        self.graph_stack.addWidget(self.graph_frame)  # Add 2D graphs to stack

        # 3D Plot Widget
        self.three_d_view = GLViewWidgetWithGrid()
        self.graph_stack.addWidget(self.three_d_view)  # Add 3D view to stack

        self.top_layout.addWidget(self.graph_stack, stretch=1)  # Graphs take 1 part

        # Slider Frame
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMaximumHeight(30)
        self.slider.setTickPosition(QSlider.NoTicks)
        self.slider.valueChanged.connect(self.slider_changed)
        self.slider.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.slider_frame = QFrame()
        slider_layout = QHBoxLayout(self.slider_frame)
        slider_layout.setContentsMargins(0, 0, 0, 0)
        slider_layout.addWidget(self.slider)
        self.slider_frame.setFrameShape(QFrame.StyledPanel)
        self.slider_frame.setFrameShadow(QFrame.Sunken)
        self.slider_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.main_layout.addWidget(self.slider_frame, stretch=1)  # Allocate less space

        # Middle Layout: Info and Buttons
        self.middle_layout = QVBoxLayout()
        self.middle_layout.setSpacing(5)

        # Info Frame
        self.info_frame = QFrame()
        info_frame_layout = QHBoxLayout(self.info_frame)
        info_frame_layout.setContentsMargins(5, 5, 5, 5)
        info_frame_layout.setSpacing(10)

        # Info Label
        self.info = QLabel("No info yet!")
        self.info.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.info.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.info.setStyleSheet("color: white;")  # Ensure text is visible on dark background

        info_frame_layout.addWidget(self.info, stretch=1)

        # Buttons Layout
        self.buttons_layout = QHBoxLayout()
        self.buttons_layout.setSpacing(10)

        # Previous Timepoint Button
        self.prev_image_button = QPushButton("Previous Timepoint")
        self.prev_image_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.prev_image_button.clicked.connect(self.show_previous_timepoint)
        self.buttons_layout.addWidget(self.prev_image_button)

        # Next Timepoint Button
        self.next_image_button = QPushButton("Next Timepoint")
        self.next_image_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.next_image_button.clicked.connect(self.show_next_timepoint)
        self.buttons_layout.addWidget(self.next_image_button)

        # Play/Pause Button
        self.toggle_video_button = QPushButton("Play")
        self.toggle_video_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.toggle_video_button.clicked.connect(self.toggle_video)
        self.buttons_layout.addWidget(self.toggle_video_button)

        # Select Data Directory Button
        self.select_image_dir_button = QPushButton("Select Data Directory")
        self.select_image_dir_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.select_image_dir_button.clicked.connect(self.open_directory)
        self.buttons_layout.addWidget(self.select_image_dir_button)

        # Toggle View Button
        self.toggle_view_button = QPushButton("Switch to 3D View")
        self.toggle_view_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.toggle_view_button.clicked.connect(self.toggle_view)
        self.buttons_layout.addWidget(self.toggle_view_button)

        # Add a spacer to push buttons to the left
        self.buttons_layout.addStretch()

        # Add buttons layout to info_frame_layout
        info_frame_layout.addLayout(self.buttons_layout)

        # Add info_frame to middle_layout
        self.middle_layout.addWidget(self.info_frame)

        # Add middle_layout to main_layout
        self.main_layout.addLayout(self.middle_layout, stretch=1)  # Allocate less space

        # Initialize Data Manager
        self.data_manager = DataManager()

        # Initialize variables
        self.current_index = 0
        self.directory = None

        # Deques for real-time graphing
        self.pressure_deque = deque(maxlen=100)
        self.resistance_deque = deque(maxlen=100)

        # Timer for playback
        self.timer = QTimer()
        self.interval = 33  # 33ms for ~30 frames per second
        self.timer.setInterval(self.interval)
        self.timer.timeout.connect(self.advance_timepoint)

        # Keyboard Shortcuts
        self.shortcut_left = QShortcut(QKeySequence(Qt.Key_Left), self)
        self.shortcut_left.activated.connect(self.show_previous_timepoint)
        self.shortcut_right = QShortcut(QKeySequence(Qt.Key_Right), self)
        self.shortcut_right.activated.connect(self.show_next_timepoint)

        # Styling
        self.apply_styles()

    def apply_styles(self):
        """Apply dark theme styling to the application."""
        self.setStyleSheet("""
            QWidget {
                background-color: #2e2e2e;
                color: white;
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

    def toggle_view(self):
        """Toggle between 2D graphs and 3D plot."""
        current_widget = self.graph_stack.currentWidget()
        if current_widget == self.graph_frame:
            self.graph_stack.setCurrentWidget(self.three_d_view)
            self.toggle_view_button.setText("Switch to 2D Graphs")
        else:
            self.graph_stack.setCurrentWidget(self.graph_frame)
            self.toggle_view_button.setText("Switch to 3D View")

    def open_directory(self):
        """Open a directory dialog to select data directory."""
        directory = QFileDialog.getExistingDirectory(self, "Open Directory", "")
        if directory:
            try:
                self.data_manager.load_directory(directory)
                self.directory = directory
                self.info.setText(f"Loaded data from {directory}")

                # Initialize timeline
                self.current_index = 0
                self.slider.setMinimum(0)
                self.slider.setMaximum(len(self.data_manager.image_paths) - 1)
                self.slider.setValue(self.current_index)

                # Display first image
                self.display_image(self.data_manager.image_paths[self.current_index])

                # Update 2D graphs
                self.update_graphs()

                # Update 3D view
                self.update_3d_view()

            except FileNotFoundError as e:
                QMessageBox.critical(self, "File Not Found", str(e))
            except IOError as e:
                QMessageBox.critical(self, "IO Error", str(e))
            except Exception as e:
                QMessageBox.critical(self, "Error", f"An unexpected error occurred: {e}")

    def check_data_loaded(self):
        """Check if data is loaded before allowing navigation."""
        if not self.data_manager.image_paths:
            QMessageBox.warning(self, "No Data Loaded", "Please load a directory with images first.")
            return False
        return True

    def show_previous_timepoint(self):
        """Show the previous timepoint."""
        if not self.check_data_loaded():
            return

        if self.current_index > 0:
            self.current_index -= 1
            self.slider.setValue(self.current_index)
            self.update_view()

    def show_next_timepoint(self):
        """Show the next timepoint."""
        if not self.check_data_loaded():
            return

        if self.current_index < len(self.data_manager.image_paths) - 1:
            self.current_index += 1
            self.slider.setValue(self.current_index)
            self.update_view()

    def toggle_video(self):
        """Toggle play/pause functionality."""
        if not self.check_data_loaded():
            return

        if self.timer.isActive():
            self.toggle_video_button.setText("Play")
            self.timer.stop()
        else:
            self.toggle_video_button.setText("Pause")
            self.timer.start()

    def advance_timepoint(self):
        """Advance to the next timepoint during playback."""
        if not self.check_data_loaded():
            return

        if self.current_index < len(self.data_manager.image_paths) - 1:
            self.current_index += 1
            self.slider.setValue(self.current_index)
            self.update_view()
        else:
            self.toggle_video_button.setText("Play")
            self.timer.stop()

    def slider_changed(self, value):
        """Update view based on slider's position."""
        if not self.check_data_loaded():
            return

        self.current_index = value
        self.update_view()

    def update_view(self):
        """Update image, 2D graphs, and 3D plot based on current index."""
        self.display_image(self.data_manager.image_paths[self.current_index])
        self.update_graphs()
        self.update_3d_view()

    def display_image(self, image_path):
        """Display the image on the QLabel."""
        if not os.path.exists(image_path):
            QMessageBox.warning(self, "Error", f"Image file {image_path} does not exist.")
            return

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

    def resizeEvent(self, event):
        """Handle window resize events to scale the image appropriately."""
        if self.data_manager.image_paths and 0 <= self.current_index < len(self.data_manager.image_paths):
            self.display_image(self.data_manager.image_paths[self.current_index])
        super().resizeEvent(event)  # Ensure the base class resizeEvent is also called

    def update_graphs(self):
        """Update the 2D graphs with the current timepoint data."""
        if not self.check_data_loaded():
            return

        # Find the closest graph_data entry for the current timestamp
        current_timestamp = self.data_manager.timestamps[self.current_index]
        graph_index = next((i for i, data in enumerate(self.data_manager.graph_data) if data['time'] >= current_timestamp), None)

        if graph_index is not None:
            graph_data = self.data_manager.graph_data[graph_index]

            # Update deques
            self.pressure_deque.clear()
            self.resistance_deque.clear()
            start_index = max(0, graph_index - 99)
            for i in range(start_index, graph_index + 1):
                data = self.data_manager.graph_data[i]
                self.pressure_deque.append(data['pressure'])
                self.resistance_deque.append(data['resistance'])

            # Update plot data
            self.plot_voltage.setData(graph_data['voltage'])
            self.plot_current.setData(graph_data['current'])
            self.plot_pressure.setData(list(self.pressure_deque))
            self.plot_resistance.setData(list(self.resistance_deque))

            # Retrieve the closest movement data entry based on current_timestamp
            movement_entry = self.data_manager.get_closest_movement_entry(current_timestamp)

            if movement_entry:
                stage_x, stage_y = movement_entry['stage']       # Correctly unpack stage
                microscope_z = movement_entry['microscope_z']     # Retrieve microscope_z separately
                pipette_x, pipette_y, pipette_z = movement_entry['pipette']
                time = current_timestamp - self.data_manager.init_image_time
                info_text = (
                    f"Time: {time:.2f} s, "
                    f"Stage: (X: {stage_x:.2f} mm, Y: {stage_y:.2f} mm, Z: {microscope_z:.2f} mm), "
                    f"Pipette: (X: {pipette_x:.2f} mm, Y: {pipette_y:.2f} mm, Z: {pipette_z:.2f} mm)<br>"
                    f"<span style='font-size: 24px; color: red;'>X "
                    f"<span style='font-size: 24px; color: green;'>Y "
                    f"<span style='font-size: 24px; color: blue;'>Z"
                )
            else:
                info_text = f"Time: {current_timestamp:.2f} s, No movement data available."

            self.info.setText(info_text)
        else:
            self.info.setText("No graph data available for this timestamp.")

    def update_3d_view(self):
        """Update the 3D visualization based on the current timepoint."""
        if not self.check_data_loaded():
            return

        if self.current_index >= len(self.data_manager.image_paths):
            return

        # Get the current image timestamp
        current_timestamp = self.data_manager.timestamps[self.current_index]

        # Retrieve the closest movement data entry based on current_timestamp
        movement_entry = self.data_manager.get_closest_movement_entry(current_timestamp)

        if movement_entry:
            time = movement_entry['time']
            stage_x, stage_y = movement_entry['stage']       # Only 2 elements
            microscope_z = movement_entry['microscope_z']     # Separate key
            pipette_x, pipette_y, pipette_z = movement_entry['pipette']

            # Update 3D objects
            self.three_d_view.update_objects(
                microscope_z=microscope_z,  # Corrected mapping
                stage_pos=(stage_x, stage_y),
                pipette_pos=(pipette_x, pipette_y, pipette_z)
            )
        else:
            # Optionally, handle cases where no movement data is available
            pass

# ------------------- Main Function -------------------

def main():
    app = QApplication(sys.argv)
    timeline = IntegratedTimeline()
    timeline.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
