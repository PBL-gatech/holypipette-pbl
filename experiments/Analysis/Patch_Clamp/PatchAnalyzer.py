import os
import sys
import csv
import numpy as np
import pyqtgraph as pg
import logging
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QPushButton,
    QFileDialog, QShortcut, QHBoxLayout, QFrame, QSlider, QMessageBox,
    QSizePolicy, QComboBox, QProgressBar
)
from PyQt5.QtGui import QColor, QIcon
from PyQt5.QtCore import Qt, QThread, pyqtSignal

# Configure logging
logging.basicConfig(
    filename='patch_analyzer.log',
    filemode='a',  # Append mode
    format='%(asctime)s - %(levelname)s - %(message)s',  # Log message format
    level=logging.INFO  # Logging level
)

# Conversion factor for CurrentProtocol command to current
C_CLAMP_AMP_PER_VOLT = 400 * 1e-12  # 400 pA / V

class DataLoaderThread(QThread):
    """Thread to load CSV data without blocking the UI."""
    progress = pyqtSignal(int)
    finished = pyqtSignal(dict)  # Emit a dictionary containing all loaded data

    def __init__(self, protocol, protocol_dir, run_groups=None):
        super().__init__()
        self.protocol = protocol
        self.protocol_dir = protocol_dir
        self.run_groups = run_groups  # Only for CurrentProtocol

    def run(self):
        loaded_data = {}
        try:
            if self.protocol == "CurrentProtocol" and self.run_groups:
                for run_number, csv_files in self.run_groups.items():
                    loaded_data[run_number] = {}
                    total_files = len(csv_files)
                    for idx, csv_file in enumerate(csv_files):
                        csv_path = os.path.join(self.protocol_dir, csv_file)
                        try:
                            with open(csv_path, 'r') as f:
                                reader = csv.reader(f, delimiter=' ')
                                time, command, response = [], [], []
                                for row in reader:
                                    if len(row) < 3:
                                        continue
                                    time.append(float(row[0]))
                                    command.append(float(row[1]) * C_CLAMP_AMP_PER_VOLT)
                                    response.append(float(row[2]))
                                loaded_data[run_number][csv_file] = {
                                    'time': np.array(time),
                                    'command': np.array(command),
                                    'response': np.array(response)
                                }
                        except Exception as e:
                            logging.error(f"Error loading {csv_file}: {e}")
                            loaded_data[run_number][csv_file] = None
                        progress_percent = int(((idx + 1) / total_files) * 100)
                        self.progress.emit(progress_percent)
            else:
                # For VoltageProtocol and HoldingProtocol
                csv_files = [f for f in os.listdir(self.protocol_dir) if f.endswith('.csv')]
                loaded_data['files'] = {}
                total_files = len(csv_files)
                for idx, csv_file in enumerate(csv_files):
                    csv_path = os.path.join(self.protocol_dir, csv_file)
                    try:
                        with open(csv_path, 'r') as f:
                            reader = csv.reader(f, delimiter=' ')
                            time, command, response = [], [], []
                            for row in reader:
                                if len(row) < 3:
                                    continue
                                time.append(float(row[0]))
                                command.append(float(row[1]))
                                response.append(float(row[2]))
                            loaded_data['files'][csv_file] = {
                                'time': np.array(time),
                                'command': np.array(command),
                                'response': np.array(response)
                            }
                    except Exception as e:
                        logging.error(f"Error loading {csv_file}: {e}")
                        loaded_data['files'][csv_file] = None
                    progress_percent = int(((idx + 1) / total_files) * 100)
                    self.progress.emit(progress_percent)
        except Exception as e:
            logging.error(f"Unexpected error during data loading: {e}")
        self.finished.emit(loaded_data)

class PatchAnalyzer(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle('PatchAnalyzer')
        self.setGeometry(100, 100, 1200, 800)
        self.setWindowIcon(QIcon())  # Set an icon if available

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
        self.graph_frame.setMinimumSize(800, 400)

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
            self.graphics_layout.addPlot(row=0, col=0, title="Command Data"),
            self.graphics_layout.addPlot(row=1, col=0, title="Response Data")
        ]

        # Set labels
        self.plots[0].setLabel('bottom', 'Time', units='s')
        self.plots[1].setLabel('bottom', 'Time', units='s')

        # Initialize plot data items for efficient updates
        self.command_plot = self.plots[0].plot([], [], pen=pg.mkPen(color='b', width=2))
        self.response_plot = self.plots[1].plot([], [], pen=pg.mkPen(color='r', width=2))

        # Set up vertical slider
        self.vertical_slider = QSlider(Qt.Vertical)
        self.vertical_slider.setTickPosition(QSlider.NoTicks)
        self.vertical_slider.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self.vertical_slider.valueChanged.connect(self.on_slider_value_changed)
        self.vertical_slider.setMinimum(0)
        self.vertical_slider.setMaximum(0)  # Will be updated after data loading

        # Create a horizontal layout to hold the graph frame and the vertical slider
        self.horizontal_layout = QHBoxLayout()
        self.horizontal_layout.addWidget(self.graph_frame)
        self.horizontal_layout.addWidget(self.vertical_slider)

        # Add the horizontal layout to the main layout
        self.main_layout.addLayout(self.horizontal_layout)

        # Create the info frame
        self.info_frame = QFrame()
        info_frame_layout = QHBoxLayout(self.info_frame)
        info_frame_layout.setContentsMargins(5, 5, 5, 5)
        info_frame_layout.setSpacing(10)

        # Add the info label to the left
        self.info = QLabel("Information: ")
        self.info.setWordWrap(True)
        info_frame_layout.addWidget(self.info, stretch=1)

        # Add the info frame to the main layout before the buttons layout
        self.main_layout.addWidget(self.info_frame)

        # Create a single row of buttons
        self.buttons_layout = QHBoxLayout()
        self.buttons_layout.setSpacing(10)

        # Navigation buttons
        self.left_button = QPushButton("←")
        self.left_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.left_button.clicked.connect(self.show_previous_protocol)
        self.buttons_layout.addWidget(self.left_button)

        self.right_button = QPushButton("→")
        self.right_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.right_button.clicked.connect(self.show_next_protocol)
        self.buttons_layout.addWidget(self.right_button)

        # Add a small spacer between navigation and control buttons
        self.buttons_layout.addSpacing(20)

        # Control buttons
        self.prev_data_button = QPushButton("Previous timepoint")
        self.prev_data_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.prev_data_button.clicked.connect(self.show_previous_timepoint)
        self.buttons_layout.addWidget(self.prev_data_button)

        self.select_data_dir_button = QPushButton("Select Data Directory")
        self.select_data_dir_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.select_data_dir_button.clicked.connect(self.open_directory)
        self.buttons_layout.addWidget(self.select_data_dir_button)

        self.next_data_button = QPushButton("Next timepoint")
        self.next_data_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.next_data_button.clicked.connect(self.show_next_timepoint)
        self.buttons_layout.addWidget(self.next_data_button)

        self.toggle_video_button = QPushButton("Play")
        self.toggle_video_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.toggle_video_button.clicked.connect(self.toggle_video)
        self.buttons_layout.addWidget(self.toggle_video_button)

        # Add a drop-down menu for run number
        self.run_dropdown = QComboBox()
        self.run_dropdown.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.run_dropdown.currentIndexChanged.connect(self.on_run_selected)
        self.run_dropdown.setEnabled(False)  # Disabled until data is loaded
        self.buttons_layout.addWidget(self.run_dropdown)

        # Add the buttons layout to the main layout
        self.main_layout.addLayout(self.buttons_layout)

        # Progress Bar for Data Loading
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)  # Hidden by default
        self.main_layout.addWidget(self.progress_bar)

        # Keyboard shortcuts
        self.shortcut_left = QShortcut(Qt.Key_Left, self)
        self.shortcut_left.activated.connect(self.show_previous_protocol)
        self.shortcut_right = QShortcut(Qt.Key_Right, self)
        self.shortcut_right.activated.connect(self.show_next_protocol)

        # Initialize protocol variables
        self.protocols = []  # List of tuples: (protocol_name, protocol_dir)
        self.current_protocol_index = 0
        self.current_protocol_name = ""
        self.current_protocol_dir = ""
        self.current_run_number = ""
        self.current_csv_files = []  # List of CSV file names for non-CurrentProtocol
        self.current_run_files = []  # List of CSV file names for CurrentProtocol

        # Data storage
        self.loaded_data = {}  # Nested dictionary to store all loaded data

    def open_directory(self):
        """Open a directory and load all data."""
        directory = QFileDialog.getExistingDirectory(self, "Open Directory", "")
        if directory:
            logging.info(f"Selected directory: {directory}")
            self.protocols = []
            self.current_protocol_index = 0

            # Check for VoltageProtocol
            volt_dir = os.path.join(directory, 'VoltageProtocol')
            if os.path.exists(volt_dir):
                self.protocols.append(('VoltageProtocol', volt_dir))
                logging.info(f"Found VoltageProtocol at: {volt_dir}")

            # Check for HoldingProtocol
            hold_dir = os.path.join(directory, 'HoldingProtocol')
            if os.path.exists(hold_dir):
                self.protocols.append(('HoldingProtocol', hold_dir))
                logging.info(f"Found HoldingProtocol at: {hold_dir}")

            # Check for CurrentProtocol
            curr_dir = os.path.join(directory, 'CurrentProtocol')
            if os.path.exists(curr_dir):
                self.protocols.append(('CurrentProtocol', curr_dir))
                logging.info(f"Found CurrentProtocol at: {curr_dir}")

            if self.protocols:
                logging.info(f"Available protocols: {self.protocols}")
                self.load_protocol(self.protocols[self.current_protocol_index])
            else:
                self.info.setText("No valid protocols found in the selected directory.")
                logging.error("No valid protocols found in the selected directory.")

    def load_protocol(self, protocol):
        """Load all CSV data for the selected protocol."""
        protocol_name, protocol_dir = protocol
        self.current_protocol_name = protocol_name
        self.current_protocol_dir = protocol_dir
        self.loaded_data = {}  # Reset loaded data

        self.info.setText(f"Loading {protocol_name}...")
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        csv_files = [f for f in os.listdir(protocol_dir) if f.endswith('.csv')]
        total_files = len(csv_files)

        if protocol_name == "CurrentProtocol":
            self.vertical_slider.setEnabled(True)  # Enable the slider for CurrentProtocol
            self.run_dropdown.setEnabled(True)
            # Group CSV files by run number
            run_groups = {}
            for csv_file in csv_files:
                parts = csv_file.split('_')
                if len(parts) >= 3:
                    run_number = parts[1]
                    if run_number not in run_groups:
                        run_groups[run_number] = []
                    run_groups[run_number].append(csv_file)
            self.loaded_data['runs'] = {}
            for idx, (run_number, files) in enumerate(sorted(run_groups.items())):
                self.loaded_data['runs'][run_number] = {}
                for file_idx, csv_file in enumerate(sorted(files)):
                    csv_path = os.path.join(protocol_dir, csv_file)
                    try:
                        with open(csv_path, 'r') as f:
                            reader = csv.reader(f, delimiter=' ')
                            time, command, response = [], [], []
                            for row in reader:
                                if len(row) < 3:
                                    continue
                                time.append(float(row[0]))
                                command.append(float(row[1]) * C_CLAMP_AMP_PER_VOLT)
                                response.append(float(row[2]))
                            self.loaded_data['runs'][run_number][csv_file] = {
                                'time': np.array(time),
                                'command': np.array(command),
                                'response': np.array(response)
                            }
                    except Exception as e:
                        logging.error(f"Error loading {csv_file}: {e}")
                        self.loaded_data['runs'][run_number][csv_file] = None
                    progress_percent = int(((idx * len(files) + file_idx + 1) / total_files) * 100)
                    self.progress_bar.setValue(progress_percent)
            # Populate run dropdown
            self.run_dropdown.blockSignals(True)
            self.run_dropdown.clear()
            for run_number in sorted(self.loaded_data['runs'].keys()):
                self.run_dropdown.addItem(f"Run {run_number}")
            self.run_dropdown.setCurrentIndex(0)
            self.run_dropdown.blockSignals(False)
            self.run_dropdown.setEnabled(True)
            # Set current run
            self.current_run_number = sorted(self.loaded_data['runs'].keys())[0]
            self.current_run_files = sorted(self.loaded_data['runs'][self.current_run_number].keys())
            # Configure slider
            self.vertical_slider.setMinimum(0)
            self.vertical_slider.setMaximum(len(self.current_run_files) - 1)
            self.vertical_slider.setValue(0)
            # Plot the first file
            self.plot_current_file(0)
        else:
            # For VoltageProtocol and HoldingProtocol, disable the slider
            self.vertical_slider.setEnabled(False)  # Disable the slider
            self.run_dropdown.setEnabled(True)  # Enable the dropdown menu
            self.loaded_data['files'] = {}
            for idx, csv_file in enumerate(sorted(csv_files)):
                csv_path = os.path.join(protocol_dir, csv_file)
                try:
                    with open(csv_path, 'r') as f:
                        reader = csv.reader(f, delimiter=' ')
                        time, command, response = [], [], []
                        for row in reader:
                            if len(row) < 3:
                                continue
                            time.append(float(row[0]))
                            command.append(float(row[1]))
                            response.append(float(row[2]))
                        self.loaded_data['files'][csv_file] = {
                            'time': np.array(time),
                            'command': np.array(command),
                            'response': np.array(response)
                        }
                except Exception as e:
                    logging.error(f"Error loading {csv_file}: {e}")
                    self.loaded_data['files'][csv_file] = None
                progress_percent = int(((idx + 1) / total_files) * 100)
                self.progress_bar.setValue(progress_percent)
            # Populate run dropdown
            self.run_dropdown.blockSignals(True)
            self.run_dropdown.clear()
            for idx, csv_file in enumerate(sorted(self.loaded_data['files'].keys())):
                self.run_dropdown.addItem(f"Run {idx + 1}")
            self.run_dropdown.setCurrentIndex(0)
            self.run_dropdown.blockSignals(False)
            self.run_dropdown.setEnabled(True)
            # Set current run files
            self.current_run_files = sorted(self.loaded_data['files'].keys())
            # Plot the first file
            self.plot_csv_file(0)

        self.progress_bar.setVisible(False)
        self.info.setText(f"Loaded {protocol_name} protocol.")


    def plot_csv_file(self, index):
        """Plot data for non-CurrentProtocol protocols."""
        if not self.current_protocol_name or self.current_protocol_name == "CurrentProtocol":
            return

        if index < 0 or index >= len(self.current_run_files):
            logging.error(f"File index {index} is out of range.")
            return

        csv_file = self.current_run_files[index]
        data = self.loaded_data['files'].get(csv_file)

        if data is None:
            self.info.setText(f"No data loaded from {csv_file}.")
            logging.warning(f"No data loaded from {csv_file}.")
            return

        time = data['time']
        command = data['command']
        response = data['response']

        # Set appropriate labels based on protocol type
        if self.current_protocol_name == "VoltageProtocol":
            self.plots[0].setLabel('left', 'Command Voltage', units='V')
            self.plots[1].setLabel('left', 'Current Response', units='A')
        elif self.current_protocol_name == "HoldingProtocol":
            self.plots[0].setLabel('left', 'Command Voltage', units='V')
            self.plots[1].setLabel('left', 'Current Response', units='A')

        # Update plot data
        self.command_plot.setData(time, command)
        self.response_plot.setData(time, response)

        # Auto-scale the plots to fit the data
        self.plots[0].enableAutoRange()
        self.plots[1].enableAutoRange()

        self.info.setText(f"Loaded data from {csv_file}")
        logging.info(f"Data from {csv_file} plotted successfully.")

    def plot_current_file(self, index):
        """Plot data for CurrentProtocol."""
        if self.current_protocol_name != "CurrentProtocol":
            return

        if not self.current_run_files:
            logging.error("No run files available to plot.")
            return

        if index < 0 or index >= len(self.current_run_files):
            logging.error(f"File index {index} is out of range.")
            return

        csv_file = self.current_run_files[index]
        data = self.loaded_data['runs'][self.current_run_number].get(csv_file)

        if data is None:
            self.info.setText(f"No data loaded from {csv_file}.")
            logging.warning(f"No data loaded from {csv_file}.")
            return

        time = data['time']
        command = data['command']
        response = data['response']

        # Set labels for CurrentProtocol
        self.plots[0].setLabel('left', 'Command Current', units='A')
        self.plots[1].setLabel('left', 'Voltage Response', units='V')

        # Update plot data
        self.command_plot.setData(time, command)
        self.response_plot.setData(time, response)

        # Auto-scale the plots to fit the data
        self.plots[0].enableAutoRange()
        self.plots[1].enableAutoRange()

        self.info.setText(f"Loaded data from {csv_file}")
        logging.info(f"Data from {csv_file} plotted successfully.")

    def on_slider_value_changed(self, value):
        """Handle the slider value change event to load a specific file."""
        logging.info(f"Slider value changed: {value}")
        if self.current_protocol_name == "CurrentProtocol":
            self.plot_current_file(value)
        else:
            self.plot_csv_file(value)

    def show_previous_protocol(self):
        """Navigate to the previous protocol."""
        if not self.protocols:
            self.info.setText("No protocols loaded. Please select a data directory.")
            logging.error("No protocols loaded. Cannot navigate to previous protocol.")
            return
        self.current_protocol_index -= 1
        if self.current_protocol_index < 0:
            self.current_protocol_index = len(self.protocols) - 1
        logging.info(f"Switching to previous protocol: {self.protocols[self.current_protocol_index][0]}")
        self.load_protocol(self.protocols[self.current_protocol_index])

    def show_next_protocol(self):
        """Navigate to the next protocol."""
        if not self.protocols:
            self.info.setText("No protocols loaded. Please select a data directory.")
            logging.error("No protocols loaded. Cannot navigate to next protocol.")
            return
        self.current_protocol_index += 1
        if self.current_protocol_index >= len(self.protocols):
            self.current_protocol_index = 0
        logging.info(f"Switching to next protocol: {self.protocols[self.current_protocol_index][0]}")
        self.load_protocol(self.protocols[self.current_protocol_index])

    def show_previous_timepoint(self):
        """Navigate to the previous timepoint (not implemented)."""
        self.info.setText("Previous timepoint functionality not implemented yet.")
        logging.info("Previous timepoint button clicked.")

    def show_next_timepoint(self):
        """Navigate to the next timepoint (not implemented)."""
        self.info.setText("Next timepoint functionality not implemented yet.")
        logging.info("Next timepoint button clicked.")

    def toggle_video(self):
        """Toggle play/pause video functionality (not implemented)."""
        self.info.setText("Play/Pause video functionality not implemented yet.")
        logging.info("Toggle video button clicked.")

    def on_run_selected(self, run_index):
        """Handle run selection from the dropdown."""
        logging.info(f"Run selected: {run_index}")
        if self.current_protocol_name == "CurrentProtocol":
            # Handle CurrentProtocol run selection
            run_number = sorted(self.loaded_data['runs'].keys())[run_index]
            self.current_run_number = run_number
            self.current_run_files = sorted(self.loaded_data['runs'][run_number].keys())
            # Configure slider for CurrentProtocol
            self.vertical_slider.setMinimum(0)
            self.vertical_slider.setMaximum(len(self.current_run_files) - 1)
            self.vertical_slider.setValue(0)
            # Plot the first file of the selected run
            self.plot_current_file(0)
        else:
            # For VoltageProtocol and HoldingProtocol, load data based on run index
            self.current_run_files = sorted(self.loaded_data['files'].keys())
            if run_index < 0 or run_index >= len(self.current_run_files):
                logging.error(f"Run index {run_index} is out of range.")
                return
            # Plot the selected run (file) from the dropdown
            self.plot_csv_file(run_index)


def main():
    app = QApplication(sys.argv)
    window = PatchAnalyzer()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
