import os
import sys
import pandas as pd
import pyqtgraph as pg
import logging
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QPushButton,
    QFileDialog, QShortcut, QHBoxLayout, QFrame, QSlider, QMessageBox, QSizePolicy, QComboBox
)
from PyQt5.QtGui import QColor
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon

# Configure logging
logging.basicConfig(
    filename='patch_analyzer.log',            # Log file name
    filemode='a',                             # Append mode
    format='%(asctime)s - %(levelname)s - %(message)s',  # Log message format
    level=logging.DEBUG                       # Logging level
)

class PatchAnalyzer(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle('PatchAnalyzer')
        self.setGeometry(100, 100, 800, 800)
        self.setWindowIcon(QIcon())  # You can set an icon if available

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
            self.graphics_layout.addPlot(row=0, col=0, title="Command Voltage"),
            self.graphics_layout.addPlot(row=1, col=0, title="Current Response")
        ]

        # Set labels
        self.plots[0].setLabel('left', 'Command Voltage', units='V')
        self.plots[0].setLabel('bottom', 'Time', units='s')
        self.plots[1].setLabel('left', 'Current Response', units='A')
        self.plots[1].setLabel('bottom', 'Time', units='s')

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
        self.buttons_layout.addWidget(self.run_dropdown)

        # Add the buttons layout to the main layout
        self.main_layout.addLayout(self.buttons_layout)

        # Keyboard shortcuts
        self.shortcut_left = QShortcut(Qt.Key_Left, self)
        self.shortcut_left.activated.connect(self.show_previous_protocol)
        self.shortcut_right = QShortcut(Qt.Key_Right, self)
        self.shortcut_right.activated.connect(self.show_next_protocol)

        # Initialize protocol variables
        self.protocols = []
        self.current_protocol_index = 0
        self.current_csv_files = []  # Initialize as empty list
        self.current_protocol_dir = ""
        self.current_protocol_name = ""

    def open_directory(self):
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
            else:
                self.info.setText("No Voltage Protocol Data Found")
                logging.warning("No Voltage Protocol Data Found")

            # Check for HoldingProtocol
            hold_dir = os.path.join(directory, 'HoldingProtocol')
            if os.path.exists(hold_dir):
                self.protocols.append(('HoldingProtocol', hold_dir))
                logging.info(f"Found HoldingProtocol at: {hold_dir}")
            else:
                self.info.setText("No Holding Protocol Data Found")
                logging.warning("No Holding Protocol Data Found")

            if self.protocols:
                logging.info(f"Available protocols: {self.protocols}")
                self.load_protocol(self.protocols[self.current_protocol_index])
            else:
                self.info.setText("No valid protocols found in the selected directory.")
                logging.error("No valid protocols found in the selected directory.")

    def load_protocol(self, protocol):
        protocol_name, protocol_dir = protocol
        logging.info(f"Loading protocol: {protocol_name} from {protocol_dir}")
        csv_files = [f for f in os.listdir(protocol_dir) if f.endswith('.csv')]

        logging.info(f"Found CSV files: {csv_files}")

        if not csv_files:
            self.info.setText(f"No valid CSV files found in {protocol_name} folder.")
            logging.error(f"No valid CSV files found in {protocol_name} folder.")
            return

        # Store protocol details before updating the dropdown
        self.current_csv_files = csv_files
        self.current_protocol_dir = protocol_dir
        self.current_protocol_name = protocol_name

        # Block signals to prevent on_run_selected from being called during initialization
        self.run_dropdown.blockSignals(True)
        self.run_dropdown.clear()
        for i in range(len(csv_files)):
            self.run_dropdown.addItem(f"Run {i + 1}")
        self.run_dropdown.setCurrentIndex(0)  # This will trigger on_run_selected
        self.run_dropdown.blockSignals(False)

        self.info.setText(f"Loading {protocol_name}...")
        logging.info(f"Loading {protocol_name} protocol...")

        # Now that current_csv_files is set, manually load the first run
        self.load_run(0)

    def load_run(self, run_index):
        if not self.current_csv_files:
            self.info.setText("No protocols loaded. Please select a data directory.")
            logging.error("No CSV files available to load.")
            return
        if run_index < 0 or run_index >= len(self.current_csv_files):
            self.info.setText(f"Run {run_index + 1} not available in {self.current_protocol_name}.")
            logging.error(f"Run index {run_index} is out of range.")
            return

        csv_file = self.current_csv_files[run_index]
        csv_path = os.path.join(self.current_protocol_dir, csv_file)
        logging.info(f"Loading CSV file: {csv_path}")

        try:
            # Read the CSV using pandas with any whitespace as delimiter
            df = pd.read_csv(csv_path, sep='\s+', header=None, names=['Time', 'CommandVoltage', 'CurrentResponse'])

            if df.empty:
                self.info.setText(f"No data loaded from {csv_file}.")
                logging.warning(f"No data loaded from {csv_file}.")
                return

            # Convert columns to lists
            time = df['Time'].astype(float).tolist()
            command_voltage = df['CommandVoltage'].astype(float).tolist()
            current_response = df['CurrentResponse'].astype(float).tolist()

            logging.info(f"Total data points loaded: {len(time)}")
            logging.debug(f"First 5 data points:\n{df.head()}")

            # Plot the data
            self.plots[0].clear()
            self.plots[1].clear()
            self.plots[0].plot(time, command_voltage, pen=pg.mkPen(color='b'), name="Command Voltage")
            self.plots[1].plot(time, current_response, pen=pg.mkPen(color='r'), name="Current Response")

            # Auto-scale the plots to fit the data
            self.plots[0].enableAutoRange()
            self.plots[1].enableAutoRange()

            self.info.setText(f"Loaded data from {csv_file}")
            logging.info(f"Data from {csv_file} plotted successfully.")
        except pd.errors.EmptyDataError:
            self.info.setText(f"No data found in {csv_file}.")
            logging.error(f"No data found in {csv_file}.")
        except pd.errors.ParserError as e:
            self.info.setText(f"Parsing error in {csv_file}: {e}")
            logging.error(f"Parsing error in {csv_file}: {e}")
        except ValueError as e:
            self.info.setText(f"Data conversion error in {csv_file}: {e}")
            logging.error(f"Data conversion error in {csv_file}: {e}")
        except Exception as e:
            self.info.setText(f"Unexpected error loading {csv_file}: {e}")
            logging.error(f"Unexpected error loading {csv_file}: {e}")

    def show_previous_protocol(self):
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
        # Implement logic to show previous timepoint
        self.info.setText("Previous timepoint functionality not implemented yet.")
        logging.info("Previous timepoint button clicked.")

    def show_next_timepoint(self):
        # Implement logic to show next timepoint
        self.info.setText("Next timepoint functionality not implemented yet.")
        logging.info("Next timepoint button clicked.")

    def toggle_video(self):
        # Implement logic to toggle video playback
        self.info.setText("Play/Pause video functionality not implemented yet.")
        logging.info("Toggle video button clicked.")

    def on_run_selected(self, run_index):
        logging.info(f"Run selected: {run_index}")
        # Ensure that run_index is within bounds
        if 0 <= run_index < len(self.current_csv_files):
            self.load_run(run_index)
        else:
            self.info.setText("Selected run index is out of range.")
            logging.error("Selected run index is out of range.")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = PatchAnalyzer()
    window.show()
    sys.exit(app.exec_())
