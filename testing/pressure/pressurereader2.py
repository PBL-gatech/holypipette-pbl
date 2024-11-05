import sys
import serial
import serial.tools.list_ports
import csv
import os
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QLabel, 
    QComboBox, QMessageBox, QTextEdit, QGroupBox, QDialog, QLineEdit, QFileDialog
)
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt, QDateTime, QTimer, QThread, pyqtSignal
import numpy as np
import pyqtgraph as pg
import time

class SerialReaderThread(QThread):
    data_received = pyqtSignal(float, float)  # Signal to emit timestamp and value
    error_occurred = pyqtSignal(str)          # Signal to emit error messages

    def __init__(self, serial_port, parent=None):
        super().__init__(parent)
        self.serial_port = serial_port
        self.is_running = True

    def run(self):
        while self.is_running and self.serial_port and self.serial_port.is_open:
            try:
                if self.serial_port.in_waiting > 0:
                    line = self.serial_port.readline().decode('utf-8').strip()
                    if line.startswith('S') and line.endswith('E'):
                        numeric_part = line[1:-1]
                        try:
                            value = float(numeric_part)
                            # value = (value - 520.72) / 0.3923  # Adjusted conversion for pressure sensor
                            # value = (value - 521)/0.213  # Adjusted conversion for pressure sensor
                            timestamp = time.time()
                            self.data_received.emit(timestamp, value)
                        except ValueError:
                            self.error_occurred.emit("Received invalid data format.")
                else:
                    self.msleep(10)  # Sleep briefly to prevent CPU overuse
            except Exception as e:
                self.error_occurred.emit(f"Error reading data: {e}")
                break

    def stop(self):
        self.is_running = False
        self.wait()

class SerialWriterThread(QThread):
    error_occurred = pyqtSignal(str)  # Signal to emit error messages

    def __init__(self, serial_port, parent=None):
        super().__init__(parent)
        self.serial_port = serial_port
        self.is_running = True

    def run(self):
        while self.is_running and self.serial_port and self.serial_port.is_open:
            try:
                self.serial_port.write(b'R')
                self.msleep(int(1000/30))  # Approximately 33 ms
            except Exception as e:
                self.error_occurred.emit(f"Error sending command: {e}")
                break

    def stop(self):
        self.is_running = False
        self.wait()

class PressureReaderApp(QWidget):
    def __init__(self):
        super().__init__()
        self.serial_port = None
        self.is_connected = False
        self.is_reading = False
        self.is_recording = False  # Initialize the recording flag
        self.csv_file = None
        self.csv_writer = None
        self.recording_count = 0  # Counter to track the number of recording sessions
        self.reader_thread = None
        self.writer_thread = None
        self.initUI()

        # Variables for plotting
        self.plot_data = np.zeros(100)  # Buffer for last 100 data points

    def initUI(self):
        self.setFixedSize(700, 500)  # Updated window size
        layout = QHBoxLayout()

        # Left side - Logo and Display
        self.left_layout = QVBoxLayout()

        # PressureReader Logo
        self.logo_label = QLabel("PressureReader", self)
        self.logo_label.setFont(QFont("Comic Sans MS", 32))  # Adjusted font size to make logo smaller
        self.logo_label.setAlignment(Qt.AlignCenter)
        self.left_layout.addWidget(self.logo_label)

        # Initialize and add plot_widget here (before data_display)
        self.init_plot()

        # Data Display (Hidden when plot is shown)
        self.data_display = QTextEdit(self)
        self.data_display.setReadOnly(True)
        self.data_display.setFixedHeight(60)  # Set height for the dialog box
        self.left_layout.addWidget(self.data_display)

        layout.addLayout(self.left_layout)

        # Right side - Buttons and Setup
        right_layout = QVBoxLayout()

        # Communication Box
        communication_box = self.create_communication_box()
        right_layout.addWidget(communication_box)

        # Data Acquisition Box
        acquisition_box = self.create_acquisition_box()
        right_layout.addWidget(acquisition_box)

        layout.addLayout(right_layout)

        self.setLayout(layout)
        self.setWindowTitle('Pressure Reader Control')

    def create_communication_box(self):
        """ Create communication box with COM port selection """
        com_box = QGroupBox("Communication")
        com_layout = QVBoxLayout()

        # COM Port dropdown
        self.com_selector = QComboBox(self)
        self.refresh_com_ports()
        com_layout.addWidget(self.com_selector)

        # Open COM Port Button
        self.open_com_button = QPushButton("Open COM Port", self)
        self.open_com_button.clicked.connect(self.open_com_port)
        com_layout.addWidget(self.open_com_button)

        com_box.setLayout(com_layout)
        return com_box

    def create_acquisition_box(self):
        """ Create data acquisition box with Start Read and Start Record buttons """
        acquisition_box = QGroupBox("Data Acquisition")
        acquisition_layout = QVBoxLayout()

        # Start Read / Stop Read Button
        self.read_button = QPushButton("Start Read", self)
        self.read_button.setCheckable(True)
        self.read_button.setEnabled(False)  # Initially disabled
        self.read_button.toggled.connect(self.toggle_read)
        acquisition_layout.addWidget(self.read_button)

        # Start Record / Stop Record Button
        self.record_button = QPushButton("Start Record", self)
        self.record_button.setCheckable(True)
        self.record_button.setEnabled(False)  # Disabled until reading starts
        self.record_button.toggled.connect(self.toggle_record)
        acquisition_layout.addWidget(self.record_button)

        acquisition_box.setLayout(acquisition_layout)
        return acquisition_box

    def refresh_com_ports(self):
        """ Refresh the list of available COM ports and set default """
        ports = serial.tools.list_ports.comports()
        self.com_selector.clear()
        default_port = "COM8"  # Adjust default port as needed
        available_ports = [port.device for port in ports]
        if default_port in available_ports:
            # set default port if available
            self.com_selector.addItem(default_port)
            # add all other available ports to the dropdown
            self.com_selector.addItems([port for port in available_ports if port != default_port])
        else:
            self.data_display.append(f"Default port {default_port} not available.")
            self.com_selector.addItems(available_ports)

    def open_com_port(self):
        """ Open the selected COM port for communication """
        selected_port = self.com_selector.currentText()
        if selected_port:
            try:
                self.serial_port = serial.Serial(selected_port, 9600, timeout=0.05)  # Adjusted baud rate and shorter timeout
                self.is_connected = True
                self.read_button.setEnabled(True)  # Enable read button when connected
                QMessageBox.information(self, "Success", f"Connected to {selected_port}")
            except serial.SerialException as e:
                QMessageBox.critical(self, "Error", f"Failed to open {selected_port}: {e}")
        else:
            QMessageBox.warning(self, "Warning", "Please select a valid COM port")

    def toggle_read(self, checked):
        """ Start or stop reading data from the pressure sensor """
        if checked:
            self.read_button.setText("Stop Read")
            self.is_reading = True

            # Start the reader thread
            self.reader_thread = SerialReaderThread(self.serial_port)
            self.reader_thread.data_received.connect(self.handle_new_data)
            self.reader_thread.error_occurred.connect(self.handle_error)
            self.reader_thread.start()

            # Start the writer thread to send 'R' commands at ~30 FPS
            self.writer_thread = SerialWriterThread(self.serial_port)
            self.writer_thread.error_occurred.connect(self.handle_error)
            self.writer_thread.start()

            # Enable recording button
            self.record_button.setEnabled(True)

            # Switch to plot view
            self.show_plot()
        else:
            self.read_button.setText("Start Read")
            self.is_reading = False

            # Stop reader thread
            if self.reader_thread and self.reader_thread.isRunning():
                self.reader_thread.stop()

            # Stop writer thread
            if self.writer_thread and self.writer_thread.isRunning():
                self.writer_thread.stop()

            # Disable recording button
            self.record_button.setEnabled(False)

            # Switch back to logo view
            self.hide_plot()

    def toggle_record(self, checked):
        """ Start or stop recording the data to a CSV file """
        if checked:
            self.record_button.setText("Stop Recording")
            self.is_recording = True  # Set recording flag to True
            directory = QFileDialog.getExistingDirectory(self, "Select Directory")
            if directory:
                self.create_csv_file(directory)
            else:
                QMessageBox.warning(self, "Warning", "No directory selected for recording.")
                self.record_button.setChecked(False)
                self.is_recording = False  # Reset recording flag
        else:
            self.record_button.setText("Start Record")
            self.is_recording = False  # Set recording flag to False
            self.close_csv_file()

    def create_csv_file(self, directory):
        """ Create a new CSV file in the selected directory with an incrementer """
        self.recording_count += 1  # Increment the recording counter
        date_str = QDateTime.currentDateTime().toString("MM_dd_yyyy")
        filename = f"PressureReaderRec_{date_str}_{self.recording_count}.csv"  # Append recording count
        filepath = os.path.join(directory, filename)
        try:
            self.csv_file = open(filepath, 'w', newline='')
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow(["Timestamp", "Pressure (mbar)"])  # Write header (adjusted for pressure sensor)
            self.data_display.append(f"Recording started: {filename}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to create CSV file: {e}")
            self.record_button.setChecked(False)
            self.is_recording = False

    def close_csv_file(self):
        """ Close the CSV file """
        if self.csv_file:
            try:
                self.csv_file.close()
                self.data_display.append("Recording stopped.")
            except Exception as e:
                QMessageBox.warning(self, "Warning", f"Failed to close CSV file: {e}")
            finally:
                self.csv_file = None
                self.csv_writer = None

    def handle_new_data(self, timestamp, value):
        """ Handle new data received from the serial port """
        # Display data
        self.data_display.append(f"time: {timestamp:.2f}s, pressure: {value:.2f}mbar")

        # Update plot data buffer
        self.plot_data[:-1] = self.plot_data[1:]
        self.plot_data[-1] = value

        # If recording, write to CSV
        if self.is_recording and self.csv_writer:
            self.csv_writer.writerow([timestamp, value])

    def handle_error(self, message):
        """ Handle errors emitted from threads """
        self.data_display.append(message)
        QMessageBox.critical(self, "Error", message)
        # Optionally, stop reading if a critical error occurs
        self.toggle_read(False)

    def init_plot(self):
        """ Initialize the PyQtGraph plot widget but keep it hidden initially """
        self.plot_widget = pg.PlotWidget(title="Pressure Sensor Data")
        # Make plot background white
        self.plot_widget.setBackground('w')
        self.plot_curve = self.plot_widget.plot(pen='k')
        self.plot_widget.hide()  # Hidden initially
        self.left_layout.addWidget(self.plot_widget)

    def update_plot(self):
        """ Update the plot with the latest data """
        if self.plot_widget.isVisible():
            self.plot_curve.setData(self.plot_data)

    def show_plot(self):
        """ Show the plot and hide the logo """
        self.logo_label.hide()
        self.plot_widget.show()

    def hide_plot(self):
        """ Hide the plot and show the logo """
        self.plot_widget.hide()
        self.logo_label.show()

    def closeEvent(self, event):
        """ Handle application exit """
        self.is_reading = False
        # Stop reader thread
        if self.reader_thread and self.reader_thread.isRunning():
            self.reader_thread.stop()
        # Stop writer thread
        if self.writer_thread and self.writer_thread.isRunning():
            self.writer_thread.stop()
        if self.serial_port and self.serial_port.is_open:
            self.serial_port.close()
        if self.csv_file:
            self.close_csv_file()
        event.accept()

    def showEvent(self, event):
        """ Initialize the plot timer when the window is shown """
        super().showEvent(event)
        # Initialize plot timer after the window is shown
        self.plot_timer = QTimer()
        self.plot_timer.timeout.connect(self.update_plot)
        self.plot_timer.start(33)  # Update plot every 50 ms

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = PressureReaderApp()
    ex.show()
    sys.exit(app.exec_())
