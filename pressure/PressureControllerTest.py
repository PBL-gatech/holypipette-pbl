'''
GUI to test the pressure controller
'''
import sys
import threading
import serial
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtCore import pyqtSignal, QObject
import pyqtgraph as pg
from datetime import datetime

class SerialReader(QObject):
    data_received = pyqtSignal(float, datetime)

    def __init__(self, port, baud_rate):
        super().__init__()
        self.port = port
        self.baud_rate = baud_rate
        self.serial = serial.Serial(port=self.port, baudrate=self.baud_rate, timeout=1)
        self.running = True

    def start(self):
        thread = threading.Thread(target=self.run)
        thread.start()

    def run(self):
        while self.running:
            if self.serial.in_waiting > 0:
                line = self.serial.readline().decode('utf-8').strip()
                try:
                    number = float(line)
                    timestamp = datetime.now()
                    self.data_received.emit(number, timestamp)
                except ValueError:
                    print("Received non-numeric data:", line)
                except KeyboardInterrupt:
                    print("STOPPING")
                    return

    def stop(self):
        self.running = False
        self.serial.close()

class SerialGrapher(QMainWindow):
    def __init__(self, com_port):
        super().__init__()

        # Graph setup
        self.graphWidget = pg.PlotWidget()
        self.setCentralWidget(self.graphWidget)
        self.graphWidget.setBackground('w')
        self.data_line = self.graphWidget.plot([], [], pen=pg.mkPen('b', width=2))
        self.data = []
        self.timestamps = []

        # Configure plot to show time on x-axis
        self.graphWidget.getPlotItem().setAxisItems({'bottom': pg.DateAxisItem()})

        # Setup serial reader
        self.reader = SerialReader(com_port, 9600)
        self.reader.data_received.connect(self.update_plot)
        self.reader.start()

    def update_plot(self, value, timestamp):
        self.data.append(value)
        self.timestamps.append(timestamp.timestamp())  # Convert datetime to timestamp
        if len(self.data) > 100:
            self.data.pop(0)
            self.timestamps.pop(0)
        self.data_line.setData(self.timestamps, self.data)

    def closeEvent(self, event):
        self.reader.stop()
        super().closeEvent(event)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    com_port = 'COM9'  # Set your COM port here
    main = SerialGrapher(com_port)
    main.show()
    sys.exit(app.exec_())
