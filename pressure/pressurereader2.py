import serial
import time
from threading import Thread
import pyqtgraph as pg
from PyQt5.QtWidgets import QApplication
import numpy as np

# Setup the serial connection (adjust the COM port as necessary)
ser = serial.Serial('COM9', 9600, timeout=0.1)  # Shorter timeout

# Initialize Qt app for plotting
app = QApplication([])

# Set up the plot window using GraphicsLayoutWidget
win = pg.GraphicsLayoutWidget(show=True, title="Real-time Sensor Data")
win.setWindowTitle('Real-time Sensor Data Plotting')
p = win.addPlot(title="Sensor Values")
curve = p.plot(pen='y')

data = np.zeros(100)  # Preallocate buffer array to hold the last 100 data points for speed

def send_command():
    while True:
        ser.write(b'R')  # Send the 'R' command to the Arduino
        time.sleep(1/30)  # Sleep for approximately 33.33 milliseconds

def receive_data():
    global data
    while True:
        if ser.in_waiting > 0:
            line = ser.readline().decode().strip()
            if line.startswith('S') and line.endswith('E'):
                numeric_part = line[1:-1]
                try:
                    value = float(numeric_part)
                    print(value)
                    data[:-1] = data[1:]  # Shift data in the array one sample left
                    data[-1] = value  # Add new sample at the end
                except ValueError:
                    continue

def update():
    curve.setData(data)  # Update the plot

timer = pg.QtCore.QTimer()
timer.timeout.connect(update)  # Timer triggers plot updates
timer.start(50)  # Redraw the plot every 50ms

# Start threads for sending and receiving data
sender_thread = Thread(target=send_command, daemon=True)
receiver_thread = Thread(target=receive_data, daemon=True)

sender_thread.start()
receiver_thread.start()

# Start Qt event loop unless running in interactive mode or using pyside
if __name__ == '__main__':
    QApplication.instance().exec_()

ser.close()
