from holypipette.devices.pressurecontroller import IBBPressureController
from serial import Serial
import time
from threading import Thread
import io
import numpy as np
import matplotlib.pyplot as plt

csv_path = 'pressureRecording.csv'
serial_device = '/dev/cu.usbmodem144201'
record_time = 90 #seconds

#setup pressure reading
readerSerial = Serial(port=serial_device, baudrate=115200, timeout=1000)
readerSerial.read_all() #clear buffer

isRunning = True
pressure_readings = []

def recordPressure():
    ''' Continuously records pressure readings from the pressure controller
        and appends them to the pressure_readings list. (Meant to be run in a seperate thread.)
    '''
    startTime = time.time()
    print('starting pressure recording...')
    while time.time() - startTime < record_time:
        #read until newline
        try:
            pressure_reading = readerSerial.read_until(b'\r\n', size=100)
            pressure_reading = pressure_reading.decode('ascii').strip()
        except:
            print('error reading pressure')
            continue

        if pressure_reading != '':
            #check if pressure_reading is a float
            pressure = -1
            pressure = float(pressure_reading)
            pressure_readings.append([time.time() - startTime, pressure])

recorderThread = Thread(target = recordPressure)
recorderThread.start()   
recorderThread.join() #wait for thread to finish

#clear file
with open(csv_path, 'w') as f:
    f.write('')

#write pressure readings to file
with open(csv_path, 'w') as f:
    for reading in pressure_readings:
        readTime = reading[0]
        readPressure = reading[1]

        f.write('{},{},\r'.format(readTime, readPressure))

#plot pressure_readings with matplotlib
pressure_readings = np.array(pressure_readings)

#plot pressure readings
plt.plot(pressure_readings[:,0], pressure_readings[:,1], 'b-')

# plt.plot(set_times[:,0], set_times[:,1], 'r-')
plt.xlabel('Time (s)')
plt.ylabel('Pressure (mbar)')
plt.title('Pressure Profile')
plt.legend(['Pressure Readings', 'Setpoints'])

plt.show()