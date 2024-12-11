from holypipette.devices.pressurecontroller import IBBPressureController
from serial import Serial
import time
from threading import Thread
import io
import numpy as np
import matplotlib.pyplot as plt

pressureBoxSerialPort = '/dev/cu.usbserial-AC012G83'
pressureReaderSerialPort = '/dev/cu.usbmodem141301'

pressureSerial = Serial(port=pressureBoxSerialPort, baudrate=9600, timeout=0)
pressure = IBBPressureController(channel=1, arduinoSerial=pressureSerial)
pressure_readings = []
freq = 15 #Hz

#setup pressure reading
readerSerial = Serial(port=pressureReaderSerialPort, baudrate=115200, timeout=1000)
readerSerial.read_all() #clear buffer

input_path = 'emulateProfile.csv'
isRunning = True

#load input profile from csv
input_profile = np.genfromtxt(input_path, delimiter=',')
raw_times = input_profile[:, 0]
raw_setpoints = input_profile[:, 1]

#make sure times start at 0
raw_times -= raw_times[0]

#we only care about setpoints at 10 Hz
lastTime = raw_times[-1]
dt = 1/freq
times = np.arange(0, lastTime, dt)

#replace last time with lastTime (so we get to proper endpoint)
times[-1] = lastTime

#interpolate setpoints at given times to get setpoints at 10 Hz
setpoints = np.interp(times, raw_times, raw_setpoints, left=raw_setpoints[0], right=raw_setpoints[-1])

#plot raw setpoints and interpolated setpoints in a new window
plt.figure()
plt.plot(raw_times, raw_setpoints, '-', label='raw setpoints')
plt.plot(times, setpoints, 'o', label='interpolated setpoints')
plt.legend()

#add title of input file
plt.title(f'input profile: {input_path}')
plt.show()

def recordPressure():
    while isRunning:
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
            pressure_readings.append([time.time(), pressure])

T = Thread(target = recordPressure)
T.start()          

#wait for pressure box to initialize
time.sleep(1)
pressure.set_ATM(False)
print('pressure set to 0')
pressure.set_pressure(0)
time.sleep(0.5)

#send setpoints to pressure box at 10 Hz
phase_lag = 0
setpoint_info = []
for setpoint in setpoints:
    print(f'pressure set to {setpoint}')
    pressure.set_pressure(setpoint)
    setpoint_info.append([time.time() + phase_lag, setpoint])
    time.sleep(1 / freq)
setpoint_info = np.array(setpoint_info)

time.sleep(0.5)
print('pressure set to 0')
pressure.set_pressure(0)
time.sleep(0.5)

isRunning = False
pressure_readings = np.array(pressure_readings)

#shift raw setpoint time to match pressure readings
print(setpoint_info.shape, pressure_readings.shape)
raw_times += (setpoint_info[0, 0] - pressure_readings[0, 0])

#pressure readings should start at 0
setpoint_info[:, 0] -= pressure_readings[0, 0]
pressure_readings[:, 0] -= pressure_readings[0, 0]

# plot pressure readings and setpoints in a new window
plt.figure()
plt.plot(setpoint_info[:, 0], setpoint_info[:, 1], 'o', color='tab:orange', label='setpoints')
plt.plot(raw_times, raw_setpoints, '--', color='tab:red', label='original recording')
plt.plot(pressure_readings[:, 0], pressure_readings[:, 1], '-', color='tab:blue', label='pressure reading')
plt.legend()

#add x and y labels
plt.xlabel('time (s)')
plt.ylabel('pressure (mbar)')
plt.title(f'pressure readings from: {input_path} (lag = {phase_lag}s, freq={freq}Hz)')
plt.show()
