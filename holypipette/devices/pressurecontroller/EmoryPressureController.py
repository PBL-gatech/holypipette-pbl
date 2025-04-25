
'''
Pressure Controller classes to communicate with the Pressure Controller Box made by the IBB.
Additionally, redesigning a closed loop pressure controller for the Emory Rig
'''

import logging
from .BasePressureController import PressureController
import serial.tools.list_ports
import serial
import time
import collections


all = ['EmoryPressureController']

class EmoryPressureController(PressureController):
    '''A PressureController child class that handles serial communication between the PC and
       the Arduino controlling the Emory Pressure box
    '''
    validProducts = ["USB Serial"] # TODO: move to a constants or json file?
    validVIDs = [0x1a86, 0x403]
    nativeZero = 2161 # The native units at a 0 pressure (y-intercept)
    nativePerMbar = float(3078/1000) # The number of native pressure transducer units from the DAC (0 to 4095) in a millibar of pressure (-400 to 700)
    serialCmdTimeout = 1 # (in sec) max time allowed between sending a serial command and expecting a response

    def __init__(self, channel, controllerSerial = None, readerSerial = None):
        super().__init__()
        # time.sleep(2) # wait for arduino to boot up

        if controllerSerial is not None:
            # no port specified, we will use the user supplied serial port
            self.controllerSerial = controllerSerial
        else:
            self.controllerSerial = None
            self.error("No controller serial port available")
        
        if readerSerial is not None:
            # no port specified, we will use the user supplied serial port
            self.readerSerial = readerSerial
        else: 
            self.readerSerial = None
            self.error("No reader serial port available")

        self.channel = channel
        self.state = None
        self.setpoint_raw = None
        self.lastVal = 0.0

        # set initial configuration of pressure controller
        self.set_ATM(False)
        self.set_pressure(0) # set initial pressure to 0 mbar
        self.start_acquisition() # start pressure acquisition thread at 60 hz


    def set_pressure(self, pressure):
        '''
        Tell pressure controller to go to a given setpoint pressure in mbar
        '''
        nativeUnits = self.mbarToNative(pressure)
        # self.info(f"Setting pressure to {nativeUnits} mbar")
        self.set_pressure_raw(nativeUnits)
    
    def mbarToNative(self, pressure):
        '''
        Comvert from a pressure in mBar to native units
        '''
        raw_pressure = int((pressure * EmoryPressureController.nativePerMbar + EmoryPressureController.nativeZero))
        return min(max(raw_pressure, 0), EmoryPressureController.nativeZero*2) # clamp native units to 0-3924


    def nativeToMbar(self, raw_pressure) -> float:
        '''
        Comvert from native units to a pressure in mBar
        '''
        pressure = (raw_pressure - EmoryPressureController.nativeZero) / EmoryPressureController.nativePerMbar
        return pressure

    def set_pressure_raw(self, raw_pressure: int):
        '''
        Tell pressure controller to go to a given setpoint pressure in native DAC units
        '''
        self.setpoint_raw = raw_pressure
        self.info(f"Setting pressure to {self.nativeToMbar(raw_pressure)} mbar (raw: {raw_pressure})")

        cmd = f"set {self.channel} {raw_pressure}\n"
        self.info(f"Sending command: {cmd}")
        self.controllerSerial.write(bytes(cmd, 'ascii'))
        self.controllerSerial.flush()
        self.info(f"Sent command: {cmd}")

    def get_pressure(self) -> float:
        '''
        Gets the current setpoint in millibar
        '''
        return self.nativeToMbar(self.setpoint_raw)

    def measure(self) -> float:
        '''
        Read the pressure sensor value from the Arduino
        '''
        pressureVal = self.lastVal

        # Send a request command to the Arduino
        self.readerSerial.write(b'R')
        # Wait for the response
        if self.readerSerial.in_waiting > 0:
            reading = self.readerSerial.readline().decode('utf-8').strip()
            if reading.startswith("S") and reading.endswith("E"):
                pressure_str = reading[1:-1]
                try:
                    pressureVal = float(pressure_str)
                    pressureVal = float((pressureVal - 516.72)/0.3923) # conversion to raw because the seeed is not working
                    self.lastVal = pressureVal
                except ValueError:
                    self.warning("Invalid pressure data received")

        else:
            self.warning("No data received from pressure sensor")
        return pressureVal
    
    def pulse(self, delayMs):
        '''Tell the onboard arduino to pulse pressure for a certain period of time
        '''
        cmd = f"pulse {self.channel} {delayMs}\n"
        self.info(f"Pulsing pressure for {delayMs} ms")
        self.controllerSerial.write(bytes(cmd, 'ascii')) #do serial writing in main thread for timing?
        self.controllerSerial.flush()
        
    def set_ATM(self, atm):
        '''Send a serial command activating or deactivating the atmosphere solenoid valve
           atm = True -> pressure output is at atmospheric pressure 
           atm = False -> pressure output comes from pressure regulator
        '''
        if atm:
            cmd = f"switchAtm {self.channel}\n" # switch to ATM command
            self.info("Switching to ATM")
        else:
            cmd = f"switchP {self.channel}\n" # switch to Pressure command
            self.info("Switching to Pressure")
        self.controllerSerial.write(bytes(cmd, 'ascii'))
        self.controllerSerial.flush()
        self.state = atm

