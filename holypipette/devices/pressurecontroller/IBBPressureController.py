'''
Pressure Controller classes to communicate with the Pressure Controller Box made by the IBB
'''
from logging import exception
import logging
from .pressurecontroller import PressureController
import serial.tools.list_ports
import serial
import time
import threading
import collections
logging.basicConfig(level=logging.INFO)

all = ['IBBPressureController']

class IBBPressureController(PressureController):
    '''A PressureController child class that handles serial communication between the PC and
       the Arduino controlling the IBB Pressure box
    '''

                    
    nativePerMbar = 0.75 # The number of native pressure transucer units from the DAC (0 to 4095) in a millibar of pressure (-700 to 700)
    nativeZero = 2048 # The native units at a 0 pressure (y-intercept)


    def __init__(self, channel, arduinoSerial=None):
        super().__init__()

        self.serial = arduinoSerial

        self.channel = channel
        self.isATM = None
        self.setpoint_raw = None

        self.serialCmdTimeout = 1 # (in sec) max time allowed between sending a serial command and expecting a response
        time.sleep(2) #wait for arduino to boot up

        #set initial configuration of pressure controller
        self.set_ATM(False)
        self.set_pressure(20)

    def set_pressure(self, pressure):
        '''Tell pressure controller to go to a given setpoint pressure in mbar
        '''
        nativeUnits = self.mbarToNative(pressure)
        self.set_pressure_raw(nativeUnits)
    
    def mbarToNative(self, pressure):
        '''Comvert from a pressure in mBar to native units
        '''
        raw_pressure = int(pressure * IBBPressureController.nativePerMbar + IBBPressureController.nativeZero)
        return min(max(raw_pressure, 0), 4095) #clamp native units to 0-4095

    def nativeToMbar(self, raw_pressure):
        '''Comvert from native units to a pressure in mBar
        '''
        pressure = (raw_pressure - IBBPressureController.nativeZero) / IBBPressureController.nativePerMbar
        return pressure

    def set_pressure_raw(self, raw_pressure):
        '''Tell pressure controller to go to a given setpoint pressure in native DAC units
        '''
        self.setpoint_raw = raw_pressure
        logging.info(f"Setting pressure to {self.nativeToMbar(raw_pressure)} mbar (raw: {raw_pressure})")

        cmd = f"set {self.channel} {raw_pressure}\n"
        logging.info(f"Sending command: {cmd}")
        
        logging.info(type(cmd))
        logging.info(cmd)
        logging.info(bytes(cmd, 'ascii'))
        self.serial.write(bytes(cmd, 'ascii'))
        self.serial.flush()

    def get_setpoint(self):
        '''Gets the current setpoint in millibar
        '''
        return self.nativeToMbar(self.setpoint_raw)

    def get_setpoint_raw(self):
        '''Gets the current setpoint in native DAC units
        '''
        logging.info(f"Current setpoint: {self.nativeToMbar(self.setpoint_raw)} mbar (raw: {self.setpoint_raw})")
        return self.setpoint_raw
    
    def get_pressure(self):
        return self.get_setpoint() #maybe add a pressure sensor down the line?
    
    
    def measure(self):
        return self.get_pressure()
    

    def pulse(self, delayMs):
        '''Tell the onboard arduino to pulse pressure for a certain period of time
        '''
        cmd = f"pulse {self.channel} {delayMs}\n"
        logging.info(f"Pulsing pressure for {delayMs} ms")
        self.serial.write(bytes(cmd, 'ascii')) #do serial writing in main thread for timing?
        self.serial.flush()
    


    def set_ATM(self, atm):
        '''Send a serial command activating or deactivating the atmosphere solenoid valve
           atm = True -> pressure output is at atmospheric pressure 
           atm = False -> pressure output comes from pressure regulator
        '''
        if atm:
            cmd = f"switchAtm {self.channel}\n" #switch to ATM command
            logging.info("Switching to ATM")
        else:
            cmd = f"switchP {self.channel}\n" #switch to Pressure command
            logging.info("Switching to Pressure")
        self.serial.write(bytes(cmd, 'ascii'))
        self.serial.flush()

        self.isATM = atm
