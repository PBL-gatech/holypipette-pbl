'''
Pressure Controller classes to communicate with the Pressure Controller Box made by the IBB
'''
from logging import exception
import logging
from .BasePressureController import PressureController
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

                    
    DEFAULT_NATIVE_PER_MBAR = 0.75  # native pressure units per mbar
    DEFAULT_NATIVE_ZERO = 2048      # native units at 0 pressure (y-intercept)
    DEFAULT_SERIAL_CMD_TIMEOUT = 1
    DEFAULT_STARTUP_PRESSURE = 20


    def __init__(self, channel, arduinoSerial=None,
                 native_zero=None, native_per_mbar=None,
                 serial_cmd_timeout=None, startup_pressure=None):
        """
        Initializes the IBB pressure controller.

        Args:
            channel (int): Channel number on the pressure box.
            arduinoSerial (serial.Serial, optional): Serial interface to Arduino. Defaults to None.
            native_zero (int, optional): DAC units at 0 mBar. Defaults to DEFAULT_NATIVE_ZERO.
            native_per_mbar (float, optional): DAC units per mBar. Defaults to DEFAULT_NATIVE_PER_MBAR.
            serial_cmd_timeout (float, optional): Timeout for serial commands in seconds. Defaults to DEFAULT_SERIAL_CMD_TIMEOUT.
            startup_pressure (float, optional): Initial pressure in mBar. Defaults to DEFAULT_STARTUP_PRESSURE.
        """
        super().__init__()

        self.serial = arduinoSerial

        self.channel = channel
        self.isATM = None
        self.setpoint_raw = None

        self.nativeZero = self.DEFAULT_NATIVE_ZERO if native_zero is None else native_zero
        self.nativePerMbar = self.DEFAULT_NATIVE_PER_MBAR if native_per_mbar is None else native_per_mbar
        self.serialCmdTimeout = self.DEFAULT_SERIAL_CMD_TIMEOUT if serial_cmd_timeout is None else serial_cmd_timeout
        self.startup_pressure = self.DEFAULT_STARTUP_PRESSURE if startup_pressure is None else startup_pressure
        time.sleep(2) #wait for arduino to boot up

        #set initial configuration of pressure controller
        self.set_ATM(False)
        self.set_pressure(self.startup_pressure)

    def set_pressure(self, pressure):
        '''Tell pressure controller to go to a given setpoint pressure in mbar
        
        Args:
            pressure (float): Target pressure in mBar.
        '''
        nativeUnits = self.mbarToNative(pressure)
        self.set_pressure_raw(nativeUnits)
    
    def mbarToNative(self, pressure):
        '''Comvert from a pressure in mBar to native units
        
        Args:
            pressure (float): Pressure in mBar.

        Returns:
            int: Clamped DAC units (0-4095).
        '''
        raw_pressure = int(pressure * self.nativePerMbar + self.nativeZero)
        return min(max(raw_pressure, 0), 4095) #clamp native units to 0-4095

    def nativeToMbar(self, raw_pressure):
        '''Comvert from native units to a pressure in mBar
        
        Args:
            raw_pressure (int): DAC units.

        Returns:
            float: Pressure in mBar.
        '''
        pressure = (raw_pressure - self.nativeZero) / self.nativePerMbar
        return pressure

    def set_pressure_raw(self, raw_pressure):
        '''Tell pressure controller to go to a given setpoint pressure in native DAC units
        
        Args:
            raw_pressure (int): Target DAC units.
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
       
        Returns:
            float: Pressure in mBar.
        '''
        return self.nativeToMbar(self.setpoint_raw)

    def get_setpoint_raw(self):
        '''Gets the current setpoint in native DAC units
        
        Returns:
            int: Pressure in DAC units.
        '''
        logging.info(f"Current setpoint: {self.nativeToMbar(self.setpoint_raw)} mbar (raw: {self.setpoint_raw})")
        return self.setpoint_raw
    
    def get_pressure(self):
        """
        Returns the current pressure in mBar.

        Returns:
            float: Current pressure in mBar.
        """
        return self.get_setpoint() #maybe add a pressure sensor down the line?
    
    
    def measure(self):
        """
        Returns the current measured pressure.

        Returns:
            float: Current pressure in mBar.
        """
        return self.get_pressure()
    

    def pulse(self, delayMs):
        '''Tell the onboard arduino to pulse pressure for a certain period of time
        
        Args:
            delayMs (int): Pulse duration in milliseconds.
        '''
        cmd = f"pulse {self.channel} {delayMs}\n"
        logging.info(f"Pulsing pressure for {delayMs} ms")
        self.serial.write(bytes(cmd, 'ascii')) #do serial writing in main thread for timing?
        self.serial.flush()
    


    def set_ATM(self, atm):
        '''Send a serial command activating or deactivating the atmosphere solenoid valve
           atm = True -> pressure output is at atmospheric pressure 
           atm = False -> pressure output comes from pressure regulator

        Args:
            atm (bool): True to switch to atmospheric pressure, False for regulated pressure.
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
