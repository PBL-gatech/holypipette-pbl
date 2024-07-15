'''
Pressure Controller classes to communicate with the Pressure Controller Box made by the IBB
'''
from logging import exception
import logging
from .BasePressureController import PressureController
import serial.tools.list_ports
import serial
import time
logging.basicConfig(level=logging.INFO)

all = ['TestPressureController']

class TestPressureController(PressureController):
    '''A PressureController child class that handles serial communication between the PC and
       the Arduino controlling the IBB Pressure box
    '''

                    
    nativePerMbar = 0.75 # The number of native pressure transucer units from the DAC (0 to 4095) in a millibar of pressure (-700 to 700)
    nativeZero = 2048 # The native units at a 0 pressure (y-intercept)


    def __init__(self, channel, controllerSerial=None, readerSerial=None):
        super().__init__()
        try:
            self.controllerSerial = controllerSerial
            self.readerSerial = readerSerial
        except Exception as e:
            logging.error(f"Error initializing pressure controller: {e}")
            return

        self.channel = channel
        self.isATM = None
        self.setpoint_raw = None
        self.lastVal = None

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
        raw_pressure = int(pressure * TestPressureController.nativePerMbar + TestPressureController.nativeZero)
        return min(max(raw_pressure, 0), 4095) #clamp native units to 0-4095

    def nativeToMbar(self, raw_pressure):
        '''Comvert from native units to a pressure in mBar
        '''
        pressure = (raw_pressure - TestPressureController.nativeZero) / TestPressureController.nativePerMbar
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
        self.controllerSerial.write(bytes(cmd, 'ascii'))
        self.controllerSerial.flush()

    def get_pressure(self):
        '''
        Get the current pressure reading from readerSerial
        '''
        # return self.lastVal
        pressureVal = self.lastVal
        if self.readerSerial.in_waiting > 0:
            reading = self.readerSerial.readline().decode('utf-8').strip()
            # print(reading)

            # check that S and E are in the string only once and that S is the first index and E is the last index
            # 
            # adding more checks results in lag somehow maybe, freezing. 
            # the GUI actuallu unfreezes due to an empty string being read in and therefore
            # pressure[0] or [-1] is indexing out of bounds, introducing a lag where the computer catches up and suddenly
            # can read values?!!
            # if "S" in pressure and "E" in pressure and pressure[0] == "S" and pressure[-1] == "E":
            # if pressure.startswith("S") and pressure.endswith("E"):
            # pressure.count("S") == 1 and pressure.count("E") == 1 and pressure.find("S") < pressure.find("E"):
            if reading[0] == "S" and reading[-1] == "E":
            # Extract the pressure reading between the markers
                pressure_str = reading[1:-1]

                # Try to convert the extracted pressure string to float
                try:
                    pressureVal = float(pressure_str)
                    # logging.info(f"Pressure: {pressureVal} mbar")
                    self.lastVal = pressureVal 
                except ValueError:
                    # pressureVal = None
                    logging.warning("Invalid pressure data received")
            else:
                # pressureVal = None
                logging.warning("Incomplete or invalid data received")
        else:
            # pressureVal = None
            logging.warning("No data received from pressure sensor")

        return pressureVal

    
    
    def getLastVal(self):
        return self.lastVal
      
    def measure(self):
        return self.get_pressure()
    
    def pulse(self, delayMs):
        '''Tell the onboard arduino to pulse pressure for a certain period of time
        '''
        cmd = f"pulse {self.channel} {delayMs}\n"
        logging.info(f"Pulsing pressure for {delayMs} ms")
        self.controllerSerial.write(bytes(cmd, 'ascii')) #do serial writing in main thread for timing?
        self.controllerSerial.flush()
    
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
        self.controllerSerial.write(bytes(cmd, 'ascii'))
        self.controllerSerial.flush()

        self.isATM = atm
    def toggle_ATM(self,atm):
        '''Toggle the atmosphere solenoid valve
        '''
        if atm: 
            self.set_ATM(False)
        else:   
            self.set_ATM(True)