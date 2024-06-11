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
    validProducts = ["USB Serial"] #TODO: move to a constants or json file?
    validVIDs = [0x1a86, 0x403]
                    
    nativePerMbar = 0.75 # The number of native pressure transucer units from the DAC (0 to 4095) in a millibar of pressure (-700 to 700)
    # nativePerMbar = 0.2925 # The number of native pressure transucer units from the DAC (0 to 4095) in a millibar of pressure (-700 to 700)
    nativeZero = 2048 # The native units at a 0 pressure (y-intercept)


    def __init__(self, channel, arduinoSerial=None):
        super(IBBPressureController, self).__init__()

        if arduinoSerial == None:
            # no port specified, we need to auto detect the arduino's serial interface
            self.serial = self.autodetectSerial()
        else:
            # user sepecified a Serial interface for the arduino
            self.serial = arduinoSerial

        self.channel = channel
        self.isATM = None
        self.setpoint_raw = None
        self.expectedResponses = collections.deque() #use a deque instead of a list for O(1) pop from beginning
        self.serialCmdTimeout = 1 # (in sec) max time allowed between sending a serial command and expecting a response

        # #setup a deamon thread to ensure arduino responses are correct
        # self.responseDeamon = threading.Thread(target = self.waitForArduinoResponses)
        # self.responseDeamon.setDaemon(True) #make sure thread dies with main thread
        # self.responseDeamon.start()

        time.sleep(2) #wait for arduino to boot up

        #set initial configuration of pressure controller
        self.set_ATM(False)
        self.set_pressure(20)

    def autodetectSerial(self):
        '''Use VID and name of serial devices to figure out which one is the IBB Pressure box
        '''

        allPorts = [COMPort for COMPort in serial.tools.list_ports.comports()]
        logging.info(f"Attempting to find IBB Pressure Box from: {[(p.product, hex(p.vid) if p.vid != None else None, p.name) for p in allPorts]}")

        possiblePorts = []
        for p in serial.tools.list_ports.comports():
            if p.product in IBBPressureController.validProducts and p.vid in IBBPressureController.validVIDs:
                possiblePorts.append(p)
        
        if len(possiblePorts) == 1:
            port = possiblePorts[0]
            logging.info(f"Auto detected IBBPressureBox on port {port.name}")
        else:
            exception(f"Could not find pressure controller serial interface from ports: {[p.name for p in possiblePorts]}")
            exit()
        
        arduSerial = serial.Serial(port=port.device, baudrate=9600, timeout=3)
        logging.info("waiting for arduino to init...")
        time.sleep(2) #wait for serial port to init #TODO: wait for serial responses instead?
        logging.info("done")

        return arduSerial

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

        #add expected arduino responces
        self.expectedResponses.append((time.time(), f"set {self.channel} {raw_pressure}"))
        self.expectedResponses.append((time.time(), f"set"))
        self.expectedResponses.append((time.time(), f"{self.channel}"))
        self.expectedResponses.append((time.time(), f"{raw_pressure}"))

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
        
        #add expected arduino responces
        self.expectedResponses.append((time.time(), f"pulse {self.channel} {delayMs}"))
        self.expectedResponses.append((time.time(), f"pulse"))
        self.expectedResponses.append((time.time(), f"{self.channel}"))
        self.expectedResponses.append((time.time(), f"{delayMs}"))


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

        #add the expected arduino responses 
        if atm:
            self.expectedResponses.append((time.time(), f"switchAtm {self.channel}"))
            self.expectedResponses.append((time.time(), f"switchAtm"))
            self.expectedResponses.append((time.time(), f"{self.channel}"))
            self.expectedResponses.append((time.time(), f"0"))
        else:
            self.expectedResponses.append((time.time(), f"switchP {self.channel}"))
            self.expectedResponses.append((time.time(), f"switchP"))
            self.expectedResponses.append((time.time(), f"{self.channel}"))
            self.expectedResponses.append((time.time(), f"0"))

    def waitForArduinoResponses(self):
        '''Continuously ensure that all expected responses are received within the timeout period.
           Runs in a deamon thread.
        '''
        while True:
            if len(self.expectedResponses) == 0 and self.serial.in_waiting == 0:
                time.sleep(0.1)
                continue #nothing to do
            
            #check for new responses
            resp = self.serial.readline().decode("ascii")
            logging.info(f"Received BEFORE PROCESSING: {resp}")
            while len(resp) > 0: #process all commands 
                #remove newlines from string
                resp = resp.replace('\n', '')
                resp = resp.replace('\r', '')
                
                #grab latest expected response
                if len(self.expectedResponses) > 0:
                    sendTime, expected = self.expectedResponses.popleft()
                else:
                    expected = None

                logging.info(f"Received: {resp}")
                logging.info(f"Expected: {expected}")
                # make what was actually received and what was expected match
                if resp != expected:
                    logging.info(f"INVALID PRESSURE COMMAND, EXPECTED RESPONSE {expected} BUT GOT {resp}")
                    self.expectedResponses.clear()
                
                #grab the next line
                resp = self.serial.readline().decode("ascii")
            
            while len(self.expectedResponses) > 0 and time.time() - self.expectedResponses[0][0] > self.serialCmdTimeout:
                #the response on top of expectedResponses has timed out!
                # ! THIS POPS THE FIRST CHAR NOT THE WORD
                self.expectedResponses.popleft() #remove timed out response
                # logging.error(self.expectedResponses.popleft()) #remove timed out response
                logging.info("PRESSURE BOX SERIAL RESPONSE TIMEOUT!")
            
            time.sleep(0.01) #sleep less when there might be things to do shortly