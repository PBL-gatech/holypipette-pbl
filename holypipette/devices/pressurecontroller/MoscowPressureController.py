
'''
Pressure Controller classes to communicate with the Pressure Controller Box made by the IBB.
Additionally, redesigning a closed loop pressure controller for the Moscow Rig
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

all = ['MoscowPressureController']

class MoscowPressureController(PressureController):
    '''A PressureController child class that handles serial communication between the PC and
       the Arduino controlling the Moscow Pressure box
    '''
    validProducts = ["USB Serial"] #TODO: move to a constants or json file?
    validVIDs = [0x1a86, 0x403]
                    
    nativePerMbar = 5 # The number of native pressure transucer units from the DAC (0 to 4095) in a millibar of pressure (-446 to 736)
    nativeZero = 2048 # The native units at a 0 pressure (y-intercept)

    serialCmdTimeout = 1 # (in sec) max time allowed between sending a serial command and expecting a response

    def __init__(self, channel, controllerSerial=None,readerSerial=None):
        super(MoscowPressureController, self).__init__()

        if controllerSerial is not None:
            # no port specified, we will use the user supplied serial port
            self.controllerSerial = controllerSerial
            
        if readerSerial is not None:
            # no port specified, we will use the user supplied serial port
            self.readerSerial= readerSerial

        self.channel = channel
        self.isATM = None
        self.setpoint_raw = None
        self.expectedResponses = collections.deque() #use a deque instead of a list for O(1) pop from beginning

        self.lastVal = 0

        #setup a deamon thread to ensure arduino responses are correct
        # self.responseDeamon = threading.Thread(target = self.waitForArduinoResponses)
        # self.responseDeamon.setDaemon(True) #make sure thread dies with main thread
        # self.responseDeamon.start()

        time.sleep(2) #wait for arduino to boot up

        #set initial configuration of pressure controller
        self.set_ATM(False)
        self.set_pressure(20)

    def autodetectSerial(self):
        '''Use VID and name of serial devices to figure out which one is the Moscow Pressure box
        '''

        allPorts = [COMPort for COMPort in serial.tools.list_ports.comports()]
        logging.info(f"Attempting to find Moscow Pressure Box from: {[(p.product, hex(p.vid) if p.vid != None else None, p.name) for p in allPorts]}")

    def set_pressure(self, pressure):
        '''Tell pressure controller to go to a given setpoint pressure in mbar
        '''
        nativeUnits = self.mbarToNative(pressure)
        self.set_pressure_raw(nativeUnits)
    
    def mbarToNative(self, pressure):
        '''Comvert from a pressure in mBar to native units
        '''
        raw_pressure = int(pressure * MoscowPressureController.nativePerMbar + MoscowPressureController.nativeZero)
        return min(max(raw_pressure, 0), 4095) #clamp native units to 0-4095

    def nativeToMbar(self, raw_pressure):
        '''Comvert from native units to a pressure in mBar
        '''
        pressure = (raw_pressure - MoscowPressureController.nativeZero) / MoscowPressureController.nativePerMbar
        return pressure

    def set_pressure_raw(self, raw_pressure):
        '''Tell pressure controller to go to a given setpoint pressure in native DAC units
        '''
        self.setpoint_raw = raw_pressure
        logging.info(f"Setting pressure to {self.nativeToMbar(raw_pressure)} mbar (raw: {raw_pressure})")

        cmd = f"set {self.channel} {raw_pressure}\n"
        # logging.info(type(cmd))
        # logging.info(cmd)
        # logging.info(bytes(cmd, 'ascii'))
        logging.info(f"Sending command: {cmd}")
        self.controllerSerial.write(bytes(cmd, 'ascii'))
        # print cmd in ascii
        self.controllerSerial.flush()
        logging.info(f"SenT command: {cmd}")

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
        # obtains pressure 
        # return self.get_setpoint() #maybe add a pressure sensor down the line?
        readvalue = self.read_sensor()
        # logging.warning(f"Pressure: {readvalue} mbar")
        # logging.warning(f"TYPE: {type(readvalue)}")
        if readvalue is None:
            readvalue = self.lastVal
        else:
            self.lastVal = readvalue
        return readvalue
    
    def read_sensor(self):
        '''Read the pressure sensor value from the arduino
        '''
        pressure = self.readerSerial.readline().decode('utf-8').strip()
        if pressure == None or pressure == "": #no data received
            logging.error("No data received from pressure sensor")
            return None
        else: 
            try: 
                pressure = float(pressure)
                # logging.info(f"Pressure: {pressure} mbar")
                return pressure
            except ValueError:
                logging.error(f"Received non-numeric data: {pressure}, type: {type(pressure)}")
                return None


    def measure(self):
        return self.get_pressure()
    

    def pulse(self, delayMs):
        '''Tell the onboard arduino to pulse pressure for a certain period of time
        '''
        cmd = f"pulse {self.channel} {delayMs}\n"
        logging.info(f"Pulsing pressure for {delayMs} ms")
        self.controllerSerial.write(bytes(cmd, 'ascii')) #do serial writing in main thread for timing?
        self.controllerSerial.flush()
        
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
        self.controllerSerial.write(bytes(cmd, 'ascii'))
        self.controllerSerial.flush()

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

    # def waitForArduinoResponses(self):
    #     '''Continuously ensure that all expected responses are received within the timeout period.
    #        Runs in a deamon thread.
    #     '''
    #     while True:
    #         if len(self.expectedResponses) == 0 and self.controllerSerial.in_waiting == 0:
    #             time.sleep(0.1)
    #             continue #nothing to do
            
    #         #check for new responses
    #         resp = self.controllerSerial.readline().decode("ascii")
    #         while len(resp) > 0: #process all commands 
    #             #remove newlines from string
    #             resp = resp.replace('\n', '')
    #             resp = resp.replace('\r', '')
                
    #             #grab latest expected response
    #             if len(self.expectedResponses) > 0:
    #                 sendTime, expected = self.expectedResponses.popleft()
    #             else:
    #                 expected = None

    #             #make what was actually received and what was expected match
    #             if resp != expected:
    #                 logging.info(f"INVALID PRESSURE COMMAND, EXPECTED RESPONSE {expected} BUT GOT {resp}")
    #                 self.expectedResponses.clear()
                
    #             #grab the next line
    #             resp = self.controllerSerial.readline().decode("ascii")
            
    #         while len(self.expectedResponses) > 0 and time.time() - self.expectedResponses[0][0] > self.serialCmdTimeout:
    #             #the response on top of expectedResponses has timed out!
    #             self.expectedResponses.popleft() #remove timed out response
    #             logging.info("PRESSURE BOX SERIAL RESPONSE TIMEOUT!")
            
    #         time.sleep(0.01) #sleep less when there might be things to do shortly