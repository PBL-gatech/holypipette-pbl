import serial
import time

import threading
from threading import Thread

from patcherbot.devices.manipulator.manipulator import Manipulator
try:
    from .CellSorterSerial import SerialCommands
except:
    print('NDA\'d CellSorter Serial Commands not available')

__all__ = ['CellSorterController']

class FakeCellSorterController():
    '''A fake cell sorter controller for testing purposes
    '''
    def __init__(self):
        #fake cell sorter state variables
        self.led = False
        self.led_ring = 1
        self.valve1 = False
        self.valve2 = False

    def is_online(self):
        '''Sends a test command to the CellSorter and returns True if the response is correct
        '''
        return True

    def set_led(self, status: bool):
        '''Sets the LED status
        '''
        print('CELLSORTER: set led {}'.format(status))
        self.led = status

    def get_led(self):
        '''Returns the LED status
        '''
        return self.led
    
    def set_led_ring(self, ring: int):
        '''Sets the LED ring to use
        '''
        print('CELLSORTER: set led ring {}'.format(ring))
        self.led_ring = ring
    
    def set_valve(self, valve: int, open: bool):
        '''Sets the given valve to be open or closed
        '''
        print('CELLSORTER: set valve {} {}'.format(valve, open))
        if valve == 1:
            self.valve1 = open
        elif valve == 2:
            self.valve2 = open

    def get_valve(self, valve: int):
        '''Returns the status of the given valve
        '''
        print('get valve {}'.format(valve))
        if valve == 1:
            return self.valve1
        elif valve == 2:
            return self.valve2

    ###  these functions are not implemented in the fake cell sorter ###
    def open_valve_for_time(self, valve: int, timeMs: int):
        print('open valve {} for {} ms'.format(valve, timeMs))

    def open_valve_for_prev_time(self, valve: int):
        print('open valve {} for prev time'.format(valve))
    
    def set_valve_delay(self, delayMs: int):
        print('set valve delay to {}'.format(delayMs))
    
    def open_valve_1_2(self):
        print('open valve 1 2')
    
    def open_valve_2_1(self):
        print('open valve 2 1')

class CellSorterController():
    '''A class to control a physical cell sorter control box over serial.
       The control box allows for LED and valve control
    '''

    def __init__(self, comPort: serial.Serial):
        self.comPort : serial.Serial = comPort

        # turn on ring 1 of the LED
        self.set_led_ring(1)
        self.set_led(True)

    def _sendCmd(self, cmd):
        for a in cmd:
            self.comPort.write(a.encode()) #send test command
        self.comPort.flush()
        time.sleep(0.1) #TODO: replace with something better

        resp = self.comPort.read_all() #read reply to message
        resp = resp.decode()
        resp = resp[:-3] #remove trailing \r\n\x00 at the end of each reply
        return resp
    
    def is_online(self):
        '''Sends a test command to the CellSorter and returns True if the response is correct
        '''
        resp = self._sendCmd(SerialCommands.TEST)
        return resp == 'R'

    def set_led(self, status: bool):
        '''Sets the LED status
        '''
        if status:
            self._sendCmd(SerialCommands.LED_ON)
        else:
            self._sendCmd(SerialCommands.LED_OFF)

    def get_led(self):
        '''Returns the LED status
        '''
        resp = self._sendCmd(SerialCommands.LED_STATUS)
        return resp == '+'
    
    def set_led_ring(self, ring: int):
        '''Sets the LED ring to use
        '''
        if ring == 1:
            self._sendCmd(SerialCommands.LED_SELECT_RING_1)
        elif ring == 2:
            self._sendCmd(SerialCommands.LED_SELECT_RING_2)
        else:
            raise ValueError('Ring must be 1 or 2')
    
    def set_valve(self, valve: int, open: bool):
        '''Sets the given valve to be open or closed
        '''
        if valve == 1:
            if open:
                self._sendCmd(SerialCommands.VALVE_1_OPEN)
            else:
                self._sendCmd(SerialCommands.VALVE_1_CLOSE)
        elif valve == 2:
            if open:
                self._sendCmd(SerialCommands.VALVE_2_OPEN)
            else:
                self._sendCmd(SerialCommands.VALVE_2_CLOSE)
        else:
            raise ValueError('Valve must be 1 or 2')

    def get_valve(self, valve: int):
        '''Returns the status of the given valve
        '''
        if valve == 1:
            resp = self._sendCmd(SerialCommands.VALVE_1_STATUS)
        elif valve == 2:
            resp = self._sendCmd(SerialCommands.VALVE_2_STATUS)
        else:
            raise ValueError('Valve must be 1 or 2')
        
        return resp == '+'

    def open_valve_for_time(self, valve: int, timeMs: int):
        '''Open a valve for the given time in milliseconds
        '''
        if valve == 1:
            self._sendCmd(SerialCommands.VALVE_1_OPEN_TIME.format(timeMs))
            self._sendCmd(SerialCommands.VALVE_1_OPEN_PREV_TIME)
        elif valve == 2:
            self._sendCmd(SerialCommands.VALVE_2_OPEN_TIME.format(timeMs))
            self._sendCmd(SerialCommands.VALVE_2_OPEN_PREV_TIME)
        else:
            raise ValueError('Valve must be 1 or 2')

    def open_valve_for_prev_time(self, valve: int):
        '''Open a valve for the last time specified
        '''
        if valve == 1:
            self._sendCmd(SerialCommands.VALVE_1_OPEN_PREV_TIME)
        elif valve == 2:
            self._sendCmd(SerialCommands.VALVE_2_OPEN_PREV_TIME)
        else:
            raise ValueError('Valve must be 1 or 2')
    
    def set_valve_delay(self, delayMs: int):
        '''Set the delay between valve openings
        '''
        self._sendCmd(SerialCommands.VALVE_DELAY.format(delayMs))
    
    def open_valve_1_2(self):
        '''Open valve 1, wait the specified delay, then open valve 2
        '''
        self._sendCmd(SerialCommands.VALVE_1_2_OPEN)
    
    def open_valve_2_1(self):
        '''Open valve 2, wait the specified delay, then open valve 1
        '''
        self._sendCmd(SerialCommands.VALVE_2_1_OPEN)

    def __del__(self):
        '''Close the serial connection when the object is destroyed
        '''
        self.comPort.close()


class CellSorterManip(Manipulator):
    '''A class to control a physical cell sorter manipulator over serial.
       The manipulator allows for z-axis commands and position feedback
    '''

    def __init__(self, comPort: serial.Serial):
        self.comPort : serial.Serial = comPort

        self.serialLock = threading.Lock()
        self.maxVel = 25
        self.maxAcc = 2

        #set init commands
        self._sendCmd(SerialCommands.SET_PITCH)
        self._sendCmd(SerialCommands.SET_CUR)
        self._sendCmd(SerialCommands.SET_REDUCTION)
        self._sendCmd(SerialCommands.SET_CUR_DELAY)
        self._sendCmd(SerialCommands.SET_SEC_VEL)

        self.zPos = 0
        #spawn z position update thread
        self.posUpdateThread = Thread(target=self._update_position_continuous, daemon=True)
        self.posUpdateThread.start()


    def _sendCmd(self, cmd):
        self.serialLock.acquire()
        self.comPort.write(cmd.encode()) #send test command
        self.comPort.flush()
        time.sleep(0.1) #TODO: replace with something better

        resp = self.comPort.read_all() #read reply to message
        resp = resp.decode()
        self.serialLock.release()
        resp = resp[:-3] #remove trailing \r\n\x00 at the end of each reply
        return resp
    
    def is_online(self):
        '''Sends a test command to the CellSorter and returns True if the response is correct
        '''
        resp = self._sendCmd(SerialCommands.GET_VERSION)
        return resp[:4] == 'Vers' # good responses start with Vers
    
    def relative_move(self, pos: float, velocity=None):
        '''Sets the position relative to the current position
        '''

        if velocity == None:
            velocity = self.maxVel

        self._sendCmd(SerialCommands.SET_VEL.format(velocity, velocity, velocity))
        self._sendCmd(SerialCommands.SET_ACCEL.format(self.maxAcc))
        self._sendCmd(SerialCommands.SET_POS_REL.format(pos))
        self._sendCmd(SerialCommands.JOY_2)
    
    def absolute_move(self, pos: float, velocity=None):
        '''Sets the absolute position
        '''
        if velocity == None:
            velocity = self.maxVel

        self._sendCmd(SerialCommands.SET_VEL.format(velocity, velocity, velocity))
        self._sendCmd(SerialCommands.SET_ACCEL.format(self.maxAcc))
        self._sendCmd(SerialCommands.SET_POS_ABS.format(pos))

    def stop(self, axis=None):
        '''Stops all movement immediately
        '''
        self._sendCmd(SerialCommands.EMERGENCY_STOP)

    def _update_position_continuous(self):
        while True:
            resp = self._sendCmd(SerialCommands.GET_POS)
            resp = resp.split(' ')
            self.zPos = float(resp[2])
            time.sleep(0.2)

    def wait_until_still(self, axes = None):
        """
        Waits until motors have stopped.

        Parameters
        ----------
        axes : list of axis numbers
        """
        previous_position = self.zPos
        new_position = None
        while new_position is None or previous_position != new_position:
            previous_position = new_position
            time.sleep(0.3)
            new_position = self.zPos

    def position(self):
        '''Returns the current position of the z-axis
        '''
        return self.zPos
        
    def set_max_speed(self, vel: float):
        '''Sets the velocity of the z-axis (velocity as a % from 0.1 to 100)
        '''
        if vel < 0.1 or vel > 100:
            raise ValueError('Velocity must be between 0.1 and 100')
        
        nativeVel = vel / 2

        self.maxVel = nativeVel
        self._sendCmd(SerialCommands.SET_VEL.format(self.maxVel, self.maxVel, self.maxVel))
    
    def set_max_accel(self, accel: float):
        '''Sets the acceleration of the z-axis from 1% to 100%
        '''
        if accel < 1 or accel > 100:
            raise ValueError('Acceleration must be between 1 and 100')

        nativeAccel = accel / 50
        self.maxAcc = nativeAccel
        self._sendCmd(SerialCommands.SET_ACCEL.format(self.maxAcc))

    def read_SW(self):
        '''Returns the current state of the switches
        '''
        resp = self._sendCmd(SerialCommands.READ_SW)
        return resp

    def __del__(self):
        '''Close the serial connection when the object is destroyed
        '''
        self.comPort.close()


class FakeCellSorterManip(Manipulator):
    ''' A fake cell sorter manipulator for testing purposes
    '''
    def __init__(self, z_pos: float = 0):
        self.z_pos = z_pos

    def is_online(self):
        return True
    
    def set_max_accel(self, accel: float):
        pass

    def set_max_speed(self, vel: float):
        pass

    def absolute_move(self, pos: float, velocity=None):
        self.z_pos = pos

    def relative_move(self, pos: float, velocity=None):
        self.z_pos += pos
    
    def position(self):
        return self.z_pos
    
    def stop(self):
        pass

    def wait_until_still(self, axes = None):
        """
        Waits until motors have stopped.

        Parameters
        ----------
        axes : list of axis numbers
        """
        return True



if __name__ == '__main__':

    testController = False

    if testController:

        # Create a serial connection to the CellSorter
        # As per documentation: 115200 baud, 8 data bits, no parity, 1 stop bit
        controllerSerial = serial.Serial('COM7', 115200, timeout=2, parity=serial.PARITY_NONE, stopbits=1, 
                                            bytesize=8, write_timeout=1, inter_byte_timeout=2)
        # controllerSerial.dtr = False
        # controllerSerial.rts = False

        cellSorterController = CellSorterController(controllerSerial)

        # Check that the CellSorter is online
        if not cellSorterController.is_online():
            print('CellSorter not online')
            exit(1)
        
        # Set the LED ring to use
        cellSorterController.set_led_ring(1)

        # Flicker the LED on and off 5 times
        for i in range(5):
            cellSorterController.set_led(True)
            time.sleep(0.5)
            cellSorterController.set_led(False)
            time.sleep(0.5)

        #close serial
        del cellSorterController
    
    else:
        manipulatorSerial = serial.Serial('COM14', 57600, timeout=2, parity=serial.PARITY_NONE, stopbits=2, 
                                            bytesize=8, write_timeout=1, inter_byte_timeout=2)
        # manipulatorSerial.dtr = True
        # manipulatorSerial.rts = False

        cellSorterManip = CellSorterManip(manipulatorSerial)

        # Check that the CellSorter is online
        if not cellSorterManip.is_online():
            print('CellSorter not online')
            exit(1)
        
        for i in range(3):
            # move down
            for i in range(3):
                cellSorterManip.set_pos_rel(-1)
                time.sleep(0.5)
            
            #move up
            for i in range(3):
                cellSorterManip.set_pos_rel(1)
                time.sleep(0.5)

        #close serial
        del cellSorterManip

