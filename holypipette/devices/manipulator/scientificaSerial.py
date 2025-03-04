
import serial
from .manipulator import Manipulator
import time
import threading

__all__ = ['ScientificaSerial']

class SerialCommands():
    GET_X_POS = 'PX\r'
    GET_Y_POS = 'PY\r'
    GET_Z_POS = 'PZ\r'
    GET_X_Y_Z = '\r'
    GET_MAX_SPEED = 'TOP\r'
    GET_MAX_ACCEL = 'ACC\r'
    GET_IS_BUSY = 's\r'

    SET_X_Y_POS_ABS = 'abs {} {}\r'
    SET_X_Y_POS_REL = 'rel {} {}\r'
    SET_X_Y_Z_POS_ABS = 'abs {} {} {}\r'
    SET_X_Z_POS_REL = 'rel {} {}\r'
    

    SET_Z_POS = 'absz {}\r'
    SET_MAX_SPEED = 'TOP {}\r'
    SET_MAX_ACCEL = 'ACC {}\r'

    SET_X_Y_Z_VEL = 'VJ {} {} {}\r'

    STOP = 'STOP\r'

class ScientificaSerialEncoder(Manipulator):

    def __init__(self, comPort: serial.Serial, zAxisComPort):
        self.comPort : serial.Serial = comPort

        self.zAxisComPort : serial.Serial = zAxisComPort
        self.stageUnitsPerEncoderPulse = 1.45
        self.encoderZ = 0

        self._lock = threading.Lock()
        self.current_pos = [0, 0, 0]

        self.set_max_accel(100)
        self.set_max_speed(10000)

        #start constantly polling position in a new thread
        self._polling_thread = threading.Thread(target=self.update_pos_continuous, daemon=True)
        self._polling_thread.start()
        self._polling_thread.deamon = True
    
    def set_max_speed(self, speed):
        '''Sets the max speed for the Scientifica Stage.  
           It seems like the range for this is around (1000, 100000)
        '''
        self._sendCmd(SerialCommands.SET_MAX_SPEED.format(int(speed)))

    def set_max_accel(self, accel):
        '''Sets the max acceleration for the Scientifica Stage.
           It seems like the range for this is around (10, 10000)
        '''
        self._sendCmd(SerialCommands.SET_MAX_ACCEL.format(int(accel)))

    def __del__(self):
        self.comPort.close()

    def _sendCmd(self, cmd):
        '''Sends a command to the stage and returns the response
        '''

        self._lock.acquire()
        self.comPort.write(cmd.encode())
        resp = self.comPort.read_until(b'\r') #read reply to message
        resp = resp[:-1]
        self._lock.release()

        return resp.decode()

    def position(self, axis=None):
        if axis == 1:
            return self.current_pos[0]
        if axis == 2:
            return self.current_pos[1]
        if axis == 3:
            return self.encoderZ
        if axis == None:
            return self.current_pos
        
    def update_pos_continuous(self, freq=10):
        '''constantly polls the device's position and updates the current_pos variable
        '''
        while True:
            startTime = time.time()
            self.zAxisComPort.read_all()
            self.zAxisComPort.read_until(b'\r\n')
            encoderZ = self.zAxisComPort.read_until(b'\r\n').strip()
            encoderZ = int(encoderZ)
            self.encoderZ = encoderZ * self.stageUnitsPerEncoderPulse

            xyz = self._sendCmd(SerialCommands.GET_X_Y_Z)
            xyz = xyz.split('\t')
            
            try:
                xPos = int(xyz[0]) / 10.0
                yPos = int(xyz[1]) / 10.0
                zPos = int(xyz[2]) / 10.0
                self.current_pos = [xPos, yPos, zPos]
            except:
                print('error reading position')

            sleepTime = 1 / freq - (time.time() - startTime)
            if sleepTime > 0:
                time.sleep(sleepTime)

    def absolute_move(self, pos, axis):
        if axis == 1:
            yPos = self.position(axis=2)
            self._sendCmd(SerialCommands.SET_X_Y_POS_ABS.format(int(pos * 10) , int(yPos * 10)))
        if axis == 2:
            xPos = self.position(axis=1)
            self._sendCmd(SerialCommands.SET_X_Y_POS_ABS.format(int(xPos * 10), int(pos * 10)))
        if axis == 3:
            stageZ = self.current_pos[2]
            setpointStage = stageZ + (pos - self.encoderZ)
            self._sendCmd(SerialCommands.SET_Z_POS.format(int(setpointStage * 10)))
            self.wait_until_still()
            time.sleep(1)
            print(f'expected encoder: {pos} actual {self.encoderZ}')
            error = pos - self.encoderZ
            print(f'error: {error}')
            if abs(error) > 2:
                print('retrying')
                self.absolute_move(pos, axis)
    
    def absolute_move_group(self, x, axes, speed=None):
        x = list(x)
        axes = list(axes)

        if 1 in axes and 2 in axes:
            # Move X and Y axes together
            xPos = x[axes.index(1)]
            yPos = x[axes.index(2)]
            print("sent cmd", xPos, yPos)
            self._sendCmd(SerialCommands.SET_X_Y_POS_ABS.format(int(xPos * 10), int(yPos * 10)))

        elif 1 in axes and 2 in axes and 3 in axes:
            # Move X, Y and Z axes together
            xPos = x[axes.index(1)]
            yPos = x[axes.index(2)]
            zPos = x[axes.index(3)]
            print("sent cmd", xPos, yPos, zPos)
            self._sendCmd(SerialCommands.SET_X_Y_Z_POS_ABS.format(int(xPos * 10), int(yPos * 10), int(zPos * 10)))

        else:
            print(f'unimplemented move group {x} {axes}')

    
    def relative_move_group(self, pos, axis, speed=None):
        if axis == 1:
            self._sendCmd(SerialCommands.SET_X_Y_POS_REL.format(pos, 0))
        if axis == 2:
            self._sendCmd(SerialCommands.SET_X_Y_POS_REL.format(0, pos))
        if axis == 3:
            absZCmd = self.position(3) + pos
            self.absolute_move(absZCmd, 3)

    def relative_move_group(self, x, axes):
        cmd = [0, 0, 0]
        for pos, axis in zip(x, axes):
            cmd[axis  - 1] = pos
        
        if cmd[0] != 0 or cmd[1] != 0:
            self._sendCmd(SerialCommands.SET_X_Y_POS_REL.format(int(cmd[0] * 10), int(cmd[1] * 10)))

        if cmd[2] != 0:
            self.relative_move(cmd[2], 3)

    def absolute_move_group_velocity(self,vel,axes):   
        try: 
         vel = list(vel)
         axes = list(axes)
         xvel = vel[axes.index(1)]
         yvel = vel[axes.index(2)]
         zvel = vel[axes.index(3)]
         self._sendCmd(SerialCommands.SET_X_Y_Z_VEL.format(xvel, yvel, zvel))
        except Exception as e:
            self.error(f"Error in absolute_move: {e}")

    def wait_until_still(self, axes = None, axis = None):
        while True:
            resp = self._sendCmd(SerialCommands.GET_IS_BUSY)
            busy = resp != '0'

            if not busy:
                break

    def stop(self):
        self._sendCmd(SerialCommands.STOP)


class ScientificaSerialNoEncoder(Manipulator):

    def __init__(self, comPort: serial.Serial):
        self.comPort : serial.Serial = comPort
        self._lock = threading.Lock()
        self.current_pos = [0, 0, 0]

        self.set_max_accel(1000)
        self.set_max_speed(100000)

        #start constantly polling position in a new thread
        self._polling_thread = threading.Thread(target=self.update_pos_continuous, daemon=True)
        self._polling_thread.start()
        self._polling_thread.deamon = True
    
    def set_max_speed(self, speed):
        '''Sets the max speed for the Scientifica Stage.  
           It seems like the range for this is around (1000, 100000)
        '''
        self._sendCmd(SerialCommands.SET_MAX_SPEED.format(int(speed)))

    def set_max_accel(self, accel):
        '''Sets the max acceleration for the Scientifica Stage.
           It seems like the range for this is around (10, 10000)
        '''
        self._sendCmd(SerialCommands.SET_MAX_ACCEL.format(int(accel)))

    def __del__(self):
        self.comPort.close()

    def _sendCmd(self, cmd):
        '''Sends a command to the stage and returns the response
        '''
        
        self._lock.acquire()
        self.comPort.write(cmd.encode())
        resp = self.comPort.read_until(b'\r') #read reply to message
        resp = resp[:-1]
        self._lock.release()
        return resp.decode()

    def position(self, axis=None):
        if axis == 1:
            return self.current_pos[0]
        if axis == 2:
            return self.current_pos[1]
        if axis == 3:
            return self.current_pos[2]
        if axis == None:
            return self.current_pos
        
    def update_pos_continuous(self, freq=100):
        '''constantly polls the device's position and updates the current_pos variable
        '''
        while True:
            startTime = time.time()
            xyz = self._sendCmd(SerialCommands.GET_X_Y_Z)
            xyz = xyz.split('\t')
            
            try:
                xPos = int(xyz[0]) / 10.0
                yPos = int(xyz[1]) / 10.0
                zPos = int(xyz[2]) / 10.0
                self.current_pos = [xPos, yPos, zPos]
            except:
                print('error reading position')

            sleepTime = 1 / freq - (time.time() - startTime)
            if sleepTime > 0:
                time.sleep(sleepTime)

    def absolute_move(self, pos, axis, speed=None):
        '''Moves the device to an absolute position in um.
        '''
        # self.abort_if_requested()
        try: 
            if axis == 1:
                yPos = self.position(axis=2)
                self._sendCmd(SerialCommands.SET_X_Y_POS_ABS.format(int(pos * 10) , int(yPos * 10)))
            if axis == 2:
                xPos = self.position(axis=1)
                self._sendCmd(SerialCommands.SET_X_Y_POS_ABS.format(int(xPos * 10), int(pos * 10)))
            if axis == 3:
                self._sendCmd(SerialCommands.SET_Z_POS.format(int(pos * 10)))
        except Exception as e:
            self.error(f"Error in absolute_move: {e}")
    
    def absolute_move_group(self, x, axes, speed=None):
        '''
        Moves the device group of axes to position x.
        Parameters
        ----------
        axes : list of axis numbers
        x : target position in um (vector or list).
        '''
        # self.abort_if_requested()
        x = list(x)
        axes = list(axes)

        if 1 in axes and 2 in axes and 3 not in axes:
            # Move X and Y axes together
            xPos = x[axes.index(1)]
            yPos = x[axes.index(2)]
            self._sendCmd(SerialCommands.SET_X_Y_POS_ABS.format(int(xPos * 10), int(yPos * 10)))

        elif 1 in axes and 2 in axes and 3 in axes:
            # Move X, Y and Z axes together
            xPos = x[axes.index(1)]
            yPos = x[axes.index(2)]
            zPos = x[axes.index(3)]
            self._sendCmd(SerialCommands.SET_X_Y_Z_POS_ABS.format(int(xPos * 10), int(yPos * 10), int(zPos * 10)))


        else:
            print(f'unimplemented move group {x} {axes}')

    def absolute_move_group_velocity(self, vel):
        '''
        Moves the device in um/s.
        Parameters
        ----------
        vel : list of velocities for each axis
        '''
        # self.abort_if_requested()   
        try: 
            vel = list(vel)
            if len(vel) != 3:
                raise ValueError("Expected velocity list of length 3: [xvel, yvel, zvel]")
            xvel, yvel, zvel = vel
            self._sendCmd(SerialCommands.SET_X_Y_Z_VEL.format(xvel, yvel, zvel))
        except Exception as e:
            self.error(f"Error in absolute_move: {e}")
        
    def relative_move_group(self, pos, axis, speed=None):
        '''Moves the device axis by relative amount pos in um.
        Parameters
        ----------
        axis : axis number starting at 0; if None, all XYZ axes
        pos : position shift in um.
        '''
        # self.abort_if_requested()
        if axis == 1:
            self._sendCmd(SerialCommands.SET_X_Y_POS_REL.format(pos, 0))
        if axis == 2:
            self._sendCmd(SerialCommands.SET_X_Y_POS_REL.format(0, pos))
        if axis == 3:
            absZCmd = self.position(3) + pos
            self.absolute_move(absZCmd, 3)

    def relative_move_group(self, x, axes, speed=None):
        '''
        Moves the device group of axes by relative amount x in um.
        Parameters
        ----------
        axes : list of axis numbers
        x : position shift in um (vector or list).
        '''
        # self.abort_if_requested()
        cmd = [0, 0, 0]
        for pos, axis in zip(x, axes):
            cmd[axis  - 1] = pos
        
        if cmd[0] != 0 or cmd[1] != 0:
            self._sendCmd(SerialCommands.SET_X_Y_POS_REL.format(int(cmd[0] * 10), int(cmd[1] * 10)))

        if cmd[2] != 0:
            self.relative_move(cmd[2], 3)

    def wait_until_still(self, axes = None, axis = None):
        while True:
            resp = self._sendCmd(SerialCommands.GET_IS_BUSY)
            busy = resp != '0'

            if not busy:
                break

    def stop(self):
        self._sendCmd(SerialCommands.STOP)


