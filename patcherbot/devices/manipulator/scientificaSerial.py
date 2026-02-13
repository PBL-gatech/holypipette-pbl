
import serial
from .manipulator import Manipulator
import time
import threading
import re

__all__ = ['ScientificaSerial']

class SerialCommands():
    GET_X_POS = 'PX\r'
    GET_Y_POS = 'PY\r'
    GET_Z_POS = 'PZ\r'
    GET_X_Y_Z = '\r'
    GET_MAX_SPEED = 'TOP\r'
    GET_MAX_SPEED_Z = 'TOPZ\r'
    GET_MAX_ACCEL = 'ACC\r'
    GET_MAX_ACCEL_Z = 'ACCZ\r'
    GET_IS_BUSY = 's\r'

    SET_X_Y_POS_ABS = 'abs {} {}\r'
    SET_X_Y_POS_REL = 'rel {} {}\r'
    SET_X_Y_Z_POS_ABS = 'abs {} {} {}\r'
    SET_X_Y_Z_POS_REL = 'rel {} {} {}\r'
    

    SET_Z_POS = 'absz {}\r'
    SET_MAX_SPEED = 'TOP {}\r'
    SET_MAX_SPEED_Z = 'TOPZ {}\r'
    SET_MAX_ACCEL = 'ACC {}\r'
    SET_MAX_ACCEL_Z = 'ACCZ {}\r'

    SET_X_Y_Z_VEL = 'VJ {} {} {}\r'

    GET_BAUD = 'BAUD\r'
    SET_BAUD = 'BAUD {}\r'

    STOP = 'STOP\r'


_number_re = re.compile(r'-?\d+')


def _parse_scientifica_int(response):
    if response is None:
        return None
    resp = str(response).strip()
    if not resp or resp.startswith('E,'):
        return None
    match = _number_re.search(resp)
    if match is None:
        return None
    return int(match.group(0))


class ScientificaSerialEncoder(Manipulator):

    def __init__(self, comPort: serial.Serial, zAxisComPort):
        self.comPort : serial.Serial = comPort

        self.zAxisComPort : serial.Serial = zAxisComPort
        self.stageUnitsPerEncoderPulse = 1.45
        self.encoderZ = 0

        self._lock = threading.Lock()
        self._supports_stage_z_profile = None
        self.current_pos = [0, 0, 0]

        # self.info(f"Baud Rate: {self.get_baud_rate()}")

        self.set_max_accel(100)
        self.set_max_speed(10000)
        

        #start constantly polling position in a new thread
        self._polling_thread = threading.Thread(target=self.update_pos_continuous, daemon=True)
        self._polling_thread.start()
        self._polling_thread.deamon = True

    def get_baud_rate(self):
        '''
        gets the baud rate of the serial port

        '''
        resp = self._sendCmd(SerialCommands.GET_BAUD)
        return (resp)

    def set_baud_rate(self, baud_rate : int):
        '''Sets the baud rate of the serial port.  
        '''
        self._sendCmd(SerialCommands.SET_BAUD.format(int(baud_rate)))

    def _get_optional_stage_profile_value(self, command):
        if self._supports_stage_z_profile is False:
            return None
        resp = self._sendCmd(command)
        value = _parse_scientifica_int(resp)
        if value is None and str(resp).startswith('E,'):
            self._supports_stage_z_profile = False
            return None
        if value is not None:
            self._supports_stage_z_profile = True
        return value

    def _set_optional_stage_profile_value(self, command):
        if self._supports_stage_z_profile is False:
            return
        resp = self._sendCmd(command)
        if str(resp).startswith('E,'):
            self._supports_stage_z_profile = False
        else:
            self._supports_stage_z_profile = True

    def get_max_speed(self):
        xy_resp = self._sendCmd(SerialCommands.GET_MAX_SPEED)
        xy_speed = _parse_scientifica_int(xy_resp)
        if xy_speed is None:
            self.warning(f"Scientifica TOP read failed: {xy_resp}")
            return None
        z_speed = self._get_optional_stage_profile_value(SerialCommands.GET_MAX_SPEED_Z)
        return min(xy_speed, z_speed) if z_speed is not None else xy_speed

    def get_max_accel(self):
        xy_resp = self._sendCmd(SerialCommands.GET_MAX_ACCEL)
        xy_accel = _parse_scientifica_int(xy_resp)
        if xy_accel is None:
            self.warning(f"Scientifica ACC read failed: {xy_resp}")
            return None
        z_accel = self._get_optional_stage_profile_value(SerialCommands.GET_MAX_ACCEL_Z)
        return min(xy_accel, z_accel) if z_accel is not None else xy_accel

    def set_max_speed(self, speed):
        speed = int(speed)
        resp = self._sendCmd(SerialCommands.SET_MAX_SPEED.format(speed))
        if str(resp).startswith('E,'):
            self.warning(f"Scientifica TOP set failed: {resp}")
            return
        self._set_optional_stage_profile_value(SerialCommands.SET_MAX_SPEED_Z.format(speed))

    def set_max_accel(self, accel):
        accel = int(accel)
        resp = self._sendCmd(SerialCommands.SET_MAX_ACCEL.format(accel))
        if str(resp).startswith('E,'):
            self.warning(f"Scientifica ACC set failed: {resp}")
            return
        self._set_optional_stage_profile_value(SerialCommands.SET_MAX_ACCEL_Z.format(accel))


    def __del__(self):
        try:
            if hasattr(self, "comPort") and self.comPort:
                self.comPort.close()
        except Exception:
            pass

    def _sendCmd(self, cmd):
        '''Sends a command to the stage and returns the response
        '''
        with self._lock:
            self.comPort.write(cmd.encode())
            resp = self.comPort.read_until(b'\r') #read reply to message
            resp = resp[:-1]

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

    
    def relative_move_group(self, x, axes, speed=None):
        """
        Relative multi‑axis move using Scientifica's `rel` command.
        Mirrors absolute_move_group but sends deltas instead of targets.
        """
        x = list(x)
        axes = list(axes)

        # Build delta vector in device order (1=X, 2=Y, 3=Z)
        dx = dy = dz = 0
        if 1 in axes:
            dx = int(x[axes.index(1)] * 10)
        if 2 in axes:
            dy = int(x[axes.index(2)] * 10)
        if 3 in axes:
            dz = int(x[axes.index(3)] * 10)

        if 1 in axes and 2 in axes and 3 in axes:
            self._sendCmd(SerialCommands.SET_X_Y_Z_POS_REL.format(dx, dy, dz))
        elif 1 in axes and 2 in axes:
            self._sendCmd(SerialCommands.SET_X_Y_POS_REL.format(dx, dy))
        elif 3 in axes:
            # Only Z move; still use the 3‑axis relative command for consistency
            self._sendCmd(SerialCommands.SET_X_Y_Z_POS_REL.format(0, 0, dz))
        else:
            print(f'unimplemented move group {x} {axes}')

    def absolute_move_group_velocity(self,vel,axes):   
        try: 
         self.info(f"Setting velocity to {vel} on axes {axes}")
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
        self._supports_stage_z_profile = None
        self.current_pos = [0, 0, 0]

        self.set_max_accel(1000)
        self.set_max_speed(100000)

        #start constantly polling position in a new thread
        self._polling_thread = threading.Thread(target=self.update_pos_continuous, daemon=True)
        self._polling_thread.start()
        self._polling_thread.deamon = True

        self.info(f"Baud Rate: {self.get_baud_rate()}")

    

    def get_baud_rate(self):
        '''
        gets the baud rate of the serial port

        '''
        resp = self._sendCmd(SerialCommands.GET_BAUD)
        return (resp)

    def set_baud_rate(self, baud_rate : int):
        '''Sets the baud rate of the serial port.  
        '''
        self._sendCmd(SerialCommands.SET_BAUD.format(int(baud_rate)))

    def _get_optional_stage_profile_value(self, command):
        if self._supports_stage_z_profile is False:
            return None
        resp = self._sendCmd(command)
        value = _parse_scientifica_int(resp)
        if value is None and str(resp).startswith('E,'):
            self._supports_stage_z_profile = False
            return None
        if value is not None:
            self._supports_stage_z_profile = True
        return value

    def _set_optional_stage_profile_value(self, command):
        if self._supports_stage_z_profile is False:
            return
        resp = self._sendCmd(command)
        if str(resp).startswith('E,'):
            self._supports_stage_z_profile = False
        else:
            self._supports_stage_z_profile = True

    def get_max_speed(self):
        xy_resp = self._sendCmd(SerialCommands.GET_MAX_SPEED)
        xy_speed = _parse_scientifica_int(xy_resp)
        if xy_speed is None:
            self.warning(f"Scientifica TOP read failed: {xy_resp}")
            return None
        z_speed = self._get_optional_stage_profile_value(SerialCommands.GET_MAX_SPEED_Z)
        return min(xy_speed, z_speed) if z_speed is not None else xy_speed

    def get_max_accel(self):
        xy_resp = self._sendCmd(SerialCommands.GET_MAX_ACCEL)
        xy_accel = _parse_scientifica_int(xy_resp)
        if xy_accel is None:
            self.warning(f"Scientifica ACC read failed: {xy_resp}")
            return None
        z_accel = self._get_optional_stage_profile_value(SerialCommands.GET_MAX_ACCEL_Z)
        return min(xy_accel, z_accel) if z_accel is not None else xy_accel

    def set_max_speed(self, speed):
        speed = int(speed)
        resp = self._sendCmd(SerialCommands.SET_MAX_SPEED.format(speed))
        if str(resp).startswith('E,'):
            self.warning(f"Scientifica TOP set failed: {resp}")
            return
        self._set_optional_stage_profile_value(SerialCommands.SET_MAX_SPEED_Z.format(speed))

    def set_max_accel(self, accel):
        accel = int(accel)
        resp = self._sendCmd(SerialCommands.SET_MAX_ACCEL.format(accel))
        if str(resp).startswith('E,'):
            self.warning(f"Scientifica ACC set failed: {resp}")
            return
        self._set_optional_stage_profile_value(SerialCommands.SET_MAX_ACCEL_Z.format(accel))

    def __del__(self):
        self.comPort.close()

    def _sendCmd(self, cmd):
        '''Sends a command to the stage and returns the response
        '''
        with self._lock:
            # start  = time.perf_counter_ns()
            self.comPort.write(cmd.encode())
            resp = self.comPort.read_until(b'\r') #read reply to message
            resp = resp[:-1]
            # if resp == b'A':
            #     print(f"command received: {resp}")
            # end = time.perf_counter_ns()
            # print(f"Time taken to send command: {(end - start)/1e6} ms")
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
        # print(f"absolute move {pos} {axis}")
        try: 
            if axis == 1:
                yPos = self.position(axis=2)
                self._sendCmd(SerialCommands.SET_X_Y_POS_ABS.format(round(pos * 10) , round(yPos * 10)))
            if axis == 2:
                xPos = self.position(axis=1)
                self._sendCmd(SerialCommands.SET_X_Y_POS_ABS.format(round(xPos * 10), round(pos * 10)))
            if axis == 3:
                self._sendCmd(SerialCommands.SET_Z_POS.format(round(pos * 10)))
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
            # end  = time.perf_counter_ns()
            # print(f"Time taken to move: {(end - start)/1e6} ms")


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
        
    def relative_move_group(self, x, axes, speed=None):
        """
        Relative multi‑axis move using the 2‑ or 3‑axis `rel` commands.
        Mirrors absolute_move_group but sends deltas instead of targets.
        """
        x = list(x)
        axes = list(axes)

        dx = dy = dz = 0
        if 1 in axes:
            dx = int(x[axes.index(1)] * 10)
        if 2 in axes:
            dy = int(x[axes.index(2)] * 10)
        if 3 in axes:
            dz = int(x[axes.index(3)] * 10)

        if 1 in axes and 2 in axes and 3 in axes:
            self._sendCmd(SerialCommands.SET_X_Y_Z_POS_REL.format(dx, dy, dz))
        elif 1 in axes and 2 in axes:
            self._sendCmd(SerialCommands.SET_X_Y_POS_REL.format(dx, dy))
        elif 3 in axes:
            self._sendCmd(SerialCommands.SET_X_Y_Z_POS_REL.format(0, 0, dz))
        else:
            print(f'unimplemented move group {x} {axes}')

    def wait_until_still(self, axes = None, axis = None):
        while True:
            resp = self._sendCmd(SerialCommands.GET_IS_BUSY)
            busy = resp != '0'

            if not busy:
                break

    def stop(self):
        self._sendCmd(SerialCommands.STOP)


