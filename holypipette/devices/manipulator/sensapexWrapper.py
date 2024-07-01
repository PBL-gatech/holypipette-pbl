from sensapex import UMP
import numpy as np
from ctypes import c_int, c_float, byref
import math
import time

from holypipette.devices.manipulator.manipulator import Manipulator

class SensapexManip(Manipulator):
    '''A wrapper class to interface between the sensapex python library and the holypipette minipulator classes
    '''
    
    def __init__(self, deviceID = None):
        Manipulator.__init__(self)
        self.ump = UMP.get_ump()

        # setup device ID
        if deviceID == None:
            umpList = self.ump.list_devices()
            assert(len(umpList) == 1, "must specify sensapex ump device id if there is more than 1 connected!")
            self.deviceID = umpList[0] #if there's only 1 device connected, use it
        else:
            self.deviceID = deviceID

        self.max_speed = 5000 # "feels good" default value
        self.max_acceleration = 1 # "feels good" default value
        self.armAngle = math.radians(-self._get_axis_angle())

    def position(self, axis=None):
        raw_pos = self.raw_position()
        if axis == None:
            return raw_pos
        else:
            return raw_pos[axis-1]

    def raw_position(self, axis=None):
        return self.ump.get_pos(self.deviceID, timeout=1)

    def absolute_move(self, x, axis, speed=None):
        setpoint = np.nan * np.ones(3)
        if axis is not None:
            setpoint[axis-1] = x
        else:
            setpoint = x

        print("moving to: {}".format(setpoint))
        speed = self.max_speed if speed is None else speed

        self.ump.goto_pos(self.deviceID, setpoint, speed, max_acceleration=self.max_acceleration, linear=True)


    def absolute_move_group(self, x, axes, speed=None):
        setpoint = np.nan * np.ones(3)
        for axis in axes:
            setpoint[axis-1] = x[axis-1]
        
        speed = self.max_speed if speed is None else speed

        self.ump.goto_pos(self.deviceID, setpoint, speed, max_acceleration=self.max_acceleration, linear=True)

        
    def stop(self, axis):
        """
        Stops current movements.
        """
        self.ump.stop()

    def _get_axis_angle(self):
        angle = c_float()
        rVal = self.ump.call("ump_get_axis_angle", self.deviceID, byref(angle))
        return angle.value

    def set_max_speed(self, speed):
        self.max_speed = speed
    
    def set_max_accel(self, accel):
        self.max_acceleration = accel