from sensapex import UMP
import numpy as np
from ctypes import c_int, c_float, byref
import math
import time

from patcherbot.devices.manipulator.manipulator import Manipulator

class SensapexManip(Manipulator):
    '''A wrapper class to interface between the sensapex python library and the patcherbot manipulator classes
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
        self.constant_z_enabled = False
        self._constant_z_anchor = None
        self._constant_z_gain = None

    def position(self, axis=None):
        raw_pos = self.raw_position()
        pos = raw_pos
        if self.constant_z_enabled:
            if self._constant_z_anchor is None:
                self._constant_z_anchor = list(raw_pos)
            gain = self._constant_z_gain
            if gain is None:
                gain = math.tan(self.armAngle)
                self._constant_z_gain = gain
            dx = raw_pos[0] - self._constant_z_anchor[0]
            corrected_z = raw_pos[2] - gain * dx
            pos = [raw_pos[0], raw_pos[1], corrected_z]
        if axis == None:
            return pos
        else:
            return pos[axis-1]

    def raw_position(self, axis=None):
        return self.ump.get_pos(self.deviceID, timeout=1)

    def enable_constant_z_readback(self, enabled=True, gain=None):
        """Optionally report a Z value that ignores virtual-axis induced Z drift."""
        self.constant_z_enabled = bool(enabled)
        if not self.constant_z_enabled:
            return
        if gain is not None:
            self._constant_z_gain = gain
        elif self._constant_z_gain is None:
            self._constant_z_gain = math.tan(self.armAngle)
        self._constant_z_anchor = list(self.raw_position())

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
