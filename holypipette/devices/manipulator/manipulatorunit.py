"""
A class for access to a particular unit managed by a device.
It is essentially a subset of a Manipulator
"""
from __future__ import absolute_import

from numpy import ones, arange
import numpy as np

from .manipulator import Manipulator

__all__ = ['ManipulatorUnit']


class ManipulatorUnit(Manipulator):
    def __init__(self, dev, axes):
        '''
        Parameters
        ----------
        dev : underlying device
        axes : list of 3 axis indexes
        '''
        Manipulator.__init__(self)
        self.dev = dev
        self.axes = axes
        # Motor ranges in um; by default +- one meter
        self.min = -ones(len(axes))*1e6
        self.max = ones(len(axes))*1e6

    def position(self, axis = None):
        '''
        Current position along an axis.

        Parameters
        ----------
        axis : axis number starting at 0; if None, all XYZ axes

        Returns
        -------
        The current position of the device axis in um.
        '''
        if axis is None: # all positions in a vector
            #return array([self.dev.position(self.axes[axis]) for axis in range(len(self.axes))])
            return self.dev.position_group(self.axes)
        else:
            return self.dev.position(self.axes[axis])

    def absolute_move(self, x, axis = None, blocking=False, speed=None):
        '''
        Moves the device axis to position x in um.

        Parameters
        ----------
        axis : axis number starting at 0; if None, all XYZ axes
        x : target position in um.
        '''
        if axis is None:
            # then we move all axes
            if blocking:
                for i, axis in enumerate(self.axes):
                    self.dev.absolute_move(x[i], axis, speed)
                    self.dev.wait_until_still([axis])
            else:
                self.dev.absolute_move_group(x, self.axes, speed)
        else:
            self.dev.absolute_move(x, self.axes[axis], speed)
            if blocking:
                self.dev.wait_until_still([self.axes[axis]])
        #self.sleep(.05)

    def absolute_move_group(self, x, axes, speed=None):
        '''
        Moves the device axes to positions x in um.
        '''
        
        self.dev.absolute_move_group(x, np.array(self.axes)[axes], speed)
        #self.sleep(.05)

    def relative_move(self, x, axis = None, speed=None):
        '''
        Moves the device axis by relative amount x in um.

        Parameters
        ----------
        axis : axis number starting at 0; if None, all XYZ axes
        x : position shift in um.
        '''
        if axis is None:
            self.dev.relative_move_group(x, self.axes, speed)
        else:
            self.dev.relative_move(x, self.axes[axis], speed)
        # self.sleep(.05)

    def absolute_move_group_velocity(self, vel, axes):
        '''
        Moves the device axes to positions x in um.
        '''
        
        self.dev.absolute_move_group_velocity(vel, np.array(self.axes)[axes])
        # self.sleep(.005)

    def stop(self, axis = None):
        """
        Stop current movements.
        """
        if axis is None:
            # then we stop all axes
            for i, axis in enumerate(self.axes):
                self.dev.stop(axis)
        else:
            self.dev.stop(self.axes[axis])

    def wait_until_still(self, axes = None):
        """
        Waits for the motors to stop.
        """
        if axes is None: # all axes
            axes = arange(len(self.axes))
        if hasattr(axes, '__len__'):  # is that useful?
            for i in axes:
                self.wait_until_still(i)
        else:
            self.dev.wait_until_still([self.axes[axes]])
        self.sleep(.05)

    def wait_until_reached(self, position, axes=None, precision=0.5, timeout=10):
        """
        Waits until position is reached within precision, and raises an error if the
        target is not reached after the time out, unless the manipulator is still moving.

        Parameters
        ----------
        position : target position in micrometer
        axes : axis number of list of axis numbers
        precision : precision in micrometer
        timeout : time out in second
        """
        self.dev.wait_until_reached(position, axes, precision, timeout)

    def set_max_speed(self, speed):
        self.dev.set_max_speed(speed)
    
    def set_max_accel(self, accel):
        self.dev.set_max_accel(accel)