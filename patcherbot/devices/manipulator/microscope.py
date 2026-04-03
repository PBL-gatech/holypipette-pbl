'''
A microscope is a manipulator with a single axis.
With methods to take a stack of images, autofocus, etc.

TODO:
* a umanager class that autoconfigures with umanager config file
* steps for stack acquisition?
'''
from patcherbot.devices.manipulator import Manipulator
import time
import warnings
try:
    import cv2
except:
    warnings.warn('OpenCV not available')

__all__ = ['Microscope']

class Microscope(Manipulator):
    '''
    A microscope Z axis, obtained here from an axis of a Manipulator.
    '''
    def __init__(self, dev, axis, microscope_units_per_um=1.0):
        '''
        Initialize the Microscope.

        Args:
            dev: The underlying device (Manipulator) to control.
            axis: Axis index corresponding to the Z-axis.
            microscope_units_per_um: Conversion factor from device units to micrometers.
        '''
        Manipulator.__init__(self)
        self.dev : Manipulator = dev
        self.axis = axis
        self.up_direction = None # Up direction, must be provided or calculated
        self.floor_Z = None # This is the Z coordinate of the coverslip
        # Motor range in um; by default +- one meter
        self.min = -1e6 # This could replace floor_Z
        self.max = 1e6
        self.microscope_units_per_um = 1.0
        self.set_units_per_um(microscope_units_per_um)

    def set_max_speed(self, speed):
        """
        Set the maximum speed for the device.

        Args:
            speed: Maximum speed to set.
        """
        self.dev.set_max_speed(speed)

    def set_units_per_um(self, units_per_um):
        """
        Set the device units per micrometer.

        Args:
            units_per_um: Conversion factor from um to device units.
        """
        if units_per_um is None:
            return
        try:
            units = float(units_per_um)
        except (TypeError, ValueError):
            return
        if units <= 0:
            return
        self.microscope_units_per_um = units

    def _scale(self):
        """
        Get the current scaling factor from um to device units.

        Returns:
            Scale factor (float)
        """
        return self.microscope_units_per_um if self.microscope_units_per_um else 1.0

    def _to_device_units(self, value_um):
        """
        Convert a value in micrometers to device units.

        Args:
            value_um: Value in micrometers.

        Returns:
            Value in device units.
        """
        return float(value_um) * self._scale()

    def _from_device_units(self, value_dev):
        """
        Convert a value in device units to micrometers.

        Args:
            value_dev: Value in device units.

        Returns:
            Value in micrometers.
        """
        return float(value_dev) / self._scale()

    def position(self):
        """
        Get the current position of the microscope Z-axis.

        Returns:
            Current position in micrometers (um).
        """
        true_position = self._from_device_units(self.dev.position(self.axis))
        
        return true_position

    def absolute_move(self, x):
        '''
        Moves the device axis to position x in um.

        Args:
            x: Target position in micrometers (um).
        '''
        ##self.abort_if_requested()
        self.dev.absolute_move(self._to_device_units(x), self.axis)
        self.sleep(.05)

    def absolute_move_velocity(self, vel):
        '''
        Moves the device axis at velocity vel in um/s.

        Args:
            vel: Velocity in micrometers per second (um/s).
        '''
        ###self.abort_if_requested()
        velarr = [0,0,self._to_device_units(vel)]
        self.dev.absolute_move_group_velocity(velarr)

        # self.sleep(.05)

    def move_to_floor(self):
        '''
        Moves the device axis to the floor position.
        '''
        ##self.abort_if_requested()
        self.dev.absolute_move(self._to_device_units(self.floor_Z), self.axis)
        self.dev.wait_until_still([self.axis])
        print(f"Moved to floor at {self.floor_Z} um")
        # self.dev.absolute_move(self.floor_Z, self.axis)
        # self.dev.wait_until_still([self.axis])

    def fix_backlash(self):
        '''
        Moves the device axis to a position and back to the original position.
        This is to fix backlash.
        '''
        ##self.abort_if_requested()
        curr_pos = self.position()
        self.absolute_move(curr_pos + 200)
        self.wait_until_still()
        self.absolute_move(curr_pos)
        self.wait_until_still()

    def relative_move(self, x):
        '''
        Moves the device axis by relative amount x in um.

        Args:
            x: Distance to move in micrometers (um).
        '''
        ##self.abort_if_requested()
        self.dev.relative_move(self._to_device_units(x), self.axis)
        self.sleep(.05)

    def step_move(self, distance):
        '''
        Moves the device axis by a fixed step distance in um.
        
        Args:
            distance: Step size in micrometers (um).
        '''
        ###self.abort_if_requested()
        self.dev.step_move(self._to_device_units(distance), self.axis)

    def stop(self):
        """
        Stop current movements.
        """
        self.dev.stop()

    def wait_until_still(self):
        """
        Waits for the motors to stop.
        """
        self.dev.wait_until_still([self.axis])
        self.sleep(.05)

    def stack(self, camera, z, preprocessing=lambda img:img, save = None, pause = 0.3):
        '''
        Take a stack of images at the positions given in the z list

        Args:
            camera: Camera object with a snap() method.
            z: List of Z positions in micrometers.
            preprocessing: Optional function to process images.
            save: Filename prefix to save images, if not None.
            pause: Pause in seconds after each movement.

        Returns:
            List of acquired images.
        '''
        position = self.position()
        images = []
        current_z = position
        for k,zi in enumerate(z):
            #self.absolute_move(zi)
            self.relative_move(zi-current_z)
            current_z = zi
            self.wait_until_still()
            # We wait a little bit because there might be mechanical oscillations
            time.sleep(pause) # also make sure the camera is in sync
            img = preprocessing(camera.snap())
            images.append(img)
            if save is not None:
                cv2.imwrite('./screenshots/'+save+'{}.jpg'.format(k), img)
        self.absolute_move(position)
        self.wait_until_still()
        return images

    def save_configuration(self):
        '''
        Outputs configuration in a dictionary.

        Returns:
            Dictionary containing configuration parameters.
        '''
        config = {'up_direction' : self.up_direction,
                  'floor_Z' : self.floor_Z}
        return config

    def load_configuration(self, config):
        '''
        Loads configuration from dictionary config.
        Variables not present in the dictionary are untouched.

        Args:
            config: Dictionary with configuration parameters.
        '''
        self.up_direction = config.get('up_direction', self.up_direction)
        #self.floor_Z = config.get('floor_Z', self.floor_Z)
