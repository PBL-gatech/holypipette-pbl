'''
A microscope is a manipulator with a single axis.
With methods to take a stack of images, autofocus, etc.

TODO:
* a umanager class that autoconfigures with umanager config file
* steps for stack acquisition?
'''
from holypipette.devices.manipulator import Manipulator
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
    def __init__(self, dev, axis):
        '''
        Parameters
        ----------
        dev : underlying device
        axis : axis index
        '''
        Manipulator.__init__(self)
        self.dev : Manipulator = dev
        self.axis = axis
        self.up_direction = None # Up direction, must be provided or calculated
        self.floor_Z = None # This is the Z coordinate of the coverslip
        # Motor range in um; by default +- one meter
        self.min = -1e6 # This could replace floor_Z
        self.max = 1e6

    def set_max_speed(self, speed):
        self.dev.set_max_speed(speed)

    def position(self):
        '''
        Current position

        Returns
        -------
        The current position of the device axis in um.
        '''
        true_position = float(5.0* self.dev.position(self.axis))
        
        return true_position

    def absolute_move(self, x):
        '''
        Moves the device axis to position x in um.

        Parameters
        ----------
        x : target position in um.
        '''
        self.dev.absolute_move(x, self.axis)
        self.sleep(.05)

    def absolute_move_velocity(self, vel):
        '''
        Moves the device axis at velocity vel in um/s.

        Parameters
        ----------
        vel : velocity in um/s.
        '''
        velarr = [0,0,vel]
        self.dev.absolute_move_group_velocity(velarr)

        # self.sleep(.05)

    def move_to_floor(self):
        self.dev.absolute_move(self.floor_Z, self.axis)
        self.dev.wait_until_still([self.axis])
        print(f"Moved to floor at {self.floor_Z} um")
        # self.dev.absolute_move(self.floor_Z, self.axis)
        # self.dev.wait_until_still([self.axis])

    def fix_backlash(self):
        curr_pos = self.position()
        self.absolute_move(curr_pos + 200)
        self.wait_until_still()
        self.absolute_move(curr_pos)
        self.wait_until_still()

    def relative_move(self, x):
        '''
        Moves the device axis by relative amount x in um.

        Parameters
        ----------
        x : position shift in um.
        '''
        self.dev.relative_move(x, self.axis)
        self.sleep(.05)

    def step_move(self, distance):
        self.dev.step_move(distance, self.axis)

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

        Parameters
        ----------
        camera : a camera, eg with a snap() method
        z : A list of z positions
        preprocessing : a function that processes the images (optional)
        save : saves images to disk if True
        pause : pause in second after each movement
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
        '''
        config = {'up_direction' : self.up_direction,
                  'floor_Z' : self.floor_Z}
        return config

    def load_configuration(self, config):
        '''
        Loads configuration from dictionary config.
        Variables not present in the dictionary are untouched.
        '''
        self.up_direction = config.get('up_direction', self.up_direction)
        #self.floor_Z = config.get('floor_Z', self.floor_Z)
