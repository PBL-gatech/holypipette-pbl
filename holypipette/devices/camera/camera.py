'''
A generic camera class

TODO:
* A stack() method which takes a series of photos along Z axis
'''
from __future__ import print_function
import collections
import os
import datetime
import time
import threading
import imageio
import logging

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import fourier_gaussian
import warnings
import traceback
from scipy.optimize import brentq
try:
    import cv2
except:
    warnings.warn('OpenCV not available')

__all__ = ['Camera', 'FakeCamera', 'RecordedVideoCamera']


class FileWriteThread(threading.Thread): # saves frames individually
    def __init__(self, *args, **kwds):
        self.queue = kwds.pop('queue')
        self.debug_write_delay = kwds.pop('debug_write_delay', 0)
        self.directory = kwds.pop('directory')
        self.file_prefix = kwds.pop('file_prefix')
        self.skip_frames = kwds.pop('skip_frames', 0)
        threading.Thread.__init__(self, *args, **kwds)
        self.first_frame = None
        self.start_time = None
        self.last_report = None
        self.written_frames = 0
        self.running = True
        self.skipped = -1

    def write_frame(self):
        frame_number, creation_time, elapsed_time, frame = self.queue.popleft()
        if frame_number is None:
            # Marker for end of recording
            return False
        
        # Make all frame numbers relative to the first frame
        if self.first_frame is None:
            self.first_frame = frame_number
            self.start_time = time.time()
            self.last_report = self.start_time
        frame_number -= self.first_frame
        # If desired, skip frames
        self.skipped += 1
        if self.skipped >= self.skip_frames:
            self.skipped = -1
            fname = os.path.join(self.directory, '{}_{:05d}.tiff'.format(self.file_prefix, frame_number))
            with imageio.get_writer(fname, software='holypipette') as writer:
                writer.append_data(frame, meta={'datetime': creation_time,
                                                'description': 'Time since start of recording: {}'.format(repr(elapsed_time))})
            self.written_frames += 1
            time.sleep(self.debug_write_delay)
            if time.time() - self.last_report > 1:
                frame_rate = self.written_frames / (time.time() - self.last_report)
                print('Writing {:.1f} fps (total frames written: {})'.format(frame_rate, frame_number))
                self.last_report = time.time()
                self.written_frames = 0

        return True

    def run(self):
        self.running = True
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
        
        while self.running:
            try:
                if len(self.queue) > self.queue.maxlen // 2:
                    print('WARNING: FileWriteThread queue is getting full ({}/{})'.format(len(self.queue),
                                                                                          self.queue.maxlen))
                if not self.write_frame():
                    break
            except IndexError:
                # queue is emtpy, wait
                time.sleep(0.01)
                # TODO: Store image metadata to file as well?

        if len(self.queue):
            print('Still need to write {} images to disk.'.format(len(self.queue)))
            while len(self.queue):
                if not self.write_frame():
                    break


class AcquisitionThread(threading.Thread):
    def __init__(self, camera, queues, raw_queues):
        self.camera = camera
        self.queues = queues
        self.raw_queues = raw_queues
        self.running = True
        
        self.last_frame_time = None
        self.fps = 0

        threading.Thread.__init__(self, name='image_acquire_thread')

        # self.recorder = FileLogger(folder_path="experiments/Data/rig_recorder_data/", isVideo=True)
        
    def get_frame_rate(self):
        # * A way to calculate FPS
        current_time = time.time()
        if self.last_frame_time is not None:
            self.fps = 1.0 / (current_time - self.last_frame_time)
        self.last_frame_time = current_time

        return self.fps

    def run(self):
        self.running = True

        start_time = time.time()

        last_report = start_time
        acquired_frames = 0
        last_frame = 0
        while self.running:
            snap_time = time.time()
            try:
                raw, processed = self.camera.snap()
            except Exception as ex:
                print('something went wrong acquiring an image, waiting for 100ms: ')
                traceback.print_exception(type(ex), ex, ex.__traceback__)

                time.sleep(.1)
                continue
            # Put image into queues for disk storage and display
            for queue in self.queues:
                queue.append((last_frame, datetime.datetime.now(), snap_time - start_time, processed))
            for queue in self.raw_queues:
                queue.append((last_frame, datetime.datetime.now(), snap_time - start_time, raw))

            # logging.info(f"FPS in camera: {self.get_frame_rate():.2f}")

            last_frame += 1
            acquired_frames += 1
            if snap_time - last_report > 1:
                frame_rate = acquired_frames / (snap_time - last_report)
                # logging.warning('Acquiring {:.1f} fps'.format(frame_rate))
                last_report = snap_time
                acquired_frames = 0
        
        # Put the end marker into the queues
        for queue in self.queues:
            queue.append((None, None, None, None))


class Camera(object):
    """
    Base class for all camera devices. At the end of the initialization, derived classes need to
    call self.start_acquisition() to start the thread that continously acquires images from the
    camera.
    """
    def __init__(self):
        super(Camera, self).__init__()
        self._file_queue = None
        self._last_frame_queue = collections.deque(maxlen=1)
        self.raw_frame_queue = collections.deque(maxlen=1)
        self._acquisition_thread = None
        self._file_thread = None
        self._debug_write_delay = 0
        self.width = 1000
        self.height = 1000
        self.flipped = False # Horizontal flip

        self.stop_show_time = 0
        self.point_to_show = None
        self.cell_list = []

    def show_point(self, point, color=(255, 0, 0), radius=10, duration=1.5, show_center=False):
        self.point_to_show = [point, radius, color, show_center]
        self.stop_show_time = time.time() + duration

    def start_acquisition(self):
        self._acquisition_thread = AcquisitionThread(camera=self,
                                                     queues=[self._last_frame_queue],
                                                     raw_queues=[self.raw_frame_queue])
        self._acquisition_thread.start()

    def stop_acquisition(self):
        self._acquisition_thread.running = False

    def start_recording(self, directory='', file_prefix='', skip_frames=0, queue_size=1000):
        if len(self._acquisition_thread.queues) > 1:
            del self._acquisition_thread.queues[1]
        self._file_queue = collections.deque(maxlen=queue_size)
        self._acquisition_thread.queues.append(self._file_queue)
        self._file_thread = FileWriteThread(queue=self._file_queue,
                                            directory=directory,
                                            file_prefix=file_prefix,
                                            skip_frames=skip_frames,
                                            debug_write_delay=self._debug_write_delay)
        self._file_thread.start()

    def stop_recording(self):
        if self._file_thread:
            self._file_thread.running = False

    def flip(self):
        self.flipped = not self.flipped

    def preprocess(self, input_img):
        # img = cv2.cvtColor(input_img.copy(), cv2.COLOR_GRAY2RGB) if len(input_img.shape) == 2 else input_img.copy()
        img = input_img.copy()

        if self.point_to_show and time.time() - self.stop_show_time < 0:
            img = cv2.circle(img, self.point_to_show[0], self.point_to_show[1], self.point_to_show[2], 3)
            if self.point_to_show[3]:
                img = cv2.circle(img, self.point_to_show[0], 2, self.point_to_show[2], 3)


        # draw cell outlines
        for cell in self.cell_list:
            img = cv2.circle(img, cell, 10, (0, 255, 0), 3)

        if self.flipped:
            img = img[:, ::-1] 
            # img = img[:, ::-1] if len(img.shape) == 2 else img[:, ::-1, :]

        return img

    def new_frame(self):
        '''
        Returns True if a new frame is available
        '''
        return True

    def snap(self):
        '''
        Returns a processed and a raw image
        '''
        raw = self.raw_snap()
        return raw, self.preprocess(raw)

    def raw_snap(self):
        return None

    def get_16bit_image(self):
        '''
        Returns the current image as a 16-bit color image
        '''
        return None

    def last_frame(self):
        '''
        Get the last snapped frame and its number

        Returns
        -------
        (frame_number, frame)
        '''
        try:
            # * the deque has a maxsize of 1, so we can do this
            # maybe we should use a list instead of a deque?
            # ? should we change this to grab the last element instead, as it is more intuitive?
            # last_entry = self._last_frame_queue[0]
            last_entry = self._last_frame_queue[-1]
            return last_entry[0], last_entry[-1]
        except IndexError:  # no frame (yet)
            return None
    
    def last_frame_data(self):
        '''
        Get the last snapped frame and its number

        Returns
        -------
        (frame_number, date, frame)
        '''
        try:
            last_entry = self._last_frame_queue[0]
            return last_entry[0], last_entry[1], last_entry[-1]
        except IndexError:  # no frame (yet)
            return None

    def close(self):
        """Shut down the camera device, free resources, etc."""
        pass

    def set_exposure(self, value):
        print('Setting exposure time not supported for this camera')

    def get_exposure(self):
        print('Getting exposure time not supported for this camera')
        return -1

    def change_exposure(self, change):
        if self.get_exposure() > 0:
            self.set_exposure(self.get_exposure() + change)

    def auto_exposure(self):
        '''
        Auto exposure assumes frames are 8 bits.
        '''
        mean_luminance = 127

        def f(value):
            self.set_exposure(value)
            time.sleep(.2+.001*value) # wait for new frame with updated value
            while not self.new_frame():
                time.sleep(0.05)
            m = self.snap().mean()
            return m-mean_luminance
        exposure = brentq(f, 0.1,100., rtol=0.1)
        self.set_exposure(exposure)

    def get_frame_rate(self):
        return -1

    def reset(self):
        pass

    def get_frame_no(self):
        raise NotImplementedError('get_frame_no not implemented for this camera')

class FakeCamera(Camera):
    def __init__(self, manipulator=None, image_z=0, paramecium=False):
        super(FakeCamera, self).__init__()
        self.width = 1024
        self.height = 768
        self.exposure_time = 30
        self.manipulator = manipulator
        self.image_z = image_z
        self.scale_factor = .5  # micrometers in pixels
        self.depth_of_field = 2.
        self.frame = np.array(np.clip(gaussian_filter(np.random.randn(self.width * 2, self.height * 2) * 0.5, 10) * 50 + 128, 0, 255), dtype=np.uint8)
        
        self.start_acquisition()

    def set_exposure(self, value):
        if 0 < value <= 200:
            self.exposure_time = value

    def get_exposure(self):
        return self.exposure_time

    def get_microscope_image(self, x, y, z):
        frame = np.roll(self.frame, int(y), axis=0)
        frame = np.roll(frame, int(x), axis=1)
        frame = frame[self.height // 2:self.height // 2 + self.height,
                      self.width // 2:self.width // 2 + self.width]
        return np.array(frame, copy=True)

    def raw_snap(self):
        '''
        Returns the current image.
        This is a blocking call (wait until next frame is available)
        '''
        if self.manipulator is not None:
            # Use the part of the image under the microscope

            stage_x, stage_y, stage_z = self.manipulator.position_group([4, 5, 6])
            stage_z -= self.image_z
            stage_x *= self.scale_factor
            stage_y *= self.scale_factor
            stage_z *= self.scale_factor
            frame = self.get_microscope_image(stage_x, stage_y, stage_z)
            if self.paramecium is not None:
                self.paramecium.update_position()
                p_x, p_y, p_z = self.paramecium.position
                p_angle = self.paramecium.angle + np.pi / 2
                p_x *= self.scale_factor
                p_y *= self.scale_factor
                p_z *= self.scale_factor
                p_width = 30 * self.scale_factor
                p_height = 100 * self.scale_factor
                xx, yy = np.meshgrid(np.arange(-self.width // 2, self.width // 2), np.arange(-self.height // 2, self.height // 2))
                frame[((xx - (p_x - stage_x)) * np.cos(p_angle) + (yy - (p_y - stage_y)) * np.sin(p_angle)) ** 2 / (p_width / 2) ** 2 +
                      ((xx - (p_x - stage_x)) * np.sin(p_angle) - (yy - (p_y - stage_y)) * np.cos(p_angle)) ** 2 / (p_height / 2) ** 2 < 1] = 50
                frame[((xx - (p_x - stage_x)) * np.cos(p_angle) + (yy - (p_y - stage_y)) * np.sin(p_angle)) ** 2 / (p_width / 2) ** 2 +
                      ((xx - (p_x - stage_x)) * np.sin(p_angle) - (yy - (p_y - stage_y)) * np.cos(p_angle)) ** 2 / (p_height / 2) ** 2 < 0.8] = 100

            for direction, axes in [(-np.pi / 2, [1, 2, 3])]:
                manipulators = np.zeros((self.height, self.width), dtype=np.int16)
                x, y, z = self.manipulator.position_group(axes)
                x = np.cos(self.manipulator.angle) * (x + 50 / self.scale_factor)
                z = np.sin(self.manipulator.angle) * (x + 50 / self.scale_factor) + z
                x *= self.scale_factor
                y *= self.scale_factor
                z *= self.scale_factor
                # cut off a tip
                # Position relative to stage
                x -= stage_x
                y -= stage_y
                z -= stage_z
                X, Y = np.meshgrid(np.arange(self.width) - self.width // 2 + x,
                                   np.arange(self.height) - self.height // 2 + y)
                angle = np.arctan2(X, Y)
                dist = np.sqrt(X ** 2 + Y ** 2)
                border = (0.075 + 0.0025 * abs(z) / self.depth_of_field)
                manipulators[(np.abs(angle - direction) < border) & (dist > 50)] = 5
                edge_width = 0.02 if z > 0 else 0.04  # Make a distinction between below and above
                manipulators[(np.abs(angle - direction) < border) & (np.abs(angle - direction) > border-edge_width) & (dist > 50)] = 75
                frame[manipulators>0] = manipulators[manipulators>0]
        else:
            img = Image.fromarray(self.frame)
            frame = np.array(img.resize((self.width, self.height)))
        exposure_factor = self.exposure_time / 30.
        frame = frame + np.random.randn(self.height, self.width) * 5

        return np.array(np.clip(frame * exposure_factor, 0, 255),
                        dtype=np.uint8)


def text_phantom(text, size):
    # Availability is platform dependent
    font = 'Arial'
    
    # Create font
    pil_font = ImageFont.truetype(font + ".ttf", size=size[0] // len(text),
                                  encoding="unic")
    text_width, text_height = pil_font.getsize(text)

    # create a blank canvas with extra space between lines
    canvas = Image.new('RGB', size, (0, 0, 0))

    # draw the text onto the canvas
    draw = ImageDraw.Draw(canvas)
    offset = ((size[0] - text_width) // 2,
              (size[1] - text_height) // 2)
    white = "#ffffff"
    draw.text(offset, text, font=pil_font, fill=white)

    # Convert the canvas into an array with values in [0, 1]
    frame = np.asarray(canvas)
    return frame


class DebugCamera(Camera):
    '''A fake camera that shows the frame number'''
    def __init__(self, frames_per_s=20, write_delay=0):
        super(DebugCamera, self).__init__()
        self.width = 1024
        self.height = 768
        self.frameno = 0
        self.last_frame_time = None
        self.delay = 1 / frames_per_s
        self._debug_write_delay = write_delay
        self.start_acquisition()

    def get_frame_rate(self):
        return 1 / self.delay

    def raw_snap(self):
        '''
        Returns the current image.
        This is a blocking call (wait until next frame is available)
        '''
        frame = text_phantom('{:05d}'.format(self.frameno), (self.width, self.height))
        self.frameno += 1
        if self.last_frame_time is not None:
            if time.time() - self.last_frame_time < self.delay:
                time.sleep(self.delay - (time.time() - self.last_frame_time))
        self.last_frame_time = time.time()
        return frame


class RecordedVideoCamera(Camera):
    def __init__(self, file_name, pixel_per_um, slowdown=1):
        super(RecordedVideoCamera, self).__init__()
        self.file_name = file_name
        self.video = cv2.VideoCapture(file_name)
        self.video.open(self.file_name)
        self.width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.pixel_per_um = pixel_per_um
        self.frame_rate = self.video.get(cv2.CAP_PROP_FPS)
        self.time_between_frames = 1 / self.frame_rate * slowdown
        self._last_frame_time = None
        self.start_acquisition()

    def get_frame_rate(self):
        return self.frame_rate

    def raw_snap(self):
        if self._last_frame_time is not None:
            if time.time() - self._last_frame_time < self.time_between_frames:
                # We are too fast, sleep a bit before returning the frame
                sleep_time = self.time_between_frames - (time.time() - self._last_frame_time)
                time.sleep(sleep_time)
        success, frame = self.video.read()
        self._last_frame_time = time.time()

        if not success and self._acquisition_thread.running:
            raise ValueError(f'Cannot read from file {self.file_name}.')

        return frame

    def close(self):
        self.video.release()
        super(RecordedVideoCamera, self).close()
