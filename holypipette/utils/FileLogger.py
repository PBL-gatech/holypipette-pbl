import logging
from datetime import datetime
import threading
from collections import deque
import imageio
import os
import logging
import pandas as pd


class FileLogger(threading.Thread):
    def __init__(self, recording_state_manager, folder_path="experiments/Data/", recorder_filename="recording", filetype="csv", isVideo=False, frame_batch_size=500):
        super().__init__()
        self.recording_state_manager = recording_state_manager
        self.time_truth = datetime.now()

        self.image_type  = "webp"

        testMode = False
        if testMode:
            folder_path = folder_path.replace("Data/", "Data/TEST_")

        self.folder_path = folder_path + self.time_truth.strftime("%Y_%m_%d-%H_%M") + "/"
        self.camera_folder_path = self.folder_path + "camera_frames/"
        self.filename = self.folder_path + recorder_filename + "." + filetype
        self.file = None
        self.last_frameno = 0
        self.frame_batch_size = frame_batch_size
        self.frame_batch_limit = int(frame_batch_size * 0.8)

        self.batch_mode_movements = False
        self.batch_mode_graph = False

        self.write_event = threading.Event()
        self.is_video = isVideo
        self.write_frame = threading.Event() if isVideo else None

        self.batch_frames = deque(maxlen=frame_batch_size)
        # self.graph_contents = deque(maxlen=frame_batch_size)
        self.movement_contents = deque(maxlen=frame_batch_size)

        self.last_movement_time = None
        self.last_graph_time = None
        self.folder_created = False  # Flag to track folder creation

        logging.info(f"FileLogger initialized. Folder path set to: {self.folder_path}")

    def create_folder(self):
        # Check if the recording is enabled before creating the folder
        if self.recording_state_manager.is_recording_enabled() and not self.folder_created:
            try:
                os.makedirs(self.camera_folder_path, exist_ok=True)
                self.folder_created = True  # Set the flag to True once folder is created
                print(f"Created folder at: {self.folder_path}")
            except OSError as exc:
                logging.error("Error creating folder for recording: %s", exc)


    def open(self):
        self.file = open(self.filename, 'a+')
        if len(self.file.readlines()) == 0:
            if self.filename == self.folder_path + "movement_recording.csv":
                self.file.write("timestamp;st_x;st_y;st_z;pi_x;pi_y;pi_z\n")
            if self.filename == self.folder_path + "graph_recording.csv":
                self.file.write("timestamp;pressure;resistance;current;voltage\n")


        print(f"Opened file at: {self.filename}")

    def _write_to_file(self, contents):
        if self.file is None:
            self.open()
        self.file.write(contents)
        self.file.flush()
        self.write_event.set()  # Signal that writing is done
        # print("Wrote file contents at path: ", self.filename)

    def _write_to_file_batch(self, contents):
        if self.file is None:
            self.open()
        self.file.writelines(contents)
        self.file.flush()
        self.write_event.set()  # Signal that writing is done
        # print("Wrote file contents at path: ", self.filename)

    def write_graph_data(self, time_value, pressure: float, resistance: float, current, voltage):
    # ? time_current is probably not necessary, will remove in a future commit when confirmed.
    # def write_graph_data(self, time_value, pressure: float, resistance: float, time_current, current):
        if not self.recording_state_manager.is_recording_enabled():
            return
        if time_value == self.last_graph_time:
            return
        self.last_graph_time = time_value
        self.create_folder()  # Create the folder if recording is enabled and it's the first time
        # content = f"timestamp:{time_value}  pressure:{pressure}  resistance:{resistance}  / current:{current}\n"
        content = f"{time_value};{pressure};{resistance};{current};{voltage}\n"
        self.write_event.clear()
        threading.Thread(target=self._write_to_file, args=(content,)).start()

    def write_movement_data_batch(self, time_value, stage_x, stage_y, stage_z, pipette_x, pipette_y, pipette_z):
        if not self.recording_state_manager.is_recording_enabled():
            return
        if time_value == self.last_movement_time:
            return
        self.last_movement_time = time_value
        self.create_folder()  # Create the folder if recording is enabled and it's the first time
        content = f"{time_value};{stage_x};{stage_y};{stage_z};{pipette_x};{pipette_y};{pipette_z}\n"

        #print('New Pos: ' + content)

        self.movement_contents.append(content)
        if len(self.movement_contents) >= self.frame_batch_limit:
            # logging.info(f"Batch size reached for MOVEMENT. Writing to disk at {datetime.now() - self.time_truth} seconds after start")
            self._flush_contents(self.movement_contents)

    def _flush_contents(self, data):
        if data:
            contents = data.copy()
            data.clear()
            self.write_event.clear()
            threading.Thread(target=self._write_to_file_batch, args=(contents,)).start()

    def _save_image(self, frame, path):
        self.batch_frames.append((frame, path))
        if len(self.batch_frames) >= self.frame_batch_limit:
            # logging.info(f"Batch size reached for FRAMES. Writing to disk at {datetime.now() - self.time_truth} seconds after start")
            self.write_frame.clear()
            threading.Thread(target=self._write_batch_to_disk).start()
    
    def _save_image_sleep(self):
        if self.batch_frames:
            self.write_frame.clear()
            threading.Thread(target=self._write_batch_to_disk).start()

    def _write_batch_to_disk(self):
        while self.batch_frames:
            frame, path = self.batch_frames.popleft()
            # imwrite(path, frame)
            imageio.imwrite(path, frame, format=self.image_type)
            # qoi.write(path, frame)
        self.write_frame.set()  # Signal that image saving is done

    def write_camera_frames(self, time_value, frame, frameno):
        if not self.recording_state_manager.is_recording_enabled():
            self._save_image_sleep()
            return

        # * Add this back in if you change where this function is called within the update_image function in the LiveFeedQT class. 
        # if frameno is None:
        #     logging.info("No frame number detected. Closing the camera recorder")
        #     self.close()
        #     return

        if frameno <= self.last_frameno:
            return
        self.create_folder()  # Create the folder if recording is enabled and it's the first time
        image_path = self.camera_folder_path + str(frameno) + '_' + str(time_value) + "." + self.image_type
        self._save_image(frame, image_path)
        self.last_frameno = frameno

    def setBatchGraph(self, value=True):
        self.batch_mode_graph = value
    def setBatchMoves(self, value=True):
        self.batch_mode_movements = value

    def close(self):
        if self.file is not None:
            logging.info("Closing file: %s", self.filename)
            # if self.batch_mode_graph and self.graph_contents:
            #     self._flush_contents(self.graph_contents)
            if self.batch_mode_movements and self.movement_contents:
                self._flush_contents(self.movement_contents)
            self.write_event.wait()  # Wait for the last task to complete

            self.file.close()
            self.file = None

        self.write_frame.wait()
        logging.info("Closing CSV recorder writing thread")
        if self.batch_frames:
            self._write_batch_to_disk()
        if self.is_video:
            logging.info("Closing frame saving thread")
