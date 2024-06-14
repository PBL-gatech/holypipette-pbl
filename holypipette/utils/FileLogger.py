import os
import logging
from datetime import datetime
from PIL import Image
import concurrent.futures
import threading

class FileLogger:
    def __init__(self, folder_path="experiments/Data", recorder_filename="recording.csv"):
        self.time_truth = datetime.now()
        self.time_truth_timestamp = self.time_truth.timestamp()
        self.folder_path = folder_path + self.time_truth.strftime("%Y_%m_%d-%H_%M") + "/"
        self.camera_folder_path = self.folder_path + "camera_frames/"
        self.filename = self.folder_path + recorder_filename
        self.file = None
        self.last_frame = 0

        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.write_event = threading.Event()
        print("FileLogger created at: ", self.time_truth_timestamp)

        self.create_folder()

    def create_folder(self):
        # check that the folder exists
        if not os.path.exists(os.path.dirname(self.folder_path)):
            try:
                # os.makedirs(os.path.dirname(self.folder_path))
                # * created a folder deeper will always create the parent folder
                os.makedirs(os.path.dirname(self.camera_folder_path))
            except OSError as exc:
                logging.error("Error creating folder for recording: %s", exc)

    def open(self):
        self.file = open(self.filename, 'w')

    def _write_to_file(self, content):
        if self.file is None:
            self.open()
        self.file.write(content)
        self.write_event.set()  # Signal that writing is done

    def write_graph_data(self, time_value, pressure, resistance, current):
        self.write_event.clear()
        content = f"{time_value - self.time_truth_timestamp}  {pressure}  {resistance}    {current}\n"
        self.executor.submit(self._write_to_file, content)

    def write_movement_data(self, time_value, stage_x, stage_y, stage_z, pipette_x, pipette_y, pipette_z):
        self.write_event.clear()
        content = f"{time_value - self.time_truth_timestamp}  {stage_x}  {stage_y}  {stage_z} {pipette_x} {pipette_y} {pipette_z}\n"
        self.executor.submit(self._write_to_file, content)

    def _save_image(self, frame, path):
        image = Image.fromarray(frame)
        image.save(path, format="webp")
        self.write_event.set()  # Signal that image saving is done

    def write_camera_frames(self, time_value, frame, frameno):
        if frameno is None:
            logging.info("No frame number detected. Closing the camera recorder")
            self.close()
            return
        
        if frameno <= self.last_frame:
            return

        self.write_event.clear()
        image_path = self.camera_folder_path + str(frameno) + '_' + str(time_value - self.time_truth_timestamp) + ".webp"
        logging.info("Saving image frame #" + str(frameno))
        self.executor.submit(self._save_image, frame, image_path)
        self.last_frame = frameno

    def close(self):
        if self.file is not None:
            logging.info("CLOSING FILE: ", self.filename)
            self.write_event.wait()  # Wait for the last task to complete
            self.file.close()
            self.file = None
        self.executor.shutdown(wait=True)
