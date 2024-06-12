import os
import logging
from datetime import datetime
from PIL import Image
import concurrent.futures

class FileLogger:
    def __init__(self, folder_path="experiments/", recorder_filename="recording.csv"):
        self.time_truth = datetime.now()
        self.time_truth_timestamp = self.time_truth.timestamp()
        self.folder_path = folder_path + self.time_truth.strftime("%Y_%m_%d-%H_%M") + "/"
        self.camera_folder_path = self.folder_path + "camera_frames/"
        self.filename = self.folder_path + recorder_filename
        self.file = None
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        print("FileLogger created at: ", self.time_truth_timestamp)

        self.create_folder()

    def create_folder(self):
        # check that the folder exists
        if not os.path.exists(os.path.dirname(self.folder_path)):
            try:
                os.makedirs(os.path.dirname(self.folder_path))
                os.makedirs(os.path.dirname(self.camera_folder_path))
            except OSError as exc:
                logging.error("Error creating folder for recording: %s", exc)

    def open(self):
        self.file = open(self.filename, 'w')

    def _write_to_file(self, content):
        if self.file is None:
            self.open()
        self.file.write(content)

    def write_graph_data(self, time_value, pressure, resistance, current):
        content = f"{time_value - self.time_truth_timestamp}  {pressure}  {resistance}    {current}\n"
        self.executor.submit(self._write_to_file, content)

    def write_movement_data(self, time_value, stage_x, stage_y, stage_z, pipette_x, pipette_y, pipette_z):
        content = f"{time_value - self.time_truth_timestamp}  {stage_x}  {stage_y}  {stage_z} {pipette_x} {pipette_y} {pipette_z}\n"
        self.executor.submit(self._write_to_file, content)

    def _save_image(self, frame, path):
        image = Image.fromarray(frame)
        image.save(path, format="png")

    def write_camera_frames(self, time_value, frame, frameno):
        image_path = self.camera_folder_path + str(frameno) + '_' + str(time_value - self.time_truth_timestamp) + ".png"
        self.executor.submit(self._save_image, frame, image_path)

    def close(self):
        if self.file is not None:
            logging.info("CLOSING FILE: ", self.filename)
            self.file.close()
            self.file = None
        self.executor.shutdown(wait=True)


