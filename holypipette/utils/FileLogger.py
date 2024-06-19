import logging
from datetime import datetime
import threading
from collections import deque
import imageio
import os

class FileLogger(threading.Thread):
    def __init__(self, folder_path="experiments/Data/", recorder_filename="recording.csv", isVideo=False, batch_size=500):
        super().__init__()
        self.time_truth = datetime.now()
        self.time_truth_timestamp = self.time_truth.timestamp()
        self.folder_path = folder_path + self.time_truth.strftime("%Y_%m_%d-%H_%M") + "/"
        self.camera_folder_path = self.folder_path + "camera_frames/"
        self.filename = self.folder_path + recorder_filename
        self.file = None
        self.last_frame = 0
        self.batch_size = batch_size

        self.batch_mode_movements = False
        self.batch_mode_graph = False

        self.write_event = threading.Event()
        self.is_video = isVideo
        self.write_frame = threading.Event() if isVideo else None

        self.batch_frames = deque(maxlen=batch_size)
        self.graph_contents = deque(maxlen=batch_size)
        self.movement_contents = deque(maxlen=batch_size)

        self.last_movement_time = None
        self.last_graph_time = None

        logging.info("FileLogger created at: %s", self.time_truth_timestamp)

        self.create_folder()

    def create_folder(self):
        if not os.path.exists(self.folder_path):
            try:
                os.makedirs(self.camera_folder_path)
                print(f"Created folder at: {self.folder_path}")
            except OSError as exc:
                logging.error("Error creating folder for recording: %s", exc)

    def open(self):
        self.file = open(self.filename, 'w')
        print(f"Opened file at: {self.filename}")

    def _write_to_file(self, contents):
        if self.file is None:
            self.open()
        self.file.write(contents)
        self.write_event.set()  # Signal that writing is done
        # print("Wrote file contents at path: ", self.filename)

    def _write_to_file_batch(self, contents):
        if self.file is None:
            self.open()
        self.file.writelines(contents)
        self.file.flush()
        self.write_event.set()  # Signal that writing is done
        # print("Wrote file contents at path: ", self.filename)

    def write_graph_data_batch(self, time_value, pressure, resistance, time_current, current):
        # content = f"{time_value - self.time_truth_timestamp}  {pressure}  {resistance}  {time_current}  {current}\n"
        if time_value == self.last_graph_time:
            return
        self.last_graph_time = time_value

        content = f"{time_value}  {pressure}  {resistance}  {time_current}  {current}\n"
        self.graph_contents.append(content)
        if len(self.graph_contents) >= self.batch_size - 50:
            self._flush_contents(self.graph_contents)
    
    def write_graph_data(self, time_value, pressure, resistance, time_current, current):
        # content = f"{time_value - self.time_truth_timestamp}  {pressure}  {resistance}  {time_current}  {current}\n"
        if time_value == self.last_graph_time:
            return
        self.last_graph_time = time_value
        content = f"timestamp:{time_value}  pressure:{pressure}  resistance:{resistance}  time:{time_current}  current:{current}\n"
        self.write_event.clear()
        threading.Thread(target=self._write_to_file, args=(content,)).start()

    def write_movement_data(self, time_value, stage_x, stage_y, stage_z, pipette_x, pipette_y, pipette_z):
        # content = f"{time_value - self.time_truth_timestamp}  {stage_x}  {stage_y}  {stage_z} {pipette_x} {pipette_y} {pipette_z}\n"
        if time_value == self.last_movement_time:
            return
        self.last_movement_time = time_value
        content = f"timestamp:{time_value}  st_x:{stage_x}  st_y:{stage_y}  st_z:{stage_z}  pi_x:{pipette_x} pi_y:{pipette_y} pi_z:{pipette_z}\n"
        self.write_event.clear()
        threading.Thread(target=self._write_to_file, args=(content,)).start()

    def write_movement_data_batch(self, time_value, stage_x, stage_y, stage_z, pipette_x, pipette_y, pipette_z):
        # content = f"{time_value - self.time_truth_timestamp}  {stage_x}  {stage_y}  {stage_z} {pipette_x} {pipette_y} {pipette_z}\n"
        if time_value == self.last_movement_time:
            return
        self.last_movement_time = time_value

        content = f"timestamp:{time_value}  st_x:{stage_x}  st_y:{stage_y}  st_z:{stage_z}  pi_x:{pipette_x} pi_y:{pipette_y} pi_z:{pipette_z}\n"

        self.movement_contents.append(content)
        if len(self.movement_contents) >= self.batch_size - 50:
            self._flush_contents(self.movement_contents)

    def _flush_contents(self, data):
        if data:
            contents = data.copy()
            data.clear()
            self.write_event.clear()
            threading.Thread(target=self._write_to_file_batch, args=(contents,)).start()

    def _save_image(self, frame, frameno, time_value, path):
        self.batch_frames.append((frameno, frame, time_value, path))
        if len(self.batch_frames) >= self.batch_size - 20:
            self.write_frame.clear()
            threading.Thread(target=self._write_batch_to_disk).start()

    def _write_batch_to_disk(self):
        # data = self.batch_frames.copy()
        # self.batch_frames.clear()
        while self.batch_frames:
            frameno, frame, time_value, path = self.batch_frames.popleft()
            imageio.imwrite(path, frame)
        self.write_frame.set()  # Signal that image saving is done

    def write_camera_frames(self, time_value, frame, frameno):
        if frameno is None:
            logging.info("No frame number detected. Closing the camera recorder")
            self.close()
            return
        
        if frameno <= self.last_frame:
            return

        image_path = self.camera_folder_path + str(frameno) + '_' + str(time_value) + ".webp"
        # image_path = self.camera_folder_path + str(frameno) + '_' + str(time_value - self.time_truth_timestamp) + ".webp"
        self._save_image(frame, frameno, time_value, image_path)
        self.last_frame = frameno

    def setBatchGraph(self, value=True):
        self.batch_mode_graph = value
    def setBatchMoves(self, value=True):
        self.batch_mode_movements = value

    def close(self):
        if self.file is not None:
            logging.info("Closing file: %s", self.filename)
            if self.batch_mode_graph and self.graph_contents:
                self._flush_contents(self.graph_contents)
            if self.batch_mode_movements and self.movement_contents:
                self._flush_contents(self.movement_contents)
            self.write_event.wait()  # Wait for the last task to complete
            self.file.close()
            self.file = None
        logging.info("Closing CSV recorder writing thread")
        if self.batch_frames:
            self._write_batch_to_disk()
        if self.is_video:
            logging.info("Closing frame saving thread")



class EPhysLogger(threading.Thread):
    def __init__(self, folder_path="experiments/Data/patch_clamp_data/", ephys_filename="ephys"):
        self.time_truth = datetime.now()
        self.time_truth_timestamp = self.time_truth.timestamp()
        self.folder_path = folder_path + self.time_truth.strftime("%Y_%m_%d-%H_%M") + "/"
        self.filename = self.folder_path + f"{ephys_filename}"
        self.file = None

        self.create_folder()
        self.write_event = threading.Event()
        print("EPhysSaver created at: ", self.time_truth_timestamp)

    def create_folder(self):
        # check that the folder exists
        if not os.path.exists(os.path.dirname(self.folder_path)):
            try:
                # os.makedirs(os.path.dirname(self.folder_path))
                # * created a folder deeper will always create the parent folder
                os.makedirs(os.path.dirname(self.folder_path))
            except OSError as exc:
                logging.error("Error creating folder for recording: %s", exc)
    
    def _write_to_file(self, data, time_value,color):
        # Create a string for each pair of values in the desired format
        lines = [f"{time_value} {color} {data[0][i]} {data[1][i]}\n" for i in range(data.shape[1])]
        # Open the file in append mode and write the formatted strings
        with open(f"{self.filename}.csv", 'a+') as file:
            file.writelines(lines)
        self.write_event.set()  # Signal that writing is done
    
    def write_ephys_data(self, time_value, data, color):
        self.write_event.clear()
        print("Writing ephys data")
        # print("len of data: ", len(data[0]))
        # print("len of data: ", len(data[1]))
        # content = f"{time_value}    {data}\n"
        # content = f"{time_value}    {data[0, :]}    {data[1, :]}\n"
        # content = f"{time_value}    {' '.join(map(str, data))}\n"
        threading.Thread(target=self._write_to_file, args=(data,time_value,color)).start()
        
    def close(self):
        if self.file is not None:
            logging.info("CLOSING FILE: ", self.filename)
            self.write_event.wait()  # Wait for the last task to complete
            self.file.close()
            self.file = None
        logging.info("Closing csv recorder writing thread")