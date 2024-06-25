# import logging
# from datetime import datetime
# import threading
# from collections import deque
# import imageio
# import os
# import logging
# from PIL import Image
# from multiprocessing import Pool
# import tracemalloc

# tracemalloc.start()



# class FileLogger(threading.Thread):
#     def __init__(self, recording_state_manager, folder_path="experiments/Data/", recorder_filename="recording", filetype="csv", isVideo=False, batch_size=500):
#         super().__init__()
#         self.recording_state_manager = recording_state_manager
#         self.time_truth = datetime.now()
#         self.time_truth_timestamp = self.time_truth.timestamp()
#         self.folder_path = folder_path + "TEST_" + self.time_truth.strftime("%Y_%m_%d-%H_%M") + "/"
#         self.camera_folder_path = self.folder_path + "camera_frames/"
#         self.filename = self.folder_path + recorder_filename + "." + filetype
#         self.file = None
#         self.last_frame = 0
#         self.batch_size = batch_size

#         self.batch_mode_movements = False
#         self.batch_mode_graph = False

#         self.write_event = threading.Event()
#         self.is_video = isVideo
#         self.write_frame = threading.Event() if isVideo else None

#         self.batch_frames = deque(maxlen=batch_size)
#         self.graph_contents = deque(maxlen=batch_size)
#         self.movement_contents = deque(maxlen=batch_size)

#         self.last_movement_time = None
#         self.last_graph_time = None

#         self.convert_pool = None
#         self.frame_convert_queue = deque()

#         logging.info("FileLogger created at: %s", self.time_truth_timestamp)
#         self.create_folder()

#     def create_folder(self):
#         try:
#             os.makedirs(self.camera_folder_path, exist_ok=True)
#             print(f"Created folder at: {self.folder_path}")
#         except OSError as exc:
#             logging.error("Error creating folder for recording: %s", exc)

#     def open(self):
#         self.file = open(self.filename, 'a+')
#         print(f"Opened file at: {self.filename}")

#     def _write_to_file(self, contents):
#         if self.file is None:
#             self.open()
#         self.file.write(contents)
#         self.file.flush()
#         self.write_event.set()  # Signal that writing is done
#         # print("Wrote file contents at path: ", self.filename)

#     def _write_to_file_batch(self, contents):
#         if self.file is None:
#             self.open()
#         self.file.writelines(contents)
#         self.file.flush()
#         self.write_event.set()  # Signal that writing is done
#         # print("Wrote file contents at path: ", self.filename)

#     # def write_graph_data_batch(self, time_value, pressure, resistance, time_current, current):
#     #     if not self.recording_state_manager.is_recording_enabled():
#     #         return
#     #     if time_value == self.last_graph_time:
#     #         return
#     #     self.last_graph_time = time_value

#     #     content = f"{time_value}  {pressure}  {resistance}  {time_current}  {current}\n"
#     #     self.graph_contents.append(content)
#     #     if len(self.graph_contents) >= self.batch_size - 50:
#     #         self._flush_contents(self.graph_contents)
    
#     def write_graph_data(self, time_value, pressure, resistance, time_current, current):
#         if not self.recording_state_manager.is_recording_enabled():
#             return
#         if time_value == self.last_graph_time:
#             return
#         self.last_graph_time = time_value
#         content = f"timestamp:{time_value}  pressure:{pressure}  resistance:{resistance}  time:{time_current}  current:{current}\n"
#         self.write_event.clear()
#         threading.Thread(target=self._write_to_file, args=(content,)).start()

#     # def write_movement_data(self, time_value, stage_x, stage_y, stage_z, pipette_x, pipette_y, pipette_z):
#     #     if not self.recording_state_manager.is_recording_enabled():
#     #         return
#     #     if time_value == self.last_movement_time:
#     #         return
#     #     self.last_movement_time = time_value
#     #     content = f"timestamp:{time_value}  st_x:{stage_x}  st_y:{stage_y}  st_z:{stage_z}  pi_x:{pipette_x} pi_y:{pipette_y} pi_z:{pipette_z}\n"
#     #     self.write_event.clear()
#     #     threading.Thread(target=self._write_to_file, args=(content,)).start()

#     def write_movement_data_batch(self, time_value, stage_x, stage_y, stage_z, pipette_x, pipette_y, pipette_z):
#         if not self.recording_state_manager.is_recording_enabled():
#             return
#         if time_value == self.last_movement_time:
#             return
#         self.last_movement_time = time_value

#         content = f"timestamp:{time_value}  st_x:{stage_x}  st_y:{stage_y}  st_z:{stage_z}  pi_x:{pipette_x} pi_y:{pipette_y} pi_z:{pipette_z}\n"

#         self.movement_contents.append(content)
#         if len(self.movement_contents) >= self.batch_size - 50:
#             self._flush_contents(self.movement_contents)

#     def _flush_contents(self, data):
#         if data:
#             contents = data.copy()
#             data.clear()
#             self.write_event.clear()
#             threading.Thread(target=self._write_to_file_batch, args=(contents,)).start()

#     def _save_image(self, frame, path):
#         self.batch_frames.append((frame, path))
#         if len(self.batch_frames) >= self.batch_size - 20:
#             self.write_frame.clear()
#             threading.Thread(target=self._write_batch_to_disk).start()

#     def _write_batch_to_disk(self):
#         # data = self.batch_frames.copy()
#         # self.batch_frames.clear()
#         while self.batch_frames:
#             frame, path = self.batch_frames.popleft()
#             # imwrite(path, frame)
#             imageio.imwrite(path, frame, format="tiff")
#             self.frame_convert_queue.append(path)
#         self.write_frame.set()  # Signal that image saving is done

#     def start_conversion(self):
#         self.convert_pool = Pool(processes=2)
#         batch = [self.frame_convert_queue.popleft() for _ in range(min(50, len(self.frame_convert_queue)))]
#         self.convert_pool.map_async(self.convert_image, batch).wait()

#     def convert_image(self, tiff_path):
#         webp_path = tiff_path.replace('.tiff', '.webp')
#         try:
#             with Image.open(tiff_path) as img:
#                 img.save(webp_path, "WEBP")
#             os.remove(tiff_path)
#         except Exception as e:
#             logging.error("Failed to convert %s: %s", tiff_path, str(e))


#     def write_camera_frames(self, time_value, frame, frameno):
#         if not self.recording_state_manager.is_recording_enabled():
#             if len(self.frame_convert_queue):
#                 logging.info("start converting frames to webp")
#                 self.start_conversion()
#             return
        
#         self.convert_pool = None

#         if frameno is None:
#             logging.info("No frame number detected. Closing the camera recorder")
#             self.close()
#             return
        
#         if frameno <= self.last_frame:
#             return

#         image_path = self.camera_folder_path + str(frameno) + '_' + str(time_value) + ".tiff"
#         # image_path = self.camera_folder_path + str(frameno) + '_' + str(time_value) + ".tiff"
#         # image_path = self.camera_folder_path + str(frameno) + '_' + str(time_value - self.time_truth_timestamp) + ".webp"
#         self._save_image(frame, image_path)
#         self.last_frame = frameno

#     def setBatchGraph(self, value=True):
#         self.batch_mode_graph = value
#     def setBatchMoves(self, value=True):
#         self.batch_mode_movements = value

#     def close(self):
#         if self.frame_convert_queue:
#             self.start_conversion()
#         if self.file is not None:
#             logging.info("Closing file: %s", self.filename)
#             if self.batch_mode_graph and self.graph_contents:
#                 self._flush_contents(self.graph_contents)
#             if self.batch_mode_movements and self.movement_contents:
#                 self._flush_contents(self.movement_contents)
#             self.write_event.wait()  # Wait for the last task to complete

#             self.file.close()
#             self.file = None

#         logging.info("Closing CSV recorder writing thread")
#         if self.batch_frames:
#             self._write_batch_to_disk()
#         if self.is_video:
#             logging.info("Closing frame saving thread")

#         if self.convert_pool:
#             self.convert_pool.close()
#             self.convert_pool.join()
import logging
from datetime import datetime
import os
from collections import deque
import imageio
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import tracemalloc

tracemalloc.start()

class FileLogger:
    def __init__(self, recording_state_manager, folder_path="experiments/Data/", recorder_filename="recording", filetype="csv", isVideo=False, batch_size=500):
        self.recording_state_manager = recording_state_manager
        self.time_truth = datetime.now()
        self.time_truth_timestamp = self.time_truth.timestamp()
        self.folder_path = folder_path + self.time_truth.strftime("%Y_%m_%d-%H_%M") + "/"
        self.camera_folder_path = self.folder_path + "camera_frames/"
        self.filename = self.folder_path + recorder_filename + "." + filetype
        self.file = None
        self.last_frame = 0
        self.batch_size = batch_size

        self.batch_mode_movements = False
        self.batch_mode_graph = False

        self.is_video = isVideo

        self.batch_frames = deque(maxlen=batch_size)
        self.graph_contents = deque(maxlen=batch_size)
        self.movement_contents = deque(maxlen=batch_size)

        self.last_movement_time = None
        self.last_graph_time = None

        self.convert_pool = None
        self.frame_convert_queue = deque()

        logging.info("FileLogger created at: %s", self.time_truth_timestamp)
        self.create_folder()

        self.executor = ThreadPoolExecutor(max_workers=4)

    def create_folder(self):
        try:
            os.makedirs(self.camera_folder_path, exist_ok=True)
            print(f"Created folder at: {self.folder_path}")
        except OSError as exc:
            logging.error("Error creating folder for recording: %s", exc)

    def open(self):
        self.file = open(self.filename, 'a+')
        print(f"Opened file at: {self.filename}")

    def _write_to_file(self, contents):
        if self.file is None:
            self.open()
        self.file.write(contents)
        self.file.flush()

    def _write_to_file_batch(self, contents):
        if self.file is None:
            self.open()
        self.file.writelines(contents)
        self.file.flush()

    def write_graph_data(self, time_value, pressure, resistance, time_current, current):
        if not self.recording_state_manager.is_recording_enabled():
            return
        if time_value == self.last_graph_time:
            return
        self.last_graph_time = time_value
        content = f"timestamp:{time_value}  pressure:{pressure}  resistance:{resistance}  time:{time_current}  current:{current}\n"
        self.executor.submit(self._write_to_file, content)

    def write_movement_data_batch(self, time_value, stage_x, stage_y, stage_z, pipette_x, pipette_y, pipette_z):
        if not self.recording_state_manager.is_recording_enabled():
            return
        if time_value == self.last_movement_time:
            return
        self.last_movement_time = time_value

        content = f"timestamp:{time_value}  st_x:{stage_x}  st_y:{stage_y}  st_z:{stage_z}  pi_x:{pipette_x} pi_y:{pipette_y} pi_z:{pipette_z}\n"

        self.movement_contents.append(content)
        if len(self.movement_contents) >= self.batch_size - 50:
            self.executor.submit(self._flush_contents, self.movement_contents.copy())
            self.movement_contents.clear()

    def _flush_contents(self, data):
        if data:
            self._write_to_file_batch(data)

    def _save_image(self, frame, path):
        self.batch_frames.append((frame, path))
        if len(self.batch_frames) >= self.batch_size - 20:
            while self.batch_frames:
                frame, path = self.batch_frames.popleft()
                imageio.imwrite(path, frame, format="tiff")
                self.frame_convert_queue.append(path)

    def start_conversion(self):
        if self.convert_pool is None:
            self.convert_pool = ProcessPoolExecutor(max_workers=2)
        while self.frame_convert_queue:
            batch = [self.frame_convert_queue.popleft() for _ in range(min(50, len(self.frame_convert_queue)))]
            self.convert_pool.map(self.convert_image, batch)

    def convert_image(self, tiff_path):
        webp_path = tiff_path.replace('.tiff', '.webp')
        try:
            with Image.open(tiff_path) as img:
                img.save(webp_path, "WEBP")
            os.remove(tiff_path)
        except Exception as e:
            logging.error("Failed to convert %s: %s", tiff_path, str(e))

    def write_camera_frames(self, time_value, frame, frameno):
        if not self.recording_state_manager.is_recording_enabled():
            if len(self.frame_convert_queue):
                logging.info("start converting frames to webp")
                self.start_conversion()
            return

        if frameno is None or frameno <= self.last_frame:
            return

        image_path = self.camera_folder_path + str(frameno) + '_' + str(time_value) + ".tiff"
        self.executor.submit(self._save_image, frame, image_path)
        self.last_frame = frameno

    def setBatchGraph(self, value=True):
        self.batch_mode_graph = value
    def setBatchMoves(self, enable):
        self.batch_mode_movements = enable

    def close(self):
        if self.frame_convert_queue:
            self.start_conversion()
        if self.file is not None:
            logging.info("Closing file: %s", self.filename)
            self.file.close()
            self.file = None

        logging.info("Closing CSV recorder writing thread")
        if self.is_video:
            logging.info("Closing frame saving thread")

        if self.convert_pool:
            self.convert_pool.shutdown(wait=True)