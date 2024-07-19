import logging
from datetime import datetime
import threading
import os
import logging

class EPhysLogger(threading.Thread):
    def __init__(self, recording_state_manager, folder_path="experiments/Data/patch_clamp_data/", ephys_filename="ephys"):
        self.recording_state_manager = recording_state_manager
        self.time_truth = datetime.now()
        self.time_truth_timestamp = self.time_truth.timestamp()
        self.folder_path = folder_path + self.time_truth.strftime("%Y_%m_%d-%H_%M") + "/"
        self.filename = self.folder_path + f"{ephys_filename}"
        self.file = None

        self.create_folder()
        self.write_event = threading.Event()
        # logging.info("EPhysLogger created at: ", self.time_truth_timestamp)

    def create_folder(self):
        # check that the folder exists
        if not os.path.exists(os.path.dirname(self.folder_path)):
            try:
                # os.makedirs(os.path.dirname(self.folder_path))
                # * created a folder deeper will always create the parent folder
                os.makedirs(os.path.dirname(self.folder_path))
            except OSError as exc:
                logging.error("Error creating folder for recording: %s", exc)
    
    def _write_to_file(self, data, index, time_value,color):
        # Create a string for each pair of values in the desired format
        lines = [f"{data[0][i]} {data[1][i]}\n" for i in range(data.shape[1])]
        # Open the file in append mode and write the formatted strings
        # logging.debug("Writing to file %s", self.filename)
        with open(f"{self.filename}_{time_value}_{index}_{color}.csv", 'a+') as file:
            file.writelines(lines)
        self.write_event.set()  # Signal that writing is done
    
    def write_ephys_data(self, index, time_value, data, color):
        # logging.info("Writing ephys data")
        self.write_event.clear()
        # ("len of data: ", len(data[0]))
        # print("len of data: ", len(data[1]))
        # content = f"{time_value}    {data}\n"
        # content = f"{time_value}    {data[0, :]}    {data[1, :]}\n"
        # content = f"{time_value}    {' '.join(map(str, data))}\n"
        threading.Thread(target=self._write_to_file, args=(data,index,time_value,color)).start()
        
    def close(self):
        if self.file is not None:
            logging.info("CLOSING FILE: ", self.filename)
            self.write_event.wait()  # Wait for the last task to complete
            self.file.close()
            self.file = None
        logging.info("Closing csv recorder writing thread")
