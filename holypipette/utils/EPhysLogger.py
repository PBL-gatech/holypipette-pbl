import logging
from datetime import datetime
import threading
import os
import logging
from PyQt5 import QtGui

class EPhysLogger(threading.Thread):
    def __init__(self, recording_state_manager, folder_path="experiments/Data/patch_clamp_data/", ephys_filename="ephys"):
        self.recording_state_manager = recording_state_manager
        self.time_truth = datetime.now()
        testMode = False
        if testMode:
            folder_path = folder_path.replace("Data/", "Data/TEST_")
        self.folder_path = folder_path + self.time_truth.strftime("%Y_%m_%d-%H_%M") + "/" +  f"{ephys_filename}" + "/"
        self.filename = self.folder_path + f"{ephys_filename}"
        self.file = None

        self.folder_created = False  # Add flag to track folder creation
        self.write_event = threading.Event()

    def create_folder(self):
        if not self.folder_created:  # Check if the folder has been created
            try:
                os.makedirs(os.path.dirname(self.folder_path), exist_ok=True)
                self.folder_created = True  # Set flag once folder is created
                logging.info(f"Created folder at: {self.folder_path}")
            except OSError as exc:
                logging.error("Error creating folder for recording: %s", exc)
        else:
            logging.info("Folder already created. Skipping creation.")

    def _write_to_file(self, index, timeData, readData, respData, color):
        # Create a string for each pair of values in the desired format
        lines = [f"{timeData[i]} {readData[i]} {respData[i]}\n" for i in range(len(timeData))]
        # Open the file in append mode and write the formatted strings
        logging.debug("Writing to file %s", self.filename)
        with open(f"{self.filename}_{index}_{color}.csv", 'a+') as file:
            file.writelines(lines)
        self.write_event.set()  # Signal that writing is done

    def write_ephys_data(self, index, timeData, readData, respData, color):
        self.create_folder()  # Ensure folder is created if it hasn't been
        
        self.write_event.clear()
        threading.Thread(target=self._write_to_file, args=(index, timeData, readData, respData, color)).start()

    def save_ephys_plot(self, index, plot):
        self.create_folder()  # Ensure folder is created if it hasn't been

        image_path = f"{self.filename}_{index}.png"
        exporter = QtGui.QImage(plot.width(), plot.height(), QtGui.QImage.Format_ARGB32)
        painter = QtGui.QPainter(exporter)
        plot.render(painter)
        painter.end()
        
        if exporter.save(image_path):
            logging.info("Saved plot to %s", image_path)
        else:
            logging.error("Failed to save plot to %s", image_path)

    def close(self):
        if self.file is not None:
            logging.info("CLOSING FILE: ", self.filename)
            self.write_event.wait()  # Wait for the last task to complete
            self.file.close()
            self.file = None
        logging.info("Closing csv recorder writing thread")
