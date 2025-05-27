import logging
from datetime import datetime
import threading
import os
from PyQt5 import QtGui
import imageio

class EPhysLogger(threading.Thread):
    def __init__(self, recording_state_manager, folder_path="experiments/Data/patch_clamp_data/", ephys_filename="ephys"):
        super().__init__()
        self.recording_state_manager = recording_state_manager
        self.time_truth = datetime.now()
        testMode = False
        if testMode:
            folder_path = folder_path.replace("Data/", "Data/TEST_")
        self.folder_path = folder_path + self.time_truth.strftime("%Y_%m_%d-%H_%M") + "/" + f"{ephys_filename}" + "/"
        self.filename = self.folder_path + f"{ephys_filename}"
        self.file = None

        # file used to store cell metadata such as coordinates
        self.cell_metadata_file = os.path.join(self.folder_path, "cell_metadata.csv")

        self.folder_created = False
        self.write_event = threading.Event()

        # Dictionary to track unique index and color combinations
        self.index_color_dict = {}
        # Lock for thread-safe access to the dictionary
        self.index_color_lock = threading.Lock()
        self.image_type  = "webp"

    def create_folder(self):
        if not self.folder_created:
            try:
                os.makedirs(os.path.dirname(self.folder_path), exist_ok=True)
                self.folder_created = True
                logging.info(f"Created folder at: {self.folder_path}")
            except OSError as exc:
                logging.error("Error creating folder for recording: %s", exc)
        else:
            pass # Folder already created, no need to create it again
            # logging.debug("Folder already created. Skipping creation.")

    def _write_to_file(self, index, timeData, readData, respData, color):
        # Check if "CurrentProtocol" is in filename
        if "CurrentProtocol" in self.filename:
            with self.index_color_lock:
                if index not in self.index_color_dict:
                    # Index is unique, create a new entry with an empty list for colors
                    self.index_color_dict[index] = []

                # Proceed only if the color is unique for the given index
                if color not in self.index_color_dict[index]:
                    # Append the color to the list for this index
                    self.index_color_dict[index].append(color)
                else:
                    # Color is not unique for this index, skip writing
                    logging.debug("Skipping write: Index %s and color %s are not unique", index, color)
                    return

        # If "CurrentProtocol" is not in filename, proceed as the original method
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

    def save_image(self, index, image):
        self.create_folder()
        image_path = f"cell_{index}.webp"
        if image is None:
            logging.error("No image to save")
            return
        else:
            imageio.imwrite(self.folder_path + image_path, image)
            logging.info("Saved image to %s", self.folder_path + image_path)

    def save_cell_metadata(self, index, stage_coords, image=None):
        """Save cell image and stage coordinates for a given protocol index."""
        self.create_folder()

        if image is None:
            logging.warning("No cell image provided; skipping cell metadata save")
            return

        img_filename = f"cell_{index}.webp"
        imageio.imwrite(os.path.join(self.folder_path, img_filename), image)

        write_header = not os.path.exists(self.cell_metadata_file)
        with open(self.cell_metadata_file, "a+") as f:
            if write_header:
                f.write("index;stage_x;stage_y;stage_z;image\n")
            f.write(
                f"{index};{stage_coords[0]};{stage_coords[1]};{stage_coords[2]};{img_filename}\n"
            )

    def hold_image(self, index, image):
        if image is None:
            logging.error("No image to hold")
            return
        else:
            self.image_path = f"cell_{index}.webp"
            self.image = image

    def close(self):
        if self.file is not None:
            logging.info("CLOSING FILE: %s", self.filename)
            self.write_event.wait()  # Wait for the last task to complete
            self.file.close()
            self.file = None
        logging.info("Closing csv recorder writing thread")
