

import logging
import csv
import os

class CSVLogHandler(logging.Handler):
    """Custom logging handler that logs to a CSV file."""

    def __init__(self, filename, mode='a'):
        super().__init__()
        self.filename = filename
        self.mode = mode
        self.output_file = open(self.filename, self.mode, newline='')
        self.csv_writer = csv.writer(self.output_file)
        
        # Write headers if the file is new/empty
        if os.stat(self.filename).st_size == 0:
            headers = ["Time(HH:MM:SS)","Time(ms)","Level", "Message", "Logger Name", "Thread ID"]
            self.csv_writer.writerow(headers)

        self.setFormatter(logging.Formatter('%(asctime)s,%(levelname)s,%(message)s,%(name)s,%(thread)d'))

    def emit(self, record):
        try:
            log_entry = self.format(record)
            self.csv_writer.writerow(log_entry.split(","))
        except Exception as e:
            self.handleError(record)

    def close(self):
        self.output_file.close()
        super().close()


class LoggingObject(object):
    @property
    def logger(self):
        if getattr(self, '_logger', None) is None:
            logger_name = f"{self.__class__.__module__}.{self.__class__.__name__}"
            self._logger = logging.getLogger(logger_name)
            self._logger.setLevel(logging.DEBUG)
        return self._logger

    def debug(self, message, *args, **kwds):
        self.logger.debug(message, *args, **kwds)

    def info(self, message, *args, **kwds):
        self.logger.info(message, *args, **kwds)

    def warn(self, message, *args, **kwds):
        self.logger.warn(message, *args, **kwds)

    def error(self, message, *args, **kwds):
        self.logger.error(message, *args, **kwds)

    def exception(self, message, *args, **kwds):
        self.logger.exception(message, *args, **kwds)


def setup_logging():
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # Console handler
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s [%(name)s - thread %(thread)d]')
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    path = r'C:\Users\sa-forest\Documents\GitHub\holypipette-pbl\experiments\Analysis\Rig_Recorder'
    # CSV file handler
    csv_handler = CSVLogHandler(path+'\logs.csv')
    root_logger.addHandler(csv_handler)

setup_logging()

logging.info("Program Started")

