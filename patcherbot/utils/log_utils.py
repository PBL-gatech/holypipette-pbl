import logging
import csv
import os
from datetime import datetime

class CSVLogHandler(logging.Handler):
    """Custom logging handler that logs to a CSV file."""

    def __init__(self, base_filename, mode='a'):
        """
        Initialize the CSVLogHandler.

        Args:
            base_filename (str): Base path and filename (without date suffix).
            mode (str, optional): File open mode.
        """
        # Generate a filename with only the date
        timestamp = datetime.now().strftime("%Y_%m_%d")
        filename = f"{base_filename}_{timestamp}.csv"
        super().__init__()
        self.filename = filename
        self.mode = mode
        self.output_file = open(self.filename, self.mode, newline='')
        self.csv_writer = csv.writer(self.output_file)
        
        # Write headers if the file is new/empty
        if os.stat(self.filename).st_size == 0:
            headers = ["Time(HH:MM:SS)", "Time(ms)", "Level", "Message", "Logger Name", "Thread ID"]
            self.csv_writer.writerow(headers)

        self.setFormatter(logging.Formatter('%(asctime)s,%(levelname)s,%(message)s,%(name)s,%(thread)d'))

    def emit(self, record):
        """
        Write a log record to the CSV file.

        Args:
            record (logging.LogRecord): Log record to write.

        Raises:
            Exception: If writing fails, handled via handleError.
        """
        try:
            log_entry = self.format(record)
            self.csv_writer.writerow(log_entry.split(","))
            # Ensure the log is flushed after every write to prevent data loss
            self.output_file.flush()
        except Exception:
            self.handleError(record)

    def close(self):
        """Flush and close the CSV file."""
        if not self.output_file.closed:
            # Flush and close the file properly when done
            self.output_file.flush()
            self.output_file.close()
        super().close()

class LoggingObject(object):
    """
    Mixin class providing a configured logger and convenience logging methods.
    """
    @property
    def logger(self):
        """
        Get or create a logger specific to the class.

        Returns:
            logging.Logger: Configured logger instance.
        """
        if getattr(self, '_logger', None) is None:
            logger_name = f"{self.__class__.__module__}.{self.__class__.__name__}"
            self._logger = logging.getLogger(logger_name)
            self._logger.setLevel(logging.DEBUG)
        return self._logger

    def debug(self, message, *args, **kwds):
        """
        Log a debug-level message.

        Args:
            message (str): Log message.
            *args: Positional arguments for formatting.
            **kwds: Keyword arguments for logging.
        """
        self.logger.debug(message, *args, **kwds)

    def info(self, message, *args, **kwds):
        """
        Log an info-level message.

        Args:
            message (str): Log message.
            *args: Positional arguments for formatting.
            **kwds: Keyword arguments for logging.
        """
        self.logger.info(message, *args, **kwds)

    def warning(self, message, *args, **kwds):
        """
        Log a warning-level message.

        Args:
            message (str): Log message.
            *args: Positional arguments for formatting.
            **kwds: Keyword arguments for logging.
        """
        self.logger.warning(message, *args, **kwds)

    def error(self, message, *args, **kwds):
        """
        Log an error-level message.

        Args:
            message (str): Log message.
            *args: Positional arguments for formatting.
            **kwds: Keyword arguments for logging.
        """
        self.logger.error(message, *args, **kwds)

    def exception(self, message, *args, **kwds):
        """
        Log an exception with traceback information.

        Args:
            message (str): Log message.
            *args: Positional arguments for formatting.
            **kwds: Keyword arguments for logging.
        """
        self.logger.exception(message, *args, **kwds)

def setup_logging():
    """
    Log an exception with traceback information.

    Args:
        message (str): Log message.
        *args: Positional arguments for formatting.
        **kwds: Keyword arguments for logging.
    """
    root_logger = logging.getLogger()

    # Prevent adding duplicate handlers if setup_logging is called multiple times.
    if getattr(root_logger, "_pb_logging_configured", False):
        return
    root_logger._pb_logging_configured = True

    root_logger.setLevel(logging.DEBUG)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s [%(name)s - thread %(thread)d]')
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    


    # Get the directory of the current file.
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Assume the repository root is one directory up from the current file.
    repo_root = os.path.abspath(os.path.join(current_dir, '..', '..'))

    # Define the log folder relative to the repository root.
    log_folder = os.path.join(repo_root, "experiments", "Data", "log_data")
    logging.info(f"log folder: {log_folder}")

    # Create the folder if it doesn't exist.
    os.makedirs(log_folder, exist_ok=True)

    # Now set up your CSVLogHandler using the relative log folder.
    csv_handler = CSVLogHandler(base_filename=os.path.join(log_folder, 'logs'))
    root_logger.addHandler(csv_handler)

# Initialize the logging
setup_logging()

logging.info("Program Started")
