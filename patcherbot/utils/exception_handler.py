# exception_handler.py
import sys
import logging

def handle_exception(exc_type, exc_value, exc_traceback):
    """
    Handle uncaught exceptions by logging them, while allowing keyboard interrupts
    to be handled by the default system behavior.

    Args:
        exc_type (type): Exception class.
        exc_value (Exception): Exception instance.
        exc_traceback (traceback): Traceback object.
    """
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    logging.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

def set_global_exception_hook():
    """Set the global exception hook to use the custom handle_exception function."""
    sys.excepthook = handle_exception