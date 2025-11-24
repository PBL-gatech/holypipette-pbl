"""
Generic Lamp class for fluorescence microscopes.

"""
all  = ['Lamp', 'FakeLamp']

from patcherbot.controller import TaskController

class Lamp(TaskController):
    """
    Base class for lamps used in fluorescence microscopy.
    """
    def __init__(self, *args, **kwargs):
        """
        Initialize the lamp with any necessary parameters.
        """
        super().__init__(*args, **kwargs)
        self._initialize()

    def _initialize(self):
        """Initialize the lamp with any necessary commands."""
        raise NotImplementedError("This method should be implemented by subclasses.")

    
    def open_shutter(self):
        """Open the lamp shutter."""
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def close_shutter(self):
        """Close the lamp shutter."""
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def get_shutter_state(self):
        """Get the current state of the lamp shutter."""
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def set_filter(self,filter= None):
        """Set the excitation filter for the lamp."""
        raise NotImplementedError("This method should be implemented by subclasses.")
    def get_filter(self):
        """Get the current excitation filter of the lamp."""
        raise NotImplementedError("This method should be implemented by subclasses.")



class FakeLamp(Lamp):
    """
    Fake Lamp that only logs actions.
    """
    def __init__(self, *args, **kwargs):
        """Initialize the fake lamp."""
        super().__init__(*args, **kwargs)
        self.filter = None
        self.shutter_status = "closed"
        self.info("FakeLamp initialized.")

    def _initialize(self):
        """Skip hardware initialization for the fake lamp."""
        self.debug("FakeLamp: _initialize called (no hardware).")

    def open_shutter(self):
        """Fake implementation of opening the lamp shutter."""
        self.shutter_status = "open"
        self.info("FakeLamp: Shutter opened.")


    def close_shutter(self):
        """Fake implementation of closing the lamp shutter."""
        self.shutter_status = "closed"
        self.info("FakeLamp: Shutter closed.")

    def get_shutter_state(self):
        """Fake implementation of getting the lamp shutter state."""
        return self.shutter_status

    def set_filter(self, filter=None):
        """Fake implementation of setting the excitation filter."""
        self.filter = filter
        self.info(f"FakeLamp: Filter set to {filter}.")

    def get_filter(self):
        """Fake implementation of getting the current excitation filter."""
        if hasattr(self, 'filter'):
            return self.filter
        else:
            return "No filter set (fake lamp)."
