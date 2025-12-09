"""
Generic Lamp class for fluorescence microscopes, including shutter-based and
color/power-addressable light engines.
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
    
    def enable(self, light=None, excitation_filter=None):
        """Enable a specific light channel/color or turn it off."""
        raise NotImplementedError("This method should be implemented by subclasses.")

    def set_power(self, power_percent: float, light=None):
        """Set the power of a given light channel/color."""
        raise NotImplementedError("This method should be implemented by subclasses.")

    def get_IIC_temp(self):
        """Get a temperature reading if supported by the lamp."""
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
        self.power_levels = {}
        self.current_light = None
        self.current_excitation_filter = None
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

    def enable(self, light=None, excitation_filter=None):
        """Fake implementation of enabling a light channel/color."""
        self.current_light = light
        if excitation_filter is not None:
            self.current_excitation_filter = excitation_filter
        if light is None or getattr(light, "name", None) == "OFF":
            self.shutter_status = "closed"
        else:
            self.shutter_status = "open"
        self.info(f"FakeLamp: Light {light} enabled with filter {excitation_filter}.")

    def set_power(self, power_percent: float, light=None):
        """Fake implementation of setting power for a light channel/color."""
        clamped = max(0, min(100, power_percent))
        self.power_levels[light] = clamped
        self.info(f"FakeLamp: Power for {light} set to {clamped}%.")

    def get_IIC_temp(self):
        """Fake implementation returning a placeholder temperature."""
        return 25.0
