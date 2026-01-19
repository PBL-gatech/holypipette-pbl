"""
Generic Laser class for optogenetic systems, including on/off and
wavelength/power-addressable light engines.
"""
__all__ = ["Laser", "FakeLaser"]

from patcherbot.controller import TaskController

class Laser(TaskController):
    """
    Base class for Lasers used in Optogenetic Systems.
    """
    def __init__(self, *args, **kwargs):
        """
        Initialize the Laser with any necessary parameters.
        """
        super().__init__(*args, **kwargs)
        self._initialize()

    def _initialize(self):
        """Initialize the Laser with any necessary commands."""
        raise NotImplementedError("This method should be implemented by subclasses.")

    def power_on(self):
        """Power on the Laser."""
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def power_off(self):
        """Power off the Laser."""
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def get_power_state(self):
        """Get the current state of the Laser power and its power level."""
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def set_wavelength(self, wavelength=None):
        """Set the excitation wavelength for the Laser."""
        raise NotImplementedError("This method should be implemented by subclasses.")
    def get_wavelength(self):
        """Get the current excitation wavelength of the Laser."""
        raise NotImplementedError("This method should be implemented by subclasses.")

    def get_laser_state(self):
        """Get the current state of the Laser (power level, wavelength, temperature)."""
        laser_state =  {
            "power_state": self.get_power_state(),
            "wavelength": self.get_wavelength(),
            "temperature": self.get_laser_temp(),
        }
        self.debug(f"Laser state: {laser_state}")
        return laser_state
    
    def excite(self, power_on=None, excitation_wavelength=None):
        """Enable a specific wavelength or turn it off."""
        laser_state = self.get_laser_state()
        if excitation_wavelength is not None:
            self.set_wavelength(excitation_wavelength)
        if power_on is not None:
            self.set_power_level(power_on)
        if laser_state["power_state"] == "on":
            self.power_off()
        else:
            self.power_on()
        return self.get_laser_state()
    
    def set_power_level(self, power_percent: float, wavelength=None):
        """Set the power level of a given excitation wavelength."""
        raise NotImplementedError("This method should be implemented by subclasses.")

    def get_laser_temp(self):
        """Get a temperature reading if supported by the Laser."""
        raise NotImplementedError("This method should be implemented by subclasses.")


class FakeLaser(Laser):
    """
    Fake Laser that only logs actions.
    """
    def __init__(self, *args, **kwargs):
        """Initialize the fake Laser."""
        super().__init__(*args, **kwargs)
        self.wavelength = None
        self.power_state = "off"
        self.power_levels = 0
        self.info("FakeLaser initialized.")

    def _initialize(self):
        """Skip hardware initialization for the fake Laser."""
        self.debug("FakeLaser: _initialize called (no hardware).")

    def power_on(self):
        """Fake implementation of powering on the Laser."""
        self.power_state = "on"
        self.info("FakeLaser: Power on.")

    def power_off(self):
        """Fake implementation of powering off the Laser."""
        self.power_state = "off"
        self.info("FakeLaser: Power off.")

    def get_power_state(self):
        """Fake implementation of getting the Laser power state."""
        return self.power_state

    def set_wavelength(self, wavelength=None):
        """Fake implementation of setting the excitation wavelength."""
        self.wavelength = wavelength
        self.info(f"FakeLaser: wavelength set to {wavelength}.")

    def get_wavelength(self):
        """Fake implementation of getting the current excitation wavelength."""
        return self.wavelength

    def set_power_level(self, power_percent: float, wavelength=None):
        """Fake implementation of setting the power level for a excitation wavelength."""
        clamped = int(max(0, min(100, round(power_percent))))
        self.power_levels = clamped
        self.info(f"FakeLaser: Power set to {clamped}%.")

    def get_laser_temp(self):
        """Fake implementation returning a placeholder temperature."""
        return 25.0
