import numpy as np

# from patcherbot.devices.amplifier.DAQ import DAQ, FakeDAQ
from patcherbot.devices.pressurecontroller.BasePressureController import PressureController


class GigasealSimulator:
    def __init__(self):
        self.resistance = 1
        self.maxResistance = 1e3
        self.pressure = 0
        self.active = False
    
    def update_pressure(self, pressure):
        self.pressure = pressure
        
    def tick(self, dt):
        if self.active:
            pressure = self.pressure
            currResistance = self.resistance
            growth = np.exp(0.1 * abs(pressure))
            newResistance = currResistance + (
                growth * (1 - currResistance / self.maxResistance) * dt
                )
            # print(f"P={self.pressure}, growth={growth}")
            self.resistance = newResistance
    
    def get_resistance(self):
        return self.resistance
    
    def start(self):
        self.active = True
    
    def stop(self):
        self.active = False