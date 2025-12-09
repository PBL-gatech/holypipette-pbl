from enum import Enum
import serial
import time
from .lamp import Lamp

class LightColor(Enum):
    '''An enum to represent the colors the Lumencore can output
    '''
    RED = 0
    GREEN = 1
    CYAN = 2
    UV = 3
    # 4th bit is for green/yellow filter (not a color)
    BLUE = 5
    TEAL = 6
    OFF = 7

class ExcitationFilter(Enum):
    '''An enum to represent the excitation filters the Lumencore can use
    '''
    YELLOW = 0
    GREEN = 1

class Lumencore(Lamp):
    '''A class to control the Lumencore Spectra X Light engine
       Documentation can be found here: https://cms.lumencor.com/system/uploads/fae/file/asset/150/57-10035_Spectra_X_Command_Reference.pdf
    '''
    def __init__(self, com: serial.Serial):
        self.com = com
        self.current_light = LightColor.OFF
        self.current_excitation_filter = ExcitationFilter.GREEN
        self.shutter_state = "closed"
        self._power_levels = {}
        super().__init__()

    def _initialize(self):
        """Send hardware init commands to the Lumencore controller."""
        self.com.write(bytearray([0x57, 0x02, 0xFF, 0x50])) #init cmd 1
        self.com.write(bytearray([0x57, 0x03, 0xAB, 0x50])) #init cmd 2
        self.info("Lumencore initialized")

    def enable(self, light : LightColor, excitation_filter : ExcitationFilter = ExcitationFilter.GREEN):
        self.current_light = light
        self.current_excitation_filter = excitation_filter

        if light == LightColor.OFF:
            cmd = bytearray([0x4F, 0x7F, 0x50])
            self.com.write(cmd)
            return
        
        light_index = 0x00
        light_index |= 1 << light.value #set the bit of the light we want to enable

        #invert the bits so that 0 is on and 1 is off
        light_index = ~light_index & 0x7F #we only want the first 7 bits

        if excitation_filter == ExcitationFilter.YELLOW:
            light_index &= 0xEF #set the 4th bit to 0

        cmd = bytearray([0x4F, light_index, 0x50])

        self.com.write(cmd)
        self.info("Lumencore color {} enabled ({} Filter)".format(light.name, excitation_filter.name))

    
    def set_power(self, power_percent : float, light : LightColor):
        '''Sets the power of a light to a percentage of the maximum
           power_percent: float between 0 and 100
           light: LightColor enum
        '''
        if power_percent > 100:
            power_percent = 100
        elif power_percent < 0:
            power_percent = 0
        
        #calculate address of the DAC given light color
        dacAddress = None
        lightAddress = 0x01
        addr18Lights = [LightColor.UV, LightColor.CYAN, LightColor.GREEN, LightColor.RED]
        addr1ALights = [LightColor.BLUE, LightColor.TEAL]
        if light in addr18Lights:
            dacAddress = 0x18
            lightAddress = lightAddress << addr18Lights.index(light)
        else:
            dacAddress = 0x1A
            lightAddress = lightAddress << addr1ALights.index(light)
        
        # convert 0-100 to 0-2^8
        power = int((power_percent/100) * 255) & 0xFF
        #invert power (0xFF is 0% power, 0x00 is 100% power)
        power = ~power
        power_highnibble = (power & 0xF0) >> 4
        power_lownibble = power & 0x0F

        #form byte array to send to DAC
        cmd = bytearray([0x53, dacAddress, 0x03, lightAddress & 0x0F, power_highnibble | 0xF0, power_lownibble << 4, 0x50])

        #send command to DAC
        self.com.write(cmd)
        self._power_levels[light] = power_percent
        self.info("Lumencore color {} set to {}%".format(light.name, power_percent))

    def get_IIC_temp(self):
        '''Gets the temperature of the IIC in degrees C
        '''
        cmd = bytearray([0x53, 0x91, 0x02, 0x50])
        self.com.write(cmd)
        time.sleep(0.1)
        temp = self.com.read(2)
        #we only want the Most Significant 11 bits
        temp = temp[1] << 3 | temp[0] >> 5
        #convert to degrees C
        temp = temp * 0.125
        return temp

    def open_shutter(self):
        """Treat shutter as binary power control for the selected light."""
        if self.current_light == LightColor.OFF:
            self.info("Lumencore: No light selected; skipping open_shutter.")
            self.shutter_state = "closed"
            return
        power = self._power_levels.get(self.current_light, 100)
        self.set_power(power, self.current_light)
        self.enable(self.current_light, self.current_excitation_filter)
        self.shutter_state = "open"

    def close_shutter(self):
        """Disable all light output."""
        self.enable(LightColor.OFF, self.current_excitation_filter)
        self.shutter_state = "closed"

    def get_shutter_state(self):
        """Return cached shutter state."""
        return self.shutter_state

    def set_filter(self, filter: LightColor | None = None, excitation_filter: ExcitationFilter | None = None):
        """Set the desired light color (acts like a filter wheel selection)."""
        if filter is None:
            self.info("Lumencore: No filter specified, skipping set_filter.")
            return
        self.current_light = filter
        if excitation_filter is not None:
            self.current_excitation_filter = excitation_filter

        if self.shutter_state == "open":
            if filter == LightColor.OFF:
                self.close_shutter()
            else:
                power = self._power_levels.get(filter, 100)
                self.set_power(power, filter)
                self.enable(filter, self.current_excitation_filter)

    def get_filter(self) -> LightColor:
        """Return the currently selected light color."""
        return self.current_light

if __name__ == '__main__':
    lampCom = serial.Serial('COM6', 9600, timeout=1, stopbits=serial.STOPBITS_ONE, parity=serial.PARITY_NONE, bytesize=serial.EIGHTBITS)
    l = Lumencore(lampCom)

    l.enable(LightColor.BLUE, ExcitationFilter.GREEN)
    l.set_power(100, LightColor.BLUE)
    time.sleep(1)

    l.enable(LightColor.RED, ExcitationFilter.GREEN)
    l.set_power(100, LightColor.RED)
    time.sleep(1)

    l.enable(LightColor.OFF, ExcitationFilter.GREEN)
    # l.set_power(48, LightColor.CYAN)
            
