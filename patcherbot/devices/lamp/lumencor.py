from enum import Enum
import serial
import time
import logging

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

class Lumencore:
    '''A class to control the Lumencore Spectra X Light engine
       Documentation can be found here: https://cms.lumencor.com/system/uploads/fae/file/asset/150/57-10035_Spectra_X_Command_Reference.pdf
    '''
    def __init__(self, com: serial.Serial):
        self.com = com
        #send init commands
        self.com.write(bytearray([0x57, 0x02, 0xFF, 0x50])) #init cmd 1
        self.com.write(bytearray([0x57, 0x03, 0xAB, 0x50])) #init cmd 2

        self.logger = logging.getLogger(__name__)
        self.logger.info("Lumencore initialized")

    def enable(self, light : LightColor, excitation_filter : ExcitationFilter = ExcitationFilter.GREEN):

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
        logging.info("Lumencore color {} enabled ({} Filter)".format(light.name, excitation_filter.name))

    
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
        logging.info("Lumencore color {} set to {}%".format(light.name, power_percent))

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
            
