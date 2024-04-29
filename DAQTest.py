import nidaqmx.system
import nidaqmx
from nidaqmx.types import CtrTime
import numpy as np
import matplotlib.pyplot as plt

ampRead = 'cDAQ1Mod1'
ampCmd = 'cDAQ1Mod4'

#list all daq devices connected
system = nidaqmx.system.System.local()
devices = system.devices
for device in system.devices:
    print(f'Device name: {device.name}')
    print('AI channels:', device.ai_physical_chans.channel_names)
    print('AO channels:', device.ao_physical_chans.channel_names)
    print('DI lines:', device.di_lines)
    print('DO lines:', device.do_lines)
# print("Devices:")
# print([d for d in devices])


ampReadDev = devices[ampRead]
ampCmdDev = devices[ampCmd]


def readAnalogInput(device, channel):
    with nidaqmx.Task() as task:
        task.ai_channels.add_ai_voltage_chan(f'{device}/{channel}')
        data = task.read(number_of_samples_per_channel=1000)
        return data

#continuously send square wave to channel 
def sendSquareWave(device, channel, wave_freq, samplesPerSec, dutyCycle):
    task = nidaqmx.Task()
    task.ao_channels.add_ao_voltage_chan(f'{device}/{channel}')
    task.timing.cfg_samp_clk_timing(samplesPerSec, sample_mode=nidaqmx.constants.AcquisitionType.CONTINUOUS)
    
    #create a wave_freq Hz square wave
    data = np.zeros(samplesPerSec)
    onTime = 1 / wave_freq * dutyCycle * samplesPerSec
    offTime = 1 / wave_freq * (1-dutyCycle) * samplesPerSec

    #calc period
    period = onTime + offTime

    #convert to int
    onTime = int(onTime)
    offTime = int(offTime)
    period = int(period)

    wavesPerSec = samplesPerSec // period

    print(onTime, offTime, wavesPerSec, samplesPerSec, wave_freq, period)

    for i in range(wavesPerSec):
        data[i * period : i * period + onTime] = 5

    
    # print(data)
    task.write(data)
    
    task.start()
    
    return task

#continuous read and plot with matplotlib
def readAnalogInputContinuous(device, channel):
    with nidaqmx.Task() as task:
        task.ai_channels.add_ai_voltage_chan(f'{device}/{channel}', max_val=10, min_val=0)
        task.timing.cfg_samp_clk_timing(10000, sample_mode=nidaqmx.constants.AcquisitionType.FINITE, samps_per_chan=1000)
        task.triggers.reference_trigger.cfg_anlg_edge_ref_trig(f'{device}/{channel}', pretrigger_samples = 10, trigger_slope=nidaqmx.constants.Slope.RISING, trigger_level = 2 )
        # task.triggers.reference_trigger.cfg_anlg_edge_ref_trig("Dev1/ai0", pretrigger_samples = 10, trigger_slope=nidaqmx.constants.Slope.FALLING, trigger_level = 5)
        squareWave = sendSquareWave(ampCmd, 'ao0', 100, 1000, 0.5)


        while True:
            task.start()
            data = task.read(number_of_samples_per_channel=1000)
            plt.plot(data)
            #fix scale between -0.1 and 0.1
            plt.ylim(-10, 10)
            plt.draw()
            #break on escape key
            if plt.waitforbuttonpress(0.001):
                break
            plt.pause(0.05)
            plt.clf()
            task.stop()

        
        #stop task
        task.stop()
        squareWave.stop()
        squareWave.close()
        print('done')
    
if __name__ == '__main__':
    import time

    print('starting continuous read')
    readAnalogInputContinuous(ampRead, 'ai1')
    # squareWave = sendSquareWave(ampCmd, 'ao1', 100, 1000, 0.5)
    # time.sleep(60)
    # squareWave.stop()