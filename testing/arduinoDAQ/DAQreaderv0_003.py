

import serial
import time



def send_command(recordingTime, wave_freq, amplitude, samplesPerSec, dutyCycle, scaling):
    # Define the command to send
     signalDurationMicros = int(recordingTime * 1e6)  # Duration in microseconds
     waveFrequencyMicros = int(1e6 / wave_freq)  # Wave period in microseconds
     waveAmplitude = int(amplitude * scaling)  # Amplitude scaled to 10-bit DAC
     sampleIntervalMicros = int(1e6 / samplesPerSec)  # Sampling interval in microseconds
     dutyCyclePercent = int(dutyCycle * 100)  # Duty cycle in percentage
     command = f"a {signalDurationMicros} {waveFrequencyMicros} {waveAmplitude} {sampleIntervalMicros} {dutyCyclePercent}\n"
     serial_port.write(command.encode('utf-8'))
     print(f"Sent command: {command.strip()}")

def read_serial_data():
    collecting_data = False
    
    # Continuously read data from the Arduino
    while True:
        if serial_port.in_waiting > 0:  # Check if data is available
            line = serial_port.readline().decode('utf-8').strip()
            
            if line == "start":
                collecting_data = True
                print("Data collection started")
            elif line == "end":
                print("Data collection ended")
                break
            elif collecting_data:
                # Split the line by comma to get command and response values
                values = line.split(',')
                if len(values) == 2:
                    command_value, response_value = values
                    print(f"Received: Command={command_value}, Response={response_value}")
                else:
                    print(f"Unexpected data format: {line}")


if __name__ == '__main__':
    # Send the command
        # Set up the serial connection (adjust COM port as needed)
    serial_port = serial.Serial('COM11', 115200, timeout=1)  # Set timeout to 1 second
    recordingTime = 0.04  # Duration of signal in seconds
    wave_freq = 250  # Frequency of the wave in Hz
    amplitude = 0.66 # Amplitude of the wave in decimal of maximum value
    samplesPerSec = 10000 # Sampling rate in Hz
    dutyCycle = 0.5 # Duty cycle of the square wave
    scaling = 1023   # Scaling factor for 10-bit DAC
    send_command(recordingTime, wave_freq, amplitude, samplesPerSec, dutyCycle, scaling)
    
    # Give the Arduino a moment to process and start sending data
    time.sleep(0.1)
    
    # Read and print incoming serial data
    print("Waiting for data from Arduino:")
    read_serial_data()
    
    # Close the serial port
    serial_port.close()
