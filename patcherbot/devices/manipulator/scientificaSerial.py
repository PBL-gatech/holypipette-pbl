
import serial
from .manipulator import Manipulator
import time
import threading

__all__ = ['ScientificaSerial']

class SerialCommands():
    """
    Holds command templates for communicating with Scientifica stages.
    """
    GET_X_POS = 'PX\r'
    GET_Y_POS = 'PY\r'
    GET_Z_POS = 'PZ\r'
    GET_X_Y_Z = '\r'
    GET_MAX_SPEED = 'TOP\r'
    GET_MAX_ACCEL = 'ACC\r'
    GET_IS_BUSY = 's\r'

    SET_X_Y_POS_ABS = 'abs {} {}\r'
    SET_X_Y_POS_REL = 'rel {} {}\r'
    SET_X_Y_Z_POS_ABS = 'abs {} {} {}\r'
    SET_X_Z_POS_REL = 'rel {} {}\r'
    

    SET_Z_POS = 'absz {}\r'
    SET_MAX_SPEED = 'TOP {}\r'
    SET_MAX_ACCEL = 'ACC {}\r'

    SET_X_Y_Z_VEL = 'VJ {} {} {}\r'

    GET_BAUD = 'BAUD\r'
    SET_BAUD = 'BAUD {}\r'

    STOP = 'STOP\r'

class EncoderCorrectionAcquisitionThread(threading.Thread):
    """
    Background acquisition loop for ScientificaSerialEncoder that fuses
    stage XYZ with external encoder Z counts for slip correction.

    Reads AMT13 series quadrature counts (per the AMT13 datasheet) streamed
    by an Arduino (UNO R4) as two-line packets:
      ENC
      <count>
    Updates encoderZ and current_pos with corrected Z at a target polling rate.
    """
    def __init__(self, parent, z_axis_port, polling_freq):
        """
        Initializes the EncoderCorrectionAcquisitionThread for continuous Z-axis encoder polling.

        Args:
            parent (ScientificaSerialEncoder): The parent stage controller instance.
            z_axis_port (serial.Serial): Serial port connected to the Z-axis encoder.
            polling_freq (float): Frequency in Hz at which encoder and stage positions are polled.
        """
        super().__init__(daemon=True, name='encoder_correction_thread')
        self._parent = parent
        self._z_axis_port = z_axis_port
        self._polling_freq = polling_freq
        self._encoder_seq = 0
        self._encoder_lock = threading.Lock()
        self._encoder_buffer = bytearray()
        self._encoder_expect_value = False
        self._stage_pos = [0, 0, 0]

    def run(self):
        """Runs the polling loop in a background thread."""
        self.run_loop()

    def run_loop(self, freq=None):
        """
        Continuously polls encoder and stage positions.

        Args:
            freq: Optional polling frequency in Hz. Defaults to self._polling_freq.
        """
        if freq is None:
            freq = self._polling_freq
        while True:
            startTime = time.time()
            try:
                self._poll_encoder_stream()
            except Exception:
                pass

            try:
                xyz = self._parent._sendCmd(SerialCommands.GET_X_Y_Z)
                xyz = xyz.split('\t')
                xPos = int(xyz[0]) / 10.0
                yPos = int(xyz[1]) / 10.0
                zPos = int(xyz[2]) / 10.0
                self._stage_pos = [xPos, yPos, zPos]
            except Exception:
                print('error reading position')

            encoder_z, _ = self.get_encoder_state()
            self._parent.current_pos = [self._stage_pos[0], self._stage_pos[1], encoder_z]

            sleepTime = 1 / freq - (time.time() - startTime)
            if sleepTime > 0:
                time.sleep(sleepTime)

    def get_encoder_state(self):
        """
        Gets the current encoder Z position and sequence number.

        Returns:
            tuple: (encoderZ, sequence_number)
        """
        with self._encoder_lock:
            return self._parent.encoderZ, self._encoder_seq

    def get_stage_z(self):
        """Returns the stage-reported Z position in micrometers."""
        return self._stage_pos[2]

    def wait_for_update(self, last_seq, timeout_s):
        """
        Waits until encoder sequence updates or timeout occurs.

        Args:
            last_seq: Last known encoder sequence number.
            timeout_s: Maximum time to wait in seconds.

        Returns:
            bool: True if encoder updated, False if timeout occurred.
        """
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            _, seq = self.get_encoder_state()
            if seq != last_seq:
                return True
            time.sleep(0.01)
        return False

    def _update_encoder_from_count(self, count):
        """
        Updates parent encoder Z using a raw encoder count.
        
        Args:
            count (int): Raw encoder count.
        """
        with self._encoder_lock:
            self._parent.encoderZ = count * self._parent.stageUnitsPerEncoderPulse
            self._encoder_seq += 1

    def _consume_encoder_bytes(self, data):
        """
        Parses incoming encoder data from serial port.
        
        Args:
            data (bytes): Incoming serial data.
        """
        if not data:
            return
        self._encoder_buffer.extend(data)
        while b'\n' in self._encoder_buffer:
            line, _, remainder = self._encoder_buffer.partition(b'\n')
            self._encoder_buffer = remainder
            line = line.strip()
            if not line:
                continue
            if line == b'ENC':
                self._encoder_expect_value = True
                continue
            if self._encoder_expect_value:
                self._encoder_expect_value = False
                try:
                    count = int(line.decode('ascii', errors='ignore').strip())
                except ValueError:
                    continue
                self._update_encoder_from_count(count)

    def _poll_encoder_stream(self):
        """Reads available bytes from encoder serial port and parses them."""
        try:
            waiting = self._z_axis_port.in_waiting
        except Exception:
            return
        if not waiting:
            return
        try:
            data = self._z_axis_port.read(waiting)
        except Exception:
            return
        self._consume_encoder_bytes(data)

class ScientificaSerialEncoder(Manipulator):
    """
    Represents a Scientifica stage with Z-axis encoder correction.

    Handles serial communication, movement commands, and encoder-based Z
    position correction.
    """
    DEFAULT_STAGE_UNITS_PER_ENCODER_PULSE = 2.178649
    DEFAULT_MAX_SPEED = 10000
    DEFAULT_MAX_ACCEL = 100
    DEFAULT_POLLING_FREQ = 10
    DEFAULT_Z_CORRECTION_TOLERANCE_UM = 2.0
    DEFAULT_Z_CORRECTION_MAX_RETRIES = 5

    def __init__(self, comPort: serial.Serial, zAxisComPort, stageUnitsPerEncoderPulse=None):
        """
        Initializes the Scientifica stage controller with encoder support.

        Args:
            comPort (serial.Serial): Serial port connected to the X/Y stage.
            zAxisComPort (serial.Serial): Serial port connected to the Z stage encoder.
            stageUnitsPerEncoderPulse (float, optional): Conversion factor from encoder counts to stage units. 
                Defaults to 1.45 if not provided.
        """
        self.comPort : serial.Serial = comPort

        self.zAxisComPort : serial.Serial = zAxisComPort
        self.stageUnitsPerEncoderPulse = stageUnitsPerEncoderPulse if stageUnitsPerEncoderPulse is not None else 1.45
    def __init__(self, comPort: serial.Serial, zAxisComPort,
                 stage_units_per_encoder_pulse=None,
                 max_speed=None,
                 max_accel=None,
                 polling_freq=None):
        """
        Initializes the Scientifica stage with Z-axis encoder correction.

        Sets up serial communication, movement limits, and a background thread
        for continuous encoder-based Z-axis position updates.

        Args:
            comPort (serial.Serial): Serial port connected to the stage controller.
            zAxisComPort (serial.Serial): Serial port connected to the Z-axis encoder.
            stage_units_per_encoder_pulse (float, optional): Conversion factor from encoder counts to stage units.
                Defaults to DEFAULT_STAGE_UNITS_PER_ENCODER_PULSE.
            max_speed (float, optional): Maximum stage speed. Defaults to DEFAULT_MAX_SPEED.
            max_accel (float, optional): Maximum stage acceleration. Defaults to DEFAULT_MAX_ACCEL.
            polling_freq (float, optional): Frequency in Hz to poll encoder updates. Defaults to DEFAULT_POLLING_FREQ.

        Raises:
            serial.SerialException: If the serial ports cannot be opened.
        """
        self.comPort : serial.Serial = comPort

        self.zAxisComPort : serial.Serial = zAxisComPort
        self.stageUnitsPerEncoderPulse = (
            self.DEFAULT_STAGE_UNITS_PER_ENCODER_PULSE
            if stage_units_per_encoder_pulse is None
            else stage_units_per_encoder_pulse
        )
        self.encoderZ = 0

        self._lock = threading.Lock()
        self.current_pos = [0, 0, 0]
        self._polling_freq = self.DEFAULT_POLLING_FREQ if polling_freq is None else polling_freq

        # self.info(f"Baud Rate: {self.get_baud_rate()}")

        self.set_max_accel(self.DEFAULT_MAX_ACCEL if max_accel is None else max_accel)
        self.set_max_speed(self.DEFAULT_MAX_SPEED if max_speed is None else max_speed)
        

        #start constantly polling position in a new thread
        self._encoder_acq = EncoderCorrectionAcquisitionThread(
            parent=self,
            z_axis_port=self.zAxisComPort,
            polling_freq=self._polling_freq
        )
        self._polling_thread = self._encoder_acq
        self._polling_thread.start()
        self._polling_thread.deamon = True

    def get_baud_rate(self):
        '''
        gets the baud rate of the serial port

        Returns:
            str: Current baud rate as reported by the stage.
        '''
        resp = self._sendCmd(SerialCommands.GET_BAUD)
        return (resp)

    def set_baud_rate(self, baud_rate : int):
        '''Sets the baud rate of the serial port.  
        
        Args:
            baud_rate (int): Desired baud rate.
        '''
        self._sendCmd(SerialCommands.SET_BAUD.format(int(baud_rate)))



    def get_max_speed(self):
        '''Gets the max speed for the Scientifica Stage.  
           It seems like the range for this is around (1000, 100000)
        
        Returns:
            int: Maximum speed.
        '''
        resp = self._sendCmd(SerialCommands.GET_MAX_SPEED)
        return int(resp)
    def get_max_accel(self):
        '''Gets the max acceleration for the Scientifica Stage.
           It seems like the range for this is around (10, 10000)
        
        Returns:
            int: Maximum acceleration.
        '''
        resp = self._sendCmd(SerialCommands.GET_MAX_ACCEL)
        return int(resp)
    
    def set_max_speed(self, speed):
        '''Sets the max speed for the Scientifica Stage.  
           It seems like the range for this is around (1000, 100000)
        
        Args:
            speed (int): Maximum speed.
        '''
        self._sendCmd(SerialCommands.SET_MAX_SPEED.format(int(speed)))

    def set_max_accel(self, accel):
        '''Sets the max acceleration for the Scientifica Stage.
           It seems like the range for this is around (10, 10000)
        
        Args:
            accel (int): Maximum acceleration.
        '''
        self._sendCmd(SerialCommands.SET_MAX_ACCEL.format(int(accel)))

    def __del__(self):
        """
        Closes serial ports on deletion.
        """
        try:
            if hasattr(self, "comPort") and self.comPort:
                self.comPort.close()
        except Exception:
            pass

    def _sendCmd(self, cmd):
        '''Sends a command to the stage and returns the response

        Args:
            cmd (str): Command string.

        Returns:
            str: Response from the stage.
        '''

        self._lock.acquire()
        self.comPort.write(cmd.encode())
        resp = self.comPort.read_until(b'\r') #read reply to message
        resp = resp[:-1]
        self._lock.release()

        return resp.decode()

    def position(self, axis=None):
        """
        Returns current position of the stage.

        Args:
            axis (int, optional): Axis index (1=X, 2=Y, 3=Z). If None, returns [X, Y, Z].

        Returns:
            float or list: Position along requested axis, or full [X, Y, Z].
        """
        if axis == 1:
            return self.current_pos[0]
        if axis == 2:
            return self.current_pos[1]
        if axis == 3:
            encoder_z, _ = self._encoder_acq.get_encoder_state()
            return encoder_z
        if axis == None:
            return self.current_pos
        
    def update_pos_continuous(self, freq=None):
        '''constantly polls the device's position and updates the current_pos variable
        
        Args:
            freq (float, optional): Polling frequency in Hz.
        '''
        self._encoder_acq.run_loop(freq=freq)

    def absolute_move(self, pos, axis):
        """
        Moves a single axis to an absolute position.

        Args:
            pos (float): Target position.
            axis (int): Axis index (1=X, 2=Y, 3=Z).

        Raises:
            RuntimeError: If Z-axis correction fails after max retries.
        """
        if axis == 1:
            yPos = self.position(axis=2)
            self._sendCmd(SerialCommands.SET_X_Y_POS_ABS.format(int(pos * 10) , int(yPos * 10)))
        if axis == 2:
            xPos = self.position(axis=1)
            self._sendCmd(SerialCommands.SET_X_Y_POS_ABS.format(int(xPos * 10), int(pos * 10)))
        if axis == 3:
            max_retries = self.DEFAULT_Z_CORRECTION_MAX_RETRIES
            tolerance = self.DEFAULT_Z_CORRECTION_TOLERANCE_UM
            for attempt in range(max_retries):
                stageZ = self._encoder_acq.get_stage_z()
                encoder_z, seq = self._encoder_acq.get_encoder_state()
                setpointStage = stageZ + (pos - encoder_z)
                self._sendCmd(SerialCommands.SET_Z_POS.format(int(setpointStage * 10)))
                self.wait_until_still()
                time.sleep(1)
                self._encoder_acq.wait_for_update(seq, max(0.1, 2.0 / self._polling_freq))
                encoder_z, _ = self._encoder_acq.get_encoder_state()
                print(f'expected encoder: {pos} actual {encoder_z}')
                error = pos - encoder_z
                print(f'error: {error}')
                if abs(error) <= tolerance:
                    break
                if attempt < max_retries - 1:
                    print('retrying')
    
    def absolute_move_group(self, x, axes, speed=None):
        """
        Moves multiple axes to absolute positions simultaneously.

        Args:
            x (list[float]): Target positions corresponding to each axis in `axes`.
            axes (list[int]): List of axes to move (1=X, 2=Y, 3=Z).
            speed (float, optional): Desired speed for the move (currently unused).

        Raises:
            NotImplementedError: If the axis combination is not supported.
        """
        x = list(x)
        axes = list(axes)

        if 1 in axes and 2 in axes and 3 in axes:
            # Move X, Y and Z axes together
            xPos = x[axes.index(1)]
            yPos = x[axes.index(2)]
            zPos = x[axes.index(3)]
            print("sent cmd", xPos, yPos, zPos)
            stageZ = self._encoder_acq.get_stage_z()
            encoder_z, _ = self._encoder_acq.get_encoder_state()
            setpointStage = stageZ + (zPos - encoder_z)
            self._sendCmd(SerialCommands.SET_X_Y_Z_POS_ABS.format(
                int(xPos * 10),
                int(yPos * 10),
                int(setpointStage * 10)
            ))

        elif 1 in axes and 2 in axes:
            # Move X and Y axes together
            xPos = x[axes.index(1)]
            yPos = x[axes.index(2)]
            print("sent cmd", xPos, yPos)
            self._sendCmd(SerialCommands.SET_X_Y_POS_ABS.format(int(xPos * 10), int(yPos * 10)))

        else:
            print(f'unimplemented move group {x} {axes}')

    
    def relative_move_group(self, pos, axis, speed=None):
        """
        Moves multiple axes relative to their current positions.

        Args:
            x (list[float]): Relative distances for each axis in `axes`.
            axes (list[int]): List of axes to move (1=X, 2=Y, 3=Z).
            speed (float, optional): Desired speed for the move (currently unused).
        """
        if axis == 1:
            self._sendCmd(SerialCommands.SET_X_Y_POS_REL.format(pos, 0))
        if axis == 2:
            self._sendCmd(SerialCommands.SET_X_Y_POS_REL.format(0, pos))
        if axis == 3:
            absZCmd = self.position(3) + pos
            self.absolute_move(absZCmd, 3)

    def relative_move_group(self, x, axes):
        """
        Moves multiple axes relative to their current positions.

        Args:
            x (list[float]): Relative distances for each axis in `axes`.
            axes (list[int]): List of axes to move (1=X, 2=Y, 3=Z).
        """
        cmd = [0, 0, 0]
        for pos, axis in zip(x, axes):
            cmd[axis  - 1] = pos
        
        if cmd[0] != 0 or cmd[1] != 0:
            print("sent cmd", cmd[0], cmd[1])
            self._sendCmd(SerialCommands.SET_X_Y_POS_REL.format(int(cmd[0] * 10), int(cmd[1] * 10)))

        if cmd[2] != 0:
            self.relative_move(cmd[2], 3)

    def absolute_move_group_velocity(self,vel,axes):
        """
        Sets the movement velocity for multiple axes simultaneously.

        Args:
            vel (list[float]): Velocity values for each axis in `axes`.
            axes (list[int]): List of axes (1=X, 2=Y, 3=Z).

        Raises:
            ValueError: If a specified axis is not 1, 2, or 3.
            Exception: If the velocity command fails to send.
        """   
        try: 
         self.info(f"Setting velocity to {vel} on axes {axes}")
         vel = list(vel)
         axes = list(axes)
         xvel = vel[axes.index(1)]
         yvel = vel[axes.index(2)]
         zvel = vel[axes.index(3)]
         self._sendCmd(SerialCommands.SET_X_Y_Z_VEL.format(xvel, yvel, zvel))
        except Exception as e:
            self.error(f"Error in absolute_move: {e}")

    def wait_until_still(self, axes = None, axis = None):
        """
        Blocks execution until the stage reports it is no longer busy.

        Args:
            axes (list or None): Ignored, for compatibility.
            axis (int or None): Ignored, for compatibility.
        """
        while True:
            resp = self._sendCmd(SerialCommands.GET_IS_BUSY)
            busy = resp != '0'

            if not busy:
                break

    def stop(self):
        """
        Stops all motion immediately.
        """
        self._sendCmd(SerialCommands.STOP)

class ScientificaSerialNoEncoder(Manipulator):
    DEFAULT_MAX_SPEED = 100000
    DEFAULT_MAX_ACCEL = 1000
    DEFAULT_POLLING_FREQ = 100

    def __init__(self, comPort: serial.Serial,
                 max_speed=None,
                 max_accel=None,
                 polling_freq=None):
        """
        Initializes the serial connection to a Scientifica stage without encoder.

        Args:
            comPort (serial.Serial): Serial port connected to the stage.
            max_speed (float, optional): Maximum stage speed.
            max_accel (float, optional): Maximum stage acceleration.
            polling_freq (float, optional): Frequency to poll the device for position updates.
        """
        self.comPort : serial.Serial = comPort
        self._lock = threading.Lock()
        self.current_pos = [0, 0, 0]
        self._polling_freq = self.DEFAULT_POLLING_FREQ if polling_freq is None else polling_freq

        self.set_max_accel(self.DEFAULT_MAX_ACCEL if max_accel is None else max_accel)
        self.set_max_speed(self.DEFAULT_MAX_SPEED if max_speed is None else max_speed)

        #start constantly polling position in a new thread
        self._polling_thread = threading.Thread(target=self.update_pos_continuous, daemon=True)
        self._polling_thread.start()
        self._polling_thread.deamon = True

        self.info(f"Baud Rate: {self.get_baud_rate()}")

    

    def get_baud_rate(self):
        '''
        gets the baud rate of the serial port

        Returns:
            str: Current baud rate reported by the stage.
        '''
        resp = self._sendCmd(SerialCommands.GET_BAUD)
        return (resp)

    def set_baud_rate(self, baud_rate : int):
        '''Sets the baud rate of the serial port.  

        Args:
            baud_rate (int): Desired baud rate.
        '''
        self._sendCmd(SerialCommands.SET_BAUD.format(int(baud_rate)))


    def set_max_speed(self, speed):
        '''Sets the max speed for the Scientifica Stage.  
           It seems like the range for this is around (1000, 100000)
        
        Args:
            speed (float): Maximum speed.
        '''
        self._sendCmd(SerialCommands.SET_MAX_SPEED.format(int(speed)))

    def set_max_accel(self, accel):
        '''Sets the max acceleration for the Scientifica Stage.
           It seems like the range for this is around (10, 10000)

        Args:
            accel (float): Maximum acceleration.
        '''
        self._sendCmd(SerialCommands.SET_MAX_ACCEL.format(int(accel)))

    def __del__(self):
        """
        Destructor; closes the serial port.
        """
        self.comPort.close()

    def _sendCmd(self, cmd):
        '''Sends a command to the stage and returns the response
        
        Args:
            cmd (str): Command string to send.

        Returns:
            str: Response from the stage.
        '''
        self._lock.acquire()
        # start  = time.perf_counter_ns()
        self.comPort.write(cmd.encode())
        resp = self.comPort.read_until(b'\r') #read reply to message
        resp = resp[:-1]
        # if resp == b'A':
        #     print(f"command received: {resp}")
        # end = time.perf_counter_ns()
        # print(f"Time taken to send command: {(end - start)/1e6} ms")
        self._lock.release()
        return resp.decode()

    def position(self, axis=None):
        """
        Returns the current position of the stage.

        Args:
            axis (int, optional): Axis to query (1=X, 2=Y, 3=Z). If None, returns all axes.

        Returns:
            float or list[float]: Position of the specified axis, or [X, Y, Z] if axis=None.
        """
        if axis == 1:
            return self.current_pos[0]
        if axis == 2:
            return self.current_pos[1]
        if axis == 3:
            return self.current_pos[2]
        if axis == None:
            return self.current_pos
        
    def update_pos_continuous(self, freq=None):
        '''constantly polls the device's position and updates the current_pos variable
        
        Args:
            freq (float, optional): Polling frequency in Hz. Defaults to `_polling_freq`.
        '''
        if freq is None:
            freq = self._polling_freq
        while True:
            startTime = time.time()
            xyz = self._sendCmd(SerialCommands.GET_X_Y_Z)
            xyz = xyz.split('\t')
            
            try:
                xPos = int(xyz[0]) / 10.0
                yPos = int(xyz[1]) / 10.0
                zPos = int(xyz[2]) / 10.0
                self.current_pos = [xPos, yPos, zPos]
            except:
                print('error reading position')

            sleepTime = 1 / freq - (time.time() - startTime)
            if sleepTime > 0:
                time.sleep(sleepTime)

    def absolute_move(self, pos, axis, speed=None):
        '''Moves the device to an absolute position in um.
        
        Args:
            pos (float): Target position in um.
            axis (int): Axis number (1=X, 2=Y, 3=Z).
            speed (float, optional): Movement speed (unused).

        Raises:
            Exception: If move fails.
        '''
        # print(f"absolute move {pos} {axis}")
        try: 
            if axis == 1:
                yPos = self.position(axis=2)
                self._sendCmd(SerialCommands.SET_X_Y_POS_ABS.format(round(pos * 10) , round(yPos * 10)))
            if axis == 2:
                xPos = self.position(axis=1)
                self._sendCmd(SerialCommands.SET_X_Y_POS_ABS.format(round(xPos * 10), round(pos * 10)))
            if axis == 3:
                self._sendCmd(SerialCommands.SET_Z_POS.format(round(pos * 10)))
        except Exception as e:
            self.error(f"Error in absolute_move: {e}")
    
    def absolute_move_group(self, x, axes, speed=None):
        '''
        Moves the device group of axes to position x.

        Args:
            x (list[float]): Target positions in um.
            axes (list[int]): Axes to move (1=X, 2=Y, 3=Z).
            speed (float, optional): Movement speed (unused).
        '''
        # self.abort_if_requested()
        x = list(x)
        axes = list(axes)

        if 1 in axes and 2 in axes and 3 not in axes:
            # Move X and Y axes together
            xPos = x[axes.index(1)]
            yPos = x[axes.index(2)]
            self._sendCmd(SerialCommands.SET_X_Y_POS_ABS.format(int(xPos * 10), int(yPos * 10)))

        elif 1 in axes and 2 in axes and 3 in axes:
            # Move X, Y and Z axes together
            xPos = x[axes.index(1)]
            yPos = x[axes.index(2)]
            zPos = x[axes.index(3)]
            self._sendCmd(SerialCommands.SET_X_Y_Z_POS_ABS.format(int(xPos * 10), int(yPos * 10), int(zPos * 10)))
            # end  = time.perf_counter_ns()
            # print(f"Time taken to move: {(end - start)/1e6} ms")


        else:
            print(f'unimplemented move group {x} {axes}')

    def absolute_move_group_velocity(self, vel):
        '''
        Moves the device in um/s.
        
        Args:
            vel (list[float]): Velocities [x, y, z] in um/s.

        Raises:
            ValueError: If vel is not length 3.
        '''
        # self.abort_if_requested()   
        try: 
            vel = list(vel)
            if len(vel) != 3:
                raise ValueError("Expected velocity list of length 3: [xvel, yvel, zvel]")
            xvel, yvel, zvel = vel
            self._sendCmd(SerialCommands.SET_X_Y_Z_VEL.format(xvel, yvel, zvel))
        except Exception as e:
            self.error(f"Error in absolute_move: {e}")
        
    def relative_move_group(self, pos, axis, speed=None):
        '''Moves the device axis by relative amount pos in um.
        
        Args:
            pos (float): Relative movement in um.
            axis (int): Axis number (1=X, 2=Y, 3=Z).
            speed (float, optional): Movement speed (unused).
        '''
        # self.abort_if_requested()
        if axis == 1:
            self._sendCmd(SerialCommands.SET_X_Y_POS_REL.format(pos, 0))
        if axis == 2:
            self._sendCmd(SerialCommands.SET_X_Y_POS_REL.format(0, pos))
        if axis == 3:
            absZCmd = self.position(3) + pos
            self.absolute_move(absZCmd, 3)

    def relative_move_group(self, x, axes, speed=None):
        '''
        Moves the device group of axes by relative amount x in um.
        
        Args:
            x (list[float]): Relative movements for each axis in `axes`.
            axes (list[int]): Axes to move (1=X, 2=Y, 3=Z).
            speed (float, optional): Movement speed (unused).
        '''
        # self.abort_if_requested()
        cmd = [0, 0, 0]
        for pos, axis in zip(x, axes):
            cmd[axis  - 1] = pos
        
        if cmd[0] != 0 or cmd[1] != 0:
            self._sendCmd(SerialCommands.SET_X_Y_POS_REL.format(int(cmd[0] * 10), int(cmd[1] * 10)))

        if cmd[2] != 0:
            self.relative_move(cmd[2], 3)

    def wait_until_still(self, axes = None, axis = None):
        """
        Blocks until the stage reports it is no longer moving.

        Args:
            axes (list[int], optional): Axes to wait for (unused).
            axis (int, optional): Single axis to wait for (unused).
        """
        while True:
            resp = self._sendCmd(SerialCommands.GET_IS_BUSY)
            busy = resp != '0'

            if not busy:
                break

    def stop(self):
        """
        Stops any ongoing movement immediately.
        """
        self._sendCmd(SerialCommands.STOP)


