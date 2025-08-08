import time
import cv2
import numpy as np
from holypipette.deepLearning.autoPatcher import CellHunter


class AutoPatchHelper():
    """
    A helper class to aid with different stages of the auto patching process.
    Neuron Hunting
    Gigasealing 
    Break in
    """
    def __init__(self,  hunter: CellHunter, ):
        self.hunter = hunter
        pass

    def hunt(self,mode,type,input):
        # check mode, if classic, send back vector with just [0,0, 0,0, 0, -10], if Agent, send to hunter class in autopatcher.
        if mode == 'classic':
             vel = [0,0, 0,0, 0, -10]
        elif mode == 'Agent': 
              # pass type and input into hunter class, get position trajectory data
              
              pos = self.hunter.inference(type,input)
              # send position data into path planning script to convert to velocities.
              vel = self.pathplan(pos)

        return vel
    
    def pathplan(self, pos):
        """
        Convert a list of position samples (e.g. [[x1, y1, z1], [x2, y2, z2], …])
        into a flat velocity profile suitable for the manipulator.

        The method:
        • Uses a fixed time step (dt = 10 ms).
        • Computes velocities with central differences (forward/backward for the edges).
        • Duplicates the 3‑D velocity into the 6‑output format.
        • Limits each 3‑D velocity vector to `max_speed`.
        """
        # Guard against too few points
        if len(pos) < 2:
            return [0, 0, 0, 0, 0, 0]

        pos_arr = np.array(pos, dtype=float)
        dt = 0.01  # 10 ms per sample

        # Initialise velocity array
        vel_arr = np.zeros((len(pos_arr), 3))

        # Central difference for middle points
        vel_arr[1:-1] = (pos_arr[2:] - pos_arr[:-2]) / (2 * dt)

        # Forward difference for the first point
        vel_arr[0] = (pos_arr[1] - pos_arr[0]) / dt

        # Backward difference for the last point
        vel_arr[-1] = (pos_arr[-1] - pos_arr[-2]) / dt

        # Flatten and duplicate each 3‑D vector into the 6‑output format
        vel_flat = []
        for v in vel_arr:
            vel_flat.extend(v.tolist())

        # Clamp each 3‑D velocity to a maximum speed
        max_speed = 30.0  # adjust as needed
        for i in range(0, len(vel_flat), 3):
            v3 = np.array(vel_flat[i:i + 3])
            speed = np.linalg.norm(v3)
            if speed > max_speed:
                scale = max_speed / speed
                vel_flat[i:i + 3] = (v3 * scale).tolist()

        return vel_flat