import time
import cv2
import numpy as np
from holypipette.deepLearning.autoPatcher import CellHunter,GigaSealer,Burglar


class AutoPatchHelper:
    """
    A helper class to aid with different stages of the auto patching process.
    Neuron Hunting
    Gigasealing 
    Break in
    """
    def __init__(self):
        self.hunter = CellHunter()
        self.gigasealer = GigaSealer()
        self.burglar = Burglar()
        self.poslist =[]
        self.hunterh0 = None
        self.hunterc0 = None
    

    def hunt(self,model_input):
        pos, self.hunterh0, self.hunterc0 = self.hunter.inference(model_input, self.hunterh0, self.hunterc0)
        # onnx returns shape (1,6); flatten to (6,) for downstream slicing
        pos = np.asarray(pos).reshape(-1)
        pos = self.clamp_positions(pos)
        return pos

    
    def gigaseal(self,mode,type,input):
       pass
    def breakin(self,mode,type,input):
        pass
    
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

    def clamp_positions(self,positions):
        '''
        restrict maximum distance model can predict for manipulators/stage to move to 10 microns
        '''
        positions = np.array(positions)
        max_distance = 10 # microns
        positions[positions > max_distance] = max_distance
        return positions.tolist()

    def prime_model(self,model,model_input):
        """
        Prepare a given model for accurate inferencing. requires loading and preloading inputs to provide an accurate prediction set
    
        """
        if model == 'hunt':
            self.hunter.load_model()
            modelactor = self.hunter
        elif model == 'gigaseal':
             self.gigasealer.load_model()
             modelactor=self.gigasealer
        elif model == 'break_in':
             self.burglar.load_model()
             modelactor = self.burglar

        predlist = []
        h0list = []
        c0list = []
        for i in range(len(model_input)):
            if i == 0:
                h0 = None
                c0 = None
            else:
                h0 = h0list[i-1]
                c0 = c0list[i-1]

            pred, h0, c0 = modelactor.inference(model_input[i], h0, c0)

            predlist.append(pred)
            h0list.append(h0)
            c0list.append(c0)
            print(f"observation primed {i}")




        
        
