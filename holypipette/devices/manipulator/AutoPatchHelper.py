import time
import cv2
import numpy as np
from holypipette.deepLearning.autoPatcher import CellHunter, GigaSealer, Burglar, PipetteFinder


class AutoPatchHelper:
    """
    A helper class to aid with different stages of the auto patching process.
    Neuron Hunting
    Gigasealing 
    Break in
    """
    def __init__(self):
        self.hunter = CellHunter()
        self.finder = PipetteFinder()
        self.gigasealer = GigaSealer()
        self.burglar = Burglar()
        self.poslist = []
        self.hunterh0 = None
        self.hunterc0 = None
        self.finderh0 = None
        self.finderc0 = None
        self._hunter_state_snapshot = None
        self._finder_state_snapshot = None
    

    def hunt(self, model_input):
        """
        Preprocess + handshake:
          • Wrapper models (obs::/goal::): build {"obs": (...), "goal": partial or omitted}
            and pass RAW HWC image + raw numerics (normalization handled inside ONNX).
          • Legacy models: pass (pip, stage, img, res); CellHunter will normalize/crop/stack.
        """
        # Ensure model is loaded and wrapper flags are known
        if not hasattr(self, "_hunter_uses_wrapper") or getattr(self.hunter, "input_names", None) is None:
            self.prepare_model("hunt")

        # Coerce observation (mild normalization of types/shapes only)
        pip, stage, img, res = model_input
        pip   = np.asarray(pip,   np.float32).reshape(-1)
        stage = np.asarray(stage, np.float32).reshape(-1)
        if stage.shape[0] == 2:
            stage = np.concatenate([stage, [0.0]]).astype(np.float32)
        else:
            stage = stage[:3].astype(np.float32)
        res = np.asarray(res, np.float32).reshape(-1)
        img = np.asarray(img)
        if img.ndim == 2:  # gray → 3‑chan
            img = np.stack([img]*3, axis=-1)

        if getattr(self, "_hunter_uses_wrapper", False):
            payload = {"obs": (pip, stage, img, res)}
            if getattr(self, "_hunter_has_goal", False) and hasattr(self, "_goal") and len(self._goal) > 0:
                # Partial goal OK (e.g., no image) — CellHunter fills missing keys from obs
                payload["goal"] = self._goal
            model_payload = payload
        else:
            # Legacy model — give tuple; CellHunter does CHW+resize+[0,1] and stacking
            model_payload = (pip, stage, img, res)
        if self._hunter_state_snapshot is not None:
            self.hunter.set_state_snapshot(self._hunter_state_snapshot)
        print(f"model payload prepared {type(model_payload)}")
        pos, h0_out, c0_out = self.hunter.inference(model_payload, self.hunterh0, self.hunterc0)
        snapshot = self.hunter.get_state_snapshot()
        self._hunter_state_snapshot = snapshot
        if snapshot:
            self.hunterh0 = snapshot.get("h0")
            self.hunterc0 = snapshot.get("c0")
        else:
            self.hunterh0, self.hunterc0 = h0_out, c0_out
        print(f"model inference returned pos {pos}")
        pos = np.asarray(pos).reshape(-1)     # (6,)
        pos = self.clamp_positions(pos)
        return pos


    
    def find_pipette(self, model_input):
        """
        Preprocess + handshake for pipette localisation.
          - Wrapper models (obs::/goal::): build {"obs": (...)} and pass raw HWC image.
          - Legacy models: pass (pip, stage, img); PipetteFinder normalises internally.
        """
        if not hasattr(self, "_finder_uses_wrapper") or getattr(self.finder, "input_names", None) is None:
            self.prepare_model("find_pipette")

        pip, stage, img = model_input
        pip = np.asarray(pip, np.float32).reshape(-1)
        stage = np.asarray(stage, np.float32).reshape(-1)
        if stage.shape[0] == 2:
            stage = np.concatenate([stage, [0.0]]).astype(np.float32)
        else:
            stage = stage[:3].astype(np.float32)
        img = np.asarray(img)
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)

        if getattr(self, "_finder_uses_wrapper", False):
            payload = {"obs": (pip, stage, img)}
            goal = getattr(self, "_finder_goal", None)
            if getattr(self, "_finder_has_goal", False) and goal:
                payload["goal"] = goal
            model_payload = payload
        else:
            model_payload = (pip, stage, img)
        if self._finder_state_snapshot is not None:
            self.finder.set_state_snapshot(self._finder_state_snapshot)
        print(f"pipette finder payload prepared {type(model_payload)}")

        pos, h0_out, c0_out = self.finder.inference(model_payload, self.finderh0, self.finderc0)
        snapshot = self.finder.get_state_snapshot()
        self._finder_state_snapshot = snapshot
        if snapshot:
            self.finderh0 = snapshot.get("h0")
            self.finderc0 = snapshot.get("c0")
        else:
            self.finderh0, self.finderc0 = h0_out, c0_out
        print(f"pipette finder inference returned pos {pos}")
        pos = np.asarray(pos).reshape(-1)
        return self.clamp_positions(pos)

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
        max_speed = 5.0  # adjust as needed
        for i in range(0, len(vel_flat), 3):
            v3 = np.array(vel_flat[i:i + 3])
            speed = np.linalg.norm(v3)
            if speed > max_speed:
                scale = max_speed / speed
                vel_flat[i:i + 3] = (v3 * scale).tolist()

        return vel_flat

    def clamp_positions(self, positions, max_distance=10.0):
        # ensure 1-D float array
        arr = np.asarray(positions, dtype=float).ravel()
        # clamp both positive and negative to ±max_distance
        arr = np.clip(arr, -max_distance, max_distance)
        return arr.tolist()  # <-- returns a plain Python list

    def prime_model(self,model,model_input):
        """
        Prepare a given model for accurate inferencing. requires loading and preloading inputs to provide an accurate prediction set
    
        """
        if model == 'hunt':
            self.hunter.load_model()
            modelactor = self.hunter
        elif model == 'find_pipette':
             self.finder.load_model()
             modelactor = self.finder
        elif model == 'gigaseal':
             self.gigasealer.load_model()
             modelactor=self.gigasealer
        elif model == 'break_in':
             self.burglar.load_model()
             modelactor = self.burglar

        modelactor.reset_state()

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
        snapshot = modelactor.get_state_snapshot()
        if model == 'find_pipette':
            self._finder_state_snapshot = snapshot
            self.finderh0 = snapshot.get("h0") if snapshot else h0
            self.finderc0 = snapshot.get("c0") if snapshot else c0
        elif model == 'hunt':
            self._hunter_state_snapshot = snapshot
            self.hunterh0 = snapshot.get("h0") if snapshot else h0
            self.hunterc0 = snapshot.get("c0") if snapshot else c0


    def prepare_model(self, which="hunt", *, onnx_path=None, providers=None):
        """
        Loads the selected model and records whether it uses new wrapper IO (obs::/goal::).
        Compatible with your call site: self.autopatchhelper.prepare_model("hunt")
        You may optionally pass an explicit `onnx_path` to control which policy is loaded.
        """
        if which == "hunt":
            self.hunter.load_model(onnx_path, providers=providers)
            self.hunter.reset_state()
            self._hunter_state_snapshot = self.hunter.get_state_snapshot()
            self.hunterh0 = self._hunter_state_snapshot.get("h0") if self._hunter_state_snapshot else None
            self.hunterc0 = self._hunter_state_snapshot.get("c0") if self._hunter_state_snapshot else None
            self._hunter_uses_wrapper = any(n.startswith("obs::") for n in self.hunter.input_names)
            self._hunter_has_goal     = any(n.startswith("goal::") for n in self.hunter.input_names)
        elif which == "find_pipette":
            self.finder.load_model(onnx_path, providers=providers)
            self.finder.reset_state()
            self._finder_state_snapshot = self.finder.get_state_snapshot()
            self.finderh0 = self._finder_state_snapshot.get("h0") if self._finder_state_snapshot else None
            self.finderc0 = self._finder_state_snapshot.get("c0") if self._finder_state_snapshot else None
            self._finder_uses_wrapper = any(n.startswith("obs::") for n in self.finder.input_names)
            self._finder_has_goal     = any(n.startswith("goal::") for n in self.finder.input_names)
        elif which == "gigaseal":
            self.gigasealer.load_model(onnx_path, providers=providers)
            self.gigasealer.reset_state()
        elif which == "break_in":
            self.burglar.load_model(onnx_path, providers=providers)
            self.burglar.reset_state()
        else:
            raise ValueError(f"Unknown model '{which}'")
        return self


    def set_goal(self, *, pip=None, stage=None, resistance=None, image=None):
        """
        Store a (possibly partial) goal. Any missing keys (e.g., image) will be mirrored
        from the current observation by the wrapper-aware path in CellHunter.
        """
        g = {}
        if pip is not None:
            g["pipette_positions"] = np.asarray(pip, np.float32).reshape(-1)
        if stage is not None:
            s = np.asarray(stage, np.float32).reshape(-1)
            if s.shape[0] == 2:
                s = np.concatenate([s, [0.0]]).astype(np.float32)
            else:
                s = s[:3].astype(np.float32)
            g["stage_positions"] = s
        if resistance is not None:
            g["resistance"] = np.asarray(resistance, np.float32).reshape(-1)
        if image is not None:
            im = np.asarray(image)
            if im.ndim == 2:  # gray → 3‑chan
                im = np.stack([im]*3, axis=-1)
            g["camera_image"] = im
        self._goal = g
        return self













