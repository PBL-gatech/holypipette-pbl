#!/usr/bin/env python
"""
autoPatcher_ort.py – ONNX Runtime version of the AutoPatcher demo.

Changes vs. the OpenCV-DNN original
-----------------------------------
* AutoPatcher.load_model → returns onnxruntime.InferenceSession
* ModelTester rewritten to call session.run(dict_inputs)
* Low-dim inputs (pipette, stage, resistance) feed only the latest frame
  while camera_image keeps the 16-frame stack.
* Actions are sliced to actions[:, -1, :] so the caller receives shape (1, 6).
"""
import time, statistics, collections
from pathlib import Path

import cv2                         # still used for resize
import h5py
import numpy as np
import onnxruntime as ort          # ★ new backend ★
import matplotlib.pyplot as plt
import numpy as np


# -------------------------------------------------
# AutoPatcher: model loader
# -------------------------------------------------
import onnxruntime as ort
from pathlib import Path
import numpy as np

class AutoPatcher:
    """
    Locate an .onnx file and return an onnxruntime.InferenceSession plus
    input/output name lists. Also provides simple inference helpers for live deque usage,
    handling LSTM states (h0, c0) if present in the model.
    """
    def __init__(self, onnx_path=None, providers=None, num_layers=2, hidden_size=400):
        self.session = None
        self.input_names = None
        self.output_names = None
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        if onnx_path is not None:
            self.load_model(onnx_path, providers)

    def load_model(self, onnx_path=None, providers=None):
        if providers is None:
            providers = ["CPUExecutionProvider"]
        if onnx_path is None or not Path(onnx_path).exists():
            model_dir = Path(__file__).parent / "patchModel"
            try:
                onnx_path = next(model_dir.glob("*.onnx"))
            except StopIteration:
                raise FileNotFoundError(f"No .onnx model found in {model_dir}")
        self.session = ort.InferenceSession(str(onnx_path), providers=providers)
        self.input_names  = [i.name for i in self.session.get_inputs()]
        self.output_names = [o.name for o in self.session.get_outputs()]
        print(f"Loaded model {onnx_path} with inputs {self.input_names} → outputs {self.output_names}")
        return self.session, self.input_names, self.output_names

    def inference(self, img_q, pip_q, stage_q, res_q, h0=None, c0=None):
        """
        Accepts four deques of length 16 (images, pipettes, stages, resistance), and
        optional LSTM state (h0, c0). Returns the (1,6) action vector and updated states.
        """
        # Prepare input arrays
        inputs = {
            "camera_image":      np.stack(img_q, 0)[None],  # (1,16,3,H,W)
            "pipette_positions": np.stack(pip_q, 0)[None],  # (1,16,3)
            "stage_positions":   np.stack(stage_q, 0)[None],# (1,16,3)
            "resistance":        np.stack(res_q, 0).reshape(1, -1), # (1,16)
        }
        # If model needs states, initialize if not provided
        if "h0" in self.input_names and h0 is None:
            h0 = np.zeros((self.num_layers, 1, self.hidden_size), np.float32)
        if "c0" in self.input_names and c0 is None:
            c0 = np.zeros((self.num_layers, 1, self.hidden_size), np.float32)
        if "h0" in self.input_names:
            inputs["h0"] = h0
        if "c0" in self.input_names:
            inputs["c0"] = c0
        # Filter unused
        filtered_inputs = {k: v for k, v in inputs.items() if k in self.input_names}
        outputs = self.session.run(None, filtered_inputs)
        output_dict = dict(zip(self.output_names, outputs))
        # Retrieve new states if available
        new_h0 = output_dict["h1"] if "h1" in output_dict else h0
        new_c0 = output_dict["c1"] if "c1" in output_dict else c0
        action = output_dict["actions"][:, -1, :]  # (1,6)
        return action, new_h0, new_c0


class ModelTester:
    """
    Feeds a 16-step history into an ONNX Runtime session and
    returns the 6-D action for the current frame.
    """
    def __init__(self, session, input_names, output_names, input_path=None, prefill_init=False):
        import h5py, numpy as np, collections, cv2
        self.session      = session
        self.input_names  = {i.name for i in session.get_inputs()}
        self.output_names = [o.name for o in session.get_outputs()]

        # -------- load HDF5 demo (optional) --------
        if input_path:
            h5 = h5py.File(str(input_path), "r")
            demo_id  = sorted(h5["data"].keys())[0]
            obs_root = f"data/{demo_id}/obs"
            act_root = f"data/{demo_id}"
            self.images            = h5[f"{obs_root}/camera_image"][:]
            self.resistance        = h5[f"{obs_root}/resistance"][:]
            self.pipette_positions = h5[f"{obs_root}/pipette_positions"][:]
            self.stage_positions   = h5[f"{obs_root}/stage_positions"][:]
            self.actions           = h5[f"{act_root}/actions"][:]
            print(f"Loaded demo '{demo_id}': images {self.images.shape}, "
                  f"resistance {self.resistance.shape}, pipette {self.pipette_positions.shape}, "
                  f"stage {self.stage_positions.shape}")
            print(f"Actions: {self.actions.shape} → {self.actions.dtype}")

        # -------- runtime buffers --------
        self.seq_len = 16
        self.img_q   = collections.deque(maxlen=self.seq_len)
        self.pip_q   = collections.deque(maxlen=self.seq_len)
        self.stage_q = collections.deque(maxlen=self.seq_len)
        self.res_q   = collections.deque(maxlen=self.seq_len)

        self.h0 = self.c0 = None          # LSTM state
        self.num_layers, self.hidden_size = 2, 400

        # resize helper
        H, W = self.images.shape[1:3]
        self.resize = lambda im: cv2.resize(im, (W, H)).astype(np.float32).transpose(2,0,1)/255.0

        # --- prefill logic ---
        self.prefill_init = prefill_init
        self._prefilled = False

    # --------------------------------------------------------------
    def _stack_3d(self, q):              # → (1,16,features)
        return np.stack(q, 0)[None]

    def _stack_2d(self, q):              # → (1,16)
        return np.stack(q, 0).reshape(1, -1)

    def run_inference(self, idx):
        img, res, pip, stage = (self.images[idx],
                                self.resistance[idx],
                                self.pipette_positions[idx],
                                self.stage_positions[idx])

        # push one step into deques
        self.img_q.append(self.resize(img))
        self.pip_q.append(pip.astype(np.float32))
        self.stage_q.append(stage.astype(np.float32))
        self.res_q.append(np.array(res, np.float32))     # scalar → ()

        # --- prefill logic ---
        if self.prefill_init and not self._prefilled and len(self.img_q) == 1:
            for _ in range(self.seq_len - 1):
                self.img_q.append(self.img_q[0].copy())
                self.pip_q.append(self.pip_q[0].copy())
                self.stage_q.append(self.stage_q[0].copy())
                self.res_q.append(np.array(self.res_q[0]))
            self._prefilled = True

        if len(self.img_q) < self.seq_len:
            return None          # warm-up

        # build input dict with correct ranks
        inputs = {
            "camera_image"     : self._stack_3d(self.img_q),      # (1,16,3,85,85)
            "pipette_positions": self._stack_3d(self.pip_q),      # (1,16,3)
            "stage_positions"  : self._stack_3d(self.stage_q),    # (1,16,3)
            "resistance"       : self._stack_2d(self.res_q),      # (1,16)
        }

        # LSTM hidden state
        if self.h0 is None:
            self.h0 = np.zeros((self.num_layers,1,self.hidden_size), np.float32)
            self.c0 = np.zeros_like(self.h0)
        inputs["h0"], inputs["c0"] = self.h0, self.c0

        # drop unused inputs (robust to future exports)
        inputs = {k:v for k,v in inputs.items() if k in self.input_names}

        # forward
        outs = self.session.run(None, inputs)
        out = dict(zip(self.output_names, outs))
        if "h1" in out: self.h0, self.c0 = out["h1"], out["c1"]

        # return 6-D command for the current frame
        return out["actions"][:, -1, :]    # (1,6)
    
    def calculate_error(self, pred, gt):
        """
        Absolute error on each of the 6 action axes.

        Parameters
        ----------
        pred : (1,6) or (6,) array
        gt   : (6,)  array
        """
        import numpy as np
        pred = np.asarray(pred).reshape(-1)    # → (6,)
        gt   = np.asarray(gt).reshape(-1)      # already (6,)

        errvector = (pred - gt)          # (6,)
        return errvector
    


if __name__ == "__main__":
    patcher = AutoPatcher()
    model_path  = Path(__file__).parent / "patchModel" / "models" / "HEKHUNTERv0_100.onnx"
    data_path   = Path(__file__).parent / "patchModel" / "test_data" / "HEKHUNTER_inference_set.hdf5"

    session, in_names, out_names = patcher.load_model(str(model_path))
    
    # Only this line is changed to enable prefill:
    tester = ModelTester(session, in_names, out_names, str(data_path), prefill_init=True)

    lat_ms = []
    error = []
    for idx in range(len(tester.images)):
        t0 = time.perf_counter()
        out =  tester.run_inference(idx)
        lat_ms.append((time.perf_counter() - t0) * 1_000)
        if out is not None:
            error.append(tester.calculate_error(out, tester.actions[idx]))

    # print(f"Inference latency over {len(lat_ms)} frames: "
    #       f"mean = {statistics.mean(lat_ms):.2f} ms  |  sd = {statistics.stdev(lat_ms):.2f} ms")
    
    # do stand alone latency plot by itself
    # if len(lat_ms) > 0:
    #     import matplotlib.pyplot as plt
    #     plt.figure(figsize=(10, 4))
    #     plt.plot(lat_ms)
    #     plt.title("Latency over time")
    #     plt.xlabel("Frame Index")
    #     plt.ylabel("Latency (ms)")
    #     plt.show()

    # also do stand alone latency histogram
    # if len(lat_ms) > 0:
    #     import matplotlib.pyplot as plt
    #     plt.figure(figsize=(10, 4))
    #     plt.hist(lat_ms, bins=np.arange(0, max(lat_ms)+5, 5))
    #     plt.title("Latency Histogram")
    #     plt.xlabel("Latency (ms)")
    #     plt.ylabel("Count")
    #     plt.show()


    # Plot error time course
    # if error:
    #     err_arr_tc = np.stack(error)  # (N,6)
    #     cumulative_err = np.cumsum(err_arr_tc, axis=0)
    #     fig2, ax2 = plt.subplots(figsize=(10, 4))
    #     labels = ["pip_x", "pip_y", "pip_z"]
    #     colors = ["black", "darkblue", "darkred"]
    #     # Plot cumulative error for pipette axes (indices 3,4,5) with specified colors
    #     for idx, lbl, color in zip([3, 4, 5], labels, colors):
    #         ax2.plot(cumulative_err[:, idx], label=lbl, color=color)
    #     ax2.set_title("Cumulative Error Time Course for Pipette Axes")
    #     ax2.set_xlabel("Frame Index")
    #     ax2.set_ylabel("Cumulative Absolute Error")
    #     ax2.legend()
    #     plt.tight_layout()
    #     plt.show()

        # 3D plot of cumulative pipette error trajectory
        # if error:
        #     err_arr_tc = np.stack(error)  # (n,6)
        #     cumulative_err = np.cumsum(err_arr_tc, axis=0)
        #     pip_cum_err = cumulative_err[:, 3:6]  # (n,3)
        #     from mpl_toolkits.mplot3d import Axes3D
        #     fig3 = plt.figure(figsize=(10, 8))
        #     ax3 = fig3.add_subplot(111, projection='3d')
        #     time_steps = np.arange(pip_cum_err.shape[0])
        #     cmap = plt.cm.viridis
        #     colors = cmap(time_steps / pip_cum_err.shape[0])
        #     ax3.scatter(pip_cum_err[:, 0], pip_cum_err[:, 1], pip_cum_err[:, 2], c=colors, marker='o', s=15)
        #     ax3.set_title('3D Cumulative Pipette Error Trajectory')
        #     ax3.set_xlabel('Cumulative Error X')
        #     ax3.set_ylabel('Cumulative Error Y')
        #     ax3.set_zlabel('Cumulative Error Z')
        #     mappable = plt.cm.ScalarMappable(cmap=cmap)
        #     mappable.set_array([])
        #     cbar = plt.colorbar(mappable, ax=ax3, pad=0.1, shrink=0.6)
        #     cbar.set_label('Time steps')
        #     plt.show()

# 3D plot showing GT and predicted pipette positions, just like above plot, need two different colorimetric time axes
        pred_pipette_positions = tester.pipette_positions
        gt_pipette_positions = tester.actions[:, 3:6]
        fig4 = plt.figure(figsize=(10, 8))
        ax4 = fig4.add_subplot(111, projection='3d')
        time_steps = np.arange(pred_pipette_positions.shape[0])
        # Use two perceptually distinct colormaps with minimal overlap
        cmap1 = plt.cm.Blues
        cmap2 = plt.cm.Oranges
        colors_pred = cmap1(time_steps / pred_pipette_positions.shape[0])
        colors_gt = cmap2(time_steps / gt_pipette_positions.shape[0])
        ax4.scatter(pred_pipette_positions[:, 0], pred_pipette_positions[:, 1], pred_pipette_positions[:, 2], c=colors_pred, marker='o', s=15, label='Predicted Pipette Positions')
        ax4.scatter(gt_pipette_positions[:, 0], gt_pipette_positions[:, 1], gt_pipette_positions[:, 2], c=colors_gt, marker='o', s=15, label='GT Pipette Positions')    
        ax4.set_title('3D Pipette Position Trajectory')
        ax4.set_xlabel('Pipette Position X')
        ax4.set_ylabel('Pipette Position Y')
        ax4.set_zlabel('Pipette Position Z')
        mappable_pred = plt.cm.ScalarMappable(cmap=cmap1)
        mappable_pred.set_array([])
        cbar_pred = plt.colorbar(mappable_pred, ax=ax4, pad=0.1, shrink=0.6)
        cbar_pred.set_label('Time steps (Predicted)')
        mappable_gt = plt.cm.ScalarMappable(cmap=cmap2)
        mappable_gt.set_array([])
        cbar_gt = plt.colorbar(mappable_gt, ax=ax4, pad=0.1, shrink=0.6)
        cbar_gt.set_label('Time steps (GT)')
        ax4.legend()
        plt.show()





        # Sum the per-frame error vectors and display cumulative error for pipette axes as bar charts.
        # if error:
            # error_array = np.stack(error)  # shape (n,6)
            # cumulative_error = np.sum(error_array, axis=0)
            # # Extract cumulative error for pipette axes (indices 3,4,5: x, y, z)
            # pip_indices = [3, 4, 5]
            # pip_labels  = ["pip_x", "pip_y", "pip_z"]
            # pip_errors  = cumulative_error[pip_indices]
            # colors      = ["black", "darkblue", "darkred"]

            # import matplotlib.pyplot as plt
            # plt.figure(figsize=(6,4))
            # plt.bar(pip_labels, pip_errors, color=colors)
            # plt.title("Cumulative Absolute Error for Pipette Axes")
            # plt.xlabel("Pipette Axes")
            # plt.ylabel("Cumulative Absolute Error")
            # plt.show()
            # Print raw predicted actions and ground truth actions
        print("Predicted actions (first 10):")
        for i, e in enumerate(error[:10]):
            print(f"Frame {i}: Predicted {tester.actions[i]}, GT {tester.actions[i] - e}")

        print("\nGround truth actions (first 10):")
        for i in range(10):
            print(f"Frame {i}: {tester.actions[i]}")