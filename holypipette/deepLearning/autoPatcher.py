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
class AutoPatcher:
    """
    Locate an .onnx file and return an onnxruntime.InferenceSession plus
    input / output name lists.
    """
    def load_model(self, onnx_path=None, providers=None):
        if providers is None:
            providers = ["CPUExecutionProvider"]      # add "CUDAExecutionProvider" if GPU build
        if onnx_path is None or not Path(onnx_path).exists():
            model_dir = Path(__file__).parent / "patchModel"
            try:
                onnx_path = next(model_dir.glob("*.onnx"))
            except StopIteration:
                raise FileNotFoundError(f"No .onnx model found in {model_dir}")
        session = ort.InferenceSession(str(onnx_path), providers=providers)
        input_names  = [i.name for i in session.get_inputs()]
        output_names = [o.name for o in session.get_outputs()]
        print(f"Loaded model {onnx_path} with inputs {input_names} → outputs {output_names}")
        return session, input_names, output_names


# -------------------------------------------------
# ModelTester: feeds video-plus-sensor data to the net
# -------------------------------------------------
class ModelTester:
    """
    Feeds a 16-step history into an ONNX Runtime session and
    returns the 6-D action for the current frame.
    """
    def __init__(self, session, input_names, output_names, input_path=None):
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
        pred = np.asarray(pred).reshape(-1)    # → (6,)
        gt   = np.asarray(gt).reshape(-1)      # already (6,)

        errvector = np.abs(pred - gt)          # (6,)
        return errvector


# -------------------------------------------------
# Main – run timing benchmark on sample demo set
# -------------------------------------------------
if __name__ == "__main__":
    patcher = AutoPatcher()
    model_path  = Path(__file__).parent / "patchModel" / "models" / "HEKHUNTERv0_050.onnx"
    data_path   = Path(__file__).parent / "patchModel" / "test_data" / "HEKHUNTER_inference_set2.hdf5"

    session, in_names, out_names = patcher.load_model(str(model_path))
    tester = ModelTester(session, in_names, out_names, str(data_path))

    lat_ms = []
    error = []
    for idx in range(len(tester.images)):
        t0 = time.perf_counter()
        out =  tester.run_inference(idx)
        lat_ms.append((time.perf_counter() - t0) * 1_000)
        if out is not None:
            # print(f"Predicted action: {out}")
            # print(f"Ground truth action: {tester.actions[idx]}")
            error.append(tester.calculate_error(out, tester.actions[idx]))
            # print(f"Error: {error[-1]}")


    print(f"Inference latency over {len(lat_ms)} frames: "
          f"mean = {statistics.mean(lat_ms):.2f} ms  |  sd = {statistics.stdev(lat_ms):.2f} ms")
    



    # -------------------------------------------------
    # Plot latency + pipette-axis error histograms
    # -------------------------------------------------
    # if error:
    #     import numpy as np, matplotlib.pyplot as plt

    #     err_arr  = np.stack(error)            # (N,6)
    #     pip_ax   = [3, 4, 5]                  # indexes for pix, piy, pz
    #     ax_names = ["pix", "piy", "pz"]
    #     bin_w    = 0.005                       # error-bin width
    #     lat_bin  = 1                          # latency-bin width in ms

    #     fig, axs = plt.subplots(1, 4, figsize=(14, 3))
    #     fig.suptitle("Latency and Pipette-Action Error Histograms")

    #     # ---- latency (leftmost subplot) ----
    #     axs[0].hist(lat_ms, bins=np.arange(0, max(lat_ms)+lat_bin, lat_bin))
    #     axs[0].set_title("Latency (ms)")
    #     axs[0].set_xlabel("ms")
    #     axs[0].set_ylabel("count")

    #     # ---- pipette actions ----
    #     for ax_i, act_i in enumerate(pip_ax, start=1):
    #         data = err_arr[:, act_i]
    #         bins = np.arange(0, data.max() + bin_w, bin_w)
    #         axs[ax_i].hist(data, bins=bins)
    #         axs[ax_i].set_title(f"{ax_names[ax_i-1]} error")
    #         axs[ax_i].set_xlabel("|pred – gt|")
    #         axs[ax_i].set_ylabel("count")

    #         # mean ± sd annotation
    #         mu, sd = data.mean(), data.std()
    #         axs[ax_i].text(0.95, 0.95,
    #                        f"μ={mu:.4f}\nσ={sd:.4f}",
    #                        transform=axs[ax_i].transAxes,
    #                        ha="right", va="top",
    #                        fontsize=8,
    #                        bbox=dict(boxstyle="round", fc="white", ec="0.8"))

    #     plt.tight_layout()
    #     plt.show()


    # ------------------------------------------------------------------
    # Absolute pipette-axis errors + latency histogram
    # ------------------------------------------------------------------
    if error:
        import numpy as np, matplotlib.pyplot as plt, math

        err_arr  = np.stack(error)                 # (N,6)
        pip_idx  = [3, 4, 5]                       # pix, piy, piz
        axis_lbl = ["pix", "piy", "piz"]

        # choose bin width automatically (10 bins between 0 and max err)
        abs_max = err_arr[:, pip_idx].max()
        bin_w   = abs_max / 20 if abs_max > 0 else 0.01
        bins    = np.arange(0, abs_max + bin_w, bin_w)

        fig, axs = plt.subplots(1, 4, figsize=(14, 3))
        fig.suptitle("Latency and Absolute Pipette-Error Histograms")

        # ---- latency ----
        axs[0].hist(lat_ms, bins=np.arange(0, max(lat_ms)+5, 5))
        axs[0].set_title("Latency (ms)")
        axs[0].set_xlabel("ms")
        axs[0].set_ylabel("count")

        # ---- absolute errors ----
        for j, name in enumerate(axis_lbl, start=1):
            data = err_arr[:, pip_idx[j-1]]
            axs[j].hist(data, bins=bins, edgecolor="k")
            axs[j].set_title(f"{name} |pred – gt|")
            axs[j].set_xlabel("absolute error")
            axs[j].set_ylabel("count")

            μ, σ = data.mean(), data.std()
            axs[j].text(0.96, 0.95, f"μ={μ:.4f}\nσ={σ:.4f}",
                        transform=axs[j].transAxes, ha="right", va="top",
                        fontsize=8,
                        bbox=dict(boxstyle="round", fc="white", ec="0.8"))

        plt.tight_layout()
        plt.show()


