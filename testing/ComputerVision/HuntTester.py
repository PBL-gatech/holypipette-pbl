from __future__ import annotations
import time, statistics, collections
from pathlib import Path

import cv2
import h5py
import numpy as np
import onnxruntime as ort
import matplotlib.pyplot as plt
import numpy as np


# -------------------------------------------------
# HuntTester: model loader
# -------------------------------------------------
import onnxruntime as ort
from pathlib import Path
import numpy as np


class HuntTester:
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
        self._h0 = None  # internal cache for recurrent state (optional)
        self._c0 = None
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
        # Detect whether model expects a sequence (legacy) or single step (new)
        cam_input = next((i for i in self.session.get_inputs() if i.name == "camera_image"), None)
        if cam_input is not None and cam_input.shape is not None:
            dims_known = [d for d in cam_input.shape if d is not None]
            self.seq_mode = len(dims_known) >= 5   # (B,T,C,H,W) → legacy
        else:
            self.seq_mode = True  # safe fallback
        print(f"Sequence mode: {self.seq_mode}")
        print(f"Loaded model {onnx_path} with inputs {self.input_names} → outputs {self.output_names}")
        return self.session, self.input_names, self.output_names

    def inference(self, img_q, pip_q, stage_q, res_q, h0=None, c0=None):
        """Run a single forward pass.
        Works with legacy sequence models and new single-step exports.
        """
        # Build inputs depending on model type
        if getattr(self, 'seq_mode', True):
            inputs = {
                "camera_image":      np.stack(img_q, 0)[None],
                "pipette_positions": np.stack(pip_q, 0)[None],
                "stage_positions":   np.stack(stage_q, 0)[None],
                "resistance":        np.stack(res_q, 0).reshape(1, -1),
            }
        else:
            # Use only the most recent step
            last_img   = img_q[-1]
            last_pip   = pip_q[-1]
            last_stage = stage_q[-1]
            last_res   = res_q[-1]
            inputs = {
                "camera_image":      last_img[None],
                "pipette_positions": np.asarray(last_pip,   np.float32).reshape(1, 3),
                "stage_positions":   np.asarray(last_stage, np.float32).reshape(1, 3),
                "resistance"            : np.asarray(last_res, np.float32).reshape(1),
            }

        # Initialize hidden states if required by model
        # [TEMP DEBUG]
        if (("h0" in self.input_names and h0 is None) or ("c0" in self.input_names and c0 is None)):
            print("[DEBUG] RNN state was None → zero-initializing for this call (expect repeated early-step actions if you do this every frame).")
        # [/TEMP DEBUG]
        if "h0" in self.input_names and h0 is None:
            h0 = np.zeros((self.num_layers, 1, self.hidden_size), np.float32)
        if "c0" in self.input_names and c0 is None:
            c0 = np.zeros_like(h0)

        if "h0" in self.input_names:
            inputs["h0"] = h0
        if "c0" in self.input_names:
            inputs["c0"] = c0


        # Filter to provided inputs only
        filtered_inputs = {k: v for k, v in inputs.items() if k in self.input_names}
        outputs = self.session.run(None, filtered_inputs)
        output_dict = dict(zip(self.output_names, outputs))

        # Updated states (handle GRU/LSTM)
        new_h0 = output_dict.get("h1", h0)
        new_c0 = output_dict.get("c1", c0)
        # [TEMP DEBUG]
        if (new_h0 is not None) and (h0 is not None):
            try:
                print("[DEBUG] ||Δh||:", float(np.linalg.norm(new_h0 - h0)))
            except Exception:
                pass
        # [/TEMP DEBUG]

        # Actions may be (1,6) or (1,T,6)
        actions = output_dict["actions"]
        if actions.ndim == 3:
            action = actions[:, -1, :]
        elif actions.ndim == 2:
            action = actions
        else:
            raise RuntimeError(f"Unexpected actions shape: {actions.shape}")
        return action, new_h0, new_c0



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

        # Detect sequence vs single-step model
        cam_input = next((i for i in self.session.get_inputs() if i.name == "camera_image"), None)
        if cam_input is not None and cam_input.shape is not None:
            dims_known = [d for d in cam_input.shape if d is not None]
            self.seq_mode = len(dims_known) >= 5
        else:
            self.seq_mode = True
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

    def _center_crop_and_resize(self, im, crop_h=78, crop_w=78):
        """
        Center-crop the HxWx3 image to (crop_h, crop_w), then resize back to original (H, W).
        Keeps ONNX input shape identical to training-time encoder output dims (post-crop).
        """
        import numpy as np, cv2
        H0, W0 = im.shape[:2]
        y0 = max(0, (H0 - crop_h) // 2);  x0 = max(0, (W0 - crop_w) // 2)
        y1, x1 = y0 + crop_h, x0 + crop_w
        im_c = im[y0:y1, x0:x1]
        im_r = cv2.resize(im_c, (W0, H0))
        # to CHW float32 in [0,1]
        return im_r.astype(np.float32).transpose(2, 0, 1) / 255.0

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
            self.img_q.append(self._center_crop_and_resize(img, 78, 78))
            self.pip_q.append(pip.astype(np.float32))
            self.stage_q.append(stage.astype(np.float32))
            self.res_q.append(np.array(res, np.float32))     # scalar → ()

            # --- Diagnostics: input drift between last two frames ---
            if len(self.img_q) >= 2:
                try:
                    d_img = float(np.linalg.norm(self.img_q[-1] - self.img_q[-2]))
                    d_pip = float(np.linalg.norm(self.pip_q[-1] - self.pip_q[-2]))
                    d_stage = float(np.linalg.norm(self.stage_q[-1] - self.stage_q[-2]))
                    d_res = float(abs(self.res_q[-1] - self.res_q[-2]))
                    print(f"[DIAG] Δimg={d_img:.4f} | Δpip={d_pip:.4f} | Δstage={d_stage:.4f} | Δres={d_res:.4f}")
                except Exception:
                    pass

            # Build inputs
            if getattr(self, 'seq_mode', True):
                # Require warm-up to full sequence
                if len(self.img_q) < self.seq_len:
                    return None
                inputs = {
                    "camera_image"     : self._stack_3d(self.img_q),    # (1,16,3,H,W)
                    "pipette_positions": self._stack_3d(self.pip_q),    # (1,16,3)
                    "stage_positions"  : self._stack_3d(self.stage_q),  # (1,16,3)
                    "resistance"       : self._stack_2d(self.res_q),    # (1,16)
                }
            else:
                # Single step: use last element only
                last_img   = self.img_q[-1]
                last_pip   = self.pip_q[-1]
                last_stage = self.stage_q[-1]
                last_res   = self.res_q[-1]
                inputs = {
                    "camera_image"     : last_img[None],                       # (1,3,H,W)
                    "pipette_positions": np.asarray(last_pip, np.float32).reshape(1, 3),
                    "stage_positions"  : np.asarray(last_stage, np.float32).reshape(1, 3),
                    "resistance"       : np.asarray(last_res, np.float32).reshape(1),
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

            # [DEBUG] check if RNN state is updating across calls  (A)
            if "h1" in out:
                try:
                    print("[DEBUG] ||Δh|| =", float(np.linalg.norm(out["h1"] - self.h0)))
                except Exception:
                    pass

            # update state if provided
            if "h1" in out:
                self.h0 = out["h1"]
                if "c1" in out: self.c0 = out["c1"]

            # return 6-D command for the current frame
            actions = out["actions"]
            # [DEBUG] print first 10 action rows to inspect repetition  (B)
            try:
                print("[DEBUG] actions sample:", actions.reshape(-1, actions.shape[-1])[:10])
            except Exception:
                pass
            if actions.ndim == 3:
                actions = actions[:, -1, :]
            return actions
 
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



"""Model analysis utilities for HEKHUNTER demo.

This module wraps the ad‑hoc logic that used to live in the
``if __name__ == "__main__":`` block inside a reusable class called
:class:`ModelAnalyzer`.  Its ``run`` method reproduces all previous
behaviour **and** writes a 60 fps animated 3‑D trajectory GIF.

Dependencies (beyond the standard ones already present in your script):
    • numpy
    • matplotlib (with Pillow or FFmpeg installed for animation saving)
    • HuntTester, ModelTester from your existing project tree

Typical usage
-------------
>>> from pathlib import Path
>>> analyzer = ModelAnalyzer(
...     model_path=Path(__file__).parent / "patchModel/models/HEKHUNTERv0_050.onnx",
...     data_path = Path(__file__).parent / "patchModel/test_data/HEKHUNTER_inference_set.hdf5",
... )
>>> analyzer.run()  # produces plots + pipette_trajectory.gif
"""

from pathlib import Path
import time
import statistics
from typing import Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, colors
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (import needed for 3‑D projection)



__all__ = ["ModelAnalyzer"]


class ModelAnalyzer:
    """Analyse ONNX model latency, accuracy and motion trajectories."""

    # ────────────────────────────────
    # Construction helpers
    # ────────────────────────────────

    def __init__(
        self,
        model_path: Path | str,
        data_path: Path | str,
        *,
        save_dir: Path | str | None = None,
        animation_fname: str = "pipette_trajectory.gif",
        animation_fps: int = 60,
    ) -> None:
        self.model_path = Path(model_path)
        self.data_path = Path(data_path)
        self.save_dir = Path(save_dir) if save_dir is not None else self.model_path.parent
        self.animation_fname = animation_fname
        self.animation_fps = animation_fps

        # Lazily‑filled during run()
        self.lat_ms: list[float] = []
        self.error_frames: list[np.ndarray] = []
        self.predicted_pip_positions: Optional[np.ndarray] = None
        self.observed_pip_positions: Optional[np.ndarray] = None

        # ── Model + tester ───────────────────────────────────────────────
        patcher = HuntTester()
        self.session, self.in_names, self.out_names = patcher.load_model(str(self.model_path))
        self.tester = ModelTester(
            self.session,
            self.in_names,
            self.out_names,
            str(self.data_path)
        )

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    def run(self) -> None:
        """Execute the full analysis pipeline and generate artefacts."""
        self._compute_latency_and_error()
        self._plot_static_trajectory()
        self._animate_trajectory(save_gif=True)

    # ------------------------------------------------------------------
    # Core analysis
    # ------------------------------------------------------------------

    def _compute_latency_and_error(self) -> None:
            """Profiles inference latency and collects prediction error."""
            print("[INFO] Running inference over frames…")
            # ensure fresh store (single pass)
            self.stored_actions = []
            for idx in range(len(self.tester.images)):
                t0 = time.perf_counter()
                out = self.tester.run_inference(idx)
                self.lat_ms.append((time.perf_counter() - t0) * 1_000)
                if out is not None:
                    # keep error for plots
                    self.error_frames.append(self.tester.calculate_error(out, self.tester.actions[idx]))
                    # store actions for later integration (avoid second model pass)
                    self.stored_actions.append(out)
                        # --- Diagnostics: cosine similarity between successive actions ---
            if hasattr(self, "stored_actions") and len(self.stored_actions) > 2:
                A = np.asarray(self.stored_actions).reshape(len(self.stored_actions), -1)  # (n,6)
                num = np.sum(A[1:] * A[:-1], axis=1)
                den = (np.linalg.norm(A[1:], axis=1) * np.linalg.norm(A[:-1], axis=1) + 1e-12)
                cos_sim = num / den
                print(f"[DIAG] action cos-sim: mean={cos_sim.mean():.6f} | p95={np.percentile(cos_sim,95):.6f} | max={cos_sim.max():.6f}")


            if self.lat_ms:
                mean_ms = statistics.mean(self.lat_ms)
                sd_ms = statistics.stdev(self.lat_ms) if len(self.lat_ms) > 1 else 0.0
                print(f"[RESULT] Inference latency — mean: {mean_ms:.2f} ms | sd: {sd_ms:.2f} ms")

            # Compute pipette positions once, reusing stored actions
            self._integrate_pipette_predictions()


    # ------------------------------------------------------------------
    # Trajectory helpers
    # ------------------------------------------------------------------

    def _integrate_pipette_predictions(self) -> None:
            """Integrate predicted pipette deltas -> absolute positions (single-pass)."""
            # 1) Build predicted delta array from stored actions (shape ~ (n, 1, 6) or (n,6))
            if not hasattr(self, 'stored_actions') or len(self.stored_actions) == 0:
                raise RuntimeError("No stored actions found — call _compute_latency_and_error() first.")
            pred_actions = np.asarray(self.stored_actions)
            if pred_actions.ndim == 3:   # (n,1,6)
                pred_actions = pred_actions[:, 0, :]
            # extract Δxyz
            pred_deltas_arr = pred_actions[:, 3:6]

            # 2) Trajectory alignment: with no prefill, first valid action aligns at seq_len-1
            start = self.tester.seq_len - 1
            end = min(start + pred_deltas_arr.shape[0], len(self.tester.pipette_positions))
            init_pos = self.tester.pipette_positions[start]
            obs_positions = self.tester.pipette_positions[start:end]

            # Equalize lengths
            n = min(pred_deltas_arr.shape[0], obs_positions.shape[0])
            pred_deltas_arr = pred_deltas_arr[:n]
            obs_positions   = obs_positions[:n]

            # 3) Integrate deltas -> absolute predicted positions
            predicted_positions = [init_pos]
            for delta in pred_deltas_arr:
                predicted_positions.append(predicted_positions[-1] + delta)

            self.predicted_pip_positions = np.stack(predicted_positions[:-1])  # (n,3)
            self.observed_pip_positions  = obs_positions

            # 4) re-zero around first observed sample for visual clarity
            anchor = self.observed_pip_positions[0]
            self.predicted_pip_positions -= anchor
            self.observed_pip_positions  -= anchor

            # debug
            print(f"[DEBUG] Predicted positions: {self.predicted_pip_positions} | Observed positions: {self.observed_pip_positions}")
   
    # Static 3‑D trajectory plot
    # ------------------------------------------------------------------

    def _plot_static_trajectory(self) -> None:
        if self.predicted_pip_positions is None or self.observed_pip_positions is None:
            raise RuntimeError("Prediction data not initialised — call run() first.")

        # Build truncated colormaps
        def _trunc_cmap(base_cmap, start=0.5, stop=1.0, n=256):
            new_colors = base_cmap(np.linspace(start, stop, n))
            return colors.LinearSegmentedColormap.from_list(f"{base_cmap.name}_trunc", new_colors)

        cmap_pred = _trunc_cmap(plt.cm.Blues, 0.5, 1.0)
        cmap_obs = _trunc_cmap(plt.cm.Oranges, 0.5, 1.0)

        n_steps = self.predicted_pip_positions.shape[0]
        norm = plt.Normalize(vmin=0, vmax=n_steps - 1)

        colors_pred = cmap_pred(norm(np.arange(n_steps)))
        colors_obs = cmap_obs(norm(np.arange(n_steps)))

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_zlim(-20, 5)

        # Z is negative in data; flip to positive‑up here for clarity
        pred_z = -self.predicted_pip_positions[:, 2]
        obs_z = -self.observed_pip_positions[:, 2]

        ax.scatter(
            self.predicted_pip_positions[:, 0],
            self.predicted_pip_positions[:, 1],
            pred_z,
            c=colors_pred,
            marker="o",
            s=20,
            label="Predicted (Integrated Actions)",
        )
        ax.scatter(
            self.observed_pip_positions[:, 0],
            self.observed_pip_positions[:, 1],
            obs_z,
            c=colors_obs,
            marker="^",
            s=20,
            label="Observed (Ground Truth)",
        )

        ax.set_title("3‑D Pipette Trajectory: Predicted vs Observed")
        ax.set_xlabel("Pipette X")
        ax.set_ylabel("Pipette Y")
        ax.set_zlabel("Pipette –Z")
        ax.legend(loc="best")

        # Colour‑bars
        mappable_pred = plt.cm.ScalarMappable(norm=norm, cmap=cmap_pred)
        mappable_pred.set_array([])
        cbar_pred = plt.colorbar(mappable_pred, ax=ax, pad=0.1, shrink=0.6)
        cbar_pred.set_label("Time steps (Predicted)")
        cbar_pred.ax.invert_yaxis()

        mappable_obs = plt.cm.ScalarMappable(norm=norm, cmap=cmap_obs)
        mappable_obs.set_array([])
        cbar_obs = plt.colorbar(mappable_obs, ax=ax, pad=0.1, shrink=0.6)
        cbar_obs.set_label("Time steps (Observed)")
        cbar_obs.ax.invert_yaxis()

        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # Animated trajectory (saves GIF)
    # ------------------------------------------------------------------



    def _animate_trajectory(self, *, save_gif: bool = True) -> None:
        if self.predicted_pip_positions is None or self.observed_pip_positions is None:
            raise RuntimeError("Prediction data not initialised — call run() first.")

        # ── pre-compute step-wise vector-magnitude error ─────────────────
        error_mag = np.linalg.norm(
            self.predicted_pip_positions - self.observed_pip_positions, axis=1
        )  # shape (n,)

        from matplotlib.lines import Line2D  # tiny local import keeps edit minimal

        n_steps = self.predicted_pip_positions.shape[0]
        norm = plt.Normalize(vmin=0, vmax=n_steps - 1)

        # Re-use truncated colormaps from static plot
        def _trunc_cmap(base_cmap, start=0.5, stop=1.0, n=256):
            new_colors = base_cmap(np.linspace(start, stop, n))
            return colors.LinearSegmentedColormap.from_list(f"{base_cmap.name}_trunc", new_colors)

        cmap_pred = _trunc_cmap(plt.cm.Blues, 0.5, 1.0)
        cmap_obs = _trunc_cmap(plt.cm.Oranges, 0.5, 1.0)

        fig_anim = plt.figure(figsize=(10, 8))
        ax_anim = fig_anim.add_subplot(111, projection="3d")
        ax_anim.set(
            xlim=(-5, 5),
            ylim=(-5, 5),
            zlim=(-20, 5),
            title="Animated 3-D Pipette Trajectory",
            xlabel="Pipette X",
            ylabel="Pipette Y",
            zlabel="Pipette –Z",
        )
        ax_anim.grid(False)

        # Artists
        sc_pred = ax_anim.scatter(
            [], [], [], c=[], cmap=cmap_pred, vmin=0, vmax=n_steps - 1, marker="o", s=20, label="Predicted"
        )
        sc_obs = ax_anim.scatter(
            [], [], [], c=[], cmap=cmap_obs, vmin=0, vmax=n_steps - 1, marker="^", s=20, label="Observed"
        )
        err_handle = Line2D([], [], linestyle="none", marker="", color="red")  # legend entry

        # Colour-bars
        for cm, pad, lbl in ((cmap_pred, 0.10, "Time steps (Predicted)"),
                             (cmap_obs, 0.03, "Time steps (Observed)")):
            m = plt.cm.ScalarMappable(norm=norm, cmap=cm); m.set_array([])
            cb = plt.colorbar(m, ax=ax_anim, pad=pad, shrink=0.6)
            cb.set_label(lbl); cb.ax.invert_yaxis()

        # Legend text handle will be captured in _init
        error_text_handle = None

        # ───── animation init/update ──────────────────────────────────
        def _init():
            nonlocal error_text_handle
            for sc in (sc_pred, sc_obs):
                sc._offsets3d = ([], [], [])
                sc.set_array(np.array([]))

            legend = ax_anim.legend(
                [sc_pred, sc_obs, err_handle],
                ["Predicted", "Observed", ""],  # placeholder for error
                loc="best",
                frameon=True,
            )
            error_text_handle = legend.get_texts()[-1]
            error_text_handle.set_color("red")
            error_text_handle.set_text(f"Error: {error_mag[0]:.3f}")
            return sc_pred, sc_obs, error_text_handle

        def _update(frame: int):
            # Predicted
            x_p, y_p, z_p = self.predicted_pip_positions[: frame + 1].T
            sc_pred._offsets3d = (x_p, y_p, -z_p)
            sc_pred.set_array(norm(np.arange(frame + 1)))

            # Observed
            x_o, y_o, z_o = self.observed_pip_positions[: frame + 1].T
            sc_obs._offsets3d = (x_o, y_o, -z_o)
            sc_obs.set_array(norm(np.arange(frame + 1)))

            # Update legend entry
            error_text_handle.set_text(f"Error: {error_mag[frame]:.3f}")
            return sc_pred, sc_obs, error_text_handle

        interval_ms = 1000 / self.animation_fps
        anim = animation.FuncAnimation(
            fig_anim,
            _update,
            init_func=_init,
            frames=n_steps,
            interval=interval_ms,
            blit=False,
        )

        if save_gif:
            out_path = self.save_dir / self.animation_fname
            try:
                anim.save(out_path, writer=animation.PillowWriter(fps=self.animation_fps))
                print(f"[INFO] Animation saved → {out_path.resolve()}")
            except Exception as exc:
                print("[WARNING] GIF not saved —", exc)

        plt.show()



if __name__ == "__main__":
    model_path = r"C:\\Users\\sa-forest\\Documents\\GitHub\\holypipette-pbl\\holypipette\\deepLearning\\patchModel\\models\\HEKHUNTERv0_187.onnx"
    data_path = r"C:\\Users\\sa-forest\\Documents\\GitHub\\holypipette-pbl\\holypipette\\deepLearning\\patchModel\\test_data\\HEKHUNTER_inference_set3.hdf5"
    analyzer = ModelAnalyzer(
        model_path=model_path,
        data_path=data_path
    )
    analyzer.run()
