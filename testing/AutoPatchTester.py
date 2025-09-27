from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Type

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, colors
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (required for 3-D projection)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from holypipette.deepLearning.autoPatcher import CellHunter, PipetteFinder


def _load_hdf5_sequence(
    data_path: Path,
    *,
    demo_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Load a single demonstration sequence from an HDF5 file."""
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    with h5py.File(str(data_path), "r") as h5:
        if "data" not in h5:
            raise ValueError(f"Expected group 'data' in {data_path}")
        keys = sorted(h5["data"].keys())
        if not keys:
            raise ValueError(f"No demonstrations stored in {data_path}")
        demo_key = demo_id or keys[0]
        if demo_key not in h5["data"]:
            raise ValueError(f"Demo '{demo_key}' not found in {data_path}")

        obs_root = f"data/{demo_key}/obs"
        act_root = f"data/{demo_key}"

        images = h5[f"{obs_root}/camera_image"][:]
        resistance = h5[f"{obs_root}/resistance"][:]
        pipette_positions = h5[f"{obs_root}/pipette_positions"][:]
        stage_positions = h5[f"{obs_root}/stage_positions"][:]
        actions = h5[f"{act_root}/actions"][:]

    stage_positions = np.asarray(stage_positions, dtype=np.float32)
    if stage_positions.ndim != 2:
        stage_positions = stage_positions.reshape(stage_positions.shape[0], -1)
    if stage_positions.shape[1] == 2:
        zeros = np.zeros((stage_positions.shape[0], 1), dtype=np.float32)
        stage_positions = np.concatenate([stage_positions, zeros], axis=1)
    elif stage_positions.shape[1] > 3:
        stage_positions = stage_positions[:, :3]

    return {
        "demo_id": demo_key,
        "images": np.asarray(images),
        "resistance": np.asarray(resistance, dtype=np.float32),
        "pipette_positions": np.asarray(pipette_positions, dtype=np.float32),
        "stage_positions": stage_positions,
        "actions": np.asarray(actions, dtype=np.float32),
    }


class HuntTester(CellHunter):
    """Dataset-driven tester built on ``CellHunter`` auto-patching policy."""

    def __init__(
        self,
        *,
        model_path: Path | str,
        data_path: Path | str,
        providers: Optional[Sequence[str]] = None,
        demo_id: Optional[str] = None,
        seq_len: int = 16,
        num_layers: int = 2,
        hidden_size: int = 400,
        prefill_init: bool = False,
        center_crop: bool = True,
    ) -> None:
        self.model_path = Path(model_path)
        self.data_path = Path(data_path)
        self.dataset = _load_hdf5_sequence(self.data_path, demo_id=demo_id)

        self.images = self.dataset["images"]
        self.resistance = self.dataset["resistance"]
        self.pipette_positions = self.dataset["pipette_positions"]
        self.stage_positions = self.dataset["stage_positions"]
        self.actions = self.dataset["actions"]
        self.demo_id = self.dataset["demo_id"]

        if self.images.ndim != 4:
            raise ValueError("Images must have shape (N, H, W, C)")
        img_h, img_w = self.images.shape[1:3]
        img_size = img_h

        super().__init__(
            onnx_path=str(self.model_path),
            providers=providers,
            num_layers=num_layers,
            hidden_size=hidden_size,
            seq_len=seq_len,
            img_size=img_size,
            prefill_init=prefill_init,
            center_crop=center_crop,
        )

        self.goal_index = len(self.images) - 1
        self.goal_data: Dict[str, np.ndarray] = {
            "camera_image": self.images[self.goal_index],
            "pipette_positions": self.pipette_positions[self.goal_index].astype(np.float32),
            "stage_positions": self.stage_positions[self.goal_index].astype(np.float32),
            "resistance": np.array(self.resistance[self.goal_index], dtype=np.float32).reshape(-1),
        }

        self.goal_input_names = [name for name in (self.input_names or []) if name.startswith("goal::")]
        self.h0 = None
        self.c0 = None

    @property
    def num_frames(self) -> int:
        return self.images.shape[0]

    def set_goal(self, index: int) -> None:
        if not 0 <= index < self.num_frames:
            raise IndexError(f"Goal index {index} out of range (0, {self.num_frames - 1})")
        self.goal_index = index
        self.goal_data = {
            "camera_image": self.images[index],
            "pipette_positions": self.pipette_positions[index].astype(np.float32),
            "stage_positions": self.stage_positions[index].astype(np.float32),
            "resistance": np.array(self.resistance[index], dtype=np.float32).reshape(-1),
        }

    def observation_at(self, idx: int) -> Dict[str, np.ndarray]:
        if not 0 <= idx < self.num_frames:
            raise IndexError(f"Frame {idx} out of range (0, {self.num_frames - 1})")
        return {
            "camera_image": self.images[idx],
            "pipette_positions": self.pipette_positions[idx].astype(np.float32),
            "stage_positions": self.stage_positions[idx].astype(np.float32),
            "resistance": np.array(self.resistance[idx], dtype=np.float32).reshape(-1),
        }

    def run_inference(self, idx: int):
        obs = self.observation_at(idx)
        model_input: Dict[str, Any]
        if self.goal_input_names:
            model_input = {"obs": obs, "goal": self.goal_data}
        else:
            model_input = obs
        actions, self.h0, self.c0 = self.inference(model_input, self.h0, self.c0)
        return actions

    @staticmethod
    def calculate_error(pred, gt) -> np.ndarray:
        pred_arr = np.asarray(pred).reshape(-1)
        gt_arr = np.asarray(gt).reshape(-1)
        return pred_arr - gt_arr



class PipetteControlTester(PipetteFinder):
    """Dataset-driven tester built on ``PipetteFinder`` auto-patching policy."""

    def __init__(
        self,
        *,
        model_path: Path | str,
        data_path: Path | str,
        providers: Optional[Sequence[str]] = None,
        demo_id: Optional[str] = None,
        seq_len: int = 16,
        num_layers: int = 2,
        hidden_size: int = 400,
        prefill_init: bool = False,
        center_crop: bool = True,
    ) -> None:
        self.model_path = Path(model_path)
        self.data_path = Path(data_path)
        self.dataset = _load_hdf5_sequence(self.data_path, demo_id=demo_id)

        self.images = self.dataset["images"]
        self.pipette_positions = self.dataset["pipette_positions"]
        self.stage_positions = self.dataset["stage_positions"]
        self.actions = self.dataset["actions"]
        self.demo_id = self.dataset["demo_id"]

        if self.images.ndim != 4:
            raise ValueError("Images must have shape (N, H, W, C)")
        img_h, img_w = self.images.shape[1:3]
        img_size = img_h

        super().__init__(
            onnx_path=str(self.model_path),
            providers=providers,
            num_layers=num_layers,
            hidden_size=hidden_size,
            seq_len=seq_len,
            img_size=img_size,
            prefill_init=prefill_init,
            center_crop=center_crop,
        )

        self.goal_index = len(self.images) - 1
        self.goal_data: Dict[str, np.ndarray] = {
            "camera_image": self.images[self.goal_index],
            "pipette_positions": self.pipette_positions[self.goal_index].astype(np.float32),
            "stage_positions": self.stage_positions[self.goal_index].astype(np.float32),
        }

        self.goal_input_names = [name for name in (self.input_names or []) if name.startswith("goal::")]
        self.h0 = None
        self.c0 = None

    @property
    def num_frames(self) -> int:
        return self.images.shape[0]

    def set_goal(self, index: int) -> None:
        if not 0 <= index < self.num_frames:
            raise IndexError(f"Goal index {index} out of range (0, {self.num_frames - 1})")
        self.goal_index = index
        self.goal_data = {
            "camera_image": self.images[index],
            "pipette_positions": self.pipette_positions[index].astype(np.float32),
            "stage_positions": self.stage_positions[index].astype(np.float32),
        }

    def observation_at(self, idx: int) -> Dict[str, np.ndarray]:
        if not 0 <= idx < self.num_frames:
            raise IndexError(f"Frame {idx} out of range (0, {self.num_frames - 1})")
        return {
            "camera_image": self.images[idx],
            "pipette_positions": self.pipette_positions[idx].astype(np.float32),
            "stage_positions": self.stage_positions[idx].astype(np.float32),
        }

    def run_inference(self, idx: int):
        obs = self.observation_at(idx)
        model_input: Dict[str, Any]
        if self.goal_input_names:
            model_input = {"obs": obs, "goal": self.goal_data}
        else:
            model_input = obs
        actions, self.h0, self.c0 = self.inference(model_input, self.h0, self.c0)
        return actions

    @staticmethod
    def calculate_error(pred, gt) -> np.ndarray:
        pred_arr = np.asarray(pred).reshape(-1)
        gt_arr = np.asarray(gt).reshape(-1)
        return pred_arr - gt_arr

class AutoPatchTester:
    """Unified tester and analyser for auto-patching models."""

    def __init__(
        self,
        *,
        model_path: Path | str,
        data_path: Path | str,
        providers: Optional[Sequence[str]] = None,
        demo_id: Optional[str] = None,
        tester_cls: Type[Any] = HuntTester,
        action_slice: slice = slice(3, 6),
        save_dir: Optional[Path | str] = None,
        animation_fname: str = "pipette_trajectory.gif",
        animation_fps: int = 60,
        tester_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.data_path = Path(data_path)
        self.providers = providers
        self.action_slice = action_slice
        self.save_dir = Path(save_dir) if save_dir is not None else self.model_path.parent
        self.animation_fname = animation_fname
        self.animation_fps = animation_fps

        tester_kwargs = dict(tester_kwargs or {})
        if "demo_id" not in tester_kwargs:
            tester_kwargs["demo_id"] = demo_id

        self.tester = tester_cls(
            model_path=self.model_path,
            data_path=self.data_path,
            providers=self.providers,
            **tester_kwargs,
        )

        self.lat_ms: list[float] = []
        self.error_frames: list[np.ndarray] = []
        self.stored_actions: list[np.ndarray] = []
        self.predicted_pip_positions: Optional[np.ndarray] = None
        self.observed_pip_positions: Optional[np.ndarray] = None

    def run(self) -> None:
        self._compute_latency_and_error()
        self._plot_static_trajectory()
        self._animate_trajectory(save_gif=True)

    def _compute_latency_and_error(self) -> None:
        print("[INFO] Running inference over frames...")
        self.lat_ms.clear()
        self.error_frames.clear()
        self.stored_actions.clear()

        for idx in range(self.tester.num_frames):
            t0 = time.perf_counter()
            out = self.tester.run_inference(idx)
            self.lat_ms.append((time.perf_counter() - t0) * 1000.0)
            if out is None:
                continue
            self.error_frames.append(self.tester.calculate_error(out, self.tester.actions[idx]))
            self.stored_actions.append(np.asarray(out))

        if self.stored_actions:
            actions_flat = np.asarray(self.stored_actions).reshape(len(self.stored_actions), -1)
            if actions_flat.shape[0] > 1:
                num = np.sum(actions_flat[1:] * actions_flat[:-1], axis=1)
                den = (
                    np.linalg.norm(actions_flat[1:], axis=1)
                    * np.linalg.norm(actions_flat[:-1], axis=1)
                    + 1e-12
                )
                cos_sim = num / den
                print(
                    "[DIAG] action cos-sim: mean={:.6f} | p95={:.6f} | max={:.6f}".format(
                        float(cos_sim.mean()),
                        float(np.percentile(cos_sim, 95)),
                        float(cos_sim.max()),
                    )
                )

        if self.lat_ms:
            mean_ms = statistics.mean(self.lat_ms)
            sd_ms = statistics.stdev(self.lat_ms) if len(self.lat_ms) > 1 else 0.0
            print(f"[RESULT] Inference latency -> mean: {mean_ms:.2f} ms | sd: {sd_ms:.2f} ms")

        self._integrate_pipette_predictions()

    def _integrate_pipette_predictions(self) -> None:
        if not self.stored_actions:
            raise RuntimeError("No actions stored; run _compute_latency_and_error() first")
        pred_actions = np.asarray(self.stored_actions)
        if pred_actions.ndim == 3:
            pred_actions = pred_actions[:, 0, :]
        pred_deltas = pred_actions[:, self.action_slice]

        start = getattr(self.tester, "seq_len", 1) - 1
        start = max(start, 0)
        obs_positions = self.tester.pipette_positions[start : start + pred_deltas.shape[0]]
        if obs_positions.size == 0:
            raise RuntimeError("Observed pipette positions empty after alignment")

        n = min(pred_deltas.shape[0], obs_positions.shape[0])
        pred_deltas = pred_deltas[:n]
        obs_positions = obs_positions[:n]

        predicted_positions = [obs_positions[0]]
        for delta in pred_deltas:
            predicted_positions.append(predicted_positions[-1] + delta)
        predicted_positions = np.stack(predicted_positions[:-1])

        anchor = obs_positions[0]
        self.predicted_pip_positions = predicted_positions - anchor
        self.observed_pip_positions = obs_positions - anchor

        print(
            "[DEBUG] Predicted positions: {} | Observed positions: {}".format(
                self.predicted_pip_positions,
                self.observed_pip_positions,
            )
        )

    def _plot_static_trajectory(self) -> None:
        if self.predicted_pip_positions is None or self.observed_pip_positions is None:
            raise RuntimeError("Trajectory data unavailable; call run() first")

        def _trunc_cmap(base_cmap, start=0.5, stop=1.0, n=256):
            new_colors = base_cmap(np.linspace(start, stop, n))
            return colors.LinearSegmentedColormap.from_list(f"{base_cmap.name}_trunc", new_colors)

        cmap_pred = _trunc_cmap(plt.cm.Blues, 0.5, 1.0)
        cmap_obs = _trunc_cmap(plt.cm.Oranges, 0.5, 1.0)

        n_steps = self.predicted_pip_positions.shape[0]
        norm = plt.Normalize(vmin=0, vmax=max(n_steps - 1, 1))

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_zlim(-20, 5)

        ax.scatter(
            self.predicted_pip_positions[:, 0],
            self.predicted_pip_positions[:, 1],
            -self.predicted_pip_positions[:, 2],
            c=cmap_pred(norm(np.arange(n_steps))),
            marker="o",
            s=20,
            label="Predicted",
        )
        ax.scatter(
            self.observed_pip_positions[:, 0],
            self.observed_pip_positions[:, 1],
            -self.observed_pip_positions[:, 2],
            c=cmap_obs(norm(np.arange(n_steps))),
            marker="^",
            s=20,
            label="Observed",
        )

        ax.set_title("3-D Pipette Trajectory: Predicted vs Observed")
        ax.set_xlabel("Pipette X")
        ax.set_ylabel("Pipette Y")
        ax.set_zlabel("Pipette Z")
        ax.legend()
        plt.show()

    def _animate_trajectory(self, *, save_gif: bool = True) -> None:
        if self.predicted_pip_positions is None or self.observed_pip_positions is None:
            raise RuntimeError("Trajectory data unavailable; call run() first")

        error_mag = np.linalg.norm(
            self.predicted_pip_positions - self.observed_pip_positions, axis=1
        )

        def _trunc_cmap(base_cmap, start=0.5, stop=1.0, n=256):
            new_colors = base_cmap(np.linspace(start, stop, n))
            return colors.LinearSegmentedColormap.from_list(f"{base_cmap.name}_trunc", new_colors)

        cmap_pred = _trunc_cmap(plt.cm.Blues, 0.5, 1.0)
        cmap_obs = _trunc_cmap(plt.cm.Oranges, 0.5, 1.0)

        n_steps = self.predicted_pip_positions.shape[0]
        norm = plt.Normalize(vmin=0, vmax=max(n_steps - 1, 1))

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        ax.set(
            xlim=(-5, 5),
            ylim=(-5, 5),
            zlim=(-20, 5),
            title="Animated 3-D Pipette Trajectory",
            xlabel="Pipette X",
            ylabel="Pipette Y",
            zlabel="Pipette Z",
        )
        ax.grid(False)

        sc_pred = ax.scatter([], [], [], c=[], cmap=cmap_pred, vmin=0, vmax=max(n_steps - 1, 1), marker="o", s=20)
        sc_obs = ax.scatter([], [], [], c=[], cmap=cmap_obs, vmin=0, vmax=max(n_steps - 1, 1), marker="^", s=20)

        from matplotlib.lines import Line2D

        error_handle = Line2D([], [], linestyle="none", marker="", color="red")

        for cm, pad, lbl in ((cmap_pred, 0.10, "Time steps (Predicted)"), (cmap_obs, 0.03, "Time steps (Observed)")):
            m = plt.cm.ScalarMappable(norm=norm, cmap=cm)
            m.set_array([])
            cb = plt.colorbar(m, ax=ax, pad=pad, shrink=0.6)
            cb.set_label(lbl)
            cb.ax.invert_yaxis()

        error_text_handle = None

        def _init():
            nonlocal error_text_handle
            for sc in (sc_pred, sc_obs):
                sc._offsets3d = ([], [], [])
                sc.set_array(np.array([]))

            legend = ax.legend([sc_pred, sc_obs, error_handle], ["Predicted", "Observed", ""], loc="best", frameon=True)
            error_text_handle = legend.get_texts()[-1]
            error_text_handle.set_color("red")
            error_text_handle.set_text(f"Error: {error_mag[0]:.3f}")
            return sc_pred, sc_obs, error_text_handle

        def _update(frame: int):
            x_p, y_p, z_p = self.predicted_pip_positions[: frame + 1].T
            sc_pred._offsets3d = (x_p, y_p, -z_p)
            sc_pred.set_array(norm(np.arange(frame + 1)))

            x_o, y_o, z_o = self.observed_pip_positions[: frame + 1].T
            sc_obs._offsets3d = (x_o, y_o, -z_o)
            sc_obs.set_array(norm(np.arange(frame + 1)))

            error_text_handle.set_text(f"Error: {error_mag[frame]:.3f}")
            return sc_pred, sc_obs, error_text_handle

        interval_ms = 1000 / self.animation_fps
        anim = animation.FuncAnimation(
            fig,
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
                print(f"[INFO] Animation saved -> {out_path.resolve()}")
            except Exception as exc:
                print(f"[WARNING] GIF not saved: {exc}")

        plt.show()










# ------------------------------------------------------------------
# Quick manual driver
# ------------------------------------------------------------------

DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "holypipette"
    / "deepLearning"
    / "patchModel"
    / "models"
    / "HEKHUNTERv0_201.onnx"
)

DEFAULT_DATA_PATH = Path(__file__).resolve().parents[1] / "testing" / "data" / "autopatch_demo.h5"

model_path = r"C:\\Users\\sa-forest\\Documents\\GitHub\\holypipette-pbl\\holypipette\\deepLearning\\patchModel\\models\\HEKHUNTERv0_201.onnx"
data_path = r"C:\\Users\\sa-forest\\Documents\\GitHub\\holypipette-pbl\\holypipette\\deepLearning\\patchModel\\test_data\\HEKHUNTER_inference_set_goal.hdf5"

def main() -> None:
    """Hard-coded replay that mirrors the original tester behaviour."""
    tester = AutoPatchTester(
        model_path=model_path if model_path else DEFAULT_MODEL_PATH,
        data_path=data_path if data_path else DEFAULT_DATA_PATH,
        providers=None,
        demo_id=None,
        tester_cls=HuntTester,
    )

    tester._compute_latency_and_error()
    tester._plot_static_trajectory()
    tester._animate_trajectory(save_gif=True)


if __name__ == "__main__":
    main()









