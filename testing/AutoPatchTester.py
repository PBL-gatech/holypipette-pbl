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

        obs_group = h5[f"{obs_root}"]
        images = obs_group["camera_image"][:]
        num_frames = images.shape[0]

        if "resistance" in obs_group:
            resistance = obs_group["resistance"][:]
        else:
            resistance = np.zeros((num_frames, 1), dtype=np.float32)

        if "pipette_positions" in obs_group:
            pipette_positions = obs_group["pipette_positions"][:]
        else:
            pipette_positions = np.zeros((num_frames, 3), dtype=np.float32)

        if "stage_positions" in obs_group:
            stage_positions = obs_group["stage_positions"][:]
        else:
            stage_positions = np.zeros((num_frames, 3), dtype=np.float32)
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
        center_crop: bool = False,
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
        position_round_decimals: Optional[int] = 2,
        tester_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.data_path = Path(data_path)
        self.providers = providers
        self.action_slice = action_slice
        self.save_dir = Path(save_dir) if save_dir is not None else self.model_path.parent
        self.animation_fname = animation_fname
        self.animation_fps = animation_fps
        self.position_round_decimals = position_round_decimals

        tester_kwargs = dict(tester_kwargs or {})
        if "demo_id" not in tester_kwargs:
            tester_kwargs["demo_id"] = demo_id

        self.tester = tester_cls(
            model_path=self.model_path,
            data_path=self.data_path,
            providers=self.providers,
            **tester_kwargs,
        )

        pip_dim, stage_dim, action_dim = self.tester._axis_dims()
        actions = getattr(self.tester, "actions", None)
        if action_dim and actions is not None and actions.shape[-1] != action_dim:
            raise ValueError(
                f"Dataset action dimension {actions.shape[-1]} does not match model output {action_dim}"
            )
        if self.tester.pipette_positions.shape[-1] < pip_dim:
            raise ValueError(
                f"Dataset pipette dimension {self.tester.pipette_positions.shape[-1]} is smaller than required {pip_dim}"
            )
        if stage_dim and self.tester.stage_positions.shape[-1] < stage_dim:
            raise ValueError(
                f"Dataset stage dimension {self.tester.stage_positions.shape[-1]} is smaller than required {stage_dim}"
            )
        if action_dim and stage_dim + pip_dim <= action_dim:
            self.action_slice = slice(stage_dim, stage_dim + pip_dim)

        self.lat_ms: list[float] = []
        self.error_frames: list[np.ndarray] = []
        self.stored_actions: list[np.ndarray] = []
        self.reference_pip_positions: Optional[np.ndarray] = None
        self.predicted_pip_positions: Optional[np.ndarray] = None
        self.observed_pip_positions: Optional[np.ndarray] = None
        self.predicted_pip_deltas: Optional[np.ndarray] = None
        self.observed_pip_deltas: Optional[np.ndarray] = None
        self.roundit: bool = False


    def _rounded_positions(self, array: np.ndarray) -> np.ndarray:
        if array is None:
            return None
        if self.position_round_decimals is None:
            return np.asarray(array)
        return np.round(np.asarray(array), self.position_round_decimals)


    def run(self) -> None:
        self._compute_latency_and_error()
        self._plot_static_trajectory()
        self._plot_raw_predictions()
        self._plot_predicted_xy_trajectory()
        self._animate_trajectory(save_gif=True)

    def _compute_latency_and_error(self) -> None:
        print("[INFO] Running inference over frames...")
        self.lat_ms.clear()
        self.error_frames.clear()
        self.stored_actions.clear()

        if hasattr(self.tester, "reset_state"):
            self.tester.reset_state()
            if hasattr(self.tester, "h0"):
                self.tester.h0 = None
            if hasattr(self.tester, "c0"):
                self.tester.c0 = None

        warmup_start = max(getattr(self.tester, 'seq_len', 1) - 1, 0)

        for idx in range(self.tester.num_frames):
            t0 = time.perf_counter()
            out = self.tester.run_inference(idx)
            self.lat_ms.append((time.perf_counter() - t0) * 1000.0)
            if out is None:
                continue
            self.error_frames.append(self.tester.calculate_error(out, self.tester.actions[idx]))
            if idx >= warmup_start:
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
            raise RuntimeError('No actions stored; run _compute_latency_and_error() first')
        pred_actions = np.asarray(self.stored_actions)
        if pred_actions.ndim == 3:
            pred_actions = pred_actions[:, 0, :]
        pred_deltas = pred_actions[:, self.action_slice]
        pip_dim = self.tester._axis_dims()[0]
        pred_deltas = pred_deltas[:, :pip_dim]

        start = getattr(self.tester, 'seq_len', 1) - 1
        start = max(start, 0)
        obs_positions = np.asarray(
            self.tester.pipette_positions[start : start + pred_deltas.shape[0]],
            dtype=np.float32,
        )
        if obs_positions.size == 0:
            raise RuntimeError('Observed pipette positions empty after alignment')
        obs_positions = obs_positions.reshape(obs_positions.shape[0], -1)[:, :pip_dim]

        n = min(pred_deltas.shape[0], obs_positions.shape[0] - 1)
        if n <= 0:
            raise RuntimeError('Not enough observed pipette positions to compute stepwise trajectories')

        base_positions = obs_positions[:n]
        next_positions = obs_positions[1 : n + 1]
        pred_deltas = pred_deltas[:n]

        predicted_positions = base_positions + pred_deltas

        pad3 = lambda a: a[:, :3] if a.shape[1] >= 3 else np.pad(a, ((0, 0), (0, 3 - a.shape[1])), mode='constant')
        base_pad = pad3(base_positions)
        next_pad = pad3(next_positions)
        pred_pad = pad3(predicted_positions)

        anchor = base_pad[0]
        self.reference_pip_positions = base_pad - anchor
        self.predicted_pip_positions = pred_pad - anchor
        self.observed_pip_positions = next_pad - anchor
        self.predicted_pip_deltas = pred_deltas
        self.observed_pip_deltas = next_positions - base_positions

        if self.roundit:
            debug_pred = self._rounded_positions(self.predicted_pip_positions)
            debug_obs = self._rounded_positions(self.observed_pip_positions)
        else:
            debug_pred = self.predicted_pip_positions
            debug_obs = self.observed_pip_positions
        print(
            '[DEBUG] Predicted endpoints: {} | Observed endpoints: {}'.format(
                debug_pred,
                debug_obs,
            )
        )

    def _plot_static_trajectory(self) -> None:
        if (
            self.reference_pip_positions is None
            or self.predicted_pip_positions is None
            or self.observed_pip_positions is None
        ):
            raise RuntimeError('Trajectory data unavailable; call run() first')

        def _trunc_cmap(base_cmap, start=0.5, stop=1.0, n=256):
            new_colors = base_cmap(np.linspace(start, stop, n))
            return colors.LinearSegmentedColormap.from_list(f'{base_cmap.name}_trunc', new_colors)

        def _limits(values: np.ndarray) -> tuple[float, float]:
            values = np.asarray(values, dtype=np.float32)
            vmin = float(values.min())
            vmax = float(values.max())
            if np.isclose(vmin, vmax):
                pad = max(abs(vmin), 1.0) * 0.5
                vmin -= pad
                vmax += pad
            else:
                pad = 0.05 * (vmax - vmin)
                pad = max(pad, 1e-3)
                vmin -= pad
                vmax += pad
            return vmin, vmax
        
        if self.roundit:
            base = self._rounded_positions(self.reference_pip_positions)
            predicted = self._rounded_positions(self.predicted_pip_positions)
            observed = self._rounded_positions(self.observed_pip_positions)
        else: 
            base = self.reference_pip_positions
            predicted = self.predicted_pip_positions
            observed = self.observed_pip_positions

        n_steps = predicted.shape[0]
        norm = plt.Normalize(vmin=0, vmax=max(n_steps, 1))

        origin = np.zeros((1, 3), dtype=predicted.dtype)
        base_plot = np.vstack((origin, base))
        predicted_plot = np.vstack((origin, predicted))
        observed_plot = np.vstack((origin, observed))

        x_all = np.concatenate((base_plot[:, 0], predicted_plot[:, 0], observed_plot[:, 0]))
        y_all = np.concatenate((base_plot[:, 1], predicted_plot[:, 1], observed_plot[:, 1]))
        z_all = np.concatenate((-base_plot[:, 2], -predicted_plot[:, 2], -observed_plot[:, 2]))

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim(*_limits(x_all))
        ax.set_ylim(*_limits(y_all))
        ax.set_zlim(*_limits(z_all))

        cmap_pred = _trunc_cmap(plt.cm.Blues, 0.5, 1.0)
        cmap_obs = _trunc_cmap(plt.cm.Oranges, 0.5, 1.0)
        pred_colors = cmap_pred(norm(np.arange(n_steps + 1)))
        obs_colors = cmap_obs(norm(np.arange(n_steps + 1)))

        base_scatter = ax.scatter(
            base_plot[:, 0],
            base_plot[:, 1],
            -base_plot[:, 2],
            c='gray',
            marker='.',
            s=15,
            alpha=0.4,
            label='Observation (t)',
        )
        pred_scatter = ax.scatter(
            predicted_plot[:, 0],
            predicted_plot[:, 1],
            -predicted_plot[:, 2],
            c=pred_colors,
            marker='o',
            s=25,
            label='Predicted (t+1)',
        )
        obs_scatter = ax.scatter(
            observed_plot[:, 0],
            observed_plot[:, 1],
            -observed_plot[:, 2],
            c=obs_colors,
            marker='^',
            s=25,
            label='Observed (t+1)',
        )

        for idx in range(n_steps):
            base_pt = base[idx]
            pred_pt = predicted[idx]
            obs_pt = observed[idx]
            color_idx = idx + 1
            ax.plot(
                [base_pt[0], pred_pt[0]],
                [base_pt[1], pred_pt[1]],
                [-base_pt[2], -pred_pt[2]],
                color=pred_colors[color_idx],
                linewidth=1.2,
                alpha=0.7,
            )
            ax.plot(
                [base_pt[0], obs_pt[0]],
                [base_pt[1], obs_pt[1]],
                [-base_pt[2], -obs_pt[2]],
                color=obs_colors[color_idx],
                linewidth=1.2,
                alpha=0.7,
            )

        ax.set_title('3-D Pipette Trajectories (per step)')
        ax.set_xlabel('Pipette X')
        ax.set_ylabel('Pipette Y')
        ax.set_zlabel('Pipette Z')

        handles = [pred_scatter, obs_scatter, base_scatter]
        ax.legend(handles=handles, loc='best')
        plt.show()




    def _plot_raw_predictions(self) -> None:
        if self.predicted_pip_deltas is None:
            raise RuntimeError('Predicted deltas unavailable; call run() first')
        if self.observed_pip_deltas is None:
            raise RuntimeError('Observed deltas unavailable; call run() first')

        if self.roundit:
            predicted = self._rounded_positions(self.predicted_pip_deltas)
            observed = self._rounded_positions(self.observed_pip_deltas)
        else:
            predicted = self.predicted_pip_deltas
            observed = self.observed_pip_deltas
        if predicted is None or predicted.size == 0:
            raise RuntimeError('Predicted delta array is empty')
        if observed is None or observed.size == 0:
            raise RuntimeError('Observed delta array is empty')

        n_steps = min(predicted.shape[0], observed.shape[0])
        if n_steps <= 0:
            raise RuntimeError('No delta steps available to plot')
        predicted = predicted[:n_steps]
        observed = observed[:n_steps]
        steps = np.arange(1, n_steps + 1, dtype=np.int32)

        fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 8))
        component_ids = (0, 1)
        axis_labels = ('dX', 'dY')
        pred_color = 'tab:blue'
        true_color = 'tab:green'

        for ax, comp, label in zip(axes, component_ids, axis_labels):
            ax.plot(steps, predicted[:, comp], color=pred_color, linewidth=1.3, label='Predicted')
            ax.plot(steps, observed[:, comp], color=true_color, linewidth=1.1, linestyle='--', label='True')
            ax.set_ylabel(label)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best')

        axes[0].set_title('Raw Predictions vs True Action Deltas (dX, dY)')
        axes[-1].set_xlabel('Step (t)')
        plt.tight_layout()
        plt.show()
    def _plot_predicted_xy_trajectory(self) -> None:
        if self.predicted_pip_positions is None:
            raise RuntimeError('Predicted trajectory unavailable; call run() first')
        

        if self.roundit:
            predicted = self._rounded_positions(self.predicted_pip_positions)
        else:
            predicted = self.predicted_pip_positions
        if predicted is None or predicted.size == 0:
            raise RuntimeError('Predicted trajectory array is empty')

        steps = np.arange(1, predicted.shape[0] + 1, dtype=np.int32)
        xs = predicted[:, 0]
        ys = predicted[:, 1]

        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')
        color_vals = np.linspace(0.4, 1.0, steps.size)
        scatter_colors = plt.cm.Blues(color_vals)

        ax.plot(steps, xs, ys, color='tab:blue', linewidth=1.1, alpha=0.6)
        ax.scatter(steps, xs, ys, c=scatter_colors, s=35, marker='o', label='Predicted (t+1)')

        ax.set_title('Predicted XY Trajectory Across Steps')
        ax.set_xlabel('Step (t)')
        ax.set_ylabel('Predicted X')
        ax.set_zlabel('Predicted Y')
        ax.legend(loc='best')
        plt.tight_layout()
        plt.show()

    def _animate_trajectory(self, *, save_gif: bool = True) -> None:
        if (
            self.reference_pip_positions is None
            or self.predicted_pip_positions is None
            or self.observed_pip_positions is None
        ):
            raise RuntimeError('Trajectory data unavailable; call run() first')
        
        if self.roundit:
            base = self._rounded_positions(self.reference_pip_positions)
            predicted = self._rounded_positions(self.predicted_pip_positions)
            observed = self._rounded_positions(self.observed_pip_positions)
        else:
            base = self.reference_pip_positions
            predicted = self.predicted_pip_positions
            observed = self.observed_pip_positions

        error_mag = np.linalg.norm(predicted - observed, axis=1)

        def _trunc_cmap(base_cmap, start=0.5, stop=1.0, n=256):
            new_colors = base_cmap(np.linspace(start, stop, n))
            return colors.LinearSegmentedColormap.from_list(f'{base_cmap.name}_trunc', new_colors)

        def _limits(values: np.ndarray) -> tuple[float, float]:
            values = np.asarray(values, dtype=np.float32)
            vmin = float(values.min())
            vmax = float(values.max())
            if np.isclose(vmin, vmax):
                pad = max(abs(vmin), 1.0) * 0.5
                vmin -= pad
                vmax += pad
            else:
                pad = 0.05 * (vmax - vmin)
                pad = max(pad, 1e-3)
                vmin -= pad
                vmax += pad
            return vmin, vmax

        cmap_pred = _trunc_cmap(plt.cm.Blues, 0.5, 1.0)
        cmap_obs = _trunc_cmap(plt.cm.Oranges, 0.5, 1.0)

        n_steps = predicted.shape[0]
        norm = plt.Normalize(vmin=0, vmax=max(n_steps, 1))

        origin = np.zeros((1, 3), dtype=predicted.dtype)
        base_plot = np.vstack((origin, base))
        predicted_plot = np.vstack((origin, predicted))
        observed_plot = np.vstack((origin, observed))
        error_series = np.concatenate(([0.0], error_mag))
        n_frames = predicted_plot.shape[0]

        x_all = np.concatenate((base_plot[:, 0], predicted_plot[:, 0], observed_plot[:, 0]))
        y_all = np.concatenate((base_plot[:, 1], predicted_plot[:, 1], observed_plot[:, 1]))
        z_all = np.concatenate((-base_plot[:, 2], -predicted_plot[:, 2], -observed_plot[:, 2]))

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.set(
            xlim=_limits(x_all),
            ylim=_limits(y_all),
            zlim=_limits(z_all),
            title='Animated 3-D Pipette Trajectory (per step)',
            xlabel='Pipette X',
            ylabel='Pipette Y',
            zlabel='Pipette Z',
        )
        ax.grid(False)

        sc_base = ax.scatter([], [], [], marker='.', c='gray', s=15, alpha=0.4)
        sc_pred = ax.scatter([], [], [], c=[], cmap=cmap_pred, vmin=0, vmax=max(n_steps, 1), marker='o', s=25)
        sc_obs = ax.scatter([], [], [], c=[], cmap=cmap_obs, vmin=0, vmax=max(n_steps, 1), marker='^', s=25)
        pred_line = ax.plot([], [], [], linewidth=1.6, alpha=0.8)[0]
        obs_line = ax.plot([], [], [], linewidth=1.6, alpha=0.8)[0]

        from matplotlib.lines import Line2D

        error_handle = Line2D([], [], linestyle='none', marker='', color='red')
        pred_conn_handle = Line2D([0], [0], color=cmap_pred(0.9), linewidth=1.6, label='Predicted delta')
        obs_conn_handle = Line2D([0], [0], color=cmap_obs(0.9), linewidth=1.6, label='Observed delta')

        for cm, pad, lbl in ((cmap_pred, 0.10, 'Time steps (Predicted)'), (cmap_obs, 0.03, 'Time steps (Observed)')):
            m = plt.cm.ScalarMappable(norm=norm, cmap=cm)
            m.set_array([])
            cb = plt.colorbar(m, ax=ax, pad=pad, shrink=0.6)
            cb.set_label(lbl)
            cb.ax.invert_yaxis()

        error_text_handle = None

        def _init():
            nonlocal error_text_handle
            sc_base._offsets3d = ([], [], [])
            sc_pred._offsets3d = ([], [], [])
            sc_obs._offsets3d = ([], [], [])
            sc_pred.set_array(np.array([]))
            sc_obs.set_array(np.array([]))
            for line in (pred_line, obs_line):
                line.set_data([], [])
                line.set_3d_properties([])

            legend = ax.legend(
                [sc_pred, sc_obs, sc_base, pred_conn_handle, obs_conn_handle, error_handle],
                ['Predicted (t+1)', 'Observed (t+1)', 'Observation (t)', 'Predicted delta', 'Observed delta', ''],
                loc='best',
                frameon=True,
            )
            error_text_handle = legend.get_texts()[-1]
            error_text_handle.set_color('red')
            error_text_handle.set_text(f'Error: {error_series[0]:.3f}')
            pred_line.set_color(cmap_pred(norm(0)))
            obs_line.set_color(cmap_obs(norm(0)))
            return sc_pred, sc_obs, sc_base, pred_line, obs_line, error_text_handle

        def _update(frame: int):
            step_ids = np.arange(frame + 1)

            base_slice = base_plot[: frame + 1].T
            sc_base._offsets3d = (base_slice[0], base_slice[1], -base_slice[2])

            pred_slice = predicted_plot[: frame + 1].T
            sc_pred._offsets3d = (pred_slice[0], pred_slice[1], -pred_slice[2])
            sc_pred.set_array(norm(step_ids))

            obs_slice = observed_plot[: frame + 1].T
            sc_obs._offsets3d = (obs_slice[0], obs_slice[1], -obs_slice[2])
            sc_obs.set_array(norm(step_ids))

            if frame > 0:
                base_pt = base_plot[frame]
                pred_pt = predicted_plot[frame]
                obs_pt = observed_plot[frame]
                pred_line.set_data([base_pt[0], pred_pt[0]], [base_pt[1], pred_pt[1]])
                pred_line.set_3d_properties([-base_pt[2], -pred_pt[2]])
                pred_line.set_color(cmap_pred(norm(frame)))

                obs_line.set_data([base_pt[0], obs_pt[0]], [base_pt[1], obs_pt[1]])
                obs_line.set_3d_properties([-base_pt[2], -obs_pt[2]])
                obs_line.set_color(cmap_obs(norm(frame)))
            else:
                pred_line.set_data([], [])
                pred_line.set_3d_properties([])
                obs_line.set_data([], [])
                obs_line.set_3d_properties([])

            error_text_handle.set_text(f'Error: {error_series[frame]:.3f}')
            return sc_pred, sc_obs, sc_base, pred_line, obs_line, error_text_handle

        interval_ms = 1000 / self.animation_fps
        anim = animation.FuncAnimation(
            fig,
            _update,
            init_func=_init,
            frames=n_frames,
            interval=interval_ms,
            blit=False,
        )

        if save_gif:
            out_path = self.save_dir / self.animation_fname
            try:
                anim.save(out_path, writer=animation.PillowWriter(fps=self.animation_fps))
                print(f'[INFO] Animation saved -> {out_path.resolve()}')
            except Exception as exc:
                print(f'[WARNING] GIF not saved: {exc}')

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

# model_path = r"C:\\Users\\sa-forest\\Documents\\GitHub\\holypipette-pbl\\holypipette\\deepLearning\\patchModel\\models\\HEKHUNTERv0_201.onnx"
# data_path = r"C:\\Users\\sa-forest\\Documents\\GitHub\\holypipette-pbl\\holypipette\\deepLearning\\patchModel\\test_data\\HEKHUNTER_inference_set_goal.hdf5"
# model_path = r"C:\Users\sa-forest\Documents\GitHub\holypipette-pbl\holypipette\deepLearning\patchModel\NeuronHunter\models\bc_HEKHunter_v0_300.onnx"
# data_path = r"C:\Users\sa-forest\Documents\GitHub\holypipette-pbl\experiments\Datasets\PatcherBot_test_dataset_v0_006\PatcherBot_test_dataset_v0_006_hunt_cell.hdf5"

model_path = r"C:\Users\sa-forest\Documents\GitHub\holypipette-pbl\holypipette\deepLearning\patchModel\NeuronHunter\models\bc_CellHunter_v0_140.onnx"
# model_path = r"C:\Users\sa-forest\Documents\GitHub\holypipette-pbl\holypipette\deepLearning\patchModel\PipetteFinder\models\bc_PipetteFinder_v0_120.onnx"
# model_path = r"C:\Users\sa-forest\Documents\GitHub\holypipette-pbl\holypipette\deepLearning\patchModel\PipetteFinder\models\df_PipetteFinder_v0_004.onnx"
# data_path = r"C:\Users\sa-forest\Documents\GitHub\holypipette-pbl\experiments\Datasets\PatcherBot_test_dataset_v0_005\PatcherBot_test_dataset_v0_005_find_pipette.hdf5"
# data_path = r"C:\Users\sa-forest\Documents\GitHub\holypipette-pbl\experiments\Datasets\PatcherBot_test_dataset_v0_120\PatcherBot_test_dataset_v0_120_find_pipette.hdf5"
data_path = r"C:\Users\sa-forest\Documents\GitHub\holypipette-pbl\experiments\Datasets\PatcherBot_test_dataset_v0_140\PatcherBot_test_dataset_v0_140_hunt_cell.hdf5"


def main() -> None:
    """Hard-coded replay that mirrors the original tester behaviour."""
    tester = AutoPatchTester(
        model_path=model_path if model_path else DEFAULT_MODEL_PATH,
        data_path=data_path if data_path else DEFAULT_DATA_PATH,
        providers=None,
        demo_id="demo_1",
        # tester_cls=PipetteControlTester,
        tester_cls=HuntTester,
    )

    tester._compute_latency_and_error()
    tester._plot_static_trajectory()
    tester._plot_raw_predictions()
    tester._plot_predicted_xy_trajectory()
    tester._animate_trajectory(save_gif=True)


if __name__ == "__main__":
    main()
