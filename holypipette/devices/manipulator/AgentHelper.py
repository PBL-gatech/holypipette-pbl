import sys
import time
from pathlib import Path
from typing import Optional, Tuple, Union, Dict, Any, List

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from holypipette.deepLearning.PatcherBotAgent import CellHunter, GigaSealer, Burglar, PipetteFinder

import h5py


class AgentHelper:

    def __init__(self):
        """Track the active agent instance and its configuration."""
        self.agent = None
        self.requires_goal = False

    def prepare_model(self, model_type):
        """Instantiate one of the supported agent subclasses."""
        self.model_type = model_type
        if model_type == "find_pipette":
            self.agent = PipetteFinder()
        elif model_type == "hunt":
            self.agent = CellHunter()
        elif model_type == "gigaseal":
            self.agent = GigaSealer()
        elif model_type == "break_in":
            self.agent = Burglar()
        else:
            raise ValueError(f"Model type '{model_type}' not supported")
        self.requires_goal = bool(getattr(self.agent, "goal_required", False))

    def run_inference(
        self,
        observation: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        goal: Optional[np.ndarray] = None,
        *,
        is_demo: bool = False,
    ) -> np.ndarray:
        """Execute inference on the prepared agent."""
        if self.agent is None:
            raise RuntimeError("Call prepare_model before run_inference")
        active_goal = goal if self.requires_goal else None
        return self.agent.inference(observation=observation, goal=active_goal, is_demo=is_demo)


class AgentTester:
    def __init__(self) -> None:
        """Prepare the agent for testing."""
        self.agent_helper = AgentHelper()
        self.predictions: List[np.ndarray] = []
        self.latencies_ms: List[float] = []
        self.errors: List[float] = []
        self.last_results: Optional[Dict[str, Any]] = None
        self.last_dataset: Optional[Dict[str, Any]] = None

    def _load_hdf5_sequence(
        self,
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

    def compute_errors(self, predictions: np.ndarray, actions: np.ndarray) -> List[float]:
        """Compute the L2 error between the predicted and actual actions per frame."""
        pred = np.asarray(predictions, dtype=np.float32)
        gt = np.asarray(actions, dtype=np.float32)

        if pred.ndim == 1:
            pred = pred.reshape(1, -1)
        if gt.ndim == 1:
            gt = gt.reshape(1, -1)
        if pred.shape[0] != gt.shape[0]:
            raise ValueError(
                f"Prediction count ({pred.shape[0]}) does not match action count ({gt.shape[0]})"
            )

        pred_flat = pred.reshape(pred.shape[0], -1)
        gt_flat = gt.reshape(gt.shape[0], -1)
        min_dim = min(pred_flat.shape[1], gt_flat.shape[1])
        if min_dim == 0:
            return [0.0] * pred_flat.shape[0]

        diff = pred_flat[:, :min_dim] - gt_flat[:, :min_dim]
        return np.linalg.norm(diff, axis=1).astype(float).tolist()
    
    def visualize(
        self,
        pred: Optional[np.ndarray] = None,
        *,
        fps: int = 30,
        save_video_path: Optional[Union[str, Path]] = None,
        save_plot_path: Optional[Union[str, Path]] = None,
        show_animation: bool = True,
        show_plot: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Overlay predicted vs. ground-truth deltas on frames and plot per-axis comparisons."""
        if self.last_results is None or self.last_dataset is None:
            raise RuntimeError("No cached results available. Run test_model before visualize.")

        try:
            import matplotlib.pyplot as plt
            from matplotlib import animation
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            from matplotlib.figure import Figure
            from matplotlib.patches import Circle
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("matplotlib is required for visualization but is not installed.") from exc

        predictions_src = pred if pred is not None else self.last_results.get("predictions")
        if predictions_src is None:
            raise RuntimeError("Predictions are not available; provide `pred` or run test_model first.")
        predictions = np.asarray(predictions_src, dtype=np.float32)
        ground_truth_src = self.last_results.get("ground_truth")
        if ground_truth_src is None:
            raise RuntimeError("Ground-truth actions are missing from the cached results.")
        ground_truth = np.asarray(ground_truth_src, dtype=np.float32)
        frames = np.asarray(self.last_dataset.get("images"))

        if predictions.ndim == 1:
            predictions = predictions[:, np.newaxis]
        if ground_truth.ndim == 1:
            ground_truth = ground_truth[:, np.newaxis]

        num_frames = min(len(frames), len(predictions), len(ground_truth))
        if num_frames == 0:
            raise RuntimeError("No frames available for visualization.")
        predictions = predictions[:num_frames]
        ground_truth = ground_truth[:num_frames]
        frames = frames[:num_frames]

        def _ensure_rgb(image: np.ndarray) -> np.ndarray:
            arr = np.asarray(image)
            if arr.ndim == 2:
                arr = np.stack([arr] * 3, axis=-1)
            elif arr.ndim == 3 and arr.shape[-1] == 1:
                arr = np.repeat(arr, 3, axis=-1)
            arr = arr.astype(np.float32)
            if arr.max() > 1.0:
                arr = arr / 255.0
            arr = np.clip(arr, 0.0, 1.0)
            return (arr * 255.0).astype(np.uint8)

        def _vector_xy(vec: np.ndarray) -> np.ndarray:
            if vec.size >= 2:
                return vec[:2]
            if vec.size == 1:
                return np.array([vec[0], 0.0], dtype=np.float32)
            return np.zeros(2, dtype=np.float32)

        frame_height = frames.shape[1]
        frame_width = frames.shape[2] if frames.ndim >= 3 else frames.shape[1]

        pipette_positions: Optional[np.ndarray] = None
        pipette_positions_src = self.last_dataset.get("pipette_positions")
        if pipette_positions_src is not None:
            pipette_positions_arr = np.asarray(pipette_positions_src, dtype=np.float32)
            if pipette_positions_arr.ndim == 1:
                pipette_positions_arr = pipette_positions_arr[:, np.newaxis]
            if (
                pipette_positions_arr.shape[1] >= 2
                and pipette_positions_arr.shape[0] >= num_frames
            ):
                pipette_positions = pipette_positions_arr[:num_frames, :2]

        overlay_frames = []
        for idx in range(num_frames):
            frame_rgb = _ensure_rgb(frames[idx])
            gt_vec = _vector_xy(ground_truth[idx])
            pred_vec = _vector_xy(predictions[idx])
            height, width = frame_rgb.shape[:2]

            if pipette_positions is not None and pipette_positions.shape[0] > idx:
                current_coords = pipette_positions[idx]
            else:
                current_coords = np.array([width / 2.0, height / 2.0], dtype=np.float32)
            if np.any(np.isnan(current_coords)):
                current_coords = np.array([width / 2.0, height / 2.0], dtype=np.float32)
            current_x = float(np.clip(current_coords[0], 0.0, max(width - 1.0, 0.0)))
            current_y = float(np.clip(current_coords[1], 0.0, max(height - 1.0, 0.0)))
            circle_radius = max(3.0, min(width, height) * 0.03)
            trail_radius = max(2.0, circle_radius * 0.7)
            future_steps = max(0, min(5, num_frames - idx - 1))

            fig = Figure(figsize=(width / 100.0, height / 100.0), dpi=100)
            canvas = FigureCanvasAgg(fig)
            ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
            ax.imshow(frame_rgb)
            ax.axis("off")

            def _future_positions(vecs: np.ndarray) -> List[np.ndarray]:
                points: List[np.ndarray] = []
                for step_idx in range(1, future_steps + 1):
                    action_idx = idx + step_idx
                    if action_idx >= vecs.shape[0]:
                        break
                    delta = _vector_xy(vecs[action_idx])
                    if np.any(np.isnan(delta)):
                        continue
                    prev_idx = action_idx - 1
                    if (
                        pipette_positions is not None
                        and prev_idx < pipette_positions.shape[0]
                    ):
                        base = np.array(
                            [
                                float(np.clip(pipette_positions[prev_idx, 0], 0.0, max(width - 1.0, 0.0))),
                                float(np.clip(pipette_positions[prev_idx, 1], 0.0, max(height - 1.0, 0.0))),
                            ],
                            dtype=np.float32,
                        )
                    elif points:
                        base = points[-1].copy()
                    else:
                        base = np.array([current_x, current_y], dtype=np.float32)
                    offset = np.array(
                        [float(delta[0]), -float(delta[1])],
                        dtype=np.float32,
                    )
                    point = base + offset
                    point[0] = float(np.clip(point[0], 0.0, max(width - 1.0, 0.0)))
                    point[1] = float(np.clip(point[1], 0.0, max(height - 1.0, 0.0)))
                    points.append(point)
                return points

            def _draw_trail(points: List[np.ndarray], color: str) -> None:
                for step_idx, point in enumerate(points):
                    alpha = max(0.3, 1.0 - step_idx * 0.15)
                    ax.add_patch(
                        Circle(
                            (float(point[0]), float(point[1])),
                            radius=trail_radius,
                            facecolor=color,
                            edgecolor="none",
                            alpha=alpha,
                        )
                    )

            if future_steps:
                _draw_trail(_future_positions(ground_truth), "tab:blue")
                _draw_trail(_future_positions(predictions), "tab:orange")

            ax.add_patch(
                Circle(
                    (current_x, current_y),
                    radius=circle_radius,
                    facecolor="white",
                    edgecolor="black",
                    linewidth=0.6,
                    alpha=0.95,
                )
            )

            canvas.draw()
            buf = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
            image = buf.reshape(canvas.get_width_height()[::-1] + (3,))
            overlay_frames.append(image)
            plt.close(fig)

        overlay_array = np.stack(overlay_frames, axis=0)

        time_axis = np.arange(num_frames)
        dims = max(predictions.shape[1], ground_truth.shape[1])
        fig_plot, axes = plt.subplots(dims, 1, sharex=True, figsize=(10, 3 * dims))
        if dims == 1:
            axes = [axes]

        for axis_idx in range(dims):
            ax_plot = axes[axis_idx]
            gt_series = ground_truth[:, axis_idx] if axis_idx < ground_truth.shape[1] else None
            pred_series = predictions[:, axis_idx] if axis_idx < predictions.shape[1] else None
            if gt_series is not None:
                ax_plot.plot(time_axis, gt_series, label="Ground truth", color="tab:blue")
            if pred_series is not None:
                ax_plot.plot(time_axis, pred_series, label="Prediction", color="tab:orange", linestyle="--")
            ax_plot.set_ylabel(f"Axis {axis_idx}")
        axes[0].set_title("Action delta comparison")
        axes[-1].set_xlabel("Frame index")
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            axes[0].legend(handles, labels, loc="upper right")
        fig_plot.tight_layout()

        canvas_plot = FigureCanvasAgg(fig_plot)
        canvas_plot.draw()
        plot_array = np.frombuffer(canvas_plot.tostring_rgb(), dtype=np.uint8)
        plot_array = plot_array.reshape(canvas_plot.get_width_height()[::-1] + (3,))

        if save_plot_path is not None:
            save_plot_path = Path(save_plot_path)
            save_plot_path.parent.mkdir(parents=True, exist_ok=True)
            fig_plot.savefig(save_plot_path, dpi=200, bbox_inches="tight")

        if save_video_path is not None:
            save_video_path = Path(save_video_path)
            save_video_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                import imageio.v3 as iio
            except ImportError:
                try:
                    import imageio as iio  # type: ignore[no-redef]
                except ImportError as exc:
                    raise RuntimeError("imageio is required to save the visualization video.") from exc
            if save_video_path.suffix.lower() == ".gif":
                iio.imwrite(save_video_path, overlay_array, fps=fps, loop=0)
            else:
                with iio.get_writer(save_video_path, fps=fps) as writer:
                    for frame in overlay_array:
                        writer.append_data(frame)

        if not show_plot:
            plt.close(fig_plot)

        if show_animation:
            fig_anim, ax_anim = plt.subplots(
                figsize=(overlay_array.shape[2] / 100.0, overlay_array.shape[1] / 100.0), dpi=100
            )
            ax_anim.axis("off")
            image_artist = ax_anim.imshow(overlay_array[0])

            def _update(frame: np.ndarray):
                image_artist.set_data(frame)
                return (image_artist,)

            anim = animation.FuncAnimation(  # keep reference to avoid garbage collection
                fig_anim,
                _update,
                frames=overlay_array,
                interval=max(1, int(1000 / max(fps, 1))),
                repeat=True,
                blit=True,
            )
            plt.show()
            plt.close(fig_anim)
            if show_plot:
                plt.close(fig_plot)
        else:
            if show_plot:
                plt.show()
                plt.close(fig_plot)

        return overlay_array, plot_array

    def test_model(
        self,
        model_type: str,
        data_path: Union[str, Path],
        demo_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run inference over a dataset and collect error/latency metrics."""
        dataset = self._load_hdf5_sequence(Path(data_path), demo_id=demo_id)
        self.last_dataset = dataset
        self.agent_helper.prepare_model(model_type)

        goal = None
        if self.agent_helper.requires_goal:
            goal = np.asarray(dataset["pipette_positions"][-1], dtype=np.float32)

        self.predictions.clear()
        self.latencies_ms.clear()
        self.errors.clear()

        num_frames = dataset["images"].shape[0]
        for idx in range(num_frames):
            observation = (
                np.asarray(dataset["pipette_positions"][idx], dtype=np.float32),
                np.asarray(dataset["stage_positions"][idx], dtype=np.float32),
                np.asarray(dataset["images"][idx]),
                np.asarray(dataset["resistance"][idx], dtype=np.float32),
            )

            start = time.perf_counter()
            predicted_action = self.agent_helper.run_inference(
                observation=observation,
                goal=goal,
                is_demo=True,
            )
            latency_ms = (time.perf_counter() - start) * 1000.0

            self.latencies_ms.append(float(latency_ms))
            self.predictions.append(np.asarray(predicted_action, dtype=np.float32).reshape(-1))

        if not self.predictions:
            raise RuntimeError("No predictions were generated during testing")

        predictions_arr = np.asarray(self.predictions, dtype=np.float32)
        actions_arr = np.asarray(dataset["actions"], dtype=np.float32)

        min_count = min(predictions_arr.shape[0], actions_arr.shape[0])
        predictions_arr = predictions_arr[:min_count]
        actions_arr = actions_arr[:min_count]

        self.errors = self.compute_errors(predictions_arr, actions_arr)

        summary = {
            "demo_id": dataset["demo_id"],
            "predictions": predictions_arr,
            "ground_truth": actions_arr,
            "errors": self.errors,
            "mean_error": float(np.mean(self.errors)) if self.errors else float("nan"),
            "max_error": float(np.max(self.errors)) if self.errors else float("nan"),
            "latencies_ms": self.latencies_ms,
            "mean_latency_ms": float(np.mean(self.latencies_ms)) if self.latencies_ms else float("nan"),
        }
        self.last_results = summary


        return summary

    def main(
        self,
        model_type: str,
        data_path: Union[str, Path],
        demo_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Entry point used by the CLI stub below."""
        results = self.test_model(model_type=model_type, data_path=data_path, demo_id=demo_id)
    
        mean_error = results.get("mean_error", float("nan"))
        mean_latency = results.get("mean_latency_ms", float("nan"))
        print(
            f"[RESULT] Demo {results['demo_id']}: mean error={mean_error:.4f}, "
            f"max error={results['max_error']:.4f}, mean latency={mean_latency:.2f} ms"
        )
        self.visualize()
        return results


if __name__ == "__main__":
    agenttester = AgentTester()

    model_type = "find_pipette"
    data_path = r"C:\Users\sa-forest\Documents\GitHub\holypipette-pbl\experiments\Datasets\PatcherBot_test_dataset_v0_170\PatcherBot_test_dataset_v0_170_find_pipette.hdf5"
    # data_path = r"C:\Users\sa-forest\Documents\GitHub\holypipette-pbl\experiments\Datasets\PatcherBot_dataset_v0_160\PatcherBot_dataset_v0_160_find_pipette.hdf5"
    demo_id = "demo_1"
    agenttester.main(model_type=model_type, data_path=data_path, demo_id=demo_id)

