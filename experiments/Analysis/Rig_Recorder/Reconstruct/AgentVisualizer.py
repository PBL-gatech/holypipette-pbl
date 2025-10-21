#!/usr/bin/env python3

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, UnidentifiedImageError

# --- USER CONFIGURATION DEFAULTS ---
DEFAULT_INPUT_DIR = Path(r"C:\Users\sa-forest\Documents\GitHub\holypipette-pbl\experiments\Data\agent_movement_data\2025_10_20-21_35")
DEFAULT_FPS = 30.0
DEFAULT_RED_THRESHOLDS = (150, 100, 100)  # (r_min, g_max, b_max)
DEFAULT_AXIS_LIMIT = 85


def natural_sort_key(path: Path) -> List[object]:
    return [int(chunk) if chunk.isdigit() else chunk.lower() for chunk in re.split(r"(\d+)", path.name)]


class AgentVisualizer:
    """Build outputs (GIF, CSV, trajectory plot) for a directory of agent frames."""

    IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".gif", ".webp"}

    def __init__(
        self,
        input_dir: Path,
        fps: float = DEFAULT_FPS,
        red_thresholds: Sequence[int] = DEFAULT_RED_THRESHOLDS,
        axis_limit: int = DEFAULT_AXIS_LIMIT,
    ) -> None:
        self.input_dir = Path(input_dir).expanduser().resolve()
        self.fps = fps
        self.r_min, self.g_max, self.b_max = red_thresholds
        self.axis_limit = axis_limit

        self.base_name = self.input_dir.name
        self.output_dir = self.input_dir
        self.output_csv = self.output_dir / f"{self.base_name}.csv"
        self.output_plot = self.output_dir / f"{self.base_name}.png"
        self.output_gif = self.output_dir / f"{self.base_name}.gif"

    def collect_image_paths(self) -> List[Path]:
        return sorted(
            (p for p in self.input_dir.iterdir() if p.is_file() and p.suffix.lower() in self.IMAGE_EXTS),
            key=natural_sort_key,
        )

    def find_red_centroid(self, arr: np.ndarray) -> Optional[Tuple[int, int]]:
        red_mask = (arr[:, :, 0] >= self.r_min) & (arr[:, :, 1] <= self.g_max) & (arr[:, :, 2] <= self.b_max)
        coords = np.argwhere(red_mask)
        if coords.size == 0:
            return None
        y, x = coords.mean(axis=0)
        return int(round(x)), int(round(y))

    def save_csv(self, rows: Sequence[Sequence[object]]) -> None:
        with open(self.output_csv, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["filename", "x", "y"])
            writer.writerows(rows)
        print(f"Saved CSV with coordinates to: {self.output_csv}")

    def save_plot(self, trajectory: Sequence[Tuple[int, int]]) -> None:
        if not trajectory:
            print("No red dot detections found; trajectory plot skipped.")
            return

        xs, ys = zip(*trajectory)
        xs_arr = np.asarray(xs, dtype=float)
        ys_arr = np.asarray(ys, dtype=float)

        dx = np.diff(xs_arr)
        dy = np.diff(ys_arr)

        r = np.hypot(xs_arr, ys_arr)
        theta_rad = np.arctan2(ys_arr, xs_arr)
        theta_unwrapped = np.unwrap(theta_rad)
        dr = np.diff(r)
        dtheta = np.diff(theta_unwrapped)

        fig, axes = plt.subplots(2, 3, figsize=(14, 8), dpi=100)

        # Top-left (1,1): XY trajectory
        ax_traj = axes[0, 0]
        ax_traj.plot(xs_arr, ys_arr, marker="o")
        ax_traj.set_title(f"Agent trajectory ({self.base_name})")
        ax_traj.set_xlabel("X coordinate (pixels)")
        ax_traj.set_ylabel("Y coordinate (pixels)")
        ax_traj.set_xlim(0, self.axis_limit)
        ax_traj.set_ylim(self.axis_limit, 0)

        # Top-center (1,2): Histogram of delta X
        ax_hist_dx = axes[0, 1]
        if dx.size > 0:
            ax_hist_dx.hist(dx, bins="auto", color="tab:blue", alpha=0.8)
            ax_hist_dx.set_xlim(-4, 4)
        else:
            ax_hist_dx.text(0.5, 0.5, "Not enough data", ha="center", va="center", transform=ax_hist_dx.transAxes)
        ax_hist_dx.set_title("Histogram of delta X")
        ax_hist_dx.set_xlabel("delta X (pixels)")
        ax_hist_dx.set_ylabel("Count")

        # Top-right (1,3): Histogram of delta Y
        ax_hist_dy = axes[0, 2]
        if dy.size > 0:
            ax_hist_dy.hist(dy, bins="auto", color="tab:orange", alpha=0.8)
            ax_hist_dy.set_xlim(-4, 4)
        else:
            ax_hist_dy.text(0.5, 0.5, "Not enough data", ha="center", va="center", transform=ax_hist_dy.transAxes)
        ax_hist_dy.set_title("Histogram of delta Y")
        ax_hist_dy.set_xlabel("delta Y (pixels)")
        ax_hist_dy.set_ylabel("Count")

        # Bottom-left (2,1): Histogram of delta r
        ax_hist_dr = axes[1, 0]
        if dr.size > 0:
            ax_hist_dr.hist(dr, bins="auto", color="tab:green", alpha=0.8)
            ax_hist_dr.set_xlim(-4, 4)
        else:
            ax_hist_dr.text(0.5, 0.5, "Not enough data", ha="center", va="center", transform=ax_hist_dr.transAxes)
        ax_hist_dr.set_title("Histogram of delta r")
        ax_hist_dr.set_xlabel("delta r (pixels)")
        ax_hist_dr.set_ylabel("Count")

        # Bottom-center (2,2): Histogram of delta theta
        ax_hist_dtheta = axes[1, 1]
        if dtheta.size > 0:
            ax_hist_dtheta.hist(dtheta, bins="auto", color="tab:red", alpha=0.8)
            theta_extent = np.max(np.abs(dtheta))
            ax_hist_dtheta.set_xlim(-theta_extent - 1.0, theta_extent + 1.0)
        else:
            ax_hist_dtheta.text(0.5, 0.5, "Not enough data", ha="center", va="center", transform=ax_hist_dtheta.transAxes)
        ax_hist_dtheta.set_title("Histogram of delta theta")
        ax_hist_dtheta.set_xlabel("delta theta (radians)")
        ax_hist_dtheta.set_ylabel("Count")

        # Bottom-right (2,3): FFT magnitude of delta X and delta Y
        ax_fft = axes[1, 2]
        sample_spacing = 1.0 / self.fps if self.fps > 0 else 1.0
        if dx.size > 0:
            freq_dx = np.fft.rfftfreq(dx.size, d=sample_spacing)
            fft_dx = np.abs(np.fft.rfft(dx - np.mean(dx))) / max(dx.size, 1)
            ax_fft.plot(freq_dx, fft_dx, label="delta X", color="tab:blue")
        if dy.size > 0:
            freq_dy = np.fft.rfftfreq(dy.size, d=sample_spacing)
            fft_dy = np.abs(np.fft.rfft(dy - np.mean(dy))) / max(dy.size, 1)
            ax_fft.plot(freq_dy, fft_dy, label="delta Y", color="tab:orange", linestyle="--")
        if dx.size == 0 and dy.size == 0:
            ax_fft.text(0.5, 0.5, "Not enough data", ha="center", va="center", transform=ax_fft.transAxes)
        ax_fft.set_title("FFT of delta X/Y")
        ax_fft.set_xlabel("Frequency (Hz)")
        ax_fft.set_ylabel("Magnitude")
        if (dx.size > 0) or (dy.size > 0):
            ax_fft.set_xlim(left=0)
        if (dx.size > 0) or (dy.size > 0):
            ax_fft.legend()

        fig.tight_layout()
        fig.savefig(self.output_plot, dpi=100)
        plt.close(fig)

        print(f"Saved trajectory plot to: {self.output_plot}")

    def save_gif(self, frames: Sequence[Image.Image]) -> None:
        if not frames:
            print("No frames available for GIF; skipping GIF creation.", file=sys.stderr)
            return

        if self.fps <= 0:
            raise ValueError("FPS must be a positive number.")

        duration_ms = max(1, int(round(1000.0 / self.fps)))
        first_frame, *other_frames = frames
        first_frame.save(
            self.output_gif,
            format="GIF",
            save_all=True,
            append_images=other_frames,
            duration=duration_ms,
            loop=0,
            disposal=2,
        )

        print(
            f"Saved GIF to {self.output_gif} with {len(frames)} frames at {self.fps:.2f} fps "
            f"(frame duration {duration_ms} ms)."
        )

    def run(self) -> None:
        if not self.input_dir.is_dir():
            raise FileNotFoundError(f"Input directory not found: {self.input_dir}")

        image_paths = self.collect_image_paths()
        if not image_paths:
            raise FileNotFoundError(f"No supported images found in {self.input_dir}")

        rows = []
        trajectory: List[Tuple[int, int]] = []
        frames: List[Image.Image] = []

        for path in image_paths:
            try:
                with Image.open(path) as img:
                    rgb_img = img.convert("RGB")
                    arr = np.array(rgb_img)
                    centroid = self.find_red_centroid(arr)

                    if centroid:
                        x, y = centroid
                        rows.append([path.name, x, y])
                        trajectory.append((x, y))
                    else:
                        rows.append([path.name, "", ""])

                    frames.append(rgb_img.convert("RGBA"))
            except UnidentifiedImageError:
                print(f"Skipping unsupported image file: {path}", file=sys.stderr)
            except OSError as exc:
                print(f"Could not process {path}: {exc}", file=sys.stderr)

        if not rows and not frames:
            raise RuntimeError("No valid image data available after processing.")

        if rows:
            self.save_csv(rows)
        else:
            print("No coordinate data to write; CSV skipped.")

        self.save_plot(trajectory)
        self.save_gif(frames)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate GIF, CSV, and trajectory plot for agent frames.")
    parser.add_argument(
        "input_dir",
        nargs="?",
        type=Path,
        default=None,
        help="Directory containing frame images. Defaults to the configured path in the script.",
    )
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS, help="Frames per second for the output GIF.")
    parser.add_argument(
        "--r-min", type=int, default=DEFAULT_RED_THRESHOLDS[0], help="Minimum red channel value for detection."
    )
    parser.add_argument(
        "--g-max", type=int, default=DEFAULT_RED_THRESHOLDS[1], help="Maximum green channel value for detection."
    )
    parser.add_argument(
        "--b-max", type=int, default=DEFAULT_RED_THRESHOLDS[2], help="Maximum blue channel value for detection."
    )
    parser.add_argument(
        "--axis-limit",
        type=int,
        default=DEFAULT_AXIS_LIMIT,
        help="Upper bound for both axes in the trajectory plot.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    input_dir = args.input_dir or DEFAULT_INPUT_DIR
    if input_dir is None:
        print("An input directory must be provided.", file=sys.stderr)
        sys.exit(1)

    visualizer = AgentVisualizer(
        input_dir=input_dir,
        fps=args.fps,
        red_thresholds=(args.r_min, args.g_max, args.b_max),
        axis_limit=args.axis_limit,
    )
    visualizer.run()


if __name__ == "__main__":
    main()
