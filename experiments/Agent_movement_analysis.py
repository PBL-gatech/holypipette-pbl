"""
Entry point to visualize agent trajectories from a rig-recorder dataset.

This script sets up paths for importing the 'AgentVisualizer' module, 
defines user-configurable parameters, and runs the visualizer on the
specified input folder.

Configuration:
    INPUT_FOLDER (Path): Directory containing the dataset to visualize.
    FPS (float): Frame rate for playback.
    RED_THRESHOLDS (tuple[int, int, int]): Thresholds for detecting red pixels.
    AXIS_LIMIT (int): Maximum axis value for visual scaling.

Usage:
    Run this script directly to launch the visualizer with the configured parameters.
"""

from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parent
_VISUALIZER_DIR = _ROOT / "Analysis" / "Rig_Recorder" / "Reconstruct"
if str(_VISUALIZER_DIR) not in sys.path:
    sys.path.insert(0, str(_VISUALIZER_DIR))

from AgentVisualizer import AgentVisualizer  # type: ignore

# --- USER CONFIGURATION ---
INPUT_FOLDER = Path(r"C:\Users\sa-forest\Documents\GitHub\PatcherBot-Agent\testing\GAv0_400_2")
FPS = 30.0
RED_THRESHOLDS = (150, 100, 100)
AXIS_LIMIT = 85


if __name__ == "__main__":
    AgentVisualizer(
        input_dir=INPUT_FOLDER,
        fps=FPS,
        red_thresholds=RED_THRESHOLDS,
        axis_limit=AXIS_LIMIT,
    ).run()
