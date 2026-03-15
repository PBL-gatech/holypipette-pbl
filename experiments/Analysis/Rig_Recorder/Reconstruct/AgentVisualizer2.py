import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.cm as cm
import matplotlib.ticker as mticker
from pathlib import Path
from typing import List, Sequence

# ----------------------------
# File paths
# ----------------------------
CSV_DIR = Path(
    r"C:\Users\sa-forest\OneDrive - Georgia Institute of Technology\Documents\Grad-school\Gatech\Fall2025\ForestLab\ML\PatcherBotAgentHEKData_v0_001\PipetteFinderAgentData"
)
IMAGE_PATH = Path(
    r"C:\Users\sa-forest\OneDrive - Georgia Institute of Technology\Documents\Grad-school\Gatech\Fall2025\ForestLab\ML\PatcherBotAgentHEKData_v0_001\PipetteFinderAgentData\2900_1762553538.011042.webp"
)
OUT_OVERLAY = CSV_DIR / "overlay_dots_white_target_large.png"
OUT_ERROR = CSV_DIR / "error_dot_plot_twilight.png"
CALIBRATION_PATH = Path(
    r"C:\Users\sa-forest\Documents\GitHub\PatcherBot-Agent\experiments\Data\Calibration_data\2025_11_10-13_19\calibration.json"
)
MAX_SAMPLES = 50
GRID_SIZE = 85.0
CAMERA_WIDTH_PX = 1280.0
CAMERA_HEIGHT_PX = 1280.0
GRID_TO_PIXEL_SCALE = np.array(
    [CAMERA_WIDTH_PX / GRID_SIZE, CAMERA_HEIGHT_PX / GRID_SIZE], dtype=np.float64
)


def load_manipulator_inverse(calibration_path: Path) -> np.ndarray:
    """
    Load and invert the manipulator calibration matrix for px-to-um conversion.
    
    Args:
        calibration_path (Path): Path to the JSON calibration file containing
            the manipulator calibration data.

    Returns:
        np.ndarray: A 3x3 NumPy array representing the inverse manipulator
        calibration matrix. If the calibration cannot be loaded or inverted,
        an identity matrix is returned.

    Raises:
        KeyError: If the JSON file does not contain a "manip" entry or the
            expected "M" matrix within it.
    """
    try:
        with calibration_path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        manip = payload.get("manip")
        if not manip or "M" not in manip:
            raise KeyError("missing manipulator entry")
        matrix = np.asarray(manip["M"], dtype=np.float64)
        return np.linalg.inv(matrix)
    except Exception as exc:  # pragma: no cover - fall back to identity if calibration fails
        print(
            f"Warning: failed to load manipulator calibration from {calibration_path}: {exc}"
        )
        return np.eye(3, dtype=np.float64)


MANIP_MINV = load_manipulator_inverse(CALIBRATION_PATH)


def find_csv_files(csv_dir: Path) -> List[Path]:
    """
    Return all CSV files directly under the directory, sorted alphabetically.
    
    Args:
        csv_dir (Path): Directory to search for CSV files.

    Returns:
        List[Path]: A list of paths to CSV files located directly inside
        the specified directory, sorted alphabetically.
    """
    return sorted(p for p in csv_dir.glob("*.csv") if p.is_file())


def load_csv_data(csv_files: Sequence[Path]) -> List[dict]:
    """
    Load each CSV and keep its DataFrame and metadata together.
    
    Args:
        csv_files (Sequence[Path]): Sequence of paths to CSV files to load.

    Returns:
        List[dict]: A list of dictionaries where each dictionary contains:
            - "path": Path to the CSV file
            - "df": pandas DataFrame containing the loaded data.
    """
    datasets: List[dict] = []
    for csv_path in csv_files:
        df = pd.read_csv(csv_path).copy()
        datasets.append({"path": csv_path, "df": df})
    return datasets


def compute_error_microns(df: pd.DataFrame) -> np.ndarray:
    """
    Convert grid-based pipette errors to microns using calibration data.
    
    Args:
        df (pd.DataFrame): DataFrame containing coordinate columns
            'red_x', 'red_y', 'white_x', and 'white_y'.

    Returns:
        np.ndarray: Array of positional errors in microns. Returns an empty
        array if required columns are missing or if no valid data is present.
    """
    required = {"red_x", "red_y", "white_x", "white_y"}
    if not required.issubset(df.columns):
        return np.empty(0, dtype=np.float64)

    deltas = (
        df[["red_x", "red_y"]].astype(np.float64).values
        - df[["white_x", "white_y"]].astype(np.float64).values
    )
    if deltas.size == 0:
        return np.empty(0, dtype=np.float64)

    deltas_px = deltas * GRID_TO_PIXEL_SCALE
    deltas_px_3d = np.column_stack([deltas_px, np.zeros(len(deltas_px))])
    deltas_um = deltas_px_3d @ MANIP_MINV.T
    return np.linalg.norm(deltas_um[:, :2], axis=1)


def compute_elapsed_seconds(filenames: pd.Series) -> np.ndarray:
    """
    Return elapsed seconds derived from camera filename timestamps.
    
    Args:
        filenames (pd.Series): Series containing camera image filenames
            with embedded timestamps.

    Returns:
        np.ndarray: Array of elapsed time values in seconds corresponding
        to each filename. If no valid timestamps are found, a sequential
        array starting from zero is returned.
    """
    if filenames.empty:
        return np.empty(0, dtype=np.float64)

    matches = (
        filenames.astype(str)
        .str.extract(r"camera_image_(\d{8}_\d{6}_\d+)")[0]
        .astype("string")
    )
    timestamps = pd.to_datetime(
        matches, format="%Y%m%d_%H%M%S_%f", errors="coerce"
    )
    valid = timestamps.dropna()
    if valid.empty:
        return np.arange(len(filenames), dtype=np.float64)

    base_time = valid.iloc[0]
    elapsed = (timestamps - base_time).dt.total_seconds()
    elapsed = elapsed.ffill().fillna(0.0)
    elapsed = elapsed.clip(lower=0.0)
    return elapsed.to_numpy(dtype=np.float64)


def main() -> None:
    """Generate trajectory overlay and positional error plots from CSV datasets."""
    csv_files = find_csv_files(CSV_DIR)
    if not csv_files:
        print(f"No CSV files found in {CSV_DIR}")
        return

    print(f"Found {len(csv_files)} CSV file(s) in {CSV_DIR}")
    datasets = load_csv_data(csv_files)
    colors = (
        cm.twilight(np.linspace(0, 1, len(datasets), endpoint=False))
        if len(datasets) > 1
        else [cm.twilight(0.5)]
    )

    with Image.open(IMAGE_PATH) as pil_img:
        img = pil_img.copy()
    w, h = img.size
    scale_x = w / 85.0
    scale_y = h / 85.0

    plt.figure(figsize=(w / 100, h / 100), dpi=100)
    plt.imshow(img, origin="upper")

    for color, dataset in zip(colors, datasets):
        df = dataset["df"]
        red_x_px = df["red_x"].astype(float) * scale_x
        red_y_px = df["red_y"].astype(float) * scale_y
        white_x_px = df["white_x"].astype(float) * scale_x
        white_y_px = df["white_y"].astype(float) * scale_y

        plt.plot(
            red_x_px,
            red_y_px,
            color=color,
            linewidth=6.0,
            alpha=0.95,
        )
        plt.scatter(
            white_x_px,
            white_y_px,
            s=200,
            facecolors=color,
            edgecolors="white",
            linewidths=1.2,
            alpha=0.95,
        )

    plt.xlim(0, w)
    plt.ylim(h, 0)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(OUT_OVERLAY, bbox_inches="tight", pad_inches=0, dpi=300)
    plt.close()

    # --- GraphPad-like time series error plot (keeps data & colors) ---
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")

    max_elapsed = 0.0

    for color, dataset in zip(colors, datasets):
        df = dataset["df"].copy()
        err_um = compute_error_microns(df)
        elapsed_seconds = compute_elapsed_seconds(df["filename"])
        limit = min(MAX_SAMPLES, len(err_um), len(elapsed_seconds))
        if limit == 0:
            continue

        err_um = err_um[:limit]
        elapsed_seconds = elapsed_seconds[:limit]
        max_elapsed = max(max_elapsed, float(elapsed_seconds[-1]))

        ax.plot(elapsed_seconds, err_um, lw=3.75, color=color, alpha=0.95)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)

    ax.tick_params(axis="both", which="major", direction="out", length=6, width=1.2)

    ax.set_ylim(bottom=0)
    if max_elapsed > 0:
        ax.set_xlim(0, max_elapsed)
    ax.margins(x=0.02, y=0.05)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(6))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(6))

    ax.set_title("Error between pipette tip and target", pad=10)
    ax.set_xlabel("Elapsed time (s)")
    ax.set_ylabel("Error (microns)")

    fig.tight_layout()
    fig.savefig(OUT_ERROR, dpi=300, bbox_inches="tight", transparent=True)
    plt.close(fig)

    print(f"Saved plots in {CSV_DIR}:\n - Overlay: {OUT_OVERLAY}\n - Error:   {OUT_ERROR}")


if __name__ == "__main__":
    main()
