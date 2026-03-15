# Batch plotting for break_in + gigaseal HDF5 in a folder (four GraphPad-style charts).
# Output: Break-in Pressure, Break-in Resistance, Gigaseal Pressure, Gigaseal Resistance
# - Twilight colormap; paired colors across pressure/resistance within each dataset
# - Boxy aspect, outward ticks, thick spines, no grid, transparent background
# - Horizontal dotted line on resistance plots (100 break-in / 1000 gigaseal)
# - Optional filtering (toggle APPLY_FILTERING)

import os
import re
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

# -------- CONFIG --------
FOLDER = r"C:\Users\sa-forest\OneDrive - Georgia Institute of Technology\Documents\Grad-school\Gatech\Fall2025\ForestLab\ML\PatcherBotAgentHEKData_v0_001\PatcherBot_classic_dataset_v0_001"  # e.g., "/mnt/data"
X_LIMIT = 500               # samples to display on x-axis
COLORMAP_NAME = "twilight"
SAVE_PLOTS = True         # toggle auto-save
PLOT_OUTPUT_DIR = os.path.join(FOLDER, "plots")
PLOT_DPI = 300
PLOT_FORMAT = "png"

# Filtering (applies to resistance only)
APPLY_FILTERING = True      # toggle
FS_HZ = 30.0                # sampling rate (Hz)
FC_HZ = 10.0                # Bessel cutoff (Hz) < FS/2
BESSEL_ORDER = 8            # 8-pole
OUTLIER_MIN = 0.0           # invalid resistance < 0
OUTLIER_MAX = 10_000.0      # invalid resistance > 10k

# Optional deps for filtering
try:
    import pandas as pd
    from scipy.signal import bessel, filtfilt
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

def interpolate_nans(y):
    """
    Linear interpolation across NaNs to keep spacing.
    
    Args:
        y (array-like): Numeric sequences possibly containing NaN values.
    
    Returns:
        np.ndarray: Array with NaN values linearly interpolated where possible.

    Raises:
        TypeError: If `y` cannot be interpreted as a numeric sequence.
    """
    if 'pd' in globals():
        return pd.Series(y).interpolate(method="linear", limit_direction="both").to_numpy()
    x = np.arange(len(y))
    mask = ~np.isnan(y)
    if mask.sum() < 2:
        return y
    return np.interp(x, x[mask], y[mask])

def bessel_lowpass_zero_phase(x, fs, fc, order):
    """Digital Bessel LPF + zero-phase filtering (SciPy)."""
    wn = fc / (fs / 2.0)
    b, a = bessel(order, wn, btype="low", analog=False, norm="phase")
    return filtfilt(b, a, x)

def style_axes(ax):
    """
    GraphPad-style cosmetics.
    
    Args:
        ax (matplotlib.axes.Axes): The Axes object to style.

    Raises:
        AttributeError: If `ax` does not support spine or tick configuration methods.
    """
    for spine in ax.spines.values():
        spine.set_linewidth(2.0)
    ax.xaxis.set_tick_params(direction='out', width=1.8, length=6)
    ax.yaxis.set_tick_params(direction='out', width=1.8, length=6)
    ax.grid(False)
    ax.set_facecolor('none')   # transparent axes bg

def save_plot(fig, label, suffix):
    """
    Persist a Matplotlib figure using sanitized filenames when SAVE_PLOTS is True.
    
    Args:
        fig (matplotlib.figure.Figure): The figure object to save.
        label (str): Primary label used to build the output filename.
        suffix (str): Additional descriptor appended to the filename.

    Raises:
        OSError: If the output directory cannot be created or the file cannot be written.
        TypeError: If `fig` is not a valid Matplotlib Figure object.
    """
    if not SAVE_PLOTS:
        return

    def _slug(text):
        cleaned = re.sub(r"[^a-z0-9]+", "_", text.strip().lower())
        return cleaned.strip("_") or "plot"

    os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)
    filename = f"{_slug(label)}_{_slug(suffix)}.{PLOT_FORMAT}"
    fig.savefig(
        os.path.join(PLOT_OUTPUT_DIR, filename),
        dpi=PLOT_DPI,
        transparent=True,
        bbox_inches="tight",
    )

def find_demos(h5):
    """
    Return the sorted list of demonstration group names in an HDF5 dataset.

    Args:
        h5: An open HDF5 file or group object containing a "/data" group.

    Returns:
        List[str]: Sorted list of demo group names (e.g., ["demo_0", "demo_1", ...]).
    """
    return sorted([k for k in h5["/data"].keys() if k.startswith("demo_")])

def plot_dataset(file_path, label, resistance_hline):
    """
    Plot resistance traces for all demonstrations in an HDF5 dataset.

    Args:
        file_path (str | Path): Path to the HDF5 dataset file.
        label (str): Label used for the dataset in the plot legend or title.
        resistance_hline (float): Resistance threshold value to as a horizontal
            reference line on the plot.
    """
    with h5py.File(file_path, "r") as h5:
        demos = find_demos(h5)
        cmap = get_cmap(COLORMAP_NAME, len(demos) if len(demos) > 0 else None)

        # ----- Pressure -----
        fig_p, ax_p = plt.subplots(figsize=(6, 6), facecolor='none')
        for i, demo in enumerate(demos):
            grp = h5[f"/data/{demo}/obs"]
            if "pressure" not in grp:
                continue
            p = grp["pressure"][()].reshape(-1).astype(float)[:X_LIMIT]
            ax_p.plot(np.arange(len(p)), p, color=cmap(i), linewidth=2.5, alpha=0.9)
        ax_p.set_xlabel("Sample #", fontsize=12)
        ax_p.set_ylabel("Pressure", fontsize=12)
        ax_p.set_title(f"{label} — Pressure (first {X_LIMIT} samples)", fontsize=14, pad=10)
        style_axes(ax_p)
        ax_p.set_box_aspect(1)
        fig_p.patch.set_alpha(0)
        fig_p.tight_layout()
        save_plot(fig_p, label, "pressure")

        # ----- Resistance -----
        fig_r, ax_r = plt.subplots(figsize=(6, 6), facecolor='none')
        for i, demo in enumerate(demos):
            grp = h5[f"/data/{demo}/obs"]
            if "resistance" not in grp:
                continue
            r = grp["resistance"][()].reshape(-1).astype(float)
            if APPLY_FILTERING:
                r[(r > OUTLIER_MAX) | (r < OUTLIER_MIN)] = np.nan
                r = interpolate_nans(r)
                if SCIPY_OK:
                    r = bessel_lowpass_zero_phase(r, FS_HZ, FC_HZ, BESSEL_ORDER)
            r = r[:X_LIMIT]
            ax_r.plot(np.arange(len(r)), r, color=cmap(i), linewidth=2.5, alpha=0.9)
        ax_r.axhline(y=resistance_hline, color='gray', linestyle=':', linewidth=1.8)
        ax_r.set_xlabel("Sample #", fontsize=12)
        ylab = "Resistance"
        if APPLY_FILTERING and SCIPY_OK:
            ylab += f" (Bessel LPF {int(FC_HZ)} Hz)"
        ax_r.set_ylabel(ylab, fontsize=12)
        ax_r.set_title(f"{label} — Resistance (first {X_LIMIT} samples{'; filtered' if APPLY_FILTERING else ''})",
                       fontsize=14, pad=10)
        style_axes(ax_r)
        ax_r.set_box_aspect(1)
        fig_r.patch.set_alpha(0)
        fig_r.tight_layout()
        save_plot(fig_r, label, "resistance")


# ---- Discover the two datasets in FOLDER ----
break_in_fp = None
gigaseal_fp = None
for name in os.listdir(FOLDER):
    low = name.lower()
    if low.endswith(".hdf5") and "break_in" in low:
        break_in_fp = os.path.join(FOLDER, name)
    if low.endswith(".hdf5") and "gigaseal" in low:
        gigaseal_fp = os.path.join(FOLDER, name)

if break_in_fp:
    plot_dataset(break_in_fp, "Break-in", resistance_hline=250)

if gigaseal_fp:
    plot_dataset(gigaseal_fp, "Gigaseal", resistance_hline=1000)

plt.show()
