#!/usr/bin/env python3
"""
Multi-CSV categorical Gantt charts (combined on one plot), no argparse.

Edit the CONFIG section below:
- FOLDER: path to the folder containing your CSVs
- PATTERN: glob for which files to include (e.g., "*.csv")
- EXCLUDE_METHODS: list of method names to remove (case-insensitive)
- Column names: ATTEMPT_COL, METHOD_COL, START_COL, END_COL

Four PNGs are produced:
- OUT_TIME:    combined_gantt_by_method.png
- OUT_ZEROED:  combined_gantt_by_method_zeroed.png
- OUT_SUCCESS: attempt_cumulative_success.png
- OUT_AVERAGE: average_gantt.png

Same attempt numbers from different files are distinguished by appending the
attempt's FIRST start time to the lane label, e.g. "attempt_12 @ 2025-11-03 01:05".
Ordering is numeric attempt ascending, then first-start-time ascending.
"""

from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
from matplotlib import transforms


# ============================ CONFIG ============================

FOLDER = Path(r"C:\Users\sa-forest\OneDrive - Georgia Institute of Technology\Documents\Grad-school\Gatech\Fall2025\ForestLab\ML\PatcherBotAgentHEKData_v0_001")     # <-- change to your folder
PATTERN = "*.csv"                 # which files to read
EXCLUDE_METHODS = ["patch"]       # remove these methods (case-insensitive)
REQUIRED_METHODS = [              # attempts must include all of these; empty list to disable
    "locate cell",
    "hunt cell",
    "gigaseal",
    "break in",
    "run",
    "escape",
]

# CSV column names
ATTEMPT_COL = "attempt"
METHOD_COL  = "method"
START_COL   = "start_utc"
END_COL     = "end_utc"

# Output files
OUT_TIME      = FOLDER / "combined_gantt_by_method.png"
OUT_ZEROED    = FOLDER / "combined_gantt_by_method_zeroed.png"
OUT_SUCCESS   = FOLDER / "attempt_cumulative_success.png"
OUT_AVERAGE   = FOLDER / "average_gantt.png"

# Bar styling
BAR_HEIGHT = 0.8
BAR_ALPHA  = 0.9

# Shared styling parameters for polished axes in summary plots.
PRISM_RC_PARAMS = {
    "font.family": "Arial",
    "axes.labelsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "axes.linewidth": 1.6,
}
PRISM_TICK_LENGTH = 8
PRISM_TICK_WIDTH = 1.8
PRISM_TICK_PAD = 6


def apply_prism_axes_style(ax):
    """
    Apply consistent fonts, spine widths, and tick spacing.
    
    Args:
        ax (matplotlib.axes.Axes): The Axes object to style.
    
        Returns:
            matplotlib.spines.Spine: The left spine of the Axes after styling,
                which can be further customized if needed.
        
        Raises:
            AttributeError: If the provided `ax` does not have spines for tick_params.
    """
    plt.rcParams.update(PRISM_RC_PARAMS)
    for side in ("right", "top"):
        ax.spines[side].set_visible(False)
    left_spine = ax.spines["left"]
    left_spine.set_linewidth(PRISM_TICK_WIDTH)
    ax.spines["bottom"].set_linewidth(PRISM_TICK_WIDTH)
    ax.tick_params(
        direction="out",
        width=PRISM_TICK_WIDTH,
        length=PRISM_TICK_LENGTH,
        pad=PRISM_TICK_PAD,
    )
    return left_spine

# ===============================================================


def normalize_method(name):
    """
    Lowercase + spacing normalized method names for consistent comparisons.
    
    Args:
        name (Any): The input method name, which can be a string, float (NaN), or None.
    
    Returns:
        str: A normalized string suitable for consistent comparisons.
    """
    if name is None or (isinstance(name, float) and np.isnan(name)):
        return ""
    if pd.isna(name):
        return ""
    text = str(name).strip().lower().replace("_", " ")
    return re.sub(r"\s+", " ", text)


def extract_first_int(s):
    """
    Extract first integer from a string like 'attempt_12' -> 12 (or None).
    
    Args:
        s (Any): Input string to search for an integer.

    Returns:
        int | None: The first integer found in the string, or None if no integer exists.
    """
    s = str(s)
    m = re.search(r"-?\d+", s)
    return int(m.group(0)) if m else None


def method_display_color(method_name: str, palette):
    """
    Return a palette color (normalized by method name) or fallback.
    
    Args:
        method_name (str): The method name to look up.
        palette (dict): Dictionary mapping normalized method names to hex colors.
        
    Returns:
        str | None: The hex color corresponding to the method, or the fallback color.

    Raises:
        TypeError: If `palette` is not a dictionary. 
    """
    if not palette:
        return None
    norm = normalize_method(method_name)
    if norm and norm in palette:
        return palette[norm]
    return palette.get("__fallback__")


def build_method_palette(data: pd.DataFrame) -> dict:
    """
    Assign each normalized method a color sampled from the cividis colormap.
    
    Args:
        data (pd.DataFrame): DataFrame containing method names in column `METHOD_COL`.

    Returns:
        dict: Mapping of normalized method names to hex color codes, including "__fallback__".
    """
    method_series = (
        data[METHOD_COL]
        .dropna()
        .astype(str)
        .map(str.strip)
    )
    norm_methods = []
    seen = set()
    for name in method_series:
        norm = normalize_method(name)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        norm_methods.append(norm)

    palette = {}
    if norm_methods:
        cmap = plt.get_cmap("twilight_shifted")
        positions = np.linspace(0.1, 0.9, len(norm_methods))
        palette.update(
            {
                norm: mcolors.to_hex(cmap(pos))
                for norm, pos in zip(norm_methods, positions)
            }
        )
    fallback_color = mcolors.to_hex(plt.get_cmap("cividis")(0.05))
    palette["__fallback__"] = fallback_color
    return palette


def read_one_csv(path: Path) -> pd.DataFrame:
    """
    Load one CSV and return parsed/filtered rows with a source_file column.
    
    Args:
        path (Path): Path to the CSV file.
    
    Returns:
        pd.DataFrame: Filtered DataFrame with columns [ATTEMPT_COL, METHOD_COL, START_COL, END_COL,
                        "__start_ts__", "__end_ts__", "source_file"].
   
    Raises:
        KeyError: If any required column (ATTEMPT_COL, METHOD_COL, START_COL, END_COL) is missing.
    """
    df = pd.read_csv(path)

    # Fail fast if columns are missing
    for col in (ATTEMPT_COL, METHOD_COL, START_COL, END_COL):
        if col not in df.columns:
            raise KeyError(f"{path.name} is missing required column: {col}")

    # Parse datetimes; keep rows that have both start & end
    start_ts = pd.to_datetime(df[START_COL], errors="coerce")
    end_ts   = pd.to_datetime(df[END_COL], errors="coerce")
    mask = start_ts.notna() & end_ts.notna()
    if not mask.any():
        return pd.DataFrame(columns=[ATTEMPT_COL, METHOD_COL, START_COL, END_COL, "source_file",
                                     "__start_ts__", "__end_ts__"])

    gdf = df.loc[mask, [ATTEMPT_COL, METHOD_COL, START_COL, END_COL]].copy()
    gdf["__start_ts__"] = start_ts.loc[mask]
    gdf["__end_ts__"]   = end_ts.loc[mask]
    gdf["source_file"]  = path.name
    return gdf


def load_folder(folder: Path) -> pd.DataFrame:
    """
    Read all CSVs, concatenate, drop excluded methods, compute numeric attempt and labels.
    
    Args:
        folder (Path): Path to the folder containing CSV files to process.

    Returns:
        pd.DataFrame: A DataFrame containing:
            - Original CSV data with additional columns:
                - "__method_norm__": Normalized method name.
                - "__is_complete__": Boolean flag indicating whether the attempt contains all required methods.
                - "__attempt_num__": Numeric attempt index for ordering.
                - "__attempt_start__": Timestamp of the first start for the attempt.
                - "__attempt_label__": Unique attempt label combining attempt number and first start time.

    Raises:
        FileNotFoundError: If no CSV files matching the pattern are found in the folder.
        RuntimeError: If none of the CSV files contain valid rows with parseable start/end times.
    """
    files = sorted(folder.glob(PATTERN))
    if not files:
        raise FileNotFoundError(f"No files matching {PATTERN!r} in {str(folder)!r}")

    parts = []
    for f in files:
        gdf = read_one_csv(f)
        if not gdf.empty:
            parts.append(gdf)

    if not parts:
        raise RuntimeError("No valid rows with parseable start/end times across all files.")

    data = pd.concat(parts, ignore_index=True)

    data["__method_norm__"] = data[METHOD_COL].map(normalize_method)

    # Exclude methods (case-insensitive)
    if EXCLUDE_METHODS:
        ex = {m.lower() for m in EXCLUDE_METHODS}
        data = data[data[METHOD_COL].astype(str).str.lower().map(lambda x: x not in ex)]

    group_cols = ["source_file", ATTEMPT_COL]
    if REQUIRED_METHODS:
        required_norms = {normalize_method(m) for m in REQUIRED_METHODS if m}

        def has_required(series):
            present = {val for val in series if val}
            return required_norms.issubset(present)

        complete_mask = data.groupby(group_cols)["__method_norm__"].transform(has_required)
    else:
        complete_mask = pd.Series(True, index=data.index)

    data["__is_complete__"] = complete_mask.astype(bool)

    # Numeric attempt for ordering
    data["__attempt_num__"] = data[ATTEMPT_COL].apply(extract_first_int)

    # First start per (source_file, attempt) to disambiguate lanes across files
    first_start = (
        data.groupby(["source_file", ATTEMPT_COL])["__start_ts__"].transform("min")
    )
    data["__attempt_start__"] = first_start

    # Unique lane label: attempt + first-start time (down to minutes for readability)
    data["__attempt_label__"] = (
        data[ATTEMPT_COL].astype(str)
        + " @ "
        + data["__attempt_start__"].dt.strftime("%Y-%m-%d %H:%M")
    )

    return data


def attempt_order(data: pd.DataFrame):
    """
    Ordering: chronological by first start time; assign sequential attempt labels.
    
    Args:
        data (pd.DataFrame): Input DataFrame containing experiment attempts with the
            following required columns:
                - "source_file": Source CSV or file identifier.
                - ATTEMPT_COL: Original attempt identifier column.
                - "__attempt_label__": Original unique attempt label.
                - "__attempt_num__": Numeric attempt index.
                - "__attempt_start__": Timestamp of the first start for the attempt.

    Returns:
        tuple[list[str], dict[str, str]]: 
            - List of new sequential attempt labels in chronological order.
            - Dictionary mapping original "__attempt_label__" values to the new sequential labels.
    """
    order_meta = (
        data[["source_file", ATTEMPT_COL, "__attempt_label__", "__attempt_num__", "__attempt_start__"]]
        .drop_duplicates()
        .sort_values(
            by=["__attempt_start__", "__attempt_num__", "source_file", ATTEMPT_COL, "__attempt_label__"],
            ascending=[True, True, True, True, True],
        )
    )
    new_labels = [f"Attempt {i + 1}" for i in range(len(order_meta))]
    label_map = dict(zip(order_meta["__attempt_label__"], new_labels))
    return new_labels, label_map


def draw_layered_by_method_datetime(
    data: pd.DataFrame,
    attempt_labels: list,
    attempt_index: dict,
    out_path: Path,
    method_palette: dict,
):
    """
    Combined Gantt with datetime x-axis; layer methods by total duration (longer first).
    
    Args:
        data (pd.DataFrame): DataFrame containing attempt records with at least the columns:
            - "__start_ts__": Attempt start datetime.
            - "__end_ts__": Attempt end datetime.
            - "__attempt_label__": Unique attempt label for plotting on the y-axis.
            - "__is_complete__": Boolean indicating whether the attempt is complete.
            - METHOD_COL: Method name associated with each attempt segment.
        attempt_labels (list): Ordered list of attempt labels to display on the y-axis.
        attempt_index (dict): Mapping from attempt labels to integer positions for plotting.
        out_path (Path): File path where the resulting figure will be saved.
        method_palette (dict): Mapping from method names to colors for consistent coloring.
    """
    width_days = (data["__end_ts__"] - data["__start_ts__"]).dt.total_seconds() / (24 * 3600)
    data = data.assign(__width_days__=width_days)

    complete = data[data["__is_complete__"]]
    incomplete = data[~data["__is_complete__"]]

    if not complete.empty:
        method_stats = (
            complete.groupby(METHOD_COL, dropna=False)["__width_days__"]
                .agg(total_days="sum", segments="count")
                .reset_index()
                .sort_values(["total_days", "segments"], ascending=[False, False])
        )
        draw_methods = method_stats[METHOD_COL].astype(str).tolist()
    else:
        draw_methods = []

    fig_height = max(1.0, 0.1 * len(attempt_labels))
    fig, ax = plt.subplots(figsize=(14, fig_height))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")

    for method in draw_methods:
        sub = complete[complete[METHOD_COL].astype(str) == method]
        if sub.empty:
            continue
        y = sub["__attempt_label__"].map(attempt_index).values
        lefts = mdates.date2num(sub["__start_ts__"].values)
        widths = mdates.date2num(sub["__end_ts__"].values) - lefts
        color = method_display_color(method, method_palette)
        bar_kwargs = dict(
            height=BAR_HEIGHT,
            align="center",
            alpha=BAR_ALPHA,
            label=method,
        )
        if color is not None:
            bar_kwargs["color"] = color
        ax.barh(y, widths, left=lefts, **bar_kwargs)

    if not incomplete.empty:
        sub = incomplete
        y = sub["__attempt_label__"].map(attempt_index).values
        lefts = mdates.date2num(sub["__start_ts__"].values)
        widths = mdates.date2num(sub["__end_ts__"].values) - lefts
        ax.barh(
            y,
            widths,
            left=lefts,
            height=BAR_HEIGHT,
            align="center",
            alpha=BAR_ALPHA,
            color="red",
            edgecolor="none",
            label="incomplete attempt",
        )

    ax.set_yticks(list(range(len(attempt_labels))), labels=attempt_labels)
    ax.set_ylabel("Attempt #)")
    ax.invert_yaxis()

    ax.set_xlabel("Time (minutes)")
    ax.set_title("Method Attempt times")
    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    ax.legend(title="Method / status", loc="best", ncol=2)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight", transparent=True)
    plt.close(fig)


def draw_zeroed_by_method_minutes(
    data: pd.DataFrame,
    attempt_labels: list,
    attempt_index: dict,
    out_path: Path,
    method_palette: dict,
):
    """
    Combined Gantt with x-axis in minutes since each attempt's first start (t=0 per attempt).
    
    Args:
        data (pd.DataFrame): DataFrame containing attempt records with at least the columns:
            - "__start_ts__": Attempt start datetime.
            - "__end_ts__": Attempt end datetime.
            - "__attempt_start__": First start datetime of the attempt for zeroing the x-axis.
            - "__attempt_label__": Unique attempt label for plotting on the y-axis.
            - "__is_complete__": Boolean indicating whether the attempt is complete.
            - METHOD_COL: Method name associated with each attempt segment.
        attempt_labels (list): Ordered list of attempt labels to display on the y-axis.
        attempt_index (dict): Mapping from attempt labels to integer positions for plotting.
        out_path (Path): File path where the resulting figure will be saved.
        method_palette (dict): Mapping from method names to colors for consistent coloring.

    Raises:
        KeyError: If any required columns ("__start_ts__", "__end_ts__", "__attempt_start__",
            "__attempt_label__", "__is_complete__", METHOD_COL) are missing from the DataFrame.
        ValueError: If `attempt_labels` or `attempt_index` do not align with the DataFrame's attempts.
    """
    left_min  = (data["__start_ts__"] - data["__attempt_start__"]).dt.total_seconds() / 60.0
    right_min = (data["__end_ts__"]   - data["__attempt_start__"]).dt.total_seconds() / 60.0
    data = data.assign(__left_min__=left_min, __width_min__=(right_min - left_min))

    complete = data[data["__is_complete__"]]
    incomplete = data[~data["__is_complete__"]]

    if not complete.empty:
        method_stats = (
            complete.groupby(METHOD_COL, dropna=False)["__width_min__"]
                .agg(total_minutes="sum", segments="count")
                .reset_index()
                .sort_values(["total_minutes", "segments"], ascending=[False, False])
        )
        draw_methods = method_stats[METHOD_COL].astype(str).tolist()
    else:
        draw_methods = []

    spacing = 0.24
    y_positions = {lab: i * spacing for i, lab in enumerate(attempt_labels)}
    fig_height = max(1.2, spacing * max(1, len(attempt_labels)) + 0.6)
    fig, ax = plt.subplots(figsize=(7, fig_height))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    bar_height = spacing * 0.30  # slightly thicker line-like bars

    for method in draw_methods:
        sub = complete[complete[METHOD_COL].astype(str) == method]
        if sub.empty:
            continue
        y = sub["__attempt_label__"].map(y_positions).values
        color = method_display_color(method, method_palette)
        bar_kwargs = dict(
            height=bar_height,
            align="center",
            alpha=BAR_ALPHA,
            label=method,
        )
        if color is not None:
            bar_kwargs["color"] = color
        ax.barh(
            y,
            sub["__width_min__"].values,
            left=sub["__left_min__"].values,
            **bar_kwargs,
        )

    if not incomplete.empty:
        sub = incomplete
        y = sub["__attempt_label__"].map(y_positions).values
        ax.barh(
            y,
            sub["__width_min__"].values,
            left=sub["__left_min__"].values,
            height=bar_height,
            align="center",
            alpha=BAR_ALPHA,
            color="red",
            edgecolor="none",
            label="incomplete attempt",
        )

    tick_positions = [y_positions[lab] for lab in attempt_labels]
    tick_labels = [re.sub(r"^attempt\s*", "", lab, flags=re.IGNORECASE) for lab in attempt_labels]
    ax.set_yticks(tick_positions, labels=tick_labels)
    ax.set_ylabel("Attempt (num)")
    ax.invert_yaxis()

    ax.set_xlabel("Time since start (minutes)")
    ax.set_title("Method Attempt times")
    ax.legend(title="Method / status", loc="best", ncol=2)
    max_minutes = max(5.0, float(right_min.max()))
    ax.set_xlim(0, max_minutes)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight", transparent=True)
    plt.close(fig)


def draw_average_attempt_gantt(
    data: pd.DataFrame,
    out_path: Path,
    method_palette: dict,
):
    """
    Single-row Gantt showing the mean timing and annotating per-method duration SD.
    
    Args:
        data (pd.DataFrame): DataFrame containing attempt records with at least the columns:
            - "__start_ts__": Attempt start datetime.
            - "__end_ts__": Attempt end datetime.
            - "__attempt_start__": First start datetime of the attempt for zeroing the x-axis.
            - "__is_complete__": Boolean indicating whether the attempt is complete.
            - METHOD_COL: Method name associated with each attempt segment.
        out_path (Path): File path where the resulting figure will be saved.
        method_palette (dict): Mapping from method names to colors for consistent coloring.

    Raises:
        RuntimeError: If there are no complete attempts with finite durations to compute 
            the average Gantt or if the resulting summary after grouping by method is empty.
        KeyError: If any required columns ("__start_ts__", "__end_ts__", "__attempt_start__", 
            "__is_complete__", METHOD_COL) are missing from the DataFrame.
    """
    left_sec = (data["__start_ts__"] - data["__attempt_start__"]).dt.total_seconds()
    right_sec = (data["__end_ts__"] - data["__attempt_start__"]).dt.total_seconds()
    data = data.assign(__left_sec__=left_sec, __width_sec__=(right_sec - left_sec))

    complete = data[data["__is_complete__"]].copy()
    complete = complete[complete["__width_sec__"] > 0]
    if complete.empty:
        raise RuntimeError("No complete attempts with finite durations to compute average Gantt.")

    complete["__method_display__"] = complete[METHOD_COL].fillna("(unknown)").astype(str)

    summary = (
        complete.groupby("__method_display__", dropna=False)[["__left_sec__", "__width_sec__"]]
        .agg(
            avg_left=("__left_sec__", "mean"),
            avg_width=("__width_sec__", "mean"),
            std_width=("__width_sec__", "std"),
        )
        .reset_index()
        .sort_values("avg_left", kind="mergesort")
    )
    if summary.empty:
        raise RuntimeError("Average Gantt summary is empty after grouping by method.")
    summary["std_width"] = summary["std_width"].fillna(0.0)

    fig, ax = plt.subplots(figsize=(7.0, 1.8))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    apply_prism_axes_style(ax)

    avg_bar_height = BAR_HEIGHT / 2.0
    y_value = 0.0
    for _, row in summary.iterrows():
        method = row["__method_display__"]
        color = method_display_color(method, method_palette)
        bar_kwargs = dict(
            height=avg_bar_height,
            align="center",
            alpha=BAR_ALPHA,
        )
        if color is not None:
            bar_kwargs["color"] = color
        ax.barh(
            y_value,
            row["avg_width"],
            left=row["avg_left"],
            **bar_kwargs,
        )

        std_width = float(row["std_width"])
        label = f"SD={std_width:.1f}s"
        ax.annotate(
            label,
            xy=(row["avg_left"] + row["avg_width"] / 2.0, y_value),
            xytext=(0, 12),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#111111",
        )

    ax.set_yticks([])
    pad = max(0.05, avg_bar_height / 2.0)
    ax.set_ylim(-avg_bar_height - pad, avg_bar_height + pad)
    ax.set_ylabel("")
    ax.set_xlabel("Time since start (s)")
    ax.set_title("Mean Durations for Successful Attempts")
    avg_finish = summary["avg_left"] + summary["avg_width"]
    max_seconds = max(30.0, float(avg_finish.max()))
    tick_step = 30.0
    left_bound = max(0.0, float(summary["avg_left"].min()) - 20.0)
    right_bound = max_seconds - 20.0
    if right_bound <= left_bound:
        right_bound = left_bound + tick_step
    ax.set_xlim(left_bound, right_bound)
    ax.set_xticks(np.arange(left_bound, right_bound + tick_step, tick_step))

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight", transparent=True)
    plt.close(fig)


def draw_cumulative_success_rate(
    data: pd.DataFrame,
    attempt_labels: list,
    out_path: Path,
):
    """
    Plot cumulative success rate vs attempt index (Attempt 1..N).
    
    Args:
        data (pd.DataFrame): DataFrame containing at least the columns:
            - "__attempt_label__": Unique attempt identifier.
            - "__is_complete__": Boolean indicating if the attempt was successfully completed.
        attempt_labels (list): Ordered list of attempt labels corresponding to the x-axis.
        out_path (Path): File path to save the generated figure.

    Raises:
        RuntimeError: If `attempt_labels` is empty.
        KeyError: If required columns "__attempt_label__" or "__is_complete__" are missing 
            from `data`.
    """
    if not attempt_labels:
        raise RuntimeError("No attempts available to plot cumulative success rate.")

    meta = (
        data.groupby("__attempt_label__")
            .agg(is_complete=("__is_complete__", "first"))
            .reindex(attempt_labels)
    )
    success_flags = meta["is_complete"].fillna(False).astype(bool)
    attempt_numbers = np.arange(1, len(success_flags) + 1)
    cumulative_successes = success_flags.astype(int).cumsum()
    success_rate = cumulative_successes / attempt_numbers

    PRISM_LINE_LW = 2.4
    PRISM_MARKER_SIZE = 6.0
    PRISM_MARKER_EDGE = 2.1
    SUCCESS_COLOR = "#003057"
    FAIL_COLOR = "red"
    UPPER_SEGMENT_MIN = 0.80  # First visible tick above the break
    BREAK_Y = 0.70  # Height at which the y-axis break is indicated
    VISIBLE_MIN = BREAK_Y - 0.07  # Extend ylim lower so the break floats above the axis base
    BREAK_SLASH_DY = 0.02  # Vertical span of each slash (data units)
    BREAK_SLASH_GAP = 0.02  # Vertical separation between the two slashes
    BREAK_SLASH_DX = 0.014  # Half-width of each slash (axes fraction)
    BREAK_AXIS_SEGMENT_PAD = 0.0  # Optional cushion so manual spine segments stop shy of the slashes

    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    left_spine = apply_prism_axes_style(ax)

    line = ax.plot(
        attempt_numbers,
        success_rate,
        color=SUCCESS_COLOR,
        lw=PRISM_LINE_LW,
        marker="o",
        ms=PRISM_MARKER_SIZE,
        mec=SUCCESS_COLOR,
        mew=PRISM_MARKER_EDGE,
        mfc=SUCCESS_COLOR,
        solid_capstyle="round",
        label="Cumulative success rate",
    )[0]

    fail_mask = ~success_flags.values
    failed_idx = attempt_numbers[fail_mask]
    fail_rates = success_rate[fail_mask]
    if len(failed_idx) > 0:
        fail_scatter = ax.scatter(
            failed_idx,
            fail_rates,
            color=FAIL_COLOR,
            s=(PRISM_MARKER_SIZE**2),
            zorder=3,
        )
        handles = [line, fail_scatter]
        labels = ["Cumulative success rate", "Attempt incomplete"]
    else:
        handles = [line]
        labels = ["Cumulative success rate"]

    x_max = max(2.0, attempt_numbers[-1] + 2.0)
    ax.set_xlim(0, x_max)
    ax.set_xticks(attempt_numbers)

    ax.set_ylim(VISIBLE_MIN, 1.05)
    ax.set_yticks([UPPER_SEGMENT_MIN, 0.9, 1.0])
    ax.set_ylabel("Cumulative Success Rate")
    ax.set_xlabel("Attempt Number")
    ax.set_title("Cumulative Attempt Success Rate")

    target_line = ax.axhline(
        0.9,
        color="#000000",
        linestyle=(0, (3.5, 2.5)),
        linewidth=1.4,
        zorder=1,
    )
    handles.append(target_line)
    labels.append("90% target")
    ax.legend(handles, labels, loc="lower right", frameon=False, fontsize=11, handlelength=1.4)

    # Manually annotate the hidden 0.0 baseline just below the break.
    axis_transform = ax.get_yaxis_transform(which="grid")
    tick_len = 0.015
    ax.plot(
        (-tick_len, 0),
        (VISIBLE_MIN, VISIBLE_MIN),
        transform=axis_transform,
        color="k",
        linewidth=PRISM_TICK_WIDTH,
        clip_on=False,
    )
    ax.text(
        -tick_len - 0.01,
        VISIBLE_MIN,
        "0.0",
        transform=axis_transform,
        fontsize=11,
        ha="right",
        va="center",
    )

    # Draw paired diagonal break marks centered at BREAK_Y to indicate the jump to 0.80.
    break_kwargs = dict(color="k", clip_on=False, linewidth=PRISM_TICK_WIDTH, zorder=5)
    break_transform = transforms.blended_transform_factory(ax.transAxes, ax.transData)

    # Replace the hidden portion of the left spine with two manual segments that
    # terminate at the outer tips of the break slashes.
    axis_color = left_spine.get_edgecolor()
    axis_width = left_spine.get_linewidth()
    left_spine.set_visible(False)
    ymin, ymax = ax.get_ylim()
    slash_outer_span = (BREAK_SLASH_GAP / 2) + (BREAK_SLASH_DY / 2)
    lower_stop = BREAK_Y - slash_outer_span - BREAK_AXIS_SEGMENT_PAD
    upper_start = BREAK_Y + slash_outer_span + BREAK_AXIS_SEGMENT_PAD
    axis_segment_kwargs = dict(
        color=axis_color,
        linewidth=axis_width,
        solid_capstyle="butt",
        transform=break_transform,
        clip_on=False,
        zorder=4,
    )
    if lower_stop > ymin:
        ax.plot((0, 0), (ymin, lower_stop), **axis_segment_kwargs)
    if upper_start < ymax:
        ax.plot((0, 0), (upper_start, ymax), **axis_segment_kwargs)

    for offset in (-BREAK_SLASH_GAP / 2, BREAK_SLASH_GAP / 2):
        y_center = BREAK_Y + offset
        ax.plot(
            (-BREAK_SLASH_DX, BREAK_SLASH_DX),
            (y_center - BREAK_SLASH_DY / 2, y_center + BREAK_SLASH_DY / 2),
            transform=break_transform,
            **break_kwargs,
        )

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight", transparent=True)
    plt.close(fig)


def main():
    """
    Generate visualizations summarizing attempt method activity.

    Loads attempt data from 'FOLDER', prepares method color assignments
    and attempt ordering, and generates several plots that describe the 
    timing and outcomes of methods across attempts.

    The following visualizations are produced:

    1. Layered Gantt chart by method using absolute timestamps.
    2. Layered Gantt chart by method with attempts zeroed to start time.
    3. Average attempt Gantt chart showing mean timing of methods.
    4. Cumulative success rate plot across attempts.

    Each plot is written to its corresponding output path:
    'OUT_TIME', 'OUT_ZEROED', 'OUT_AVERAGE', and 'OUT_SUCCESS'.
    """
    data = load_folder(FOLDER)
    method_palette = build_method_palette(data)
    attempt_labels, label_map = attempt_order(data)
    data = data.assign(__attempt_label__=data["__attempt_label__"].map(label_map))
    attempt_index = {lab: i for i, lab in enumerate(attempt_labels)}
    draw_layered_by_method_datetime(data, attempt_labels, attempt_index, OUT_TIME, method_palette)
    draw_zeroed_by_method_minutes(data, attempt_labels, attempt_index, OUT_ZEROED, method_palette)
    draw_average_attempt_gantt(data, OUT_AVERAGE, method_palette)
    draw_cumulative_success_rate(data, attempt_labels, OUT_SUCCESS)
    print(
        "Saved:\n"
        f"  - {OUT_TIME}\n"
        f"  - {OUT_ZEROED}\n"
        f"  - {OUT_AVERAGE}\n"
        f"  - {OUT_SUCCESS}"
    )


if __name__ == "__main__":
    main()
