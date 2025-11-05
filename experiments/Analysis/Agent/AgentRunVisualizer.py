#!/usr/bin/env python3
"""
Multi-CSV categorical Gantt charts (combined on one plot), no argparse.

Edit the CONFIG section below:
- FOLDER: path to the folder containing your CSVs
- PATTERN: glob for which files to include (e.g., "*.csv")
- EXCLUDE_METHODS: list of method names to remove (case-insensitive)
- Column names: ATTEMPT_COL, METHOD_COL, START_COL, END_COL

Two PNGs are produced:
- OUT_TIME:   combined_gantt_by_method.png
- OUT_ZEROED: combined_gantt_by_method_zeroed.png

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

# Bar styling
BAR_HEIGHT = 0.8
BAR_ALPHA  = 0.9

# ===============================================================


def normalize_method(name):
    """Lowercase + spacing normalized method names for consistent comparisons."""
    if name is None or (isinstance(name, float) and np.isnan(name)):
        return ""
    if pd.isna(name):
        return ""
    text = str(name).strip().lower().replace("_", " ")
    return re.sub(r"\s+", " ", text)


def extract_first_int(s):
    """Extract first integer from a string like 'attempt_12' -> 12 (or None)."""
    s = str(s)
    m = re.search(r"-?\d+", s)
    return int(m.group(0)) if m else None


def method_display_color(method_name: str):
    """Return an override color for a given method, or None to use defaults."""
    norm = normalize_method(method_name)
    if norm == "locate cell":
        return "tab:orange"
    return None


def read_one_csv(path: Path) -> pd.DataFrame:
    """Load one CSV and return parsed/filtered rows with a source_file column."""
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
    """Read all CSVs, concatenate, drop excluded methods, compute numeric attempt and labels."""
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
    """Ordering: chronological by first start time; assign sequential attempt labels."""
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
):
    """Combined Gantt with datetime x-axis; layer methods by total duration (longer first)."""
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

    fig, ax = plt.subplots(figsize=(14, max(5, 0.5 * len(attempt_labels))))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")

    for method in draw_methods:
        sub = complete[complete[METHOD_COL].astype(str) == method]
        if sub.empty:
            continue
        y = sub["__attempt_label__"].map(attempt_index).values
        lefts = mdates.date2num(sub["__start_ts__"].values)
        widths = mdates.date2num(sub["__end_ts__"].values) - lefts
        color = method_display_color(method)
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
    ax.set_ylabel("Attempt (num, time-disambiguated)")
    ax.invert_yaxis()

    ax.set_xlabel("Time")
    ax.set_title("Combined Categorical Gantt (colored by method)")
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
):
    """Combined Gantt with x-axis in minutes since each attempt's first start (t=0 per attempt)."""
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

    fig, ax = plt.subplots(figsize=(14, max(5, 0.5 * len(attempt_labels))))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")

    for method in draw_methods:
        sub = complete[complete[METHOD_COL].astype(str) == method]
        if sub.empty:
            continue
        y = sub["__attempt_label__"].map(attempt_index).values
        color = method_display_color(method)
        bar_kwargs = dict(
            height=BAR_HEIGHT,
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
        y = sub["__attempt_label__"].map(attempt_index).values
        ax.barh(
            y,
            sub["__width_min__"].values,
            left=sub["__left_min__"].values,
            height=BAR_HEIGHT,
            align="center",
            alpha=BAR_ALPHA,
            color="red",
            edgecolor="none",
            label="incomplete attempt",
        )

    ax.set_yticks(list(range(len(attempt_labels))), labels=attempt_labels)
    ax.set_ylabel("Attempt (num, time-disambiguated)")
    ax.invert_yaxis()

    ax.set_xlabel("Minutes since attempt start (t = 0 per attempt)")
    ax.set_title("Combined Categorical Gantt (by method, start time zeroed per attempt)")
    ax.legend(title="Method / status", loc="best", ncol=2)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight", transparent=True)
    plt.close(fig)


def draw_cumulative_success_rate(
    data: pd.DataFrame,
    attempt_labels: list,
    out_path: Path,
):
    """Plot cumulative success rate vs attempt index (Attempt 1..N)."""
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

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    ax.plot(
        attempt_numbers,
        success_rate,
        marker="o",
        color="tab:blue",
        label="Cumulative success rate",
    )

    failed_idx = attempt_numbers[~success_flags.values]
    if len(failed_idx) > 0:
        ax.scatter(
            failed_idx,
            success_rate[~success_flags.values],
            color="red",
            zorder=3,
            label="Attempt incomplete",
        )

    ax.set_xlim(1, max(1, attempt_numbers[-1]))
    ax.set_ylim(0, 1.1,)
    ax.set_xticks(attempt_numbers)
    ax.set_xlabel("Attempt number")
    ax.set_ylabel("Cumulative success rate")
    ax.set_title("Cumulative Attempt Success Rate")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax.legend(loc="best")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight", transparent=True)
    plt.close(fig)


def main():
    data = load_folder(FOLDER)
    attempt_labels, label_map = attempt_order(data)
    data = data.assign(__attempt_label__=data["__attempt_label__"].map(label_map))
    attempt_index = {lab: i for i, lab in enumerate(attempt_labels)}
    draw_layered_by_method_datetime(data, attempt_labels, attempt_index, OUT_TIME)
    draw_zeroed_by_method_minutes(data, attempt_labels, attempt_index, OUT_ZEROED)
    draw_cumulative_success_rate(data, attempt_labels, OUT_SUCCESS)
    print(f"Saved:\n  - {OUT_TIME}\n  - {OUT_ZEROED}\n  - {OUT_SUCCESS}")


if __name__ == "__main__":
    main()
