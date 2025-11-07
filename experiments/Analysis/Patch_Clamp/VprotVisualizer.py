#!/usr/bin/env python3
# Usage: python VprotVisualizer.py /path/to/VoltageProtocol [output.png]

import sys
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd

DEFAULT_VOLTAGE_PROTOCOL_FOLDER = Path(
    r"C:\Users\sa-forest\Documents\GitHub\PatchAnalyzer\Data\Forest_HEK_exp\corrected\2025_11_02-21_00\VoltageProtocol"
)

def robust_read(path: Path) -> pd.DataFrame:
    """Try several common CSV dialects before normalizing whitespace runs to commas."""
    # 1) Let pandas sniff the delimiter
    try:
        df = pd.read_csv(path, engine="python", sep=None, header=None)
        if df.shape[1] >= 2:
            return df
    except Exception:
        pass
    # 2) Tab, semicolon, whitespace
    for sep in ["\t", ";"]:
        try:
            df = pd.read_csv(path, sep=sep, header=None)
            if df.shape[1] >= 2:
                return df
        except Exception:
            pass
    try:
        df = pd.read_csv(path, delim_whitespace=True, engine="python", header=None)
        if df.shape[1] >= 2:
            return df
    except Exception:
        pass
    # 3) Fallback: normalize runs of spaces/tabs to commas
    text = path.read_text(encoding="utf-8", errors="ignore")
    import re
    from io import StringIO

    text_csvish = re.sub(r"[ \t]+", ",", text)
    return pd.read_csv(StringIO(text_csvish), engine="python", header=None)


def looks_like_number(s) -> bool:
    try:
        float(str(s))
        return True
    except Exception:
        return False


def load_trace(csv_path: Path) -> Optional[pd.DataFrame]:
    df = robust_read(csv_path)

    # If headers look numeric, rename them to generic names
    if all(looks_like_number(c) for c in df.columns):
        df.columns = [f"col{i+1}" for i in range(df.shape[1])]

    if df.shape[1] < 3:
        print(f"Skipping {csv_path.name}: need at least three columns (time index, stimulus, response).", file=sys.stderr)
        return None

    x_col = df.columns[0]
    y_col = df.columns[2]

    x = pd.to_numeric(df[x_col], errors="coerce")
    y = pd.to_numeric(df[y_col], errors="coerce")

    plot_df = (
        pd.DataFrame({"x": x, "y": y})
        .dropna()
        .sort_values("x")
    )

    if plot_df.empty:
        print(f"Skipping {csv_path.name}: no numeric samples after cleaning.", file=sys.stderr)
        return None

    return plot_df


def collect_voltage_traces(folder: Path) -> List[Tuple[Path, pd.DataFrame]]:
    csv_files = sorted(p for p in folder.glob("*.csv") if p.is_file())
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {folder}")

    traces: List[Tuple[Path, pd.DataFrame]] = []
    for csv_path in csv_files:
        trace_df = load_trace(csv_path)
        if trace_df is not None:
            traces.append((csv_path, trace_df))

    if not traces:
        raise ValueError(f"CSV files were found in {folder}, but none contained usable voltage traces.")

    return traces


def plot_voltage_traces(traces: List[Tuple[Path, pd.DataFrame]], outfile: Path) -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "axes.linewidth": 1.4,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
    })

    fig, ax = plt.subplots(figsize=(6, 4.5))
    cmap = plt.get_cmap("twilight")
    denom = max(1, len(traces) - 1)

    for idx, (_, plot_df) in enumerate(traces):
        color = cmap(idx / denom)
        ax.plot(plot_df["x"], plot_df["y"], linewidth=3.0, color=color)

    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=6, width=1.2)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Response (A)")

    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    ax.yaxis.get_offset_text().set_visible(False)
    ax.xaxis.get_offset_text().set_visible(False)

    ax.text(-0.07, 1.02, "1e-10", transform=ax.transAxes, ha="left", va="bottom", fontsize=11)
    ax.text(1.00, -0.08, "1e-3", transform=ax.transAxes, ha="right", va="top", fontsize=11)

    ax.set_box_aspect(1)
    ax.set_facecolor("none")
    fig.patch.set_facecolor("none")
    plt.tight_layout()

    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, dpi=300, bbox_inches="tight", transparent=True)
    print(f"Saved: {outfile}")


def main(folder: str, outfile: Optional[str] = None) -> None:
    folder_path = Path(folder)
    if not folder_path.is_dir():
        raise NotADirectoryError(f"{folder} is not a directory.")

    traces = collect_voltage_traces(folder_path)

    output_path = Path(outfile) if outfile else folder_path / "voltage_protocol.png"
    plot_voltage_traces(traces, output_path)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        folder_arg = DEFAULT_VOLTAGE_PROTOCOL_FOLDER
        output_arg = None
    else:
        folder_arg = Path(sys.argv[1])
        output_arg = sys.argv[2] if len(sys.argv) >= 3 else None

    main(str(folder_arg), output_arg)
