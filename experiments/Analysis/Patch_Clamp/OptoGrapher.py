#!/usr/bin/env python3
"""
Plot optogenetic traces around each stim window and save WEBP plots.
Voltage is plotted in mV and current in pA.
"""

from pathlib import Path
import re
import sys
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

DEFAULT_FOLDER = Path(
    r"C:\Users\sa-forest\Documents\GitHub\PatcherBot-Agent\experiments\Data\patch_clamp_data\2026_01_21-14_05\OptogeneticProtocol"
)
OUTPUT_NAME = "opto_plot.webp"
OUTPUT_PREFIX = "opto_plot"
DATA_PREFIX = "OptogeneticProtocol"
ACTIVE_STATE = "on"
PRE_STIM_S = 0.025
POST_STIM_S = 0.025
DOWN_SAMPLE: Optional[int] = None
CHUNK_ROWS = 1_000_000
# Column mapping for this dataset: time, response, command.
TIME_COL = 0
COMMAND_COL = 2
RESPONSE_COL = 1


def detect_separator(path: Path) -> str:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            if "," in stripped:
                return ","
            if "\t" in stripped:
                return "\t"
            return "whitespace"
    return "whitespace"


def resolve_downsample(path: Path, requested: Optional[int]) -> int:
    if requested is not None:
        return max(1, int(requested))

    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb <= 100:
        return 1
    return max(1, int(round(size_mb / 100)))


def load_wavelength_trace(
    path: Path,
    downsample: int,
    chunk_rows: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    sep = detect_separator(path)
    read_kwargs = {
        "header": None,
        "usecols": sorted({TIME_COL, COMMAND_COL, RESPONSE_COL}),
        "chunksize": chunk_rows,
    }

    if sep == "whitespace":
        read_kwargs["delim_whitespace"] = True
    else:
        read_kwargs["sep"] = sep

    time_parts: List[np.ndarray] = []
    command_parts: List[np.ndarray] = []
    response_parts: List[np.ndarray] = []

    for chunk in pd.read_csv(path, **read_kwargs):
        chunk = chunk.apply(pd.to_numeric, errors="coerce").dropna()
        if chunk.empty:
            continue
        if downsample > 1:
            chunk = chunk.iloc[::downsample]
        time_parts.append(chunk.iloc[:, TIME_COL].to_numpy())
        command_parts.append(chunk.iloc[:, COMMAND_COL].to_numpy())
        response_parts.append(chunk.iloc[:, RESPONSE_COL].to_numpy())

    if not time_parts:
        raise ValueError(f"No numeric data found in {path}")

    return (
        np.concatenate(time_parts),
        np.concatenate(command_parts),
        np.concatenate(response_parts),
    )


def load_stim_windows(path: Path, active_state: str) -> List[Tuple[float, float, str]]:
    df = pd.read_csv(path)
    if df.empty:
        return []

    columns = {col.lower(): col for col in df.columns}
    start_col = columns.get("start_s") or columns.get("start") or df.columns[0]
    end_col = columns.get("end_s") or columns.get("end") or df.columns[1]
    state_col = columns.get("state") or (df.columns[2] if len(df.columns) > 2 else None)
    wavelength_col = columns.get("wavelength") or (df.columns[3] if len(df.columns) > 3 else None)

    starts = pd.to_numeric(df[start_col], errors="coerce")
    ends = pd.to_numeric(df[end_col], errors="coerce")
    if state_col:
        states = df[state_col].astype(str).str.lower()
    else:
        states = pd.Series(["on"] * len(df), index=df.index)
    if wavelength_col:
        wavelengths = df[wavelength_col].astype(str).str.lower()
    else:
        wavelengths = pd.Series(["stim"] * len(df), index=df.index)

    keep = states == active_state.lower()
    windows_df = pd.DataFrame(
        {"start": starts, "end": ends, "wavelength": wavelengths}
    ).loc[keep]
    windows_df = windows_df.dropna(subset=["start", "end"])

    windows: List[Tuple[float, float, str]] = []
    for row in windows_df.itertuples(index=False):
        start = float(row.start)
        end = float(row.end)
        if end < start:
            start, end = end, start
        windows.append((start, end, str(row.wavelength)))

    return windows


def wavelength_color(name: str) -> str:
    name = name.lower()
    if "red" in name:
        return "#d62728"
    if "infra" in name or "ir" in name:
        return "#ff7f0e"
    if "blue" in name:
        return "#1f77b4"
    if "green" in name:
        return "#2ca02c"
    return "#7f7f7f"


def is_data_csv(path: Path) -> bool:
    name = path.name.lower()
    return (
        name.endswith(".csv")
        and name.startswith(DATA_PREFIX.lower())
        and not name.endswith("_stim.csv")
    )


def protocol_key(path: Path) -> str:
    stem = path.stem
    stem_lower = stem.lower()
    if stem_lower.endswith("_stim"):
        stem = stem[: -len("_stim")]
        stem_lower = stem.lower()

    if stem_lower.startswith(DATA_PREFIX.lower() + "_"):
        stem = stem[len(DATA_PREFIX) + 1 :]

    stem = re.sub(r"^\d+_", "", stem)
    stem = stem.strip("_")
    return stem or path.stem


def sanitize_filename(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", name.strip())
    return cleaned.strip("_") or "protocol"


Segment = Tuple[Path, np.ndarray, np.ndarray, np.ndarray, float, str]


def segment_trace(
    path: Path,
    time_s: np.ndarray,
    command_v: np.ndarray,
    response_a: np.ndarray,
    stim_windows: List[Tuple[float, float, str]],
) -> List[Segment]:
    segments: List[Segment] = []
    for start, end, wavelength in stim_windows:
        window_start = start - PRE_STIM_S
        window_end = end + POST_STIM_S
        mask = (time_s >= window_start) & (time_s <= window_end)
        if not np.any(mask):
            continue
        time_rel = time_s[mask] - start
        cmd_segment = command_v[mask]
        resp_segment = response_a[mask]
        baseline_mask = time_rel < 0
        if np.any(baseline_mask):
            cmd_baseline = float(np.nanmean(cmd_segment[baseline_mask]))
            resp_baseline = float(np.nanmean(resp_segment[baseline_mask]))
        else:
            cmd_baseline = float(np.nanmean(cmd_segment[:1]))
            resp_baseline = float(np.nanmean(resp_segment[:1]))

        segments.append(
            (
                path,
                time_rel,
                cmd_segment - cmd_baseline,
                resp_segment - resp_baseline,
                end - start,
                wavelength,
            )
        )
    return segments


def gather_traces(folder: Path) -> Dict[str, List[Segment]]:
    data_files = sorted(path for path in folder.glob("*.csv") if is_data_csv(path))
    if not data_files:
        raise FileNotFoundError(f"No data CSV files found in {folder}")

    grouped: Dict[str, List[Segment]] = {}
    for data_path in data_files:
        downsample = resolve_downsample(data_path, DOWN_SAMPLE)
        if DOWN_SAMPLE is None and downsample > 1:
            size_mb = data_path.stat().st_size / (1024 * 1024)
            print(
                f"Auto downsample={downsample} for {data_path.name} ({size_mb:.1f} MB). "
                "Set DOWN_SAMPLE to override."
            )

        time_s, command_v, response = load_wavelength_trace(
            data_path, downsample=downsample, chunk_rows=CHUNK_ROWS
        )

        stim_path = data_path.with_name(f"{data_path.stem}_stim.csv")
        stim_windows = load_stim_windows(stim_path, ACTIVE_STATE) if stim_path.exists() else []
        if not stim_windows:
            continue

        segments = segment_trace(data_path, time_s, command_v, response, stim_windows)
        if not segments:
            continue
        group = protocol_key(data_path)
        grouped.setdefault(group, []).extend(segments)

    return grouped


def plot_opto(
    traces: List[Segment],
    title: Optional[str],
) -> Tuple[plt.Figure, plt.Axes]:
    fig, (ax_cmd, ax_resp) = plt.subplots(
        2, 1, sharex=True, figsize=(11, 7.5), gridspec_kw={"height_ratios": [1, 1]}
    )

    cmap = plt.get_cmap("tab10")
    sources = sorted({trace[0] for trace in traces})
    denom = max(1, len(sources) - 1)
    source_colors = {
        source: cmap(idx / denom) for idx, source in enumerate(sources)
    }

    cmd_legend_items = []
    cmd_legend_labels = []
    stim_legend: Dict[str, str] = {}
    stim_spans = set()
    xmin = None
    xmax = None

    for path, time_s, command_v, response, duration_s, wavelength in traces:
        color = source_colors[path]
        label = path.stem.replace("_", " ")
        cmd_line, = ax_cmd.plot(
            time_s, command_v * 1e3, color=color, linewidth=1.0, label=label
        )
        resp_line, = ax_resp.plot(
            time_s, response * 1e12, color=color, linewidth=1.0, linestyle="--", label=label
        )
        if label not in cmd_legend_labels:
            cmd_legend_items.append(cmd_line)
            cmd_legend_labels.append(label)

        if time_s.size:
            local_min = float(np.nanmin(time_s))
            local_max = float(np.nanmax(time_s))
            xmin = local_min if xmin is None else min(xmin, local_min)
            xmax = local_max if xmax is None else max(xmax, local_max)

        span_key = (round(duration_s, 9), wavelength)
        if span_key not in stim_spans:
            stim_spans.add(span_key)
            stim_color = wavelength_color(wavelength)
            for axis in (ax_cmd, ax_resp):
                axis.axvspan(0.0, duration_s, color=stim_color, alpha=0.18, linewidth=0)
                axis.axvline(0.0, color=stim_color, alpha=0.6, linewidth=1.0)
                axis.axvline(duration_s, color=stim_color, alpha=0.6, linewidth=1.0)
            stim_label = f"Stim {wavelength}"
            stim_legend.setdefault(stim_label, stim_color)

    for stim_label, color in stim_legend.items():
        cmd_legend_items.append(mpatches.Patch(color=color, alpha=0.25, label=stim_label))
        cmd_legend_labels.append(stim_label)

    ax_resp.set_xlabel("Time (s)")
    ax_cmd.set_ylabel("Command voltage (mV)")
    ax_resp.set_ylabel("Response current (pA)")
    if title:
        ax_cmd.set_title(title)
    if xmin is not None and xmax is not None:
        ax_cmd.set_xlim(xmin, xmax)

    ax_cmd.legend(cmd_legend_items, cmd_legend_labels, loc="upper right", frameon=False, fontsize=8)
    fig.tight_layout()
    return fig, ax_cmd


def find_protocol_folders(root: Path) -> List[Path]:
    folders = {path.parent for path in root.rglob("*.csv") if is_data_csv(path)}
    return sorted(folders)


def save_figure(fig: plt.Figure, output_path: Path) -> None:
    output_path = output_path.with_suffix(".webp")
    try:
        fig.savefig(output_path, dpi=300, bbox_inches="tight", format="webp")
    except Exception as exc:
        raise RuntimeError(
            "WEBP save failed. Install Pillow or change OUTPUT_NAME to a supported format."
        ) from exc
    print(f"Saved: {output_path}")


def process_folder(folder: Path) -> None:
    grouped_traces = gather_traces(folder)
    multi_group = len(grouped_traces) > 1

    for group, traces in grouped_traces.items():
        title = group if multi_group else None
        fig, _ = plot_opto(traces, title=title)
        if multi_group:
            output_name = f"{OUTPUT_PREFIX}_{sanitize_filename(group)}.webp"
        else:
            output_name = OUTPUT_NAME
        output_path = folder / output_name
        save_figure(fig, output_path)
        plt.close(fig)


def main(folder: Path) -> None:
    if not folder.is_dir():
        raise NotADirectoryError(f"{folder} is not a folder.")

    protocol_folders = find_protocol_folders(folder)
    if not protocol_folders:
        raise FileNotFoundError(f"No OptogeneticProtocol CSVs found in {folder}")

    for protocol_folder in protocol_folders:
        process_folder(protocol_folder)


if __name__ == "__main__":
    folder_arg = Path(sys.argv[1]) if len(sys.argv) >= 2 else DEFAULT_FOLDER
    main(folder_arg)
