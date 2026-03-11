#!/usr/bin/env python3
"""
Plot optogenetic traces around each stim window and save WEBP plots.
Voltage is plotted in mV and current in pA.
One plot is saved per data/stim CSV pair.
"""

from pathlib import Path
import re
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import bessel, filtfilt

# DEFAULT_FOLDER = Path(R"C:\Users\sa-forest\Documents\GitHub\PatcherBot-Agent\experiments\Data\patch_clamp_data\2026_02_27-17_15")
DEFAULT_FOLDER = Path(R"C:\Users\sa-forest\Documents\GitHub\PatcherBot-Agent\experiments\Data\patch_clamp_data\2026_03_03-13_42")
OUTPUT_NAME = "opto_plot.webp"
OUTPUT_PREFIX = "opto_plot"
PROCESSED_SUBFOLDER = "Processed"
DATA_PREFIX = "OptogeneticProtocol"
ACTIVE_STATE = "on"
PRE_STIM_S = 1.0
POST_STIM_S = 1.0
DOWN_SAMPLE: Optional[int] = None
CHUNK_ROWS = 1_000_000
BESSEL_CUTOFF_HZ = 3000.0
BESSEL_ORDER = 8
COLOR_ORDER = (
    ("uv", "ultraviolet"),
    ("violet", "purple"),
    ("blue",),
    ("cyan",),
    ("green",),
    ("yellow", "amber"),
    ("orange",),
    ("red",),
    ("infra", "ir"),
)
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


def apply_bessel_filter(
    time_s: np.ndarray,
    signal: np.ndarray,
    cutoff_hz: float,
    order: int,
) -> np.ndarray:
    if signal.size < order * 3:
        return signal

    diffs = np.diff(time_s)
    if diffs.size == 0:
        return signal
    dt = float(np.nanmedian(diffs))
    if not np.isfinite(dt) or dt <= 0:
        return signal

    fs = 1.0 / dt
    nyquist = 0.5 * fs
    if not np.isfinite(nyquist) or nyquist <= 0:
        return signal

    norm_cutoff = cutoff_hz / nyquist
    if not np.isfinite(norm_cutoff) or norm_cutoff >= 1.0 or norm_cutoff <= 0:
        return signal

    try:
        b, a = bessel(order, norm_cutoff, btype="low", norm="phase")
        return filtfilt(b, a, signal)
    except Exception:
        return signal


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
        read_kwargs["sep"] = r"\s+"
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


def load_stim_windows(
    path: Path, active_state: str
) -> List[Tuple[float, float, str, Optional[float]]]:
    df = pd.read_csv(path)
    if df.empty:
        return []

    columns = {col.lower(): col for col in df.columns}
    start_col = columns.get("start_s") or columns.get("start") or df.columns[0]
    end_col = columns.get("end_s") or columns.get("end") or df.columns[1]
    state_col = columns.get("state") or (df.columns[2] if len(df.columns) > 2 else None)
    wavelength_col = columns.get("wavelength") or (df.columns[3] if len(df.columns) > 3 else None)
    power_col = columns.get("power_percent") or columns.get("power")

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
    if power_col:
        powers = pd.to_numeric(df[power_col], errors="coerce")
    else:
        powers = pd.Series([np.nan] * len(df), index=df.index)

    keep = states == active_state.lower()
    windows_df = pd.DataFrame(
        {"start": starts, "end": ends, "wavelength": wavelengths, "power": powers}
    ).loc[keep]
    windows_df = windows_df.dropna(subset=["start", "end"])

    windows: List[Tuple[float, float, str, Optional[float]]] = []
    for row in windows_df.itertuples(index=False):
        start = float(row.start)
        end = float(row.end)
        if end < start:
            start, end = end, start
        power = None if pd.isna(row.power) else float(row.power)
        windows.append((start, end, str(row.wavelength), power))

    return windows


def load_stim_table(path: Path) -> pd.DataFrame:
    """
    Load full stimulation table (on/off) with normalized column names.
    """
    df = pd.read_csv(path)
    if df.empty:
        return pd.DataFrame(
            columns=["start_s", "end_s", "state", "wavelength", "power_percent"]
        )

    columns = {col.lower(): col for col in df.columns}
    start_col = columns.get("start_s") or columns.get("start") or df.columns[0]
    end_col = columns.get("end_s") or columns.get("end") or df.columns[1]
    state_col = columns.get("state") or (df.columns[2] if len(df.columns) > 2 else None)
    wavelength_col = columns.get("wavelength") or (df.columns[3] if len(df.columns) > 3 else None)
    power_col = columns.get("power_percent") or columns.get("power")

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
    if power_col:
        powers = pd.to_numeric(df[power_col], errors="coerce")
    else:
        powers = pd.Series([np.nan] * len(df), index=df.index)

    table = pd.DataFrame(
        {
            "start_s": starts,
            "end_s": ends,
            "state": states,
            "wavelength": wavelengths,
            "power_percent": powers,
        }
    ).dropna(subset=["start_s", "end_s"])

    if table.empty:
        return table

    start_vals = table["start_s"].to_numpy(dtype=float)
    end_vals = table["end_s"].to_numpy(dtype=float)
    swap_mask = end_vals < start_vals
    if np.any(swap_mask):
        tmp = start_vals[swap_mask].copy()
        start_vals[swap_mask] = end_vals[swap_mask]
        end_vals[swap_mask] = tmp
        table.loc[:, "start_s"] = start_vals
        table.loc[:, "end_s"] = end_vals

    table = table.sort_values("start_s", kind="mergesort").reset_index(drop=True)
    return table


def summarize_stim_profile(stim_table: pd.DataFrame) -> dict:
    """
    Compute protocol timing metrics from the stimulation table.
    """
    if stim_table.empty:
        return {
            "events_total": 0,
            "events_on": 0,
            "protocol_start_s": 0.0,
            "protocol_end_s": 0.0,
            "protocol_duration_s": 0.0,
            "first_on_start_s": np.nan,
            "last_on_end_s": np.nan,
            "on_span_s": np.nan,
            "mean_on_duration_s": np.nan,
            "median_on_duration_s": np.nan,
        }

    on_table = stim_table[stim_table["state"].astype(str).str.lower() == "on"]
    if on_table.empty:
        on_table = stim_table

    start_min = float(stim_table["start_s"].min())
    end_max = float(stim_table["end_s"].max())
    on_start_min = float(on_table["start_s"].min())
    on_end_max = float(on_table["end_s"].max())
    on_durations = (on_table["end_s"] - on_table["start_s"]).to_numpy(dtype=float)

    return {
        "events_total": int(len(stim_table)),
        "events_on": int(len(on_table)),
        "protocol_start_s": start_min,
        "protocol_end_s": end_max,
        "protocol_duration_s": end_max - start_min,
        "first_on_start_s": on_start_min,
        "last_on_end_s": on_end_max,
        "on_span_s": on_end_max - on_start_min,
        "mean_on_duration_s": float(np.nanmean(on_durations)),
        "median_on_duration_s": float(np.nanmedian(on_durations)),
    }


def plot_stim_profile(
    stim_table: pd.DataFrame, metrics: dict, title: Optional[str]
) -> Tuple[plt.Figure, plt.Axes]:
    if stim_table.empty:
        raise ValueError("No stimulation table rows to plot.")

    fig, ax = plt.subplots(figsize=(12, 2.8))
    has_off = False

    for row in stim_table.itertuples(index=False):
        start = float(row.start_s)
        end = float(row.end_s)
        width = max(0.0, end - start)
        state = str(row.state).lower()
        wavelength = str(row.wavelength)

        if state == "on":
            y_base = 0.75
            color = wavelength_color(wavelength)
            alpha = 0.85
        else:
            has_off = True
            y_base = 0.15
            color = "#b0b0b0"
            alpha = 0.5

        ax.broken_barh([(start, width)], (y_base, 0.18), facecolors=color, alpha=alpha)

    x_left = min(0.0, float(metrics["protocol_start_s"]))
    x_right = float(metrics["protocol_end_s"])
    if x_right <= x_left:
        x_right = x_left + 1.0
    ax.set_xlim(x_left, x_right)
    ax.set_ylim(0.0, 1.1)
    ax.set_xlabel("Absolute protocol time (s)")
    ax.set_yticks([0.24, 0.84] if has_off else [0.84])
    ax.set_yticklabels(["off", "on"] if has_off else ["on"])
    ax.grid(axis="x", alpha=0.25, linestyle="--", linewidth=0.8)

    subtitle = (
        f"duration={metrics['protocol_duration_s']:.3f}s | "
        f"events_on={metrics['events_on']} | "
        f"first_on={metrics['first_on_start_s']:.3f}s | "
        f"last_on={metrics['last_on_end_s']:.3f}s"
    )
    if title:
        ax.set_title(f"{title}\n{subtitle}", fontsize=10)
    else:
        ax.set_title(subtitle, fontsize=10)

    fig.tight_layout()
    return fig, ax


def wavelength_color(name: str) -> str:
    name = name.lower()
    if "uv" in name or "ultraviolet" in name:
        return "#9467bd"
    if "red" in name:
        return "#d62728"
    if "infra" in name or "ir" in name:
        return "#ff7f0e"
    if "blue" in name:
        return "#1f77b4"
    if "green" in name:
        return "#2ca02c"
    if "cyan" in name:
        return "#7ec9f1"
    return "#7f7f7f"


def is_data_csv(path: Path) -> bool:
    name = path.name.lower()
    return (
        name.endswith(".csv")
        and name.startswith(DATA_PREFIX.lower())
        and not name.endswith("_stim.csv")
    )


def sanitize_filename(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", name.strip())
    return cleaned.strip("_") or "protocol"


def wavelength_sort_key(name: str) -> int:
    name = name.lower()
    for idx, tokens in enumerate(COLOR_ORDER):
        if any(token in name for token in tokens):
            return idx
    return len(COLOR_ORDER)


Segment = Tuple[
    Path,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    float,
    str,
    Optional[float],
]


def segment_trace(
    path: Path,
    time_s: np.ndarray,
    command_v: np.ndarray,
    response_a: np.ndarray,
    stim_windows: List[Tuple[float, float, str, Optional[float]]],
    apply_filter: bool,
    cutoff_hz: float,
    order: int,
) -> List[Segment]:
    segments: List[Segment] = []
    for start, end, wavelength, power in stim_windows:
        window_start = start - PRE_STIM_S
        window_end = end + POST_STIM_S
        time_axis = time_s
        mask = (time_axis >= window_start) & (time_axis <= window_end)
        if not np.any(mask):
            # Local per-trace files store time from ~0; align them to the requested
            # absolute window so plotting reflects protocol time.
            t_min = float(np.nanmin(time_s))
            shift = window_start - t_min
            time_axis = time_s + shift
            mask = (time_axis >= window_start) & (time_axis <= window_end)
        if not np.any(mask):
            continue
        time_abs = time_axis[mask]
        if apply_filter:
            cmd_segment = apply_bessel_filter(
                time_s[mask], command_v[mask], cutoff_hz, order
            )
            resp_segment = apply_bessel_filter(
                time_s[mask], response_a[mask], cutoff_hz, order
            )
        else:
            cmd_segment = command_v[mask]
            resp_segment = response_a[mask]
        baseline_mask = time_abs < start
        if np.any(baseline_mask):
            cmd_baseline = float(np.nanmean(cmd_segment[baseline_mask]))
            resp_baseline = float(np.nanmean(resp_segment[baseline_mask]))
        else:
            cmd_baseline = float(np.nanmean(cmd_segment[:1]))
            resp_baseline = float(np.nanmean(resp_segment[:1]))

        segments.append(
            (
                path,
                time_abs,
                cmd_segment - cmd_baseline,
                resp_segment - resp_baseline,
                start,
                end,
                wavelength,
                power,
            )
        )
    return segments


def gather_trace_pairs(
    folder: Path,
    apply_filter: bool,
    cutoff_hz: float,
    order: int,
    order_by_color: bool,
) -> List[Tuple[Path, List[Segment]]]:
    data_files = sorted(path for path in folder.glob("*.csv") if is_data_csv(path))
    if not data_files:
        raise FileNotFoundError(f"No data CSV files found in {folder}")

    pairs: List[Tuple[Path, List[Segment]]] = []
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

        stim_path = resolve_stim_path(data_path)
        stim_windows = load_stim_windows(stim_path, ACTIVE_STATE) if stim_path else []
        stim_windows = select_stim_windows_for_trace(data_path, stim_windows)
        if not stim_windows:
            continue

        segments = segment_trace(
            data_path,
            time_s,
            command_v,
            response,
            stim_windows,
            apply_filter,
            cutoff_hz,
            order,
        )
        if not segments:
            continue
        has_power = any(seg[7] is not None for seg in segments)
        if has_power:
            segments = sorted(
                segments,
                key=lambda seg: (float("inf") if seg[7] is None else seg[7]),
            )
        elif order_by_color:
            segments = sorted(segments, key=lambda seg: wavelength_sort_key(seg[6]))
        pairs.append((data_path, segments))

    return pairs


def resolve_stim_path(data_path: Path) -> Optional[Path]:
    # Prefer shared stim file per protocol, e.g. OptogeneticProtocol_3_wavelength_stim.csv
    match = re.match(
        rf"^({re.escape(DATA_PREFIX)}_\d+_(?:wavelength|power))",
        data_path.stem,
        flags=re.IGNORECASE,
    )
    if match:
        shared = data_path.with_name(f"{match.group(1)}_stim.csv")
        if shared.exists():
            return shared

    # Fallback: legacy per-trace stim files.
    per_trace = data_path.with_name(f"{data_path.stem}_stim.csv")
    if per_trace.exists():
        return per_trace

    return None


def protocol_group_key(data_path: Path) -> str:
    match = re.match(
        rf"^({re.escape(DATA_PREFIX)}_\d+_(?:wavelength|power))",
        data_path.stem,
        flags=re.IGNORECASE,
    )
    if match:
        return sanitize_filename(match.group(1))
    return sanitize_filename(data_path.stem)


def select_stim_windows_for_trace(
    data_path: Path,
    stim_windows: List[Tuple[float, float, str, Optional[float]]],
) -> List[Tuple[float, float, str, Optional[float]]]:
    if not stim_windows:
        return stim_windows

    match = re.match(
        rf"^{re.escape(DATA_PREFIX)}_\d+_(?:wavelength|power)(?:_(.+))?$",
        data_path.stem,
        flags=re.IGNORECASE,
    )
    suffix = match.group(1) if match else None
    if not suffix:
        return stim_windows

    suffix = suffix.lower()
    tokens = [token for token in re.split(r"[_\s]+", suffix) if token]

    token_map = {
        "uv": ("uv", "ultraviolet"),
        "ultraviolet": ("uv", "ultraviolet"),
        "infrared": ("infra", "ir"),
        "ir": ("infra", "ir"),
    }

    color_tokens: Optional[Tuple[str, ...]] = None
    for token in tokens:
        if token in token_map:
            color_tokens = token_map[token]
            break
        if token in {
            "violet",
            "purple",
            "blue",
            "cyan",
            "green",
            "yellow",
            "amber",
            "orange",
            "red",
            "infra",
        }:
            color_tokens = (token,)
            break

    filtered = stim_windows
    if color_tokens:
        by_color = [
            window
            for window in filtered
            if any(token in window[2].lower() for token in color_tokens)
        ]
        if by_color:
            filtered = by_color

    rep_match = re.search(r"rep(\d+)", suffix, flags=re.IGNORECASE)
    if rep_match and filtered:
        rep_idx = int(rep_match.group(1))
        if rep_idx < len(filtered):
            filtered = [filtered[rep_idx]]
        else:
            filtered = [filtered[-1]]

    return filtered


def localize_stim_windows_if_needed(
    time_s: np.ndarray,
    stim_windows: List[Tuple[float, float, str, Optional[float]]],
) -> List[Tuple[float, float, str, Optional[float]]]:
    if not stim_windows or time_s.size == 0:
        return stim_windows

    has_overlap = False
    for start, end, _, _ in stim_windows:
        mask = (time_s >= (start - PRE_STIM_S)) & (time_s <= (end + POST_STIM_S))
        if np.any(mask):
            has_overlap = True
            break
    if has_overlap:
        return stim_windows

    t_min = float(np.nanmin(time_s))
    t_max = float(np.nanmax(time_s))
    trace_duration = max(0.0, t_max - t_min)

    localized: List[Tuple[float, float, str, Optional[float]]] = []
    for start, end, wavelength, power in stim_windows:
        duration = max(0.0, float(end - start))
        if trace_duration > 0:
            local_end = t_min + min(duration, trace_duration)
        else:
            local_end = t_min + duration
        localized.append((t_min, local_end, wavelength, power))
    return localized


def plot_opto(
    traces: List[Segment],
    title: Optional[str],
) -> Tuple[plt.Figure, plt.Axes]:
    if not traces:
        raise ValueError("No traces to plot.")

    trace_count = len(traces)
    fig_height = max(6.5, trace_count * 2.6)
    fig, axes = plt.subplots(
        trace_count,
        2,
        sharex=True,
        figsize=(11, fig_height),
        gridspec_kw={"width_ratios": [1, 1]},
    )
    axes = np.atleast_2d(axes)
    xmin = 0.0
    xmax = None

    for _, time_s, _, _, _, _, _, _ in traces:
        if time_s.size:
            local_min = float(np.nanmin(time_s))
            local_max = float(np.nanmax(time_s))
            span = max(0.0, local_max - local_min)
            xmax = span if xmax is None else max(xmax, span)

    for idx, (
        _,
        time_s,
        command_v,
        response,
        stim_start_s,
        stim_end_s,
        wavelength,
        power,
    ) in enumerate(traces):
        ax_cmd = axes[idx, 0]
        ax_resp = axes[idx, 1]
        stim_color = wavelength_color(wavelength)
        if not time_s.size:
            continue
        t0 = float(np.nanmin(time_s))
        time_plot = time_s - t0
        stim_start_plot = stim_start_s - t0
        stim_end_plot = stim_end_s - t0

        ax_cmd.plot(time_plot, command_v * 1e3, color=stim_color, linewidth=1.0)
        ax_resp.plot(time_plot, response * 1e12, color=stim_color, linewidth=1.0)

        for axis in (ax_cmd, ax_resp):
            axis.axvspan(stim_start_plot, stim_end_plot, color=stim_color, alpha=0.18, linewidth=0)
            axis.axvline(
                stim_start_plot, color="#111111", alpha=0.85, linewidth=1.2, linestyle="--", zorder=6
            )
            axis.axvline(
                stim_end_plot,
                color="#111111",
                alpha=0.85,
                linewidth=1.2,
                linestyle="--",
                zorder=6,
            )

        label = f"Stim {wavelength}"
        if power is not None:
            label = f"{label} ({power:g}%)"
        duration_s = max(0.0, float(stim_end_s - stim_start_s))
        if duration_s > 0:
            label = f"{label} ({duration_s:.4g} s)"
        ax_cmd.set_title(label, fontsize=9)
        ax_cmd.set_ylabel("Command voltage (mV)")
        ax_resp.set_ylabel("Response current (pA)")

        if idx < trace_count - 1:
            ax_cmd.tick_params(labelbottom=False)
            ax_resp.tick_params(labelbottom=False)

    if xmax is not None:
        for row in axes:
            for axis in row:
                axis.set_xlim(xmin, xmax)

    axes[-1, 0].set_xlabel("Time (s, zeroed per trace)")
    axes[-1, 1].set_xlabel("Time (s, zeroed per trace)")
    if title:
        fig.suptitle(title, y=0.995)
        fig.tight_layout(rect=[0, 0, 1, 0.985])
    else:
        fig.tight_layout()
    return fig, axes[0, 0]


def find_protocol_folders(root: Path) -> List[Path]:
    folders = {path.parent for path in root.rglob("*.csv") if is_data_csv(path)}
    return sorted(folders)


def save_figure(fig: plt.Figure, output_path: Path) -> None:
    output_path = output_path.with_suffix(".webp")
    target_path = output_path
    try:
        fig.savefig(target_path, dpi=300, bbox_inches="tight", format="webp")
    except PermissionError:
        # Windows preview/thumbnail handlers can lock files. Fall back to a
        # unique filename so repeated runs still complete.
        stem = output_path.stem
        suffix = output_path.suffix
        for i in range(1, 1000):
            candidate = output_path.with_name(f"{stem}_{i}{suffix}")
            if candidate.exists():
                continue
            target_path = candidate
            fig.savefig(target_path, dpi=300, bbox_inches="tight", format="webp")
            break
        else:
            raise RuntimeError("WEBP save failed: no available fallback filename.")
    except Exception as exc:
        raise RuntimeError(
            "WEBP save failed. Install Pillow or change OUTPUT_NAME to a supported format."
        ) from exc
    print(f"Saved: {target_path}")


def process_folder(
    folder: Path,
    apply_filter: bool,
    cutoff_hz: float,
    order: int,
    order_by_color: bool,
) -> None:
    trace_pairs = gather_trace_pairs(
        folder, apply_filter, cutoff_hz, order, order_by_color
    )
    processed_dir = folder / PROCESSED_SUBFOLDER
    processed_dir.mkdir(parents=True, exist_ok=True)
    grouped_traces: dict[str, List[Segment]] = {}
    stim_tables: List[pd.DataFrame] = []
    seen_stim_paths: set[Path] = set()

    for data_path, traces in trace_pairs:
        group_key = protocol_group_key(data_path)
        if traces:
            grouped_traces.setdefault(group_key, []).extend(traces)

        stim_path = resolve_stim_path(data_path)
        if (
            stim_path is not None
            and stim_path.exists()
            and stim_path not in seen_stim_paths
        ):
            seen_stim_paths.add(stim_path)
            stim_table = load_stim_table(stim_path)
            if not stim_table.empty:
                stim_tables.append(stim_table)

    for group_key, traces in grouped_traces.items():
        traces = sorted(traces, key=lambda seg: (seg[4], seg[5], str(seg[6])))
        if traces:
            fig, _ = plot_opto(traces, title=group_key)
            output_path = processed_dir / f"{OUTPUT_PREFIX}_{group_key}_traces"
            save_figure(fig, output_path)
            plt.close(fig)

    if stim_tables:
        combined_stim = pd.concat(stim_tables, ignore_index=True)
        combined_stim = combined_stim.drop_duplicates(
            subset=["start_s", "end_s", "state", "wavelength", "power_percent"]
        )
        combined_stim = combined_stim.sort_values("start_s", kind="mergesort").reset_index(drop=True)
        metrics = summarize_stim_profile(combined_stim)
        profile_fig, _ = plot_stim_profile(combined_stim, metrics, title=folder.name)
        profile_path = processed_dir / "stim_profile"
        save_figure(profile_fig, profile_path)
        plt.close(profile_fig)


def main(
    folder: Path,
    apply_filter: bool,
    cutoff_hz: float,
    order: int,
    order_by_color: bool,
) -> None:
    if not folder.is_dir():
        raise NotADirectoryError(f"{folder} is not a folder.")

    protocol_folders = find_protocol_folders(folder)
    if not protocol_folders:
        raise FileNotFoundError(f"No OptogeneticProtocol CSVs found in {folder}")

    for protocol_folder in protocol_folders:
        process_folder(protocol_folder, apply_filter, cutoff_hz, order, order_by_color)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Plot optogenetic traces around each stim window."
    )
    parser.add_argument(
        "folder",
        nargs="?",
        default=DEFAULT_FOLDER,
        type=Path,
        help="Root folder to search for OptogeneticProtocol CSVs.",
    )
    parser.add_argument(
        "--cutoff-hz",
        type=float,
        default=BESSEL_CUTOFF_HZ,
        help="Bessel low-pass cutoff frequency in Hz.",
    )
    parser.add_argument(
        "--order",
        type=int,
        default=BESSEL_ORDER,
        help="Bessel filter order.",
    )
    parser.add_argument(
        "--no-filter",
        action="store_true",
        help="Disable the Bessel low-pass filter.",
    )
    parser.add_argument(
        "--color-order",
        action="store_true",
        help="Order stim windows by wavelength color gate instead of stim file order.",
    )
    args = parser.parse_args()

    apply_filter = not args.no_filter
    if apply_filter:
        print(f"Filtering: {args.cutoff_hz} Hz, order={args.order} (Bessel, filtfilt)")
    else:
        print("Filtering disabled.")
    if args.color_order:
        print("Ordering traces by color gate.")

    main(args.folder, apply_filter, args.cutoff_hz, args.order, args.color_order)
