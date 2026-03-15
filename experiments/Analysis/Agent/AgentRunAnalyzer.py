#!/usr/bin/env python3
"""
Summarize patch-attempt method timings with a hardcoded parent directory.
Adds 'outcome' and 'mode' columns to the table.

Output: attempt_summary.csv in the parent directory with columns:
attempt, method, start_utc, end_utc, duration_s, within_patch_interval,
overlap_with_patch_s, outcome, mode
"""

import json
import csv
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Dict, Tuple, Optional

# 🔧 CHANGE THIS to your actual parent folder
# PARENT_DIR = Path(r"C:\Users\sa-forest\OneDrive - Georgia Institute of Technology\Documents\Grad-school\Gatech\Fall2025\ForestLab\ML\PatcherBotAgent10runData").resolve()
PARENT_DIR = Path(r"C:\Users\sa-forest\OneDrive - Georgia Institute of Technology\Documents\Grad-school\Gatech\Fall2025\ForestLab\ML\PatcherBotAgentHEKData_v0_001\2025_11_02-19_54").resolve()
# PARENT_DIR = Path(r"C:\Users\sa-forest\OneDrive - Georgia Institute of Technology\Documents\Grad-school\Gatech\Fall2025\ForestLab\ML\PatcherBotAgentHEKData_v0_001\2025_11_02-21_05").resolve()

FILENAME_METHOD_HINTS = {
    "patch": "patch",
    "locate_cell": "locate_cell",
    "hunt_cell": "hunt_cell",
    "gigaseal": "gigaseal",
    "break_in": "break_in",
    "breakin": "break_in",
    "escape": "escape",
}

@dataclass
class MethodRecord:
    """
    Represents a recorded method attempt with timing and outcome metadata.

    Attributes:
        method (str): Name of the method executed.
        start (float): Start timestamp (seconds since epoch).
        end (float); End timestamp (seconds since epoch).
        outcome (Optional[str]): Outcome label for the method.
    """
    method: str
    start: float
    end: float
    outcome: Optional[str] = None
    mode: Optional[str] = None

    @property
    def duration_s(self) -> float:
        """
        Return the duration of the method attempt in seconds.

        Returns:
            float: Duration computed as end - start.
        """
        return float(self.end - self.start)

    @property
    def start_utc(self) -> str:
        """
        Return the start timestamp as a UTC ISO-8601 string.

        Returns:
            str: Start time formatted in UTC.
        """
        return datetime.fromtimestamp(self.start, tz=timezone.utc).isoformat()

    @property
    def end_utc(self) -> str:
        """
        Returns the start timestamp as a UTC ISO-8601 string.

        Returns:
            str: End time formatted in UTC.
        """
        return datetime.fromtimestamp(self.end, tz=timezone.utc).isoformat()


def infer_method_name_from_filename(p: Path) -> Optional[str]:
    """
    Infer the method name from the filename using hints.
    
    Args:
        p (Path): Path to the file whose name will be inspected.

    Returns:
        Optional[str]: The canonical method names if a matching hint
            is found; otherwise None.
    """
    name = p.name.lower()
    for hint, canonical in FILENAME_METHOD_HINTS.items():
        if hint in name:
            return canonical
    return None


def read_json_fields(p: Path) -> Optional[Tuple[float, float, Optional[str], Optional[str]]]:
    """
    Read a JSON file and return (started, finished, outcome, mode).
    If any required field is missing, None is returned.

    Args:
        p (Path): Path to the JSON file to read.

    Returns:
        Optional[Tuple[float, float, Optional[str], Optional[str]]]:
            A tuple containing:
            - start (float): Start timestamp.
            - end (float): End timestamp.
            - outcome (Optional[str]): Outcome label if present.
            - mode (Optional[str]): Mode label if present.
        None if file cannot be read ot required fields are missing.
    """
    try:
        with p.open("r") as f:
            d = json.load(f)
        start = float(d["started"])
        end = float(d["finished"])
        outcome = d.get("outcome")  # e.g., "success", "fail", etc.
        mode = d.get("mode")        # e.g., "auto", "manual", etc.
        return start, end, outcome, mode
    except Exception as e:
        print(f"[WARN] Skipping {p}: {e}")
        return None


def choose_patch_window(records: List[MethodRecord]) -> Optional[Tuple[float, float]]:
    """
    Among records whose method == 'patch', choose the patch interval.
    If multiple exist, pick the widest window (max duration).

    Args:
        records (List[MethodRecord]): List of method records to search.
    
    Returns:
        Optional[Tuple[float, float]]: A tuple containing the start and end timestamps
            of the selected patch interval.
        None if no patch records are present.
    """
    patch_records = [r for r in records if r.method == "patch"]
    if not patch_records:
        return None
    patch_records.sort(key=lambda r: r.duration_s, reverse=True)
    return (patch_records[0].start, patch_records[0].end)


def summarize_attempt(attempt_dir: Path) -> List[Dict[str, object]]:
    """
    Build rows for a single attempt directory.
    Returns a list of dict rows ready for CSV.

    Args:
        attempt_dir (Path): Directory containing JSOn files for a single attempt.
    
    Returns:
        List[Dict[str, object]]: A list of row dictionaries ready for CSV output.
        Empty list if no valid records are found.
    """
    records: List[MethodRecord] = []

    for p in attempt_dir.glob("*.json"):
        method = infer_method_name_from_filename(p)
        if not method:
            # If we can't infer, attempt a best-guess: token after first underscore (optional)
            parts = p.stem.split("_")
            if len(parts) >= 2:
                method = parts[1].lower()
            else:
                continue

        vals = read_json_fields(p)
        if not vals:
            continue
        start, end, outcome, mode = vals
        records.append(MethodRecord(method=method, start=start, end=end, outcome=outcome, mode=mode))

    if not records:
        return []

    # Determine patch window (if present)
    patch_window = choose_patch_window(records)

    rows = []
    for rec in records:
        within_patch = False
        overlap = 0.0
        if patch_window:
            p_start, p_end = patch_window
            within_patch = (p_start <= rec.start) and (rec.end <= p_end)
            overlap = max(0.0, min(p_end, rec.end) - max(p_start, rec.start))

        rows.append({
            "attempt": attempt_dir.name,
            "method": rec.method,
            "start_utc": rec.start_utc,
            "end_utc": rec.end_utc,
            "duration_s": round(rec.duration_s, 6),
            "within_patch_interval": within_patch,
            "overlap_with_patch_s": round(overlap, 6),
            "outcome": rec.outcome if rec.outcome is not None else "",
            "mode": rec.mode if rec.mode is not None else "",
        })

    # Sort methods by start time for readability
    return sorted(rows, key=lambda r: (r["attempt"], r["start_utc"]))


def main():
    """
    Aggregate method records from attempt directories and writes a summary CSV.

    Scans 'PARENT_DIR' for subdirectories named with the prefix "attempt_". For
    each attempt directory, it extracts method records using 'summarize_attempt',
    aggregates the results, and writes them to "attempt_summary.csv" in the 
    parent directory.

    If the parent directory is invalid, no attempt directories are found, or no
    records are extracted, the function prints a message and exits without creating
    a CSV.
    """
    if not PARENT_DIR.exists() or not PARENT_DIR.is_dir():
        print(f"Error: '{PARENT_DIR}' is not a directory.")
        return

    attempt_dirs = sorted([d for d in PARENT_DIR.iterdir() if d.is_dir() and d.name.lower().startswith("attempt_")])
    if not attempt_dirs:
        print(f"No attempt_* directories found under {PARENT_DIR}")
        return

    all_rows: List[Dict[str, object]] = []
    for attempt_dir in attempt_dirs:
        rows = summarize_attempt(attempt_dir)
        if not rows:
            print(f"[WARN] No valid JSONs found in {attempt_dir}")
        all_rows.extend(rows)

    if not all_rows:
        print("No data extracted.")
        return

    out_csv = PARENT_DIR / "attempt_summary.csv"
    fieldnames = [
        "attempt",
        "method",
        "start_utc",
        "end_utc",
        "duration_s",
        "within_patch_interval",
        "overlap_with_patch_s",
        "outcome",
        "mode",
    ]
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"✅ Wrote {len(all_rows)} rows to {out_csv}")
    for attempt in sorted(set(r["attempt"] for r in all_rows)):
        count = sum(1 for r in all_rows if r["attempt"] == attempt)
        print(f"  {attempt}: {count} rows")


if __name__ == "__main__":
    main()
