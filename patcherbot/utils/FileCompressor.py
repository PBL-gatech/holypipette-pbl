import pandas as pd
import numpy as np
import os

def load_csv_rig(csv_path):
    print("Loading CSV:", csv_path)

    # Always try semicolon first — your rig ALWAYS uses semicolon
    try:
        df = pd.read_csv(csv_path, sep=";", engine="python")
        print("Loaded with semicolon delimiter:", df.shape)
        return df
    except Exception:
        print("Semicolon failed (unexpected). Trying delimiter detection...")

    # Fallback: automatic delimiter detection
    try:
        import csv
        with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
            sample = f.read(4096)
            f.seek(0)
            dialect = csv.Sniffer().sniff(sample)
            df = pd.read_csv(csv_path, sep=dialect.delimiter, engine="python")
            print("Loaded with detected delimiter", dialect.delimiter, df.shape)
            return df
    except Exception:
        print("Delimiter detection failed — trying to skip bad lines...")

    # Last resort: skip bad lines and try semicolon again
    return pd.read_csv(csv_path, sep=";", engine="python", on_bad_lines="skip")


def convert_csv_to_npz(csv_path: str) -> str:
    # Load correctly
    df = load_csv_rig(csv_path)

    # Construct .npz path
    base, _ = os.path.splitext(csv_path)
    npz_path = base + ".npz"

    # Save NPZ
    np.savez_compressed(
        npz_path,
        data=df.to_numpy(),
        columns=df.columns.to_numpy()
    )

    print("Saved NPZ:", npz_path)
    return npz_path


# Example call
csv_path = r"C:\Users\sa-forest\Documents\GitHub\PatcherBot-Agent\experiments\Data\rig_recorder_data\2025_09_25-20_23\graph_recording.csv"
convert_csv_to_npz(csv_path)
