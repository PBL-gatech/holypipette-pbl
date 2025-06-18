import os
import glob
import pandas as pd
import numpy as np
from pyabf.abfWriter import writeABF1

# --- your conversion constants ---
C_CLAMP_AMP_PER_VOLT   = 400e-12   # 400 pA per DAQ-V (current path)
C_CLAMP_VOLT_PER_VOLT  = (1000e-3)   # 10 mV per DAQ-V (voltage path)

# Folders

csv_folder = r"C:\Users\sa-forest\Documents\GitHub\holypipette-pbl\experiments\Analysis\Patch_Clamp\rowanexample\CurrentProtocol"


out_folder = csv_folder
os.makedirs(out_folder, exist_ok=True)

# 1) find & group CSVs by the text before the first '#'
all_csvs = glob.glob(os.path.join(csv_folder, "*.csv"))
groups = {}
for path in all_csvs:
    prefix = os.path.basename(path).split("#", 1)[0]
    groups.setdefault(prefix, []).append(path)

# 2) process each group
for prefix, paths in groups.items():
    paths.sort()
    raws_current = []
    raws_voltage = []
    times = None
    sample_rate = None

    for p in paths:
        df = pd.read_csv(
            p,
            sep=r"\s+", header=None,
            names=["time_s", "raw_current", "raw_voltage"],
            engine="python"
        )
        if times is None:
            times = df["time_s"].to_numpy()
            dt = times[1] - times[0]
            sample_rate = 1.0 / dt
        else:
            assert len(df) == len(times), f"Length mismatch in {p}"

        raws_current.append(df["raw_current"].to_numpy())
        raws_voltage.append(df["raw_voltage"].to_numpy())

    # scale to physical units
    currents_pa = [r * C_CLAMP_AMP_PER_VOLT * 1e12 for r in raws_current]     # pA
    voltages_mv = [r * C_CLAMP_VOLT_PER_VOLT * 1e3 for r in raws_voltage]     # mV

    # stack into (nSweeps × nPoints)
    I_stack = np.vstack(currents_pa)
    V_stack = np.vstack(voltages_mv)

    # write current ABF1 as *_Command.abf
    cmd_abf = os.path.join(out_folder, f"{prefix.rstrip('_')}_Command.abf")
    writeABF1(I_stack, cmd_abf, sample_rate, units="pA")
    print(f"Wrote {len(I_stack)} sweeps → {os.path.basename(cmd_abf)} (current in pA)")

    # write voltage ABF1 as *_Response.abf
    resp_abf = os.path.join(out_folder, f"{prefix.rstrip('_')}_Response.abf")
    writeABF1(V_stack, resp_abf, sample_rate, units="mV")
    print(f"Wrote {len(V_stack)} sweeps → {os.path.basename(resp_abf)} (voltage in mV)")
