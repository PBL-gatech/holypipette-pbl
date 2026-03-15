import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def plot_csv_folder(folder_path):
    """
    Generate plots for each CSV file in a folder and save them as PNG images.
    
    Args:
    folder_path (str or Path): Path to the directory containing CSV files
        to be plotted.
    """
    folder = Path(folder_path)
    out_dir = folder / "plots"
    out_dir.mkdir(exist_ok=True)

    csv_files = list(folder.glob("*.csv"))
    if not csv_files:
        print("No CSV files found in the folder.")
        return

    for csv_file in csv_files:
        try:
            # Load CSV (whitespace-delimited)
            df = pd.read_csv(csv_file, header=None, delim_whitespace=True)
            
            x = df.iloc[:, 0]
            ycols = df.columns[1:]

            for c in ycols:
                plt.figure()
                plt.plot(x, df.iloc[:, c])
                plt.xlabel("Column 0")
                plt.ylabel(f"Column {c}")
                plt.title(f"{csv_file.name} - Column {c}")
                plt.tight_layout()

                out_path = out_dir / f"{csv_file.stem}_col{c}.png"
                plt.savefig(out_path, dpi=150)
                plt.close()
            
            print(f"✅ Plotted {csv_file.name}")

        except Exception as e:
            print(f"❌ Failed to plot {csv_file.name}: {e}")

    print(f"All plots saved in: {out_dir}")

# Example usage:
plot_csv_folder(r"C:\Users\sa-forest\Documents\GitHub\PatcherBot-Agent\experiments\Data\patch_clamp_data\2025_10_19-16_46\VoltageProtocol")
