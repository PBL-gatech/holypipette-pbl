# Build counts of positive pipette actions along X, Y, and Z, then visualise in 3‑D
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

file_path = r"C:\\Users\\sa-forest\\Documents\\GitHub\\holypipette-pbl\\experiments\\Datasets\\HEK_dataset.hdf5"

# ------------------------------------------------------------------
#  ❱❱  Count positive action components per demonstration  ❰❰
# ------------------------------------------------------------------
action_counts = {}
with h5py.File(file_path, 'r') as hdf:
    for demo_key in hdf['data'].keys():
        demo_group = hdf['data'][demo_key]

        # Fetch the action dataset (shape: T × 3). Adjust the key if needed.
        if 'action' in demo_group:
            actions = demo_group['action'][:]
        else:
            # Fallback: finite‑difference of pipette positions as proxy actions
            pos = demo_group['obs']['pipette_positions'][:]
            actions = np.diff(pos, axis=0, prepend=pos[:1])

        # Count strictly positive moves along each axis
        action_counts[demo_key] = (actions > 0).sum(axis=0)

# ------------------------------------------------------------------
#  ❱❱  3‑D scatter of (count⁺X, count⁺Y, count⁺Z)  ❰❰
# ------------------------------------------------------------------
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

labels = list(action_counts.keys())
counts = np.vstack(list(action_counts.values()))  # shape (N, 3)

scatter = ax.scatter(counts[:, 0], counts[:, 1], counts[:, 2],
                     c=counts.sum(axis=1), cmap='viridis', s=45, depthshade=True)

for idx, demo in enumerate(labels):
    ax.text(counts[idx, 0], counts[idx, 1], counts[idx, 2], demo, size=8)

ax.set_xlabel('# +X actions')
ax.set_ylabel('# +Y actions')
ax.set_zlabel('# +Z actions')
ax.set_title('Positive Pipette Actions per Demonstration')
ax.grid(False)

cbar = plt.colorbar(scatter, pad=0.1)
cbar.set_label('Total positive actions')

plt.tight_layout()
plt.show()
