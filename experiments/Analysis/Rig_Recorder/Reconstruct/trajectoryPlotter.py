import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

file_path = r"C:\Users\sa-forest\Documents\GitHub\holypipette-pbl\experiments\Datasets\HEK_dataset_fullStage.hdf5"

pipette_positions_data = {}
resistance_data = {}
stage_positions_data = {}

with h5py.File(file_path, 'r') as hdf:
    for demo_key in hdf['data'].keys():
        demo_group = hdf['data'][demo_key]['obs']
        if 'pipette_positions' in demo_group:
            pipette_positions = demo_group['pipette_positions'][:]
            pipette_positions_data[demo_key] = pipette_positions
        if 'resistance' in demo_group:
            resistance = demo_group['resistance'][:]
            resistance_data[demo_key] = resistance
        if 'stage_positions' in demo_group:
            stage_positions = demo_group['stage_positions'][:]
            stage_positions_data[demo_key] = stage_positions

# Plot pipette positions
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.grid(False)
cmap = cm.viridis
for positions in pipette_positions_data.values():
    time_steps = np.arange(positions.shape[0])
    colors = cmap(time_steps / positions.shape[0])
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c=colors, marker='o', s=15)
ax.set_xlim([-20, 20])
ax.set_ylim([-20, 20])
ax.set_zlim([30, 0])
ax.set_title('3D Pipette Motion with Time-Colorimetric Encoding')
ax.set_xlabel('X position')
ax.set_ylabel('Y position')
ax.set_zlabel('Z position (top-down)')
mappable = cm.ScalarMappable(cmap=cmap)
mappable.set_array([])
cbar = plt.colorbar(mappable, ax=ax, pad=0.1, shrink=0.6)
cbar.set_label('Time steps')
cbar.ax.invert_yaxis()  # Flip color legend to start from the top

# Plot resistance change
fig, ax = plt.subplots(figsize=(12, 4))
for resistance in resistance_data.values():
    ax.plot(resistance)
ax.set_title('Resistance Change over Time')
ax.set_xlabel('Time steps')
ax.set_ylabel('Resistance')

# to be done 

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.grid(False)
cmap = cm.viridis
for positions in stage_positions_data.values():
    time_steps = np.arange(positions.shape[0])
    colors = cmap(time_steps / positions.shape[0])
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c=colors, marker='o', s=15)
ax.set_xlim([-20, 20])
ax.set_ylim([-20, 20])
ax.set_zlim([30, 0])
ax.set_title('3D stage Motion with Time-Colorimetric Encoding')
ax.set_xlabel('X position')
ax.set_ylabel('Y position')
ax.set_zlabel('Z position (top-down)')
mappable = cm.ScalarMappable(cmap=cmap)
mappable.set_array([])
cbar = plt.colorbar(mappable, ax=ax, pad=0.1, shrink=0.6)
cbar.set_label('Time steps')
cbar.ax.invert_yaxis()  # Flip color legend to start from the top


# # Plot stage positions
# fig, ax = plt.subplots(figsize=(12, 4))
# for stage_positions in stage_positions_data.values():
#     ax.plot(stage_positions)
# ax.set_title('Stage Positions over Time')
# ax.set_xlabel('Time steps')
# ax.set_ylabel('Stage Positions')

plt.show()

# import h5py

# # # file_path = 'initial_train_dataset.hdf5'

# # # List demo folders to delete
# demo_folders_to_delete =  [
#     'demo_38', 'demo_41', 'demo_43', 'demo_49', 'demo_52', 'demo_55', 
#     'demo_57', 'demo_59', 'demo_60', 'demo_64', 'demo_66', 'demo_68', 
#     'demo_73', 'demo_76', 'demo_78', 'demo_79'
# ]


# with h5py.File(file_path, 'a') as hdf:  # 'a' mode is required for modifications
#     data_group = hdf['data']
#     for demo_key in demo_folders_to_delete:
#         if demo_key in data_group:
#             del data_group[demo_key]
#             print(f"Deleted {demo_key}")
#         else:
#             print(f"{demo_key} not found")
