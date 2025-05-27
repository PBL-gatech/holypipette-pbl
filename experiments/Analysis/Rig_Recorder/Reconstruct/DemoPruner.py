import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

file_path = r"C:\Users\sa-forest\Documents\GitHub\holypipette-pbl\experiments\Datasets\HEK_dataset.hdf5"


# # List demo folders to delete
demo_folders_to_delete =  [
'demo_11',
'demo_14',
'demo_28',
'demo_29',
'demo_30',
'demo_31',
'demo_33',
'demo_36',
'demo_37',
'demo_38',
'demo_4',
'demo_49',
'demo_50',
'demo_52',
'demo_53',
'demo_58',
'demo_60',
'demo_65',
'demo_66',
'demo_67',
'demo_70',
'demo_71',
'demo_79',
'demo_8',
'demo_80',
'demo_84',
'demo_86',
'demo_87',
'demo_89',
'demo_90'
]
demo_folders_to_delete = [
    'demo_0'
]

with h5py.File(file_path, 'a') as hdf:  # 'a' mode is required for modifications
    data_group = hdf['data']
    for demo_key in demo_folders_to_delete:
        if demo_key in data_group:
            del data_group[demo_key]
            print(f"Deleted {demo_key}")
        else:
            print(f"{demo_key} not found")
