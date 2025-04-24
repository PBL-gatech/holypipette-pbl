import h5py
import numpy as np

from experiments.Analysis.DatasetBuilder import DatasetBuilder



if False:
    # Create a new HDF5 file (overwrites if it exists)
    with h5py.File('my_data.hdf5', 'w') as hf:
        # Create a dataset within the file
        data = np.random.rand(100, 10)
        group = hf.create_group('data')
        group.create_dataset('dataset_1', data=data)


    # Open an existing HDF5 file in read mode
    with h5py.File('my_data.hdf5', 'r') as hf:
        # Access a dataset
        data = hf['data']['dataset_1'][:]  # Read all data into a NumPy array
        print(data)


    # Create a new HDF5 file (overwrites if it exists)
    with h5py.File('experiments/Datasets/dataset_1.hdf5', 'w') as hf:
        # Create a dataset within the file
        group = hf.create_group('data')
        group.attrs['num_demos'] = 0


    with h5py.File('experiments/Datasets/dataset_1.hdf5', 'a') as hf:
        # Create a dataset within the file
        print(hf['data'])

