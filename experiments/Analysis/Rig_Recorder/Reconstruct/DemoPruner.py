import h5py
import numpy as np

file_path = r"C:\Users\sa-forest\Documents\GitHub\holypipette-pbl\experiments\Datasets\HEK_dataset_v0_020.hdf5"
demo_folders_to_delete = ["demo_40"]        # add as many keys as you like

with h5py.File(file_path, "a") as hf:
    data_grp  = hf["data"]
    mask_grp  = hf.get("mask")              # returns None if the group isn’t there

    for demo_key in demo_folders_to_delete:
        # -------- delete the raw demo --------
        if demo_key in data_grp:
            del data_grp[demo_key]
            print(f"✅  deleted data/{demo_key}")
        else:
            print(f"⚠️  data/{demo_key} not found")

        # -------- remove the demo from every mask split --------
        if mask_grp is None:
            continue                         # nothing to clean up

        for split in ("train", "valid"):
            if split not in mask_grp:
                continue

            ds        = mask_grp[split]
            keys_orig = ds[()].astype(str)   # bytes → str for comparison
            keys_new  = [k for k in keys_orig if k != demo_key]

            if len(keys_new) == len(keys_orig):
                continue                     # demo_key wasn’t in this split

            # overwrite the dataset with the filtered list
            del mask_grp[split]
            mask_grp.create_dataset(split, data=np.asarray(keys_new, dtype="S"))
            print(f"✅  removed {demo_key} from mask/{split}")

print("🎉  clean-up complete")
