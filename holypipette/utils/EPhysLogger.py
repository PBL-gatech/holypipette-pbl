# import os
# import logging
# from datetime import datetime
# import concurrent.futures
# import threading


# class EPhysLogger:
#     def __init__(self, folder_path="experiments/Data/patch_clamp_data/", ephys_filename="ephys_data.csv"):
#         self.time_truth = datetime.now()
#         self.time_truth_timestamp = self.time_truth.timestamp()
#         self.folder_path = folder_path + self.time_truth.strftime("%Y_%m_%d-%H_%M") + "/"
        
#         # os.makedirs(self.folder_path, exist_ok=True)
#         # self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
#         # self.write_event = threading.Event()
#         # print("EPhysSaver created at: ", self.time_truth_timestamp)

#     def write_ephys_data(self, time_value, type, data):
#         filename = os.path.join(self.folder_path, f"{type}_data.csv")
#         with open(filename, 'a') as file:
#             file.write(f"{time_value},{','.join(map(str, data))}\n")

