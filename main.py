import subprocess
from multiprocessing import Process
import os
import numpy as np
from numba import cuda
from Factor_Analyse import factor_analysis_main  # Import function

def work(device, all_stock_data, all_index_data, unique_stock_codes):
    print(f"Running on device {device}")
    cuda.select_device(device)
    # Now you can call your function
    results = factor_analysis_main()
    # You might want to save the results to a file,
    # or use some other method to collect the results
    np.save(f'results_device_{device}.npy', results)


# The device ids for 7 RTX 3090 GPUs might be 0, 1, 2, 3, 4, 5, 6
devices = [6]

# Load your data
all_stock_data_np = np.load('all_stock_data_np.npy', allow_pickle=True)
all_index_data_np = np.load('all_index_data_np.npy', allow_pickle=True)
unique_stock_codes = np.load('20230512_unique_stock_codes.npy', allow_pickle=True)
print("Data successfully loaded.")

processes = []
for device in devices:
    p = Process(target=work, args=(device, all_stock_data_np, all_index_data_np, unique_stock_codes))
    p.start()
    processes.append(p)

for p in processes:
    p.join()

results = []

for device in devices:
    result = np.load(f'results_device_{device}.npy', allow_pickle=True)
    results.append(result)

# Now `results` is a list of the result arrays from each GPU
# Combine all results into a single array
combined_results = np.array(results)


# If you're running on only one GPU, there will be only one set of results
# So you can just take the first element of the list
print(combined_results)
np.save('20230512_combined_results.npy', combined_results)
