import json
import os
from os.path import abspath, dirname
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from pr3d.de import ConditionalGaussianMixtureEVM, ConditionalGaussianMM, GaussianMM, GaussianMixtureEVM, GammaMixtureEVM


folder_addr = "ep5g-1/gmm/"
src_model_name = "model_0.h5"
src_info_name = "info.json"
dest_name = 'model_0_parameters.json'

delay_model = GaussianMM(h5_addr=folder_addr + src_model_name)
params_dict = delay_model.get_parameters()
for key in params_dict.keys():
    params_dict[key] = np.array(params_dict[key]).tolist()

print(f"model is loaded. Parameters: {params_dict}")

# Save dictionary to JSON file
with open(folder_addr + src_info_name, 'r') as f:
    info_dict = json.load(f)

mean = float(info_dict["mean"])
scale = float(info_dict["scale"])

print(f"info is loaded. mean: {mean}, scale: {scale}")

params_dict["mixture_locations"] = (np.array(params_dict["mixture_locations"]) + mean*scale).tolist()

# Save dictionary to JSON file
with open(folder_addr + dest_name, 'w') as f:
    json.dump(params_dict, f)



