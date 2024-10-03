import numpy as np
import re
import os

def get_best_model(dir_path):
    pattern = re.compile(r"model_(\d+)\.pth")

    # Initialize the maximum model number
    max_num = -1

    # Traverse through all subdirectories of the parent directory
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            model_path = os.path.join((root,file))
            print(model_path)