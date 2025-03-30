import numpy as np
import os


def checknan(filepath):
    for file in os.listdir(filepath):
        curve = np.load(os.path.join(filepath, file))
        checknan = np.sum(curve)
        isnan = np.isnan(checknan)
        if(isnan == True):
            print(isnan)
            print(file)