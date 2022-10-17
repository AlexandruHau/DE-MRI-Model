import numpy as np

from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, path_to_curve, path_to_params):
        self.curves = np.load(path_to_curve)
        self.params = np.load(path_to_params)
        
    def __getitem__(self, index):
        curve = self.curves[index]
        param = self.params[index]
        return {"X": curve, "Y": param}

    def __len__(self):
        return self.curves.shape[0]

# one np