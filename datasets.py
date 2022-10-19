import numpy as np

from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, curves_file_list, params_file_list):
        self.curves_file_list = read_filelist(curves_file_list)
        self.params_file_list = read_filelist(params_file_list)
        
    def __getitem__(self, index):

        curve_path = self.curves_file_list[index]
        param_path = self.params_file_list[index]

        curve = np.load(curve_path)
        param = np.load(param_path)
        return {"X": curve, "Y": param}

    def __len__(self):
        return len(self.curves_file_list)


def read_filelist(file_list):
    with open(file_list, 'r') as file:
        content = file.readlines()
        filepath_list = [x.strip() for x in content if x.strip()]        
        return filepath_list