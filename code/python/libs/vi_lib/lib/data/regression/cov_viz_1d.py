import os
import torch
import torch.utils.data as data
import numpy as np
import re

datafile_for_dataset = {
    'cov_viz_1d':'1d-cov-viz-dataset.csv',
    'cov_viz_1d_outlier':'1d-cov-viz-dataset_outlier.csv',
    'cov_viz_1d_gap':'1d-cov-viz-dataset_gap.csv',
}

class CovViz1D(data.Dataset):

    def __init__(self, root, data_set):
        self.data_folder = os.path.expanduser(root)
        self.data_file = os.path.join(self.data_folder, datafile_for_dataset[data_set])
        data = np.loadtxt(self.data_file, delimiter=",")

        x = data[:,0].reshape(-1,1)
        y = data[:,1].reshape(-1,)
        
        self.train_data, self.train_labels = torch.FloatTensor(x), torch.FloatTensor(y)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        return self.train_data[index], self.train_labels[index]

    def __len__(self):
        return len(self.train_data)
