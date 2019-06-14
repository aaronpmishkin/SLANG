import os
import math
import torch
import torch.utils.data as data
import numpy as np

class Australian(data.Dataset):

    def __init__(self, root, train=True):
        self.root = os.path.expanduser(root)
        self.train = train  # training set or test set

        self.file_x = os.path.join(self.root, "australian/australian_scale_X.csv")
        self.file_y = os.path.join(self.root, "australian/australian_scale_y.csv")
        X = np.loadtxt(self.file_x)
        y = np.loadtxt(self.file_y)
        self.data, self.labels = torch.FloatTensor(X), torch.LongTensor(y)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        x, y = self.data[index], self.labels[index]

        return x, y

    def __len__(self):
        return len(self.data)
