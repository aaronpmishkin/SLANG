import os
import torch
import torch.utils.data as data
import numpy as np
from sklearn.datasets import load_svmlight_file

class Axa(data.Dataset):

    def __init__(self, root, data_set = 'a1a', train=True):
        self.root = os.path.expanduser(root)
        self.train = train  # training set or test set

        if self.train:
            self.training_file = os.path.join(self.root, "axa", data_set,  data_set)
            X_train, y_train = load_svmlight_file(self.training_file, n_features=123)
            X_train = np.array(X_train.todense())
            y_train = (y_train+1)/2
            self.train_data, self.train_labels = torch.FloatTensor(X_train), torch.LongTensor(y_train)
        else:
            self.test_file = os.path.join(self.root, "axa", data_set,  data_set + ".t")
            X_test, y_test = load_svmlight_file(self.test_file, n_features=123)
            X_test = np.array(X_test.todense())
            y_test = (y_test+1)/2
            self.test_data, self.test_labels = torch.FloatTensor(X_test), torch.LongTensor(y_test)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            x, y = self.train_data[index], self.train_labels[index]
        else:
            x, y = self.test_data[index], self.test_labels[index]

        return x, y

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)
